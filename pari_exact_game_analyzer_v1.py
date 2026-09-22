from __future__ import annotations

import json
import math
import sqlite3
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

try:
    from curl_cffi import requests as http_requests  # type: ignore
    HTTP_BACKEND = "curl_cffi"
except Exception:
    import requests as http_requests  # type: ignore
    HTTP_BACKEND = "requests"

try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass


LIVE_EVENTS_URL = "https://line-lb51-a.pb06e2-resources.com/line/liveEvents"
ACTRANS_URL = "https://line-lb01-w.pb06e2-resources.com/sportscast/actrans"
SPORTCAST_EVENTS_URL = "https://line-lb01-w.pb06e2-resources.com/sportscast/events"
EVENT_DETAILS_URL = "https://line-lb51-w.pb06e2-resources.com/events/event"
EVENT_DETAILS_IP_URL = "https://212.41.30.103/events/event"

MOBILE_UA = "Paribet/6.119.2-pru-i-r (Android 28; Phone; ru.paribet; Google; google Pixel 2)"
WEBVIEW_UA = (
    "Mozilla/5.0 (Linux; Android 9; google Pixel 2 Build/LMY47I; wv) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 "
    "Chrome/138.0.7204.179 Mobile Safari/537.36"
)

# ========================= НАСТРОЙКИ =========================
RUN_MODE = "live"             # "live" или "self-test"
INTERVAL = 2.0                # секунд между циклами
TIMEOUT = 8.0
OUTPUT = "exact_game_live.json"
DB = "pari_exact_game_stats.sqlite3"
EVENT_API_VERSION = "76749944487"
MISSING_CYCLES_TO_UNKNOWN = 15
STALE_HOURS = 12.0
PROCESSED_GAME_RETENTION_HOURS = 48.0
HISTORY_LIMIT = 50000
UNKNOWN_SAMPLE_LIMIT = 30
UNKNOWN_SAMPLES_FILE = "exact_game_unknown_samples.json"

# Честный режим: прогноз создаётся только в начале гейма при 0:0.
# Если скрипт впервые увидел гейм уже на 15:0/0:15 и т.п., этот гейм в статистику НЕ идёт.
SIGNAL_ONLY_AT_ZERO_ZERO = True
# ============================================================

OUTCOMES = ("40:0", "40:15", "40:30", "+:40", "0:40", "15:40", "30:40", "40:+")

POINT_RANK = {0: 0, 15: 1, 30: 2, 40: 3}
DEFAULT_DEUCE_RATE = 0.285
DEFAULT_SERVER_POINT_P = 0.62
GLOBAL_DEUCE_PRIOR_STRENGTH = 30.0
PLAYER_DEUCE_PRIOR_STRENGTH = 18.0
GLOBAL_POINT_PRIOR_STRENGTH = 80.0
PLAYER_POINT_PRIOR_STRENGTH = 45.0


@dataclass
class PlayerProfile:
    player: str
    service_games: int = 0
    deuce_games: int = 0
    service_points: int = 0
    points_won: int = 0


@dataclass
class CompletedGame:
    event_id: str
    set_num: int
    game_num: int
    server: str
    deuce: bool
    service_points: int
    points_won: int
    exact_outcome: str | None


@dataclass
class ParsedMatch:
    event_id: str
    player1: str
    player2: str
    first_server_match: int
    current_set: int
    current_game: int
    raw_score_a: Any
    raw_score_b: Any
    completed_games: list[CompletedGame]
    max_game_by_set: dict[int, int]
    events: list[dict[str, Any]]


@dataclass
class ExactPrediction:
    event_id: str
    player1: str
    player2: str
    set_num: int
    game_num: int
    server: str
    current_score: str
    p_server: float
    p_player1_point: float
    probabilities: dict[str, float]
    top1: str
    top1_probability: float
    top2: list[str]
    data_quality: str
    match_link: str = "https://pari.ru/"
    created_at: float = 0.0


def _int(v: Any, default: int | None = None) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def data_quality_label(service_games: int, service_points: int) -> str:
    evidence = service_games + service_points / 12.0
    if evidence >= 35:
        return "много данных"
    if evidence >= 12:
        return "достаточно данных"
    return "мало данных"


def deuce_rate_from_point_p(p: float) -> float:
    q = 1.0 - p
    return 20.0 * (p ** 3) * (q ** 3)


def point_p_from_deuce_rate(target_rate: float) -> float:
    target_rate = clamp(target_rate, 0.001, deuce_rate_from_point_p(0.5))
    lo, hi = 0.5, 0.90
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if deuce_rate_from_point_p(mid) > target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def get_server_for_game(
    set_num: int,
    game_num: int,
    first_server_match: int,
    max_game_by_set: dict[int, int],
) -> int:
    if set_num < 1 or game_num < 1 or first_server_match not in (1, 2):
        raise ValueError("Некорректные данные подачи")
    games_before = sum(max(0, int(max_game_by_set.get(s, 0))) for s in range(1, set_num))
    absolute_game = games_before + game_num
    return first_server_match if absolute_game % 2 == 1 else 3 - first_server_match


def exact_game_distribution(p1: float) -> dict[str, float]:
    """Распределение точного финала стандартного теннисного гейма.

    p1 = вероятность, что первое имя матча выигрывает отдельный розыгрыш.
    +:40 = первый игрок выигрывает гейм после деуса.
    40:+ = второй игрок выигрывает гейм после деуса.
    """
    p = clamp(float(p1), 0.001, 0.999)
    q = 1.0 - p
    reach_deuce = 20.0 * (p ** 3) * (q ** 3)
    denom = p * p + q * q
    p1_after_deuce = (p * p / denom) if denom else 0.5

    d = {
        "40:0": p ** 4,
        "40:15": 4.0 * (p ** 4) * q,
        "40:30": 10.0 * (p ** 4) * (q ** 2),
        "+:40": reach_deuce * p1_after_deuce,
        "0:40": q ** 4,
        "15:40": 4.0 * (q ** 4) * p,
        "30:40": 10.0 * (q ** 4) * (p ** 2),
        "40:+": reach_deuce * (1.0 - p1_after_deuce),
    }
    total = sum(d.values()) or 1.0
    return {k: v / total for k, v in d.items()}


def _normalize_score_token(v: Any) -> int | str | None:
    if isinstance(v, str):
        s = v.strip().upper()
        if s in {"+", "A", "AD", "ADV", "ADVANTAGE"}:
            return "ADV"
        try:
            v = int(s)
        except ValueError:
            return None
    try:
        i = int(v)
    except (TypeError, ValueError):
        return None
    if i in POINT_RANK:
        return i
    # Некоторые фиды кодируют advantage числом 50.
    if i in {45, 50}:
        return "ADV"
    return None


def _raw_score_states_for_game(events: list[dict[str, Any]], set_num: int, game_num: int) -> list[tuple[Any, Any]]:
    states: list[tuple[Any, Any]] = []
    for ev in events:
        if _int(ev.get("i1")) != set_num or _int(ev.get("i2")) != game_num:
            continue
        if "i8" not in ev or "i9" not in ev:
            continue
        a = _normalize_score_token(ev.get("i8"))
        b = _normalize_score_token(ev.get("i9"))
        if a is None or b is None:
            continue
        st = (a, b)
        if not states or states[-1] != st:
            states.append(st)
    return states


def _numeric_states_for_game(events: list[dict[str, Any]], set_num: int, game_num: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for a, b in _raw_score_states_for_game(events, set_num, game_num):
        if isinstance(a, int) and isinstance(b, int) and a in POINT_RANK and b in POINT_RANK:
            if not out or out[-1] != (a, b):
                out.append((a, b))
    return out


def _service_point_stats_from_states(states: list[tuple[int, int]], server_num: int) -> tuple[int, int]:
    total = 0
    won = 0
    for (a1, b1), (a2, b2) in zip(states, states[1:]):
        ra1, rb1 = POINT_RANK[a1], POINT_RANK[b1]
        ra2, rb2 = POINT_RANK[a2], POINT_RANK[b2]
        winner: int | None = None
        if ra2 == ra1 + 1 and rb2 == rb1:
            winner = 1
        elif rb2 == rb1 + 1 and ra2 == ra1:
            winner = 2
        if winner is None:
            continue
        total += 1
        if winner == server_num:
            won += 1
    return total, won


def exact_outcome_from_states(states: list[tuple[Any, Any]]) -> str | None:
    """Определяет рынок 'Как закончится гейм' по последнему состоянию перед окончанием.

    Для гейма без деуса последняя точка перед победным розыгрышем должна быть
    40:0 / 40:15 / 40:30 или зеркальная.

    Для гейма после деуса пробуем определить последнюю advantage-позицию.
    Если фид не отдаёт advantage — честно возвращаем None, а не придумываем победителя.
    """
    if not states:
        return None
    deuce_seen = any(a == 40 and b == 40 for a, b in states)

    if not deuce_seen:
        last = states[-1]
        mapping = {
            (40, 0): "40:0",
            (40, 15): "40:15",
            (40, 30): "40:30",
            (0, 40): "0:40",
            (15, 40): "15:40",
            (30, 40): "30:40",
        }
        return mapping.get(last)

    # После последнего деуса ищем последнее наблюдаемое преимущество.
    last_deuce_idx = max(i for i, st in enumerate(states) if st == (40, 40))
    after = states[last_deuce_idx + 1:]
    for a, b in reversed(after):
        if a == "ADV" and b == 40:
            return "+:40"
        if a == 40 and b == "ADV":
            return "40:+"
    return None


def parse_match(event_id: int | str, player1: str, player2: str, events: list[dict[str, Any]]) -> ParsedMatch | None:
    first_server_match: int | None = None
    current_set: int | None = None
    current_game: int | None = None
    latest_score_by_game: dict[tuple[int, int], tuple[Any, Any]] = {}
    max_game_by_set: dict[int, int] = {}

    for ev in events:
        typ = _int(ev.get("type"))
        s = _int(ev.get("i1"))
        g = _int(ev.get("i2"))
        if typ == 1123 and first_server_match is None:
            fs = _int(ev.get("i3"))
            if fs in (1, 2):
                first_server_match = fs
        if s and g:
            max_game_by_set[s] = max(max_game_by_set.get(s, 0), g)
        if typ == 1125 and s and g:
            current_set, current_game = s, g
        if s and g and "i8" in ev and "i9" in ev:
            latest_score_by_game[(s, g)] = (ev.get("i8"), ev.get("i9"))

    if not (current_set and current_game and first_server_match in (1, 2)):
        return None

    completed_keys: set[tuple[int, int]] = set()
    for ev in events:
        s = _int(ev.get("i1"))
        g = _int(ev.get("i2"))
        if s and g and (s, g) < (current_set, current_game):
            completed_keys.add((s, g))

    completed: list[CompletedGame] = []
    for s, g in sorted(completed_keys):
        try:
            server_num = get_server_for_game(s, g, first_server_match, max_game_by_set)
        except ValueError:
            continue
        server = player1 if server_num == 1 else player2
        numeric_states = _numeric_states_for_game(events, s, g)
        service_points, points_won = _service_point_stats_from_states(numeric_states, server_num)
        raw_states = _raw_score_states_for_game(events, s, g)
        outcome = exact_outcome_from_states(raw_states)
        deuce = any(a == 40 and b == 40 for a, b in raw_states)
        completed.append(CompletedGame(
            event_id=str(event_id), set_num=s, game_num=g, server=server, deuce=deuce,
            service_points=service_points, points_won=points_won, exact_outcome=outcome,
        ))

    raw_a, raw_b = latest_score_by_game.get((current_set, current_game), (None, None))
    return ParsedMatch(
        event_id=str(event_id), player1=player1, player2=player2,
        first_server_match=first_server_match, current_set=current_set, current_game=current_game,
        raw_score_a=raw_a, raw_score_b=raw_b, completed_games=completed,
        max_game_by_set=max_game_by_set, events=events,
    )


class PariClient:
    def __init__(self, timeout: float = 8.0):
        self.timeout = timeout
        self.session = http_requests.Session()
        self.link_cache: dict[str, str] = {}
        self.link_retry_after: dict[str, float] = {}

    @staticmethod
    def _headers_mobile() -> dict[str, str]:
        return {"User-Agent": MOBILE_UA}

    @staticmethod
    def _headers_webview() -> dict[str, str]:
        return {
            "User-Agent": WEBVIEW_UA,
            "Accept": "*/*",
            "Origin": "https://pari.ru",
            "X-Requested-With": "ru.paribet",
            "Referer": "https://pari.ru/",
            "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        }

    @staticmethod
    def _headers_event_details(use_ip: bool = False) -> dict[str, str]:
        h = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:150.0) Gecko/20100101 Firefox/150.0",
            "Accept": "*/*",
            "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
            "Referer": "https://pari.ru/",
            "Origin": "https://pari.ru",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "cross-site",
        }
        if use_ip:
            h["Host"] = "line-lb51-w.pb06e2-resources.com"
        return h

    def _get_json(self, url: str, *, params: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        kwargs: dict[str, Any] = dict(params=params, headers=headers, verify=False, timeout=self.timeout)
        if HTTP_BACKEND == "curl_cffi":
            kwargs["impersonate"] = "chrome"
        r = self.session.get(url, **kwargs)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, dict):
            raise ValueError(f"Ожидался JSON-объект от {url}")
        return data

    def live_events(self) -> list[dict[str, Any]]:
        data = self._get_json(
            LIVE_EVENTS_URL,
            params={"lang": "ru", "scopeMarket": "2300"},
            headers=self._headers_mobile(),
        )
        return data.get("eventMiscs", []) or []

    def actrans(self, event_id: int | str) -> list[dict[str, Any]]:
        data = self._get_json(
            ACTRANS_URL,
            params={"fonid": str(event_id)},
            headers=self._headers_webview(),
        )
        return data.get("items", []) or []

    def sportscast_events(self, code: Any) -> list[dict[str, Any]]:
        data = self._get_json(
            SPORTCAST_EVENTS_URL,
            params={"code": str(code), "lastid": "0"},
            headers=self._headers_webview(),
        )
        return data.get("events", []) or []

    def event_link(self, event_id: int | str) -> str:
        key = str(event_id)
        if key in self.link_cache:
            return self.link_cache[key]
        if time.time() < self.link_retry_after.get(key, 0.0):
            return "https://pari.ru/"
        params = {"lang": "ru", "version": EVENT_API_VERSION, "eventId": key, "scopeMarket": "2300"}
        data: dict[str, Any] | None = None
        try:
            data = self._get_json(EVENT_DETAILS_URL, params=params, headers=self._headers_event_details(False))
        except Exception:
            try:
                data = self._get_json(EVENT_DETAILS_IP_URL, params=params, headers=self._headers_event_details(True))
            except Exception:
                data = None
        link = "https://pari.ru/"
        if data:
            for ev in data.get("events", []) or []:
                sport_id = ev.get("sportId")
                parent_id = ev.get("parentId")
                if sport_id and parent_id:
                    link = f"https://pari.ru/sports/tennis/{sport_id}/{parent_id}"
                    break
        if link != "https://pari.ru/":
            self.link_cache[key] = link
            self.link_retry_after.pop(key, None)
        else:
            self.link_retry_after[key] = time.time() + 30.0
        return link


class ExactGameTracker:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.missing_cycles: dict[str, int] = {}
        self._last_maintenance = 0.0
        self._init_db()

    def _init_db(self) -> None:
        prob_cols = ",\n".join(f"p_{i} REAL NOT NULL" for i in range(8))
        self.conn.executescript(f"""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;

            CREATE TABLE IF NOT EXISTS pending_predictions (
                key TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                player1 TEXT NOT NULL,
                player2 TEXT NOT NULL,
                set_num INTEGER NOT NULL,
                game_num INTEGER NOT NULL,
                server TEXT NOT NULL,
                p_server REAL NOT NULL,
                p_player1 REAL NOT NULL,
                top1 TEXT NOT NULL,
                top2 TEXT NOT NULL,
                data_quality TEXT NOT NULL,
                match_link TEXT NOT NULL,
                {prob_cols},
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_exact_pending_event ON pending_predictions(event_id);

            CREATE TABLE IF NOT EXISTS signal_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                event_id TEXT NOT NULL,
                player1 TEXT NOT NULL,
                player2 TEXT NOT NULL,
                set_num INTEGER NOT NULL,
                game_num INTEGER NOT NULL,
                server TEXT NOT NULL,
                p_server REAL NOT NULL,
                p_player1 REAL NOT NULL,
                top1 TEXT NOT NULL,
                top2 TEXT NOT NULL,
                data_quality TEXT NOT NULL,
                {prob_cols},
                actual_outcome TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                top1_hit INTEGER,
                top2_hit INTEGER,
                brier REAL,
                log_loss REAL,
                created_at REAL NOT NULL,
                resolved_at REAL
            );
            CREATE INDEX IF NOT EXISTS idx_exact_history_status ON signal_history(status);

            CREATE TABLE IF NOT EXISTS processed_games (
                event_id TEXT NOT NULL,
                set_num INTEGER NOT NULL,
                game_num INTEGER NOT NULL,
                processed_at REAL NOT NULL,
                PRIMARY KEY(event_id,set_num,game_num)
            );

            CREATE TABLE IF NOT EXISTS player_profiles (
                player TEXT PRIMARY KEY,
                service_games INTEGER NOT NULL DEFAULT 0,
                deuce_games INTEGER NOT NULL DEFAULT 0,
                service_points INTEGER NOT NULL DEFAULT 0,
                points_won INTEGER NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS global_profile (
                id INTEGER PRIMARY KEY CHECK(id=1),
                service_games INTEGER NOT NULL DEFAULT 0,
                deuce_games INTEGER NOT NULL DEFAULT 0,
                service_points INTEGER NOT NULL DEFAULT 0,
                points_won INTEGER NOT NULL DEFAULT 0
            );
            INSERT OR IGNORE INTO global_profile(id) VALUES(1);

            CREATE TABLE IF NOT EXISTS overall_stats (
                id INTEGER PRIMARY KEY CHECK(id=1),
                checked INTEGER NOT NULL DEFAULT 0,
                top1_hits INTEGER NOT NULL DEFAULT 0,
                top2_hits INTEGER NOT NULL DEFAULT 0,
                unknown INTEGER NOT NULL DEFAULT 0,
                brier_sum REAL NOT NULL DEFAULT 0,
                logloss_sum REAL NOT NULL DEFAULT 0
            );
            INSERT OR IGNORE INTO overall_stats(id) VALUES(1);

            CREATE TABLE IF NOT EXISTS outcome_stats (
                outcome TEXT PRIMARY KEY,
                resolved INTEGER NOT NULL DEFAULT 0,
                actual_count INTEGER NOT NULL DEFAULT 0,
                predicted_sum REAL NOT NULL DEFAULT 0,
                top_pick_count INTEGER NOT NULL DEFAULT 0,
                top_pick_hits INTEGER NOT NULL DEFAULT 0
            );
        """)
        for outcome in OUTCOMES:
            self.conn.execute("INSERT OR IGNORE INTO outcome_stats(outcome) VALUES(?)", (outcome,))
        self.conn.commit()

    @staticmethod
    def _key(event_id: str, set_num: int, game_num: int) -> str:
        return f"{event_id}:{set_num}:{game_num}"

    def player_profile(self, player: str) -> PlayerProfile:
        r = self.conn.execute(
            "SELECT player,service_games,deuce_games,service_points,points_won FROM player_profiles WHERE player=?",
            (player,),
        ).fetchone()
        if r is None:
            return PlayerProfile(player=player)
        return PlayerProfile(str(r["player"]), int(r["service_games"]), int(r["deuce_games"]), int(r["service_points"]), int(r["points_won"]))

    def global_profile(self) -> PlayerProfile:
        r = self.conn.execute("SELECT service_games,deuce_games,service_points,points_won FROM global_profile WHERE id=1").fetchone()
        return PlayerProfile("__GLOBAL__", int(r["service_games"]), int(r["deuce_games"]), int(r["service_points"]), int(r["points_won"]))

    def estimate_server_p(self, server: str) -> tuple[float, PlayerProfile, PlayerProfile]:
        profile = self.player_profile(server)
        glob = self.global_profile()
        global_deuce = (DEFAULT_DEUCE_RATE * GLOBAL_DEUCE_PRIOR_STRENGTH + glob.deuce_games) / (GLOBAL_DEUCE_PRIOR_STRENGTH + glob.service_games)
        server_deuce = (global_deuce * PLAYER_DEUCE_PRIOR_STRENGTH + profile.deuce_games) / (PLAYER_DEUCE_PRIOR_STRENGTH + profile.service_games)
        server_deuce = clamp(server_deuce, 0.001, deuce_rate_from_point_p(0.5))
        p_from_deuce = point_p_from_deuce_rate(server_deuce)

        global_point_p = (DEFAULT_SERVER_POINT_P * GLOBAL_POINT_PRIOR_STRENGTH + glob.points_won) / (GLOBAL_POINT_PRIOR_STRENGTH + glob.service_points)
        server_point_p = (global_point_p * PLAYER_POINT_PRIOR_STRENGTH + profile.points_won) / (PLAYER_POINT_PRIOR_STRENGTH + profile.service_points)
        blend = 0.0 if profile.service_points < 8 else min(0.75, profile.service_points / (profile.service_points + 80.0))
        p_server = (1.0 - blend) * p_from_deuce + blend * server_point_p
        return clamp(p_server, 0.50, 0.85), profile, glob

    def ingest_completed_games(self, parsed: ParsedMatch) -> None:
        for g in parsed.completed_games:
            if self.conn.execute(
                "SELECT 1 FROM processed_games WHERE event_id=? AND set_num=? AND game_num=?",
                (g.event_id, g.set_num, g.game_num),
            ).fetchone() is not None:
                continue
            self.conn.execute(
                "INSERT INTO processed_games(event_id,set_num,game_num,processed_at) VALUES(?,?,?,?)",
                (g.event_id, g.set_num, g.game_num, time.time()),
            )
            self.conn.execute(
                """
                INSERT INTO player_profiles(player,service_games,deuce_games,service_points,points_won,updated_at)
                VALUES(?,1,?,?,?,?)
                ON CONFLICT(player) DO UPDATE SET
                  service_games=service_games+1,
                  deuce_games=deuce_games+excluded.deuce_games,
                  service_points=service_points+excluded.service_points,
                  points_won=points_won+excluded.points_won,
                  updated_at=excluded.updated_at
                """,
                (g.server, int(g.deuce), g.service_points, g.points_won, time.time()),
            )
            self.conn.execute(
                """UPDATE global_profile SET service_games=service_games+1,deuce_games=deuce_games+?,
                   service_points=service_points+?,points_won=points_won+? WHERE id=1""",
                (int(g.deuce), g.service_points, g.points_won),
            )
        self.conn.commit()

    def build_prediction(self, parsed: ParsedMatch, match_link: str) -> ExactPrediction:
        server_num = get_server_for_game(parsed.current_set, parsed.current_game, parsed.first_server_match, parsed.max_game_by_set)
        server = parsed.player1 if server_num == 1 else parsed.player2
        p_server, profile, _ = self.estimate_server_p(server)
        p1 = p_server if server_num == 1 else 1.0 - p_server
        dist = exact_game_distribution(p1)
        ordered = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
        return ExactPrediction(
            event_id=parsed.event_id,
            player1=parsed.player1,
            player2=parsed.player2,
            set_num=parsed.current_set,
            game_num=parsed.current_game,
            server=server,
            current_score=f"{parsed.raw_score_a}:{parsed.raw_score_b}",
            p_server=round(p_server, 5),
            p_player1_point=round(p1, 5),
            probabilities={k: round(v * 100.0, 2) for k, v in dist.items()},
            top1=ordered[0][0],
            top1_probability=round(ordered[0][1] * 100.0, 2),
            top2=[ordered[0][0], ordered[1][0]],
            data_quality=data_quality_label(profile.service_games, profile.service_points),
            match_link=match_link,
            created_at=time.time(),
        )

    def _prob_values(self, pred: ExactPrediction) -> list[float]:
        return [float(pred.probabilities[o]) / 100.0 for o in OUTCOMES]

    def create_signal_if_allowed(self, parsed: ParsedMatch, pred: ExactPrediction) -> bool:
        key = self._key(parsed.event_id, parsed.current_set, parsed.current_game)
        if self.conn.execute("SELECT 1 FROM signal_history WHERE key=?", (key,)).fetchone() is not None:
            return False
        if self.conn.execute("SELECT 1 FROM pending_predictions WHERE key=?", (key,)).fetchone() is not None:
            return False

        # Не прогнозируем тай-брейк как обычный гейм.
        if parsed.current_game >= 13:
            return False

        a = _int(parsed.raw_score_a)
        b = _int(parsed.raw_score_b)
        if SIGNAL_ONLY_AT_ZERO_ZERO and (a, b) != (0, 0):
            return False

        probs = self._prob_values(pred)
        top2 = json.dumps(pred.top2, ensure_ascii=False)
        common = [
            key, parsed.event_id, pred.player1, pred.player2, pred.set_num, pred.game_num,
            pred.server, pred.p_server, pred.p_player1_point, pred.top1, top2,
            pred.data_quality, pred.match_link,
        ] + probs
        placeholders = ",".join("?" for _ in range(len(common) + 1))
        self.conn.execute(
            f"INSERT INTO pending_predictions(key,event_id,player1,player2,set_num,game_num,server,p_server,p_player1,top1,top2,data_quality,match_link,{','.join(f'p_{i}' for i in range(8))},created_at) VALUES({placeholders})",
            tuple(common + [pred.created_at]),
        )
        history_values = common[:12] + probs + [pred.created_at]
        placeholders2 = ",".join("?" for _ in range(len(history_values)))
        self.conn.execute(
            f"INSERT INTO signal_history(key,event_id,player1,player2,set_num,game_num,server,p_server,p_player1,top1,top2,data_quality,{','.join(f'p_{i}' for i in range(8))},created_at) VALUES({placeholders2})",
            tuple(history_values),
        )
        self.conn.commit()
        return True

    def pending_for(self, event_id: str, set_num: int, game_num: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM pending_predictions WHERE key=?",
            (self._key(event_id, set_num, game_num),),
        ).fetchone()

    def _row_probs(self, row: sqlite3.Row) -> dict[str, float]:
        return {outcome: float(row[f"p_{i}"]) for i, outcome in enumerate(OUTCOMES)}

    def resolve_completed_games(self, parsed: ParsedMatch) -> None:
        outcomes = {(g.set_num, g.game_num): g.exact_outcome for g in parsed.completed_games}
        rows = self.conn.execute(
            "SELECT * FROM pending_predictions WHERE event_id=?",
            (parsed.event_id,),
        ).fetchall()
        current = (parsed.current_set, parsed.current_game)
        for row in rows:
            sg = (int(row["set_num"]), int(row["game_num"]))
            if sg >= current:
                continue
            actual = outcomes.get(sg)
            self._resolve_row(row, actual)

    def _resolve_row(self, row: sqlite3.Row, actual: str | None) -> None:
        key = str(row["key"])
        if actual not in OUTCOMES:
            self.conn.execute(
                "UPDATE signal_history SET status='unknown',resolved_at=? WHERE key=?",
                (time.time(), key),
            )
            self.conn.execute("UPDATE overall_stats SET unknown=unknown+1 WHERE id=1")
            self.conn.execute("DELETE FROM pending_predictions WHERE key=?", (key,))
            self.conn.commit()
            return

        probs = self._row_probs(row)
        top1 = str(row["top1"])
        try:
            top2 = list(json.loads(str(row["top2"])))
        except Exception:
            top2 = [top1]
        top1_hit = int(actual == top1)
        top2_hit = int(actual in top2)
        y = {o: 1.0 if o == actual else 0.0 for o in OUTCOMES}
        brier = sum((probs[o] - y[o]) ** 2 for o in OUTCOMES) / len(OUTCOMES)
        p_actual = clamp(probs[actual], 1e-9, 1.0)
        log_loss = -math.log(p_actual)

        self.conn.execute(
            """UPDATE signal_history SET actual_outcome=?,status='resolved',top1_hit=?,top2_hit=?,
               brier=?,log_loss=?,resolved_at=? WHERE key=?""",
            (actual, top1_hit, top2_hit, brier, log_loss, time.time(), key),
        )
        self.conn.execute(
            """UPDATE overall_stats SET checked=checked+1,top1_hits=top1_hits+?,top2_hits=top2_hits+?,
               brier_sum=brier_sum+?,logloss_sum=logloss_sum+? WHERE id=1""",
            (top1_hit, top2_hit, brier, log_loss),
        )
        for outcome in OUTCOMES:
            self.conn.execute(
                """UPDATE outcome_stats SET resolved=resolved+1,actual_count=actual_count+?,predicted_sum=predicted_sum+?,
                   top_pick_count=top_pick_count+?,top_pick_hits=top_pick_hits+? WHERE outcome=?""",
                (int(actual == outcome), probs[outcome], int(top1 == outcome), int(top1 == outcome and actual == outcome), outcome),
            )
        self.conn.execute("DELETE FROM pending_predictions WHERE key=?", (key,))
        self.conn.commit()

    def finish_missing_matches(self, active_event_ids: set[str]) -> None:
        tracked = {str(r[0]) for r in self.conn.execute("SELECT DISTINCT event_id FROM pending_predictions").fetchall()}
        for event_id in tracked:
            if event_id in active_event_ids:
                self.missing_cycles[event_id] = 0
                continue
            self.missing_cycles[event_id] = self.missing_cycles.get(event_id, 0) + 1
            if self.missing_cycles[event_id] < MISSING_CYCLES_TO_UNKNOWN:
                continue
            rows = self.conn.execute("SELECT * FROM pending_predictions WHERE event_id=?", (event_id,)).fetchall()
            for row in rows:
                self._resolve_row(row, None)
            self.missing_cycles.pop(event_id, None)

        now = time.time()
        if now - self._last_maintenance >= 900.0:
            cutoff = now - PROCESSED_GAME_RETENTION_HOURS * 3600.0
            self.conn.execute("DELETE FROM processed_games WHERE processed_at < ?", (cutoff,))
            stale = now - STALE_HOURS * 3600.0
            rows = self.conn.execute("SELECT * FROM pending_predictions WHERE created_at < ?", (stale,)).fetchall()
            for row in rows:
                self._resolve_row(row, None)
            self._cleanup_history()
            self.conn.commit()
            self._last_maintenance = now

    def _cleanup_history(self) -> None:
        total = int(self.conn.execute("SELECT COUNT(*) FROM signal_history").fetchone()[0])
        excess = max(0, total - HISTORY_LIMIT)
        if excess <= 0:
            return
        ids = [int(r[0]) for r in self.conn.execute(
            "SELECT id FROM signal_history WHERE status!='pending' ORDER BY id ASC LIMIT ?", (excess,)
        ).fetchall()]
        if ids:
            self.conn.executemany("DELETE FROM signal_history WHERE id=?", [(i,) for i in ids])

    def row_to_prediction(self, row: sqlite3.Row, current_score: str, match_link: str) -> ExactPrediction:
        probs = {o: round(float(row[f"p_{i}"]) * 100.0, 2) for i, o in enumerate(OUTCOMES)}
        top1 = str(row["top1"])
        try:
            top2 = list(json.loads(str(row["top2"])))
        except Exception:
            top2 = [top1]
        return ExactPrediction(
            event_id=str(row["event_id"]), player1=str(row["player1"]), player2=str(row["player2"]),
            set_num=int(row["set_num"]), game_num=int(row["game_num"]), server=str(row["server"]),
            current_score=current_score, p_server=float(row["p_server"]), p_player1_point=float(row["p_player1"]),
            probabilities=probs, top1=top1, top1_probability=probs[top1], top2=top2,
            data_quality=str(row["data_quality"]), match_link=match_link or str(row["match_link"]),
            created_at=float(row["created_at"]),
        )

    def summary(self) -> dict[str, Any]:
        r = self.conn.execute("SELECT * FROM overall_stats WHERE id=1").fetchone()
        checked = int(r["checked"])
        return {
            "checked": checked,
            "top1_hits": int(r["top1_hits"]),
            "top2_hits": int(r["top2_hits"]),
            "unknown": int(r["unknown"]),
            "pending": int(self.conn.execute("SELECT COUNT(*) FROM pending_predictions").fetchone()[0]),
            "top1_accuracy": (int(r["top1_hits"]) / checked * 100.0) if checked else 0.0,
            "top2_accuracy": (int(r["top2_hits"]) / checked * 100.0) if checked else 0.0,
            "brier": (float(r["brier_sum"]) / checked) if checked else None,
            "log_loss": (float(r["logloss_sum"]) / checked) if checked else None,
        }

    def close(self) -> None:
        self.conn.close()


def save_unknown_sample(parsed: ParsedMatch, game: CompletedGame) -> None:
    if game.exact_outcome is not None:
        return
    states = _raw_score_states_for_game(parsed.events, game.set_num, game.game_num)
    # Только полезные поля, без всего сырого матча.
    related = []
    for ev in parsed.events:
        if _int(ev.get("i1")) == game.set_num and _int(ev.get("i2")) == game.game_num:
            related.append({k: ev.get(k) for k in ("type", "i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9", "i10") if k in ev})
    path = Path(UNKNOWN_SAMPLES_FILE)
    try:
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        if not isinstance(existing, list):
            existing = []
    except Exception:
        existing = []
    key = f"{parsed.event_id}:{game.set_num}:{game.game_num}"
    if any(x.get("key") == key for x in existing if isinstance(x, dict)):
        return
    existing.append({
        "key": key,
        "players": [parsed.player1, parsed.player2],
        "set": game.set_num,
        "game": game.game_num,
        "server": game.server,
        "states": states,
        "events": related[-40:],
    })
    existing = existing[-UNKNOWN_SAMPLE_LIMIT:]
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")


def scan_live(client: PariClient, tracker: ExactGameTracker) -> tuple[list[ExactPrediction], set[str]]:
    predictions: list[ExactPrediction] = []
    seen_codes: set[str] = set()
    live = client.live_events()
    active_event_ids = {str(x.get("id")) for x in live if x.get("id") is not None}

    for misc in live:
        if "(" not in str(misc.get("comment", "")):
            continue
        event_id = misc.get("id")
        if event_id is None:
            continue
        try:
            items = client.actrans(event_id)
        except Exception as e:
            print(f"[actrans] event={event_id}: {e}", file=sys.stderr)
            continue

        for item in items:
            player1 = str(item.get("fon_team1") or "Игрок1")
            player2 = str(item.get("fon_team2") or "Игрок2")
            code = item.get("code")
            if code is None or str(code) in seen_codes:
                continue
            seen_codes.add(str(code))
            try:
                events = client.sportscast_events(code)
                parsed = parse_match(event_id, player1, player2, events)
            except Exception as e:
                print(f"[events] code={code}: {e}", file=sys.stderr)
                continue
            if parsed is None:
                continue

            # Сначала закрываем старые прогнозы исходом завершившегося гейма.
            tracker.resolve_completed_games(parsed)

            # Профиль игроков обновляем после определения исхода.
            tracker.ingest_completed_games(parsed)
            for g in parsed.completed_games:
                if g.exact_outcome is None:
                    save_unknown_sample(parsed, g)

            link = client.event_link(event_id)
            fresh = tracker.build_prediction(parsed, link)
            tracker.create_signal_if_allowed(parsed, fresh)
            pending = tracker.pending_for(parsed.event_id, parsed.current_set, parsed.current_game)
            if pending is not None:
                pred = tracker.row_to_prediction(
                    pending,
                    current_score=f"{parsed.raw_score_a}:{parsed.raw_score_b}",
                    match_link=link,
                )
                predictions.append(pred)

    predictions.sort(key=lambda p: p.top1_probability, reverse=True)
    return predictions, active_event_ids


def self_test() -> None:
    for p in (0.2, 0.5, 0.62, 0.8):
        d = exact_game_distribution(p)
        assert abs(sum(d.values()) - 1.0) < 1e-12
        assert set(d) == set(OUTCOMES)

    assert exact_outcome_from_states([(0, 0), (15, 0), (30, 0), (40, 0)]) == "40:0"
    assert exact_outcome_from_states([(0, 0), (0, 15), (15, 15), (15, 30), (15, 40)]) == "15:40"
    assert exact_outcome_from_states([(40, 40), ("ADV", 40)]) == "+:40"
    assert exact_outcome_from_states([(40, 40), (40, "ADV")]) == "40:+"
    assert exact_outcome_from_states([(40, 40)]) is None

    db = ":memory:"
    tr = ExactGameTracker(db)
    # Сигнал при 0:0 создаётся.
    parsed = ParsedMatch("e1", "A", "B", 1, 1, 1, 0, 0, [], {1: 1}, [])
    pred = tr.build_prediction(parsed, "https://pari.ru/")
    assert tr.create_signal_if_allowed(parsed, pred) is True
    assert tr.pending_for("e1", 1, 1) is not None
    # Повторно один и тот же гейм не создаётся.
    assert tr.create_signal_if_allowed(parsed, pred) is False
    tr.close()

    tr = ExactGameTracker(":memory:")
    # Если впервые увидели не 0:0 — в статистику гейм не попадает.
    parsed2 = ParsedMatch("e2", "A", "B", 1, 1, 1, 15, 0, [], {1: 1}, [])
    pred2 = tr.build_prediction(parsed2, "https://pari.ru/")
    assert tr.create_signal_if_allowed(parsed2, pred2) is False
    assert tr.summary()["pending"] == 0
    tr.close()


def run_live() -> int:
    client = PariClient(TIMEOUT)
    tracker = ExactGameTracker(DB)
    print("PARI — точный итог гейма (8 исходов)")
    print(f"HTTP backend: {HTTP_BACKEND}")
    print(f"БД: {DB} | JSON: {OUTPUT}")
    print("Честный режим: новый прогноз создаётся только при счёте 0:0.")
    print("Если точный финал гейма не удалось восстановить — исход UNKNOWN, не ошибка модели.")
    print("Ctrl+C для остановки\n")
    try:
        while True:
            started = time.time()
            try:
                preds, active = scan_live(client, tracker)
                tracker.finish_missing_matches(active)
                Path(OUTPUT).write_text(json.dumps([asdict(p) for p in preds], ensure_ascii=False, indent=2), encoding="utf-8")
                s = tracker.summary()
                print("\n" + "=" * 92)
                print(time.strftime("%Y-%m-%d %H:%M:%S"), f"| живых прогнозов: {len(preds)}")
                print(
                    f"Проверено: {s['checked']} | TOP-1: {s['top1_hits']} ({s['top1_accuracy']:.1f}%) | "
                    f"TOP-2: {s['top2_hits']} ({s['top2_accuracy']:.1f}%) | UNKNOWN: {s['unknown']} | pending: {s['pending']}"
                )
                for p in preds[:15]:
                    top = sorted(p.probabilities.items(), key=lambda kv: kv[1], reverse=True)
                    print(f"🎾 {p.player1} — {p.player2} | сет {p.set_num}, гейм {p.game_num} | подаёт {p.server} | счёт {p.current_score}")
                    print("   " + " | ".join(f"{k} {v:.1f}%" for k, v in top[:4]))
                    print(f"   TOP: {p.top1} ({p.top1_probability:.1f}%) | p подачи={p.p_server*100:.1f}% | {p.data_quality}")
            except KeyboardInterrupt:
                print("\nСТОП")
                return 0
            except Exception as e:
                print(f"[цикл] {type(e).__name__}: {e}", file=sys.stderr)
            time.sleep(max(0.0, INTERVAL - (time.time() - started)))
    finally:
        tracker.close()


def main() -> int:
    mode = str(RUN_MODE).strip().lower()
    if mode in {"self-test", "selftest", "test"}:
        self_test()
        print("SELF-TEST: OK")
        return 0
    if mode != "live":
        raise ValueError("RUN_MODE должен быть 'live' или 'self-test'")
    return run_live()


if __name__ == "__main__":
    raise SystemExit(main())
