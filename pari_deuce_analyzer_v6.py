from __future__ import annotations

# v6.0 STRICT: direct 30:30 model, pressure-state transitions, receiver context,
# first-30:30 / mid-set selection. Forward statistics are kept in a fresh DB.

import argparse
import json
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent

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

# Стартовые priors нужны только пока собственной статистики мало.
# По мере накопления global/player profile их влияние быстро уменьшается.
DEFAULT_DEUCE_RATE = 0.285
GLOBAL_DEUCE_PRIOR_STRENGTH = 30.0
PLAYER_DEUCE_PRIOR_STRENGTH = 18.0
DEFAULT_SERVER_POINT_P = 0.62
GLOBAL_POINT_PRIOR_STRENGTH = 80.0
PLAYER_POINT_PRIOR_STRENGTH = 45.0
EMPIRICAL_MIN_SIGNALS = 30
EMPIRICAL_FULL_WEIGHT = 200
MAX_CALIBRATION_ADJUSTMENT_PP = 12.0

POINT_TO_NUM = {0: 0, 15: 1, 30: 2, 40: 3, "0": 0, "15": 1, "30": 2, "40": 3}
POINT_RANK = {0: 0, 15: 1, 30: 2, 40: 3}

# В PARI рынок на достижение 40:40 уже недоступен при 40:30 / 30:40.
BLOCKED_MARKET_SCORES = {(40, 30), (30, 40), (40, 40)}


# ========================= НАСТРОЙКИ ЗАПУСКА =========================
RUN_MODE = "live"          # "live", "offline" или "self-test"
OFFLINE_FILE = "tennis.json"

INTERVAL = 2.0             # пауза между live-циклами, сек
TIMEOUT = 8.0              # HTTP timeout, сек
MIN_PROBABILITY = 50.5     # V6 STRICT: ниже этого новый сигнал не выдаём
TOP = 20                   # максимум сигналов на экран
OUTPUT = str(BASE_DIR / "pari_predictions.json")
LEGACY_OUTPUT = str(BASE_DIR / "tennis.json")  # для существующей веб-морды
DB = str(BASE_DIR / "pari_deuce_v6.sqlite3")  # только forward-статистика V6
TRAINING_DB = str(BASE_DIR / "pari_model_stats_v5.sqlite3")  # старые ФАКТЫ используются только как prior
STRICT_GAME_MIN = 4
STRICT_GAME_MAX = 6
LEGACY_GLOBAL_WEIGHT = 0.50
LEGACY_CONTEXT_WEIGHT = 1.00
PRESSURE_CACHE_SECONDS = 15.0
MISSING_CYCLES_TO_UNKNOWN = 15       # ~30 секунд при INTERVAL=2
EVENT_API_VERSION = "76749944487"     # значение из твоего старого рабочего скрипта
STALE_HOURS = 12.0
MIN_BEST_CONDITION_SIGNALS = 50
PROCESSED_GAME_RETENTION_HOURS = 48.0
SIGNAL_HISTORY_LIMIT = 50000          # только компактные выданные сигналы, без сырых событий  # защита от двойного учета при кратком исчезновении матча
# ====================================================================


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
    deuce_games: set[tuple[int, int]]
    completed_games: list[CompletedGame]
    max_game_by_set: dict[int, int]
    events: list[dict[str, Any]]


@dataclass
class Prediction:
    event_id: int | str | None
    player1: str
    player2: str
    set_num: int
    game_num: int
    server: str
    current_score: str
    game_band: str
    server_service_games: int
    server_deuce_games: int
    server_service_points: int
    server_points_won: int
    global_service_games: int
    posterior_deuce_rate: float
    estimated_server_point_p: float
    raw_probability_deuce: float
    probability_deuce: float
    calibration_adjustment: float
    data_quality: str
    confidence: str  # совместимость со старым интерфейсом
    server_score_state: str = ""  # счёт относительно подающего: первым всегда очки подающего
    already_deuce: bool = False
    market_available: bool = True
    match_link: str = "https://pari.ru/"
    signal_score: str = ""
    live_probability_deuce: float = 0.0
    strict_eligible: bool = False
    transition_probability: float = 0.0
    context_probability: float = 0.0
    signal_reason: str = ""

    def pretty(self) -> str:
        status = "УЖЕ БЫЛО 40:40" if self.already_deuce else f"{self.probability_deuce:.1f}%"
        market = "доступен" if self.market_available else "НЕДОСТУПЕН"
        adjustment = (
            f"{self.calibration_adjustment:+.1f} п.п."
            if abs(self.calibration_adjustment) >= 0.05
            else "0.0 п.п."
        )
        lines = [
            f"🎾 {self.player1} — {self.player2}",
            f"   Сет {self.set_num}, гейм {self.game_num} | подаёт: {self.server}",
            f"   Текущий счёт: {self.current_score} | рынок: {market}",
            f"   Счёт относительно подающего: {self.server_score_state or '—'}",
        ]
        if self.signal_score:
            lines.append(f"   Сигнал зафиксирован при: {self.signal_score}")
        lines.append(f"   📊 Оценка вероятности 40:40 при сигнале: {status}")
        if self.signal_score:
            lines.append(f"   Текущая оценка модели: {self.live_probability_deuce:.1f}%")
        lines.extend([
            f"   Базовая модель: {self.raw_probability_deuce:.1f}% | поправка статистики: {adjustment}",
            f"   Подающий: {self.server_deuce_games}/{self.server_service_games} его геймов дошли до 40:40",
            f"   Оценка p(очко подающего): {self.estimated_server_point_p * 100:.1f}%",
            f"   V6 transition: {self.transition_probability:.1f}% | context: {self.context_probability:.1f}%",
            f"   Режим: {self.signal_reason or 'наблюдение'}",
            f"   Объём данных: {self.data_quality}",
            f"   🔗 {self.match_link}",
        ])
        return "\n".join(lines)



@dataclass
class MatchObservation:
    event_id: str
    prediction: Prediction
    deuce_games: set[tuple[int, int]]


def _int(v: Any, default: int | None = None) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def game_band(game_num: int) -> str:
    if game_num <= 3:
        return "Геймы 1–3"
    if game_num <= 9:
        return "Геймы 4–9"
    return "Геймы 10+"


def set_label(set_num: int) -> str:
    suffix = "-й"
    if set_num == 3:
        suffix = "-й"
    return f"{set_num}{suffix} сет"


def score_label(raw_a: Any, raw_b: Any) -> str:
    return f"{raw_a}:{raw_b}"


def server_score_label(raw_a: Any, raw_b: Any, server_num: int) -> str:
    """Счёт с точки зрения подающего: сначала очки подающего, затем принимающего."""
    a = _int(raw_a)
    b = _int(raw_b)
    if a not in POINT_RANK or b not in POINT_RANK or server_num not in (1, 2):
        return "неизвестно"
    return f"{a}:{b}" if server_num == 1 else f"{b}:{a}"


def get_current_server(set_num: int, game_num: int, first_server_match: int) -> int:
    # Оставлено только для обратной совместимости. Для реального live используем
    # get_server_for_game(), потому что первый подающий следующего сета зависит
    # от количества геймов в предыдущих сетах, а не просто от номера сета.
    if set_num < 1 or game_num < 1 or first_server_match not in (1, 2):
        raise ValueError("Неверные входные данные для определения подачи")
    first_in_set = first_server_match if set_num % 2 == 1 else 3 - first_server_match
    return first_in_set if game_num % 2 == 1 else 3 - first_in_set


def get_server_for_game(
    set_num: int,
    game_num: int,
    first_server_match: int,
    max_game_by_set: dict[int, int],
) -> int:
    """Точный порядок подачи через абсолютный номер гейма матча.

    В теннисе подача чередуется по геймам непрерывно между сетами. Поэтому
    нельзя просто менять первого подающего на каждом новом сете: после 6:4
    первый подающий следующего сета будет тем же, а после 6:3 — другим.
    """
    if set_num < 1 or game_num < 1 or first_server_match not in (1, 2):
        raise ValueError("Неверные входные данные для определения подачи")
    games_before = sum(max(0, int(max_game_by_set.get(s, 0))) for s in range(1, set_num))
    absolute_game_num = games_before + game_num
    return first_server_match if absolute_game_num % 2 == 1 else 3 - first_server_match


def deuce_rate_from_point_p(p: float) -> float:
    q = 1.0 - p
    return 20.0 * (p ** 3) * (q ** 3)


def point_p_from_deuce_rate(target_rate: float) -> float:
    # Частота деуса симметрична относительно p=0.5. Для подачи выбираем ветку >= 0.5.
    max_rate = deuce_rate_from_point_p(0.5)
    target_rate = clamp(target_rate, 0.001, max_rate)
    lo, hi = 0.5, 0.90
    for _ in range(80):
        mid = (lo + hi) / 2
        rate = deuce_rate_from_point_p(mid)
        if rate > target_rate:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def prob_reach_deuce_from_score(a: int, b: int, p1_point: float) -> float:
    memo: dict[tuple[int, int], float] = {}

    def rec(x: int, y: int) -> float:
        if x == 3 and y == 3:
            return 1.0
        if x >= 4 and y <= 2:
            return 0.0
        if y >= 4 and x <= 2:
            return 0.0
        if x > 3 or y > 3:
            return 1.0
        key = (x, y)
        if key in memo:
            return memo[key]
        memo[key] = p1_point * rec(x + 1, y) + (1.0 - p1_point) * rec(x, y + 1)
        return memo[key]

    return rec(a, b)


def data_quality_label(service_games: int, service_points: int) -> str:
    evidence = service_games + service_points / 12.0
    if evidence >= 35:
        return "много данных"
    if evidence >= 12:
        return "достаточно данных"
    return "мало данных"


def valid_point_score(v: Any) -> int | None:
    return POINT_TO_NUM.get(v)


def is_market_available(raw_a: Any, raw_b: Any) -> bool:
    """Сигнал разрешён только для известного валидного счёта и открытого рынка."""
    a = _int(raw_a)
    b = _int(raw_b)
    if a not in POINT_RANK or b not in POINT_RANK:
        return False
    return (a, b) not in BLOCKED_MARKET_SCORES


def extract_deuce_games(events: list[dict[str, Any]]) -> set[tuple[int, int]]:
    result: set[tuple[int, int]] = set()
    for ev in events:
        s = _int(ev.get("i1"))
        g = _int(ev.get("i2"))
        if not (s and g):
            continue
        if _int(ev.get("i8")) == 40 and _int(ev.get("i9")) == 40:
            result.add((s, g))
    return result


def _unique_score_states_for_game(events: list[dict[str, Any]], set_num: int, game_num: int) -> list[tuple[int, int]]:
    states: list[tuple[int, int]] = []
    for ev in events:
        if _int(ev.get("i1")) != set_num or _int(ev.get("i2")) != game_num:
            continue
        a = _int(ev.get("i8"))
        b = _int(ev.get("i9"))
        if a not in POINT_RANK or b not in POINT_RANK:
            continue
        st = (a, b)
        if not states or states[-1] != st:
            states.append(st)
    return states


def _service_point_stats_from_states(states: list[tuple[int, int]], server_num: int) -> tuple[int, int]:
    """Считает только однозначно наблюдаемые переходы счёта до/в момент деуса."""
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


def parse_match(
    event_id: int | str,
    player1: str,
    player2: str,
    events: list[dict[str, Any]],
) -> ParsedMatch | None:
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

    deuce_games = extract_deuce_games(events)
    completed_keys: set[tuple[int, int]] = set()
    for ev in events:
        s = _int(ev.get("i1"))
        g = _int(ev.get("i2"))
        if not (s and g):
            continue
        if (s, g) < (current_set, current_game):
            completed_keys.add((s, g))

    completed_games: list[CompletedGame] = []
    for s, g in sorted(completed_keys):
        try:
            srv_num = get_server_for_game(s, g, first_server_match, max_game_by_set)
        except ValueError:
            continue
        srv_name = player1 if srv_num == 1 else player2
        states = _unique_score_states_for_game(events, s, g)
        points_total, points_won = _service_point_stats_from_states(states, srv_num)
        completed_games.append(
            CompletedGame(
                event_id=str(event_id),
                set_num=s,
                game_num=g,
                server=srv_name,
                deuce=(s, g) in deuce_games,
                service_points=points_total,
                points_won=points_won,
            )
        )

    raw_a, raw_b = latest_score_by_game.get((current_set, current_game), (None, None))
    return ParsedMatch(
        event_id=str(event_id),
        player1=player1,
        player2=player2,
        first_server_match=first_server_match,
        current_set=current_set,
        current_game=current_game,
        raw_score_a=raw_a,
        raw_score_b=raw_b,
        deuce_games=deuce_games,
        completed_games=completed_games,
        max_game_by_set=max_game_by_set,
        events=events,
    )



def _states_server_perspective(
    events: list[dict[str, Any]],
    set_num: int,
    game_num: int,
    server_num: int,
) -> list[tuple[int, int]]:
    raw = _unique_score_states_for_game(events, set_num, game_num)
    if server_num == 1:
        return raw
    return [(b, a) for a, b in raw]


def _pressure_features_from_states(states: list[tuple[int, int]]) -> dict[str, int | None]:
    """Извлекает именно то, что нужно для события 'дойдёт ли 30:30 до 40:40'.

    Все счета уже относительно подающего.
    """
    had_3030 = int((30, 30) in states)
    reached_deuce = int((40, 40) in states)
    first_after_server: int | None = None
    if had_3030:
        i = states.index((30, 30))
        for st in states[i + 1:]:
            if st == (40, 30):
                first_after_server = 1
                break
            if st == (30, 40):
                first_after_server = 0
                break
            if st == (40, 40):
                # Фид мог пропустить промежуточные 40:30/30:40.
                break

    def branch(state: tuple[int, int]) -> tuple[int, int]:
        try:
            i = states.index(state)
        except ValueError:
            return 0, 0
        # Считаем возврат к 40:40 только если этот state был ДО первого deuce.
        try:
            d = states.index((40, 40))
        except ValueError:
            d = -1
        if d >= 0 and i > d:
            return 0, 0
        return 1, int(d > i)

    lead_n, lead_return = branch((40, 30))   # подающий ведёт: вернёт ли принимающий к deuce
    trail_n, trail_save = branch((30, 40))   # подающий уступает: спасёт ли break-point
    return {
        "had_3030": had_3030,
        "reached_deuce": reached_deuce,
        "first_after_server": first_after_server,
        "lead_n": lead_n,
        "lead_return": lead_return,
        "trail_n": trail_n,
        "trail_save": trail_save,
    }


def _has_prior_3030(parsed: ParsedMatch) -> bool:
    cur = (parsed.current_set, parsed.current_game)
    keys: set[tuple[int, int]] = set()
    for ev in parsed.events:
        ss = _int(ev.get("i1"))
        gg = _int(ev.get("i2"))
        if ss and gg and (ss, gg) < cur:
            keys.add((ss, gg))
    for ss, gg in sorted(keys):
        try:
            srv = get_server_for_game(ss, gg, parsed.first_server_match, parsed.max_game_by_set)
        except Exception:
            continue
        if (30, 30) in _states_server_perspective(parsed.events, ss, gg, srv):
            return True
    return False


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
            raise ValueError(f"Ожидался JSON-объект от {url}, получено {type(data).__name__}")
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

        params = {
            "lang": "ru",
            "version": EVENT_API_VERSION,
            "eventId": key,
            "scopeMarket": "2300",
        }
        data: dict[str, Any] | None = None
        try:
            data = self._get_json(
                EVENT_DETAILS_URL,
                params=params,
                headers=self._headers_event_details(False),
            )
        except Exception:
            try:
                data = self._get_json(
                    EVENT_DETAILS_IP_URL,
                    params=params,
                    headers=self._headers_event_details(True),
                )
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
            # Не запоминаем неудачу навсегда: повторим попытку позже.
            self.link_retry_after[key] = time.time() + 30.0
        return link


class StatsTracker:
    """Компактная БД: pending/processed только для живых матчей, постоянны лишь агрегаты."""

    def __init__(self, db_path: str, missing_cycles_to_unknown: int = 15):
        self.db_path = db_path
        self.missing_cycles_to_unknown = max(1, missing_cycles_to_unknown)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.missing_cycles: dict[str, int] = {}
        self._last_maintenance = 0.0
        self._init_db()

    def _init_db(self) -> None:
        self.conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;

            CREATE TABLE IF NOT EXISTS pending_predictions (
                key TEXT PRIMARY KEY,
                event_id TEXT NOT NULL,
                player1 TEXT NOT NULL,
                player2 TEXT NOT NULL,
                set_num INTEGER NOT NULL,
                game_num INTEGER NOT NULL,
                game_band TEXT NOT NULL,
                server TEXT NOT NULL,
                score_state TEXT NOT NULL,
                server_score_state TEXT NOT NULL DEFAULT '',
                probability REAL NOT NULL,
                raw_probability REAL NOT NULL,
                data_quality TEXT NOT NULL,
                server_service_games INTEGER NOT NULL DEFAULT 0,
                server_deuce_games INTEGER NOT NULL DEFAULT 0,
                server_service_points INTEGER NOT NULL DEFAULT 0,
                server_points_won INTEGER NOT NULL DEFAULT 0,
                global_service_games INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_pending_event ON pending_predictions(event_id);

            CREATE TABLE IF NOT EXISTS processed_games (
                event_id TEXT NOT NULL,
                set_num INTEGER NOT NULL,
                game_num INTEGER NOT NULL,
                server TEXT NOT NULL,
                processed_at REAL NOT NULL DEFAULT 0,
                PRIMARY KEY(event_id, set_num, game_num)
            );
            CREATE INDEX IF NOT EXISTS idx_processed_event ON processed_games(event_id);

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
            INSERT OR IGNORE INTO global_profile(id) VALUES (1);

            CREATE TABLE IF NOT EXISTS segment_stats (
                dimension TEXT NOT NULL,
                label TEXT NOT NULL,
                total INTEGER NOT NULL DEFAULT 0,
                hits INTEGER NOT NULL DEFAULT 0,
                misses INTEGER NOT NULL DEFAULT 0,
                unknown INTEGER NOT NULL DEFAULT 0,
                probability_sum REAL NOT NULL DEFAULT 0,
                PRIMARY KEY(dimension, label)
            );

            CREATE TABLE IF NOT EXISTS signal_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                event_id TEXT NOT NULL,
                player1 TEXT NOT NULL,
                player2 TEXT NOT NULL,
                set_num INTEGER NOT NULL,
                game_num INTEGER NOT NULL,
                game_band TEXT NOT NULL,
                server TEXT NOT NULL,
                score_state TEXT NOT NULL,
                server_score_state TEXT NOT NULL DEFAULT '',
                probability REAL NOT NULL,
                raw_probability REAL NOT NULL,
                data_quality TEXT NOT NULL,
                server_service_games INTEGER NOT NULL DEFAULT 0,
                server_deuce_games INTEGER NOT NULL DEFAULT 0,
                server_service_points INTEGER NOT NULL DEFAULT 0,
                server_points_won INTEGER NOT NULL DEFAULT 0,
                global_service_games INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at REAL NOT NULL,
                resolved_at REAL
            );
            CREATE INDEX IF NOT EXISTS idx_signal_history_status ON signal_history(status);
            CREATE INDEX IF NOT EXISTS idx_signal_history_created ON signal_history(created_at);
            """
        )
        # Миграция старой v5-БД без потери статистики.
        cols = {str(r[1]) for r in self.conn.execute("PRAGMA table_info(processed_games)").fetchall()}
        if "processed_at" not in cols:
            self.conn.execute("ALTER TABLE processed_games ADD COLUMN processed_at REAL NOT NULL DEFAULT 0")

        pending_cols = {str(r[1]) for r in self.conn.execute("PRAGMA table_info(pending_predictions)").fetchall()}
        pending_migrations = {
            "server_score_state": "TEXT NOT NULL DEFAULT ''",
            "server_service_games": "INTEGER NOT NULL DEFAULT 0",
            "server_deuce_games": "INTEGER NOT NULL DEFAULT 0",
            "server_service_points": "INTEGER NOT NULL DEFAULT 0",
            "server_points_won": "INTEGER NOT NULL DEFAULT 0",
            "global_service_games": "INTEGER NOT NULL DEFAULT 0",
        }
        for name, ddl in pending_migrations.items():
            if name not in pending_cols:
                self.conn.execute(f"ALTER TABLE pending_predictions ADD COLUMN {name} {ddl}")

        # Backfill счёта относительно подающего для уже ожидающих сигналов.
        rows = self.conn.execute(
            "SELECT key,player1,player2,server,score_state,server_score_state FROM pending_predictions"
        ).fetchall()
        for row in rows:
            if str(row["server_score_state"] or "").strip():
                continue
            try:
                a_s, b_s = str(row["score_state"]).split(":", 1)
                a, b = int(a_s), int(b_s)
                label = f"{a}:{b}" if str(row["server"]) == str(row["player1"]) else f"{b}:{a}"
            except Exception:
                label = "неизвестно"
            self.conn.execute(
                "UPDATE pending_predictions SET server_score_state=? WHERE key=?",
                (label, str(row["key"])),
            )

        # Существующие pending тоже являются реально выданными сигналами.
        self.conn.execute(
            """
            INSERT OR IGNORE INTO signal_history(
                key,event_id,player1,player2,set_num,game_num,game_band,server,score_state,server_score_state,
                probability,raw_probability,data_quality,server_service_games,server_deuce_games,
                server_service_points,server_points_won,global_service_games,status,created_at
            )
            SELECT key,event_id,player1,player2,set_num,game_num,game_band,server,score_state,server_score_state,
                   probability,raw_probability,data_quality,server_service_games,server_deuce_games,
                   server_service_points,server_points_won,global_service_games,'pending',created_at
            FROM pending_predictions
            """
        )
        self.conn.commit()

    @staticmethod
    def _key(event_id: str, set_num: int, game_num: int) -> str:
        return f"{event_id}:{set_num}:{game_num}"

    @staticmethod
    def _probability_bucket(probability: float) -> str:
        lo = max(0, min(95, int(probability // 5) * 5))
        return f"{lo:02d}-{lo + 5:02d}%"

    def ingest_completed_games(self, parsed: ParsedMatch) -> None:
        """Обновляет профиль подающих ровно один раз на каждый завершённый гейм."""
        for g in parsed.completed_games:
            exists = self.conn.execute(
                "SELECT 1 FROM processed_games WHERE event_id=? AND set_num=? AND game_num=?",
                (g.event_id, g.set_num, g.game_num),
            ).fetchone()
            if exists is not None:
                continue

            self.conn.execute(
                "INSERT INTO processed_games(event_id,set_num,game_num,server,processed_at) VALUES(?,?,?,?,?)",
                (g.event_id, g.set_num, g.game_num, g.server, time.time()),
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
                """
                UPDATE global_profile SET
                    service_games=service_games+1,
                    deuce_games=deuce_games+?,
                    service_points=service_points+?,
                    points_won=points_won+?
                WHERE id=1
                """,
                (int(g.deuce), g.service_points, g.points_won),
            )
        self.conn.commit()

    def player_profile(self, player: str) -> PlayerProfile:
        row = self.conn.execute(
            "SELECT player,service_games,deuce_games,service_points,points_won FROM player_profiles WHERE player=?",
            (player,),
        ).fetchone()
        if row is None:
            return PlayerProfile(player=player)
        return PlayerProfile(
            player=str(row["player"]),
            service_games=int(row["service_games"]),
            deuce_games=int(row["deuce_games"]),
            service_points=int(row["service_points"]),
            points_won=int(row["points_won"]),
        )

    def global_profile(self) -> PlayerProfile:
        row = self.conn.execute(
            "SELECT service_games,deuce_games,service_points,points_won FROM global_profile WHERE id=1"
        ).fetchone()
        return PlayerProfile(
            player="__GLOBAL__",
            service_games=int(row["service_games"]),
            deuce_games=int(row["deuce_games"]),
            service_points=int(row["service_points"]),
            points_won=int(row["points_won"]),
        )

    def estimate_server_parameters(self, server: str) -> tuple[float, float, PlayerProfile, PlayerProfile]:
        profile = self.player_profile(server)
        glob = self.global_profile()

        global_deuce = (
            DEFAULT_DEUCE_RATE * GLOBAL_DEUCE_PRIOR_STRENGTH + glob.deuce_games
        ) / (GLOBAL_DEUCE_PRIOR_STRENGTH + glob.service_games)
        server_deuce = (
            global_deuce * PLAYER_DEUCE_PRIOR_STRENGTH + profile.deuce_games
        ) / (PLAYER_DEUCE_PRIOR_STRENGTH + profile.service_games)
        server_deuce = min(deuce_rate_from_point_p(0.5), max(0.001, server_deuce))

        global_point_p = (
            DEFAULT_SERVER_POINT_P * GLOBAL_POINT_PRIOR_STRENGTH + glob.points_won
        ) / (GLOBAL_POINT_PRIOR_STRENGTH + glob.service_points)
        server_point_p = (
            global_point_p * PLAYER_POINT_PRIOR_STRENGTH + profile.points_won
        ) / (PLAYER_POINT_PRIOR_STRENGTH + profile.service_points)

        p_from_deuce = point_p_from_deuce_rate(server_deuce)
        if profile.service_points < 8:
            blend = 0.0
        else:
            blend = min(0.75, profile.service_points / (profile.service_points + 80.0))
        p_server = (1.0 - blend) * p_from_deuce + blend * server_point_p
        p_server = clamp(p_server, 0.50, 0.85)
        return server_deuce, p_server, profile, glob

    def _segment_row(self, dimension: str, label: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM segment_stats WHERE dimension=? AND label=?",
            (dimension, label),
        ).fetchone()

    def _calibration_correction(self, dimension: str, label: str, multiplier: float = 1.0) -> tuple[float, float] | None:
        row = self._segment_row(dimension, label)
        if row is None:
            return None
        total = int(row["total"])
        if total < EMPIRICAL_MIN_SIGNALS:
            return None
        actual = int(row["hits"]) / total * 100.0
        model_avg = float(row["probability_sum"]) / total
        correction = actual - model_avg
        weight = min(1.0, total / float(EMPIRICAL_FULL_WEIGHT)) * multiplier
        return correction, weight

    def calibrate_probability(
        self,
        raw_probability: float,
        raw_score_state: str,
        server_score_state: str,
        set_num: int,
        band: str,
    ) -> tuple[float, float]:
        """Консервативная online-калибровка только по прошлым выданным сигналам.

        Главные признаки: диапазон СЫРОЙ вероятности и счёт относительно подающего.
        Сет/номер гейма получают маленький вес, потому что по текущей выборке их
        различия слабее и легче переобучиться на шум.
        """
        corrections: list[tuple[float, float]] = []

        # 1) Самый важный контроль систематического завышения/занижения модели.
        c = self._calibration_correction("raw_probability", self._probability_bucket(raw_probability), 1.0)
        if c:
            corrections.append(c)

        # 2) Счёт именно относительно подающего: не смешиваем "подающий 30:15"
        #    и "принимающий 30:15" в одну статистику.
        c = self._calibration_correction("score_server", server_score_state, 1.0)
        if c:
            corrections.append(c)
        else:
            # Пока новая статистика не накопилась, старый raw-score используем лишь как слабый fallback.
            c = self._calibration_correction("score", raw_score_state, 0.35)
            if c:
                corrections.append(c)

        # 3) Сет и диапазон гейма — только небольшая дополнительная поправка.
        c_set = self._calibration_correction("cal_set", set_label(set_num), 0.25)
        if c_set:
            corrections.append(c_set)
        else:
            c = self._calibration_correction("set", set_label(set_num), 0.10)
            if c:
                corrections.append(c)

        c_game = self._calibration_correction("cal_game_band", band, 0.25)
        if c_game:
            corrections.append(c_game)
        else:
            c = self._calibration_correction("game_band", band, 0.10)
            if c:
                corrections.append(c)

        if not corrections:
            return raw_probability, 0.0
        denom = sum(w for _, w in corrections)
        adjustment = sum(corr * w for corr, w in corrections) / denom if denom else 0.0
        adjustment = clamp(adjustment, -MAX_CALIBRATION_ADJUSTMENT_PP, MAX_CALIBRATION_ADJUSTMENT_PP)
        return clamp(raw_probability + adjustment, 0.1, 99.9), adjustment

    def build_prediction(self, parsed: ParsedMatch, match_link: str) -> Prediction:
        server_num = get_server_for_game(
            parsed.current_set, parsed.current_game, parsed.first_server_match, parsed.max_game_by_set
        )
        server = parsed.player1 if server_num == 1 else parsed.player2
        posterior_deuce, p_server, profile, glob = self.estimate_server_parameters(server)

        raw_a, raw_b = parsed.raw_score_a, parsed.raw_score_b
        a, b = valid_point_score(raw_a), valid_point_score(raw_b)
        current_score = score_label(raw_a, raw_b)
        server_score_state = server_score_label(raw_a, raw_b, server_num)
        already = (parsed.current_set, parsed.current_game) in parsed.deuce_games
        market_available = is_market_available(raw_a, raw_b)

        if already:
            raw_probability = 100.0
        elif a is not None and b is not None:
            p1_point = p_server if server_num == 1 else 1.0 - p_server
            raw_probability = prob_reach_deuce_from_score(a, b, p1_point) * 100.0
        else:
            raw_probability = posterior_deuce * 100.0

        band = game_band(parsed.current_game)
        calibrated, adjustment = self.calibrate_probability(
            raw_probability, current_score, server_score_state, parsed.current_set, band
        )
        quality = data_quality_label(profile.service_games, profile.service_points)

        return Prediction(
            event_id=parsed.event_id,
            player1=parsed.player1,
            player2=parsed.player2,
            set_num=parsed.current_set,
            game_num=parsed.current_game,
            server=server,
            current_score=current_score,
            game_band=band,
            server_service_games=profile.service_games,
            server_deuce_games=profile.deuce_games,
            server_service_points=profile.service_points,
            server_points_won=profile.points_won,
            global_service_games=glob.service_games,
            posterior_deuce_rate=posterior_deuce,
            estimated_server_point_p=p_server,
            raw_probability_deuce=round(raw_probability, 1),
            probability_deuce=round(calibrated, 1),
            calibration_adjustment=round(adjustment, 1),
            data_quality=quality,
            confidence=quality,
            server_score_state=server_score_state,
            already_deuce=already,
            market_available=market_available,
            match_link=match_link,
            live_probability_deuce=round(calibrated, 1),
        )

    def _segment_entries_from_row(self, row: sqlite3.Row) -> list[tuple[str, str, float]]:
        final_p = float(row["probability"])
        raw_p = float(row["raw_probability"])
        set_num = int(row["set_num"])
        score_state = str(row["score_state"])
        server_score_state = str(row["server_score_state"] or "неизвестно")
        band = str(row["game_band"])
        server = str(row["server"])
        return [
            # Основные пользовательские отчёты — по реально показанной вероятности.
            ("overall", "Все сигналы", final_p),
            ("probability", self._probability_bucket(final_p), final_p),
            ("score", score_state, final_p),
            ("set", set_label(set_num), final_p),
            ("game_band", band, final_p),
            ("server", server, final_p),
            ("data_quality", str(row["data_quality"]), final_p),
            ("combo", f"{server_score_state} подающий | {set_label(set_num)} | {band}", final_p),

            # Отдельные чистые сегменты для будущей калибровки: сравниваем факт именно с RAW-моделью,
            # чтобы калибровка не обучалась на собственной уже скорректированной вероятности.
            ("raw_probability", self._probability_bucket(raw_p), raw_p),
            ("score_server", server_score_state, raw_p),
            ("cal_set", set_label(set_num), raw_p),
            ("cal_game_band", band, raw_p),
        ]

    def _update_segment(self, dimension: str, label: str, probability: float, status: str) -> None:
        if status == "unknown":
            self.conn.execute(
                """
                INSERT INTO segment_stats(dimension,label,unknown)
                VALUES(?,?,1)
                ON CONFLICT(dimension,label) DO UPDATE SET unknown=unknown+1
                """,
                (dimension, label),
            )
            return

        hit = int(status == "hit")
        miss = int(status == "miss")
        self.conn.execute(
            """
            INSERT INTO segment_stats(dimension,label,total,hits,misses,probability_sum)
            VALUES(?,?,1,?,?,?)
            ON CONFLICT(dimension,label) DO UPDATE SET
                total=total+1,
                hits=hits+excluded.hits,
                misses=misses+excluded.misses,
                probability_sum=probability_sum+excluded.probability_sum
            """,
            (dimension, label, hit, miss, probability),
        )

    def _resolve_key(self, key: str, status: str) -> None:
        row = self.conn.execute("SELECT * FROM pending_predictions WHERE key=?", (key,)).fetchone()
        if row is None:
            return
        for dimension, label, model_probability in self._segment_entries_from_row(row):
            self._update_segment(dimension, label, model_probability, status)
        self.conn.execute(
            "UPDATE signal_history SET status=?, resolved_at=? WHERE key=?",
            (status, time.time(), key),
        )
        self.conn.execute("DELETE FROM pending_predictions WHERE key=?", (key,))
        self.conn.commit()

    def pending_for(self, event_id: str, set_num: int, game_num: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM pending_predictions WHERE key=?",
            (self._key(event_id, set_num, game_num),),
        ).fetchone()

    def observe(self, observation: MatchObservation, min_probability: float) -> None:
        event_id = observation.event_id
        pred = observation.prediction
        current = (pred.set_num, pred.game_num)
        self.missing_cycles[event_id] = 0
        current_key = self._key(event_id, pred.set_num, pred.game_num)

        # Если текущий гейм дошёл до 40:40 — ранее показанный сигнал сработал.
        if current in observation.deuce_games:
            self._resolve_key(current_key, "hit")

        # Более ранние геймы точно завершились. Здесь MISS уже надёжен.
        rows = self.conn.execute(
            "SELECT key,set_num,game_num FROM pending_predictions WHERE event_id=?",
            (event_id,),
        ).fetchall()
        for row in rows:
            sg = (int(row["set_num"]), int(row["game_num"]))
            if sg < current:
                self._resolve_key(row["key"], "hit" if sg in observation.deuce_games else "miss")

        # Ровно один сигнал на гейм: первый момент, когда модель выше порога и рынок открыт.
        if (
            pred.already_deuce
            or current in observation.deuce_games
            or not pred.market_available
            or pred.probability_deuce < min_probability
        ):
            return

        # Если этот гейм уже когда-либо выдавался как сигнал (даже UNKNOWN после обрыва),
        # повторно его не считаем и не показываем как новый сигнал.
        if self.conn.execute("SELECT 1 FROM signal_history WHERE key=?", (current_key,)).fetchone() is not None:
            return

        if self.conn.execute("SELECT 1 FROM pending_predictions WHERE key=?", (current_key,)).fetchone() is None:
            created = time.time()
            values = (
                current_key, event_id, pred.player1, pred.player2, pred.set_num, pred.game_num,
                pred.game_band, pred.server, pred.current_score, pred.server_score_state,
                pred.probability_deuce, pred.raw_probability_deuce, pred.data_quality,
                pred.server_service_games, pred.server_deuce_games, pred.server_service_points,
                pred.server_points_won, pred.global_service_games, created,
            )
            self.conn.execute(
                """
                INSERT INTO pending_predictions(
                    key,event_id,player1,player2,set_num,game_num,game_band,server,score_state,server_score_state,
                    probability,raw_probability,data_quality,server_service_games,server_deuce_games,
                    server_service_points,server_points_won,global_service_games,created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                values,
            )
            self.conn.execute(
                """
                INSERT INTO signal_history(
                    key,event_id,player1,player2,set_num,game_num,game_band,server,score_state,server_score_state,
                    probability,raw_probability,data_quality,server_service_games,server_deuce_games,
                    server_service_points,server_points_won,global_service_games,status,created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'pending',?)
                """,
                values,
            )
            self.conn.commit()

    def finish_missing_matches(self, active_event_ids: set[str]) -> None:
        # Для UNKNOWN следим только за матчами, где реально был выдан сигнал.
        tracked_ids = {
            str(r[0]) for r in self.conn.execute(
                "SELECT DISTINCT event_id FROM pending_predictions"
            ).fetchall()
        }
        for event_id in tracked_ids:
            if event_id in active_event_ids:
                self.missing_cycles[event_id] = 0
                continue

            self.missing_cycles[event_id] = self.missing_cycles.get(event_id, 0) + 1
            if self.missing_cycles[event_id] < self.missing_cycles_to_unknown:
                continue

            # Исчезновение матча НЕ считается проигрышем.
            rows = self.conn.execute(
                "SELECT key FROM pending_predictions WHERE event_id=?",
                (event_id,),
            ).fetchall()
            for row in rows:
                self._resolve_key(str(row["key"]), "unknown")
            self.missing_cycles.pop(event_id, None)

        # processed_games держим ограниченное время. Это защищает профиль от
        # двойного учета, если матч на 30–60 секунд пропал из live и вернулся.
        now = time.time()
        if now - self._last_maintenance >= 900.0:
            self.cleanup_old_processed_games(PROCESSED_GAME_RETENTION_HOURS)
            # Даже если eventId формально остаётся в liveEvents, зависший сигнал
            # не может висеть в pending бесконечно. Старые сигналы → UNKNOWN.
            self.cleanup_stale_pending(STALE_HOURS)
            self.cleanup_signal_history(SIGNAL_HISTORY_LIMIT)
            self._last_maintenance = now

    def cleanup_old_processed_games(self, max_age_hours: float = 48.0) -> int:
        cutoff = time.time() - max_age_hours * 3600.0
        cur = self.conn.execute(
            "DELETE FROM processed_games WHERE processed_at > 0 AND processed_at < ?",
            (cutoff,),
        )
        # Строки старой схемы processed_at=0 безопасно удаляем при первой уборке.
        cur2 = self.conn.execute("DELETE FROM processed_games WHERE processed_at = 0")
        self.conn.commit()
        return int(cur.rowcount) + int(cur2.rowcount)

    def cleanup_stale_pending(self, max_age_hours: float = 12.0) -> int:
        cutoff = time.time() - max_age_hours * 3600
        rows = self.conn.execute(
            "SELECT key FROM pending_predictions WHERE created_at < ?", (cutoff,)
        ).fetchall()
        for row in rows:
            self._resolve_key(str(row["key"]), "unknown")
        return len(rows)

    def cleanup_signal_history(self, max_rows: int = SIGNAL_HISTORY_LIMIT) -> int:
        total = int(self.conn.execute("SELECT COUNT(*) FROM signal_history").fetchone()[0])
        excess = max(0, total - max_rows)
        if excess <= 0:
            return 0
        ids = [int(r[0]) for r in self.conn.execute(
            "SELECT id FROM signal_history WHERE status != 'pending' ORDER BY id ASC LIMIT ?",
            (excess,),
        ).fetchall()]
        if not ids:
            return 0
        self.conn.executemany("DELETE FROM signal_history WHERE id=?", [(i,) for i in ids])
        self.conn.commit()
        return len(ids)

    def pending_count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM pending_predictions").fetchone()[0])

    def overall_summary(self) -> dict[str, Any]:
        row = self._segment_row("overall", "Все сигналы")
        if row is None:
            return {"total": 0, "hits": 0, "misses": 0, "unknown": 0, "rate": 0.0}
        total = int(row["total"])
        hits = int(row["hits"])
        misses = int(row["misses"])
        unknown = int(row["unknown"])
        return {
            "total": total,
            "hits": hits,
            "misses": misses,
            "unknown": unknown,
            "rate": hits / total * 100.0 if total else 0.0,
        }

    def print_summary(self) -> None:
        s = self.overall_summary()
        print(
            f"📚 Проверено: {s['total']} | ✅ {s['hits']} | ❌ {s['misses']} | "
            f"⚪ неизвестно: {s['unknown']} | ⏳ ожидание: {self.pending_count()} | "
            f"проходимость: {s['rate']:.1f}%"
        )

    def close(self) -> None:
        self.conn.close()



class V6Tracker(StatsTracker):
    """V6: не пытается угадывать deuce через одну p(очко).

    Основная вероятность на 30:30 строится напрямую по фактам 30:30 -> 40:40
    и по pressure-переходам 30:30 -> 40:30/30:40 -> 40:40.
    Старый v5 signal_history используется только как обучающий prior.
    Forward hit/miss V6 хранится в отдельной БД.
    """

    def __init__(self, db_path: str, missing_cycles_to_unknown: int = 15, training_db: str = TRAINING_DB):
        self.training_db = training_db
        self._pressure_cache_at = 0.0
        self._pressure_cache: dict[str, Any] | None = None
        self.legacy_global = (0, 0)
        self.legacy_premium = (0, 0)
        self.legacy_server: dict[str, tuple[int, int]] = {}
        self.legacy_receiver: dict[str, tuple[int, int]] = {}
        self.legacy_pair: dict[tuple[str, str], tuple[int, int]] = {}
        super().__init__(db_path, missing_cycles_to_unknown)
        self._init_v6_tables()
        self._load_legacy_training()

    def _init_v6_tables(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS research_pressure (
                event_id TEXT NOT NULL,
                set_num INTEGER NOT NULL,
                game_num INTEGER NOT NULL,
                server TEXT NOT NULL,
                receiver TEXT NOT NULL,
                had_3030 INTEGER NOT NULL DEFAULT 0,
                reached_deuce INTEGER NOT NULL DEFAULT 0,
                first_after_server INTEGER,
                lead_n INTEGER NOT NULL DEFAULT 0,
                lead_return INTEGER NOT NULL DEFAULT 0,
                trail_n INTEGER NOT NULL DEFAULT 0,
                trail_save INTEGER NOT NULL DEFAULT 0,
                first_3030_in_match INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                PRIMARY KEY(event_id,set_num,game_num)
            );
            CREATE INDEX IF NOT EXISTS idx_pressure_server ON research_pressure(server);
            CREATE INDEX IF NOT EXISTS idx_pressure_receiver ON research_pressure(receiver);
            CREATE INDEX IF NOT EXISTS idx_pressure_3030 ON research_pressure(had_3030,first_3030_in_match,game_num);
            """
        )
        self.conn.commit()

    @staticmethod
    def _add_count(d: dict[Any, list[int]], key: Any, hit: int) -> None:
        if key not in d:
            d[key] = [0, 0]
        d[key][0] += int(hit)
        d[key][1] += 1

    def _load_legacy_training(self) -> None:
        p = Path(self.training_db)
        if not p.exists() or p.resolve() == Path(self.db_path).resolve():
            return
        try:
            c = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=2)
            c.row_factory = sqlite3.Row
            rows = c.execute(
                """SELECT event_id,player1,player2,server,game_num,server_score_state,score_state,status,created_at
                   FROM signal_history
                   WHERE status IN ('hit','miss')
                   ORDER BY created_at,id"""
            ).fetchall()
        except Exception as e:
            print(f"[V6 prior] Не удалось прочитать {p.name}: {e}", file=sys.stderr)
            return
        finally:
            try:
                c.close()
            except Exception:
                pass

        g_h = g_n = 0
        premium_h = premium_n = 0
        server_d: dict[str, list[int]] = {}
        recv_d: dict[str, list[int]] = {}
        pair_d: dict[tuple[str, str], list[int]] = {}
        seen_event: set[str] = set()

        for r in rows:
            state = str(r["server_score_state"] or r["score_state"] or "")
            if state != "30:30":
                continue
            hit = int(str(r["status"]) == "hit")
            event_id = str(r["event_id"])
            server = str(r["server"])
            p1, p2 = str(r["player1"]), str(r["player2"])
            receiver = p2 if server == p1 else p1
            g_h += hit
            g_n += 1
            self._add_count(server_d, server, hit)
            self._add_count(recv_d, receiver, hit)
            self._add_count(pair_d, (server, receiver), hit)
            if event_id not in seen_event:
                seen_event.add(event_id)
                gn = int(r["game_num"])
                if STRICT_GAME_MIN <= gn <= STRICT_GAME_MAX:
                    premium_h += hit
                    premium_n += 1

        self.legacy_global = (g_h, g_n)
        self.legacy_premium = (premium_h, premium_n)
        self.legacy_server = {k: (v[0], v[1]) for k, v in server_d.items()}
        self.legacy_receiver = {k: (v[0], v[1]) for k, v in recv_d.items()}
        self.legacy_pair = {k: (v[0], v[1]) for k, v in pair_d.items()}
        if g_n:
            print(
                f"[V6 prior] v5 факты 30:30: {g_h}/{g_n} = {g_h/g_n*100:.1f}% | "
                f"первый 30:30, геймы {STRICT_GAME_MIN}-{STRICT_GAME_MAX}: "
                f"{premium_h}/{premium_n} = {(premium_h/premium_n*100 if premium_n else 0):.1f}%"
            )

    def ingest_completed_games(self, parsed: ParsedMatch) -> None:
        # Старые профили оставляем как дополнительную телеметрию.
        super().ingest_completed_games(parsed)

        completed = sorted(
            {(g.set_num, g.game_num) for g in parsed.completed_games}
        )
        seen_3030 = False
        changed = False
        for ss, gg in completed:
            try:
                srv_num = get_server_for_game(ss, gg, parsed.first_server_match, parsed.max_game_by_set)
            except Exception:
                continue
            server = parsed.player1 if srv_num == 1 else parsed.player2
            receiver = parsed.player2 if srv_num == 1 else parsed.player1
            states = _states_server_perspective(parsed.events, ss, gg, srv_num)
            f = _pressure_features_from_states(states)
            first = int(bool(f["had_3030"]) and not seen_3030)
            if f["had_3030"]:
                seen_3030 = True
            before = self.conn.total_changes
            self.conn.execute(
                """INSERT OR IGNORE INTO research_pressure(
                     event_id,set_num,game_num,server,receiver,had_3030,reached_deuce,
                     first_after_server,lead_n,lead_return,trail_n,trail_save,first_3030_in_match,created_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    parsed.event_id, ss, gg, server, receiver,
                    int(f["had_3030"]), int(f["reached_deuce"]),
                    f["first_after_server"], int(f["lead_n"]), int(f["lead_return"]),
                    int(f["trail_n"]), int(f["trail_save"]), first, time.time(),
                ),
            )
            changed = changed or (self.conn.total_changes > before)
        self.conn.commit()
        if changed:
            self._pressure_cache = None

    @staticmethod
    def _acc_init() -> dict[str, float]:
        return dict(n3030=0.0,h3030=0.0,nfirst=0.0,hfirst=0.0,nlead=0.0,hlead=0.0,ntrail=0.0,htrail=0.0)

    @staticmethod
    def _acc_add(a: dict[str, float], r: sqlite3.Row) -> None:
        if int(r["had_3030"]):
            a["n3030"] += 1
            a["h3030"] += int(r["reached_deuce"])
        if r["first_after_server"] is not None:
            a["nfirst"] += 1
            a["hfirst"] += int(r["first_after_server"])
        if int(r["lead_n"]):
            a["nlead"] += 1
            a["hlead"] += int(r["lead_return"])
        if int(r["trail_n"]):
            a["ntrail"] += 1
            a["htrail"] += int(r["trail_save"])

    def _snapshot(self) -> dict[str, Any]:
        now = time.time()
        if self._pressure_cache is not None and now - self._pressure_cache_at < PRESSURE_CACHE_SECONDS:
            return self._pressure_cache
        rows = self.conn.execute("SELECT * FROM research_pressure").fetchall()
        glob = self._acc_init()
        server: dict[str, dict[str, float]] = {}
        receiver: dict[str, dict[str, float]] = {}
        pair: dict[tuple[str, str], dict[str, float]] = {}
        premium = self._acc_init()
        for r in rows:
            self._acc_add(glob, r)
            sk, rk = str(r["server"]), str(r["receiver"])
            if sk not in server:
                server[sk] = self._acc_init()
            if rk not in receiver:
                receiver[rk] = self._acc_init()
            if (sk, rk) not in pair:
                pair[(sk, rk)] = self._acc_init()
            self._acc_add(server[sk], r)
            self._acc_add(receiver[rk], r)
            self._acc_add(pair[(sk, rk)], r)
            if int(r["first_3030_in_match"]) and STRICT_GAME_MIN <= int(r["game_num"]) <= STRICT_GAME_MAX:
                self._acc_add(premium, r)
        self._pressure_cache = dict(global_=glob, server=server, receiver=receiver, pair=pair, premium=premium, n_rows=len(rows))
        self._pressure_cache_at = now
        return self._pressure_cache

    @staticmethod
    def _smooth(h: float, n: float, prior: float, k: float) -> float:
        return (h + prior * k) / (n + k) if n + k else prior

    @staticmethod
    def _weighted_rate(parts: list[tuple[float, float]], fallback: float) -> float:
        den = sum(w for _, w in parts if w > 0)
        return sum(p*w for p, w in parts if w > 0) / den if den else fallback

    def _legacy_context_rate(self) -> tuple[float, float]:
        gh, gn = self.legacy_global
        global_p = gh / gn if gn else 0.464
        ph, pn = self.legacy_premium
        # Контекст 4–6 был сильным в истории, но shrink=80 не даёт 81 наблюдению
        # сразу превратиться в "62% истинной вероятности".
        ctx = self._smooth(ph * LEGACY_CONTEXT_WEIGHT, pn * LEGACY_CONTEXT_WEIGHT, global_p, 80.0)
        return global_p, ctx

    def _rate_for_scope(
        self,
        agg: dict[str, float] | None,
        field_h: str,
        field_n: str,
        prior: float,
        k: float,
    ) -> tuple[float, float]:
        if not agg:
            return prior, 0.0
        n = float(agg[field_n])
        h = float(agg[field_h])
        return self._smooth(h, n, prior, k), n

    def estimate_v6_probability(self, parsed: ParsedMatch, server: str, receiver: str) -> tuple[float, float, float, int]:
        snap = self._snapshot()
        legacy_global, legacy_ctx = self._legacy_context_rate()

        # Новая глобальная research-выборка постепенно перетягивает старый prior.
        g = snap["global_"]
        old_h, old_n = self.legacy_global
        eff_old_n = old_n * LEGACY_GLOBAL_WEIGHT
        eff_old_h = old_h * LEGACY_GLOBAL_WEIGHT
        new_n, new_h = g["n3030"], g["h3030"]
        global_direct = (eff_old_h + new_h) / (eff_old_n + new_n) if eff_old_n + new_n else legacy_global

        # Прямой контекст: первый 30:30 матча в геймах 4–6.
        pg = snap["premium"]
        old_ph, old_pn = self.legacy_premium
        context_direct = self._smooth(
            old_ph * LEGACY_CONTEXT_WEIGHT + pg["h3030"],
            old_pn * LEGACY_CONTEXT_WEIGHT + pg["n3030"],
            global_direct,
            80.0,
        )

        # Персональные поправки по прямому факту 30:30->40:40. Сильное shrinkage.
        sp, sn = self._rate_for_scope(snap["server"].get(server), "h3030", "n3030", global_direct, 45.0)
        rp, rn = self._rate_for_scope(snap["receiver"].get(receiver), "h3030", "n3030", global_direct, 45.0)
        pp, pn = self._rate_for_scope(snap["pair"].get((server, receiver)), "h3030", "n3030", global_direct, 80.0)

        # Старые player-сигналы используем только как слабую добавку.
        lsh, lsn = self.legacy_server.get(server, (0, 0))
        lrh, lrn = self.legacy_receiver.get(receiver, (0, 0))
        lph, lpn = self.legacy_pair.get((server, receiver), (0, 0))
        if lsn:
            sp = 0.7*sp + 0.3*self._smooth(lsh*0.5, lsn*0.5, global_direct, 45.0)
        if lrn:
            rp = 0.7*rp + 0.3*self._smooth(lrh*0.5, lrn*0.5, global_direct, 45.0)
        if lpn:
            pp = 0.8*pp + 0.2*self._smooth(lph*0.5, lpn*0.5, global_direct, 80.0)

        personal_parts = [(global_direct, 1.0)]
        personal_parts.append((sp, min(0.7, sn/45.0)))
        personal_parts.append((rp, min(0.7, rn/45.0)))
        personal_parts.append((pp, min(0.4, pn/80.0)))
        personal_direct = self._weighted_rate(personal_parts, global_direct)

        # Контекст — главный сигнал отбора; персонализация только аккуратно двигает его.
        direct = clamp(context_direct + 0.35*(personal_direct-global_direct), 0.30, 0.75)

        # Pressure-transition модель:
        # P(deuce | 30:30) =
        # P(S wins next)*P(R returns from 40:30) +
        # P(R wins next)*P(S saves from 30:40)
        def global_rate(hf: str, nf: str, fallback: float) -> float:
            return self._smooth(g[hf], g[nf], fallback, 50.0)

        gf = global_rate("hfirst","nfirst",0.50)
        gl = global_rate("hlead","nlead",0.50)
        gt = global_rate("htrail","ntrail",0.50)

        def combine_transition(hf: str, nf: str, gp: float, k: float) -> tuple[float, float]:
            sa = snap["server"].get(server)
            ra = snap["receiver"].get(receiver)
            pa = snap["pair"].get((server, receiver))
            sp2,sn2=self._rate_for_scope(sa,hf,nf,gp,k)
            rp2,rn2=self._rate_for_scope(ra,hf,nf,gp,k)
            pp2,pn2=self._rate_for_scope(pa,hf,nf,gp,k*1.6)
            parts=[(gp,1.0),(sp2,min(.8,sn2/k)),(rp2,min(.8,rn2/k)),(pp2,min(.4,pn2/(k*1.6)))]
            return self._weighted_rate(parts,gp), int(sn2+rn2+pn2)

        p_first, e1 = combine_transition("hfirst","nfirst",gf,35.0)
        p_lead, e2 = combine_transition("hlead","nlead",gl,35.0)
        p_trail, e3 = combine_transition("htrail","ntrail",gt,35.0)
        transition = clamp(p_first*p_lead + (1.0-p_first)*p_trail, 0.20, 0.80)

        global_transition_evidence = g["nfirst"] + g["nlead"] + g["ntrail"]
        tw = min(0.40, 0.40 * global_transition_evidence / 500.0)
        final = (1.0-tw)*direct + tw*transition
        evidence = int(new_n + e1 + e2 + e3)
        return clamp(final, 0.20, 0.80), transition, context_direct, evidence

    def build_prediction(self, parsed: ParsedMatch, match_link: str) -> Prediction:
        server_num = get_server_for_game(
            parsed.current_set, parsed.current_game, parsed.first_server_match, parsed.max_game_by_set
        )
        server = parsed.player1 if server_num == 1 else parsed.player2
        receiver = parsed.player2 if server_num == 1 else parsed.player1

        raw_a, raw_b = parsed.raw_score_a, parsed.raw_score_b
        current_score = score_label(raw_a, raw_b)
        server_state = server_score_label(raw_a, raw_b, server_num)
        already = (parsed.current_set, parsed.current_game) in parsed.deuce_games
        market_available = is_market_available(raw_a, raw_b)

        first_3030 = not _has_prior_3030(parsed)
        strict = (
            server_state == "30:30"
            and first_3030
            and STRICT_GAME_MIN <= parsed.current_game <= STRICT_GAME_MAX
            and not already
            and market_available
        )

        final, transition, context_p, evidence = self.estimate_v6_probability(parsed, server, receiver)
        profile = self.player_profile(server)
        glob = self.global_profile()
        quality = "много данных" if evidence >= 250 else "достаточно данных" if evidence >= 80 else "мало данных"
        reason = (
            f"STRICT: первый 30:30 матча, гейм {STRICT_GAME_MIN}–{STRICT_GAME_MAX}"
            if strict else
            "не проходит STRICT-фильтр"
        )
        return Prediction(
            event_id=parsed.event_id,
            player1=parsed.player1,
            player2=parsed.player2,
            set_num=parsed.current_set,
            game_num=parsed.current_game,
            server=server,
            current_score=current_score,
            game_band=game_band(parsed.current_game),
            server_service_games=profile.service_games,
            server_deuce_games=profile.deuce_games,
            server_service_points=profile.service_points,
            server_points_won=profile.points_won,
            global_service_games=glob.service_games,
            posterior_deuce_rate=final,
            estimated_server_point_p=0.5,  # больше не выдаём фиктивную "силу p" за главный параметр
            raw_probability_deuce=round(transition*100.0,1),
            probability_deuce=round(final*100.0,1),
            calibration_adjustment=round((final-transition)*100.0,1),
            data_quality=quality,
            confidence=quality,
            server_score_state=server_state,
            already_deuce=already,
            market_available=market_available,
            match_link=match_link,
            live_probability_deuce=round(final*100.0,1),
            strict_eligible=strict,
            transition_probability=round(transition*100.0,1),
            context_probability=round(context_p*100.0,1),
            signal_reason=reason,
        )

    def observe(self, observation: MatchObservation, min_probability: float) -> None:
        event_id = observation.event_id
        pred = observation.prediction
        current = (pred.set_num, pred.game_num)
        self.missing_cycles[event_id] = 0
        current_key = self._key(event_id, pred.set_num, pred.game_num)

        # Разрешение ранее выданных сигналов — как в v5.2.
        if current in observation.deuce_games:
            self._resolve_key(current_key, "hit")

        rows = self.conn.execute(
            "SELECT key,set_num,game_num FROM pending_predictions WHERE event_id=?",
            (event_id,),
        ).fetchall()
        for row in rows:
            sg = (int(row["set_num"]), int(row["game_num"]))
            if sg < current:
                self._resolve_key(row["key"], "hit" if sg in observation.deuce_games else "miss")

        # НОВЫЙ сигнал только STRICT. Не позволяем калибровке вытянуть 15:30 и т.п. выше порога.
        if (
            not pred.strict_eligible
            or pred.already_deuce
            or current in observation.deuce_games
            or not pred.market_available
            or pred.probability_deuce < min_probability
        ):
            return

        if self.conn.execute("SELECT 1 FROM signal_history WHERE key=?", (current_key,)).fetchone() is not None:
            return
        if self.conn.execute("SELECT 1 FROM pending_predictions WHERE key=?", (current_key,)).fetchone() is not None:
            return

        created = time.time()
        values = (
            current_key, event_id, pred.player1, pred.player2, pred.set_num, pred.game_num,
            pred.game_band, pred.server, pred.current_score, pred.server_score_state,
            pred.probability_deuce, pred.raw_probability_deuce, pred.data_quality,
            pred.server_service_games, pred.server_deuce_games, pred.server_service_points,
            pred.server_points_won, pred.global_service_games, created,
        )
        self.conn.execute(
            """INSERT INTO pending_predictions(
               key,event_id,player1,player2,set_num,game_num,game_band,server,score_state,server_score_state,
               probability,raw_probability,data_quality,server_service_games,server_deuce_games,
               server_service_points,server_points_won,global_service_games,created_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            values,
        )
        self.conn.execute(
            """INSERT INTO signal_history(
               key,event_id,player1,player2,set_num,game_num,game_band,server,score_state,server_score_state,
               probability,raw_probability,data_quality,server_service_games,server_deuce_games,
               server_service_points,server_points_won,global_service_games,status,created_at
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'pending',?)""",
            values,
        )
        self.conn.commit()


def build_legacy_listing(pred: Prediction, parsed: ParsedMatch) -> list[Any]:
    first_server_name = pred.player1 if parsed.first_server_match == 1 else pred.player2
    listing: list[Any] = [
        f"Начал подачу: {first_server_name} | Гейм: {pred.game_num} | Cет: {pred.set_num}  |  {pred.player1} | {pred.player2}"
    ]

    for s, g in sorted(parsed.deuce_games):
        try:
            srv_num = get_server_for_game(s, g, parsed.first_server_match, parsed.max_game_by_set)
            srv_name = pred.player1 if srv_num == 1 else pred.player2
        except Exception:
            srv_name = "-"
        listing.append({
            "score_chel_1": 40,
            "score_chel_2": 40,
            "game": f"{g} | Подача: {srv_name}",
            "set_": s,
            "link": pred.match_link,
        })

    report = (
        f"🎾 Прогноз: Гейм {pred.game_num}, Сет {pred.set_num}\n"
        f"📊 Вероятность 40:40: {pred.probability_deuce:.1f}%\n"
        f"🎯 Подаёт: {pred.server}\n"
        f"📚 Объём данных: {pred.data_quality}\n"
        f"📍 Счёт при сигнале: {pred.signal_score or pred.current_score}\n"
        f"🎾 Счёт относительно подающего: {pred.server_score_state or '—'}\n"
        f"⏱ Текущий счёт: {pred.current_score}\n\n"
        f"🔍 Ход анализа:\n"
        f"   👥 {pred.player1} vs {pred.player2}\n"
        f"   🎾 Подающий: {pred.server_deuce_games}/{pred.server_service_games} его геймов дошли до 40:40\n"
        f"   🎯 Наблюдаемые очки на подаче: {pred.server_points_won}/{pred.server_service_points}\n"
        f"   📈 Базовая модель: {pred.raw_probability_deuce:.1f}%\n"
        f"   🧪 Поправка по накопленной статистике: {pred.calibration_adjustment:+.1f} п.п.\n"
        f"   ✅ Оценка при сигнале: {pred.probability_deuce:.1f}%\n"
        f"   ⏱ Текущая оценка модели: {pred.live_probability_deuce:.1f}%\n"
        f"   🔗 Матч: {pred.match_link}"
    )
    listing.append(report)
    return listing


def scan_live(
    client: PariClient,
    tracker: StatsTracker,
    min_probability: float = 0.0,
) -> tuple[list[Prediction], set[str], list[list[Any]]]:
    predictions: list[Prediction] = []
    seen_codes: set[str] = set()
    legacy_rows: list[list[Any]] = []

    live = client.live_events()
    active_event_ids = {str(x.get("id")) for x in live if x.get("id") is not None}

    for misc in live:
        comment = str(misc.get("comment", ""))
        if "(" not in comment:
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

            tracker.ingest_completed_games(parsed)
            pred = tracker.build_prediction(parsed, "https://pari.ru/")
            obs = MatchObservation(
                event_id=str(event_id),
                prediction=pred,
                deuce_games=set(parsed.deuce_games),
            )

            # Сначала фиксируем/закрываем сигнал. Это делает статистику и отображение
            # одним и тем же событием: в БД хранится именно тот первый сигнал, который увидел пользователь.
            tracker.observe(obs, min_probability)
            pending = tracker.pending_for(str(event_id), pred.set_num, pred.game_num)

            # Показываем только живой pending-сигнал, пока рынок ещё доступен.
            if pending is not None and not pred.already_deuce and pred.market_available:
                live_p = pred.probability_deuce
                pred.signal_score = str(pending["score_state"])
                pred.server_score_state = str(pending["server_score_state"] or pred.server_score_state)
                pred.live_probability_deuce = live_p
                pred.probability_deuce = float(pending["probability"])
                pred.raw_probability_deuce = float(pending["raw_probability"])
                pred.data_quality = str(pending["data_quality"])
                pred.confidence = pred.data_quality
                pred.match_link = client.event_link(event_id)
                predictions.append(pred)
                legacy_rows.append(build_legacy_listing(pred, parsed))

    predictions.sort(key=lambda x: x.probability_deuce, reverse=True)
    legacy_rows.sort(
        key=lambda row: float(re.search(r"Вероятность 40:40:\s*([0-9.]+)%", row[-1]).group(1))
        if row and isinstance(row[-1], str) and re.search(r"Вероятность 40:40:\s*([0-9.]+)%", row[-1]) else 0.0,
        reverse=True,
    )
    return predictions, active_event_ids, legacy_rows


def analyze_legacy_listing(listing: list[Any]) -> Prediction | None:
    """Упрощённый offline-разбор старого tennis.json только для просмотра."""
    if not listing or not isinstance(listing[0], str):
        return None
    header = listing[0]
    gm = re.search(r"Гейм:\s*(\d+)", header, re.I)
    sm = re.search(r"[СсCc]ет:\s*(\d+)", header, re.I)
    if not gm or not sm:
        return None
    game_num, set_num = int(gm.group(1)), int(sm.group(1))
    parts = [p.strip() for p in header.split("|")]
    player1 = parts[-2] if len(parts) >= 2 else "Игрок1"
    player2 = parts[-1] if len(parts) >= 1 else "Игрок2"
    deuce_rows = [x for x in listing[1:] if isinstance(x, dict)]
    deuce_keys = {(int(x.get("set_", 0)), int(str(x.get("game", "0")).split()[0])) for x in deuce_rows}
    deuces = sum(1 for s, g in deuce_keys if s < set_num or (s == set_num and g < game_num))
    max_by_set: dict[int, int] = {}
    for s, g in deuce_keys:
        max_by_set[s] = max(max_by_set.get(s, 0), g)
    total = max(game_num - 1, 0)
    for s, max_g in max_by_set.items():
        if 0 < s < set_num:
            total += max(6, max_g)
    rate = (DEFAULT_DEUCE_RATE * 10 + deuces) / (10 + max(total, 0))
    rate = min(deuce_rate_from_point_p(0.5), max(0.001, rate))
    p = point_p_from_deuce_rate(rate)
    m = re.search(r"Начал подачу:\s*(.*?)\s*\|", header)
    first_name = m.group(1).strip() if m else player1
    first_server = 1 if first_name == player1 else 2
    server_num = get_current_server(set_num, game_num, first_server)
    server = player1 if server_num == 1 else player2
    return Prediction(
        event_id=None,
        player1=player1,
        player2=player2,
        set_num=set_num,
        game_num=game_num,
        server=server,
        current_score="нет в legacy JSON",
        game_band=game_band(game_num),
        server_service_games=total,
        server_deuce_games=deuces,
        server_service_points=0,
        server_points_won=0,
        global_service_games=total,
        posterior_deuce_rate=rate,
        estimated_server_point_p=p,
        raw_probability_deuce=round(rate * 100, 1),
        probability_deuce=round(rate * 100, 1),
        calibration_adjustment=0.0,
        data_quality=data_quality_label(total, 0),
        confidence=data_quality_label(total, 0),
    )


def run_offline(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("В offline-файле ожидается JSON-массив")
    results = [p for row in data if isinstance(row, list) for p in [analyze_legacy_listing(row)] if p]
    results.sort(key=lambda x: x.probability_deuce, reverse=True)
    print(f"Проверено записей: {len(results)}\n")
    for p in results:
        print(p.pretty())
        print("-" * 72)
    return 0


def run_live(args: argparse.Namespace) -> int:
    client = PariClient(timeout=args.timeout)
    tracker = V6Tracker(args.db, missing_cycles_to_unknown=args.missing_cycles_to_unknown, training_db=TRAINING_DB)
    stale = tracker.cleanup_stale_pending(args.stale_hours)

    print(f"HTTP backend: {HTTP_BACKEND}")
    print("Источник: PARI live/sportscast")
    print(f"Статистика: {args.db} | старых pending → UNKNOWN: {stale}")
    print(f"V6 STRICT: новый сигнал ТОЛЬКО на первом 30:30 матча и только в геймах {STRICT_GAME_MIN}–{STRICT_GAME_MAX}.")
    print("15:30 / 30:15 / 15:40 / 40:15 больше НЕ становятся ставочными сигналами.")
    print("Исчезновение матча больше НЕ считается проигрышем: это UNKNOWN.")
    print("V6 учит прямой 30:30→40:40 + pressure-переходы 40:30/30:40, server/receiver/pair.")
    print("Старая v5 история используется только как prior; hit/miss V6 считаются с нуля в отдельной БД.")
    print(f"Минимальный сигнал: {args.min_probability:.1f}% (меньше сигналов, выше ожидаемая точность отбора).")
    print("Сырые события матчей не сохраняются; сохраняется только компактная история реально выданных сигналов.")
    print("Ctrl+C для остановки\n")

    try:
        while True:
            started = time.time()
            try:
                predictions, active_event_ids, legacy_rows = scan_live(
                    client, tracker, min_probability=args.min_probability
                )
                tracker.finish_missing_matches(active_event_ids)

                print("\n" + "=" * 92)
                print(time.strftime("%Y-%m-%d %H:%M:%S"), f"| найдено сигналов: {len(predictions)}")
                for p in predictions[: args.top]:
                    print(p.pretty())
                    print("-" * 76)
                tracker.print_summary()

                Path(args.output).write_text(
                    json.dumps([asdict(p) for p in predictions], ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                Path(LEGACY_OUTPUT).write_text(
                    json.dumps(legacy_rows, ensure_ascii=False),
                    encoding="utf-8",
                )
            except KeyboardInterrupt:
                print("\nСТОП")
                return 0
            except Exception as e:
                print(f"[цикл] {type(e).__name__}: {e}", file=sys.stderr)

            elapsed = time.time() - started
            time.sleep(max(0.0, args.interval - elapsed))
    finally:
        tracker.close()


def self_test() -> None:
    # 1) Чистая математика деуса.
    for p in (0.5, 0.58, 0.65):
        got = prob_reach_deuce_from_score(0, 0, p)
        exp = deuce_rate_from_point_p(p)
        assert abs(got - exp) < 1e-12, (p, got, exp)
    assert abs(prob_reach_deuce_from_score(3, 2, 0.62) - 0.38) < 1e-12
    assert abs(prob_reach_deuce_from_score(2, 3, 0.62) - 0.62) < 1e-12
    assert is_market_available(40, 30) is False
    assert is_market_available(30, 40) is False
    assert is_market_available(40, 40) is False
    assert is_market_available(30, 30) is True
    assert is_market_available(None, None) is False
    assert server_score_label(30, 15, 1) == "30:15"
    assert server_score_label(30, 15, 2) == "15:30"

    # Порядок подачи между сетами зависит от числа геймов в предыдущем сете.
    # После 6:4 (10 геймов) первый подающий 2-го сета тот же.
    assert get_server_for_game(2, 1, 1, {1: 10, 2: 1}) == 1
    # После 6:3 (9 геймов) — другой.
    assert get_server_for_game(2, 1, 1, {1: 9, 2: 1}) == 2

    # 2) Парсинг матча и очков.
    events = [
        {"type": 1123, "i3": 1},
        {"type": 1125, "i1": 1, "i2": 1, "i8": 0, "i9": 0},
        {"type": 999, "i1": 1, "i2": 1, "i8": 15, "i9": 0},
        {"type": 999, "i1": 1, "i2": 1, "i8": 15, "i9": 15},
        {"type": 999, "i1": 1, "i2": 1, "i8": 30, "i9": 15},
        {"type": 999, "i1": 1, "i2": 1, "i8": 30, "i9": 30},
        {"type": 999, "i1": 1, "i2": 1, "i8": 40, "i9": 30},
        {"type": 999, "i1": 1, "i2": 1, "i8": 40, "i9": 40},
        {"type": 1125, "i1": 1, "i2": 2, "i8": 0, "i9": 0},
    ]
    parsed = parse_match(123, "A", "B", events)
    assert parsed is not None
    assert parsed.current_set == 1 and parsed.current_game == 2
    assert len(parsed.completed_games) == 1
    cg = parsed.completed_games[0]
    assert cg.server == "A" and cg.deuce is True
    assert cg.service_points == 6 and cg.points_won == 3

    # V6 pressure-state extraction.
    pf = _pressure_features_from_states([(0,0),(15,0),(15,15),(30,15),(30,30),(40,30),(40,40)])
    assert pf["had_3030"] == 1 and pf["reached_deuce"] == 1
    assert pf["first_after_server"] == 1 and pf["lead_return"] == 1

    # 3) БД: HIT/MISS/UNKNOWN, профили, сегменты и очистка временных матчей.
    tr = StatsTracker(":memory:", missing_cycles_to_unknown=1)
    tr.ingest_completed_games(parsed)
    prof = tr.player_profile("A")
    assert prof.service_games == 1 and prof.deuce_games == 1
    assert prof.service_points == 6 and prof.points_won == 3

    pred = tr.build_prediction(parsed, "https://pari.ru/sports/tennis/1/2")
    # Принудительно создаём сигнал текущего гейма.
    pred.probability_deuce = 60.0
    obs = MatchObservation("123", pred, set(parsed.deuce_games))
    tr.observe(obs, 35.0)
    assert tr.pending_count() == 1
    assert tr.conn.execute("SELECT COUNT(*) FROM signal_history").fetchone()[0] == 1
    # Матч исчез → UNKNOWN, а не MISS.
    tr.finish_missing_matches(set())
    assert tr.pending_count() == 0
    s = tr.overall_summary()
    assert s["total"] == 0 and s["unknown"] == 1
    assert tr.conn.execute("SELECT status FROM signal_history WHERE event_id='123'").fetchone()[0] == "unknown"
    # Если тот же гейм после обрыва снова появился, второй сигнал не создаём.
    tr.observe(obs, 35.0)
    assert tr.pending_for("123", pred.set_num, pred.game_num) is None
    assert tr.conn.execute("SELECT COUNT(*) FROM signal_history WHERE event_id='123'").fetchone()[0] == 1

    # Новый сигнал и следующий гейм без деуса → настоящий MISS.
    pred2 = tr.build_prediction(parsed, "https://pari.ru/sports/tennis/1/2")
    pred2.probability_deuce = 60.0
    tr.observe(MatchObservation("124", pred2, set()), 35.0)
    assert tr.pending_count() == 1
    next_pred = Prediction(**{**asdict(pred2), "event_id": "124", "game_num": 3, "current_score": "0:0"})
    tr.observe(MatchObservation("124", next_pred, set()), 99.0)
    s = tr.overall_summary()
    assert s["total"] == 1 and s["misses"] == 1

    # 4) Если сигнал НЕ был выдан, исход не должен попасть в статистику.
    hist_before = tr.conn.execute("SELECT COUNT(*) FROM signal_history").fetchone()[0]
    no_signal_pred = Prediction(**{**asdict(pred2), "event_id": "200", "game_num": 4, "current_score": "0:0", "probability_deuce": 20.0})
    tr.observe(MatchObservation("200", no_signal_pred, set()), 35.0)
    assert tr.pending_for("200", no_signal_pred.set_num, no_signal_pred.game_num) is None
    next_no_signal = Prediction(**{**asdict(no_signal_pred), "event_id": "200", "game_num": 5, "current_score": "0:0"})
    tr.observe(MatchObservation("200", next_no_signal, set()), 99.0)
    s2 = tr.overall_summary()
    assert s2["total"] == 1 and s2["hits"] == 0 and s2["misses"] == 1
    assert tr.conn.execute("SELECT COUNT(*) FROM signal_history").fetchone()[0] == hist_before

    # 5) На закрытых счетах новый сигнал не создаётся.
    for eid, score in (("301", "40:30"), ("302", "30:40"), ("303", "40:40"), ("304", "None:None")):
        a_raw, b_raw = score.split(":", 1)
        a_val = None if a_raw == "None" else int(a_raw)
        b_val = None if b_raw == "None" else int(b_raw)
        closed_parsed = ParsedMatch(
            event_id=eid, player1="A", player2="B", first_server_match=1,
            current_set=1, current_game=2, raw_score_a=a_val, raw_score_b=b_val,
            deuce_games={(1, 2)} if score == "40:40" else set(), completed_games=[],
            max_game_by_set={1: 2}, events=[]
        )
        cp = tr.build_prediction(closed_parsed, "https://pari.ru/")
        cp.probability_deuce = 99.0
        tr.observe(MatchObservation(eid, cp, set(closed_parsed.deuce_games)), 35.0)
        assert tr.pending_for(eid, 1, 2) is None

    tr.close()


def main() -> int:
    args = argparse.Namespace(
        interval=INTERVAL,
        timeout=TIMEOUT,
        min_probability=MIN_PROBABILITY,
        top=TOP,
        output=OUTPUT,
        db=DB,
        missing_cycles_to_unknown=MISSING_CYCLES_TO_UNKNOWN,
        stale_hours=STALE_HOURS,
    )

    mode = str(RUN_MODE).strip().lower()
    if mode in {"self-test", "selftest", "test"}:
        self_test()
        print("SELF-TEST: OK")
        return 0
    if mode == "offline":
        return run_offline(Path(OFFLINE_FILE))
    if mode != "live":
        raise ValueError(f"Неизвестный RUN_MODE={RUN_MODE!r}. Используй: 'live', 'offline' или 'self-test'.")
    return run_live(args)


if __name__ == "__main__":
    raise SystemExit(main())
