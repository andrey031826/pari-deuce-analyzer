from __future__ import annotations

# V7.4 SEEK DUAL BRAIN: Model A = online logistic with fear+search drive, Model B = XGBoost shadow comparison.
# Online learner получает reward/punishment после каждого завершённого гейма.
# Координаты PARI фиксированы: player1 всегда слева, player2 всегда справа.

import argparse
import json
import re
import math
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
MIN_PROBABILITY = 0.0      # V7 не использует фиксированный порог вероятности; оставлено для совместимости
MIN_SIGNAL_STRENGTH = 76.0  # стартовый порог силы сигнала 0..100; дальше политика адаптирует его сама
V7_THRESHOLD_FLOOR = 68.0
V7_THRESHOLD_CEIL = 92.0
V7_EARLY_SCORES = {(0, 0), (15, 0), (0, 15), (15, 15)}
V7_MIN_MATCH_GAMES = 2
V7_MIN_SAME_ORIENTATION_GAMES = 1
V7_MODEL_LR = 0.06
V7_MODEL_L2 = 0.002

# V7.3 shadow-model comparison. XGBoost NEVER controls the live signal yet.
V73_XGB_MIN_GAMES = 80
V73_XGB_RETRAIN_EVERY = 20
V73_XGB_MAX_TRAIN_GAMES = 5000
V73_XGB_MIN_POSITIVES = 12
V73_DUAL_BUILD = 'V7.4-SEEK-DUAL-R5-2026-09-23'

# V7.4 SEEK: fear and search are separate states.
# A false confident signal must still hurt, but fear is NOT allowed to solve the task
# simply by closing the gate. Missed real deuces create search pressure and are
# deliberately strong positive training examples for the online learner.
V7_HIT_REWARD = 4.0
V7_MISS_PENALTY = -4.0
V7_MISSED_DEUCE_PENALTY = -0.50
V7_CORRECT_SILENCE_REWARD = 0.01
V7_PAIN_MAX = 100.0
V7_SEARCH_MAX = 100.0
V7_TARGET_PRECISION = 0.18      # with ~8.0 odds, above 12.5% break-even with a safety margin
V7_MIN_ACCEPTABLE_PRECISION = 0.14
V7_TARGET_RECALL = 0.28         # do not allow a precise model to catch only a tiny fraction of deuces
V7_POLICY_WINDOW = 120
V7_POLICY_VERSION = 2.0
TOP = 20                   # максимум сигналов на экран
OUTPUT = str(BASE_DIR / "pari_predictions.json")
LEGACY_OUTPUT = str(BASE_DIR / "tennis.json")  # для существующей веб-морды
DB = str(BASE_DIR / "pari_deuce_v7.sqlite3")  # свежая forward-статистика V7
PREVIOUS_V6_DB = str(BASE_DIR / "pari_deuce_v6.sqlite3")  # только research/profile bootstrap, НЕ hit/miss
TRAINING_DB = str(BASE_DIR / "pari_model_stats_v5.sqlite3")  # старые ФАКТЫ используются только как prior
STRICT_GAME_MIN = 4
STRICT_GAME_MAX = 6
STRICT_SET_NUM = 1
STYLE_PRIOR_GAMES = 20.0
STYLE_EFFECT_MAX_PP = 2.5
CONTEXT_PRIOR_STRENGTH = 8.0
TRANSITION_MAX_WEIGHT = 0.25
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
    signal_strength: float = 0.0       # V7: рейтинг кандидата 0..100, НЕ вероятность
    online_probability: float = 0.0    # V7: online-logistic оценка вероятности deuce
    policy_threshold: float = 0.0      # V7: текущий адаптивный порог силы
    matchup_state: str = ""            # P1 serve/P2 return или P1 return/P2 serve
    history_summary: str = ""          # краткая история нужной ориентации подачи
    learning_reward: float = 0.0
    training_games: int = 0

    def pretty(self) -> str:
        if str(self.signal_reason).startswith("V7"):
            return "\n".join([
                f"🎾 {self.player1} — {self.player2}",
                f"   Сет {self.set_num}, гейм {self.game_num} | подаёт: {self.server}",
                f"   Текущий счёт PARI: {self.current_score}",
                f"   🧭 {self.matchup_state}",
                f"   🧠 Сила сигнала: {self.signal_strength:.1f}/100 | порог: {self.policy_threshold:.1f}",
                (f"   📈 Online learner: разогрев ({self.training_games}/50) | history-ranker: {self.context_probability:.1f}/100"
                 if self.training_games < 50 else
                 f"   📈 Online P(deuce): {self.online_probability:.1f}% | history-ranker: {self.context_probability:.1f}/100"),
                f"   📚 {self.history_summary}",
                f"   Режим: {self.signal_reason}",
                f"   🔗 {self.match_link}",
            ])
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

    v7_mode = str(pred.signal_reason).startswith("V7")
    if v7_mode:
        online_line = (
            f"📈 Online learner: разогрев ({pred.training_games}/50)\n"
            if pred.training_games < 50
            else f"📈 Online-оценка deuce: {pred.online_probability:.1f}%\n"
        )
        headline = (
            f"🧠 Сила сигнала V7: {pred.signal_strength:.1f}/100\n"
            + online_line
            + f"🎚 Порог: {pred.policy_threshold:.1f}/100\n"
        )
        details = (
            f"   🧠 History-ranker: {pred.context_probability:.1f}/100\n"
            + (f"   🤖 Online learner: разогрев {pred.training_games}/50\n"
               if pred.training_games < 50 else f"   🤖 Online learner: {pred.online_probability:.1f}%\n")
            + f"   🧩 Итоговая сила: {pred.signal_strength:.1f}/100\n"
            + f"   🎚 Порог модели: {pred.policy_threshold:.1f}/100\n"
        )
    else:
        headline = f"📊 Вероятность 40:40: {pred.probability_deuce:.1f}%\n"
        details = (
            f"   📈 Базовая модель: {pred.raw_probability_deuce:.1f}%\n"
            f"   🧪 Поправка по накопленной статистике: {pred.calibration_adjustment:+.1f} п.п.\n"
            f"   ✅ Оценка при сигнале: {pred.probability_deuce:.1f}%\n"
        )
    report = (
        f"🎾 Прогноз: Гейм {pred.game_num}, Сет {pred.set_num}\n"
        + headline +
        f"🎯 Подаёт: {pred.server}\n"
        f"📚 Объём данных: {pred.data_quality}\n"
        f"📍 Счёт при сигнале: {pred.signal_score or pred.current_score}\n"
        f"🎾 Счёт относительно подающего: {pred.server_score_state or '—'}\n"
        f"⏱ Текущий счёт: {pred.current_score}\n"
        f"🧭 {pred.matchup_state or ''}\n"
        f"🧠 {pred.history_summary or ''}\n\n"
        f"🔍 Ход анализа:\n"
        f"   👥 {pred.player1} vs {pred.player2}\n"
        + details +
        f"   🔗 Матч: {pred.match_link}"
    )
    listing.append(report)
    return listing




class V61Tracker(V6Tracker):
    """V6.1: precision-first extension of V6.

    Important: the spectacular historical split by *final* player_profiles is not
    treated as a backtest because it leaks future games into old signals. We do use
    that profile as a frozen prior for FUTURE games, with strong shrinkage and a
    capped effect. The set-1 filter and pressure transition model remain causal.
    """

    def __init__(
        self,
        db_path: str,
        missing_cycles_to_unknown: int = 15,
        training_db: str = TRAINING_DB,
        previous_v6_db: str | None = PREVIOUS_V6_DB,
    ):
        self.previous_v6_db = previous_v6_db
        self.legacy_style: dict[str, tuple[int, int, int, int]] = {}
        self.legacy_style_global = (0, 0, 0, 0)  # games,deuces,points,points_won
        self.legacy_premium_set1 = (0, 0)
        self._last_diag: dict[str, Any] = {}
        super().__init__(db_path, missing_cycles_to_unknown, training_db)
        self._bootstrap_previous_v6()
        self._load_v61_priors()

    def _bootstrap_previous_v6(self) -> None:
        """Carry forward research facts/profiles only; never copy old V6 hit/miss stats."""
        if not self.previous_v6_db:
            return
        p = Path(self.previous_v6_db)
        if not p.exists() or p.resolve() == Path(self.db_path).resolve():
            return
        if int(self.conn.execute("SELECT COUNT(*) FROM research_pressure").fetchone()[0]) > 0:
            return
        try:
            old = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=2)
            old.row_factory = sqlite3.Row
            rrows = old.execute("SELECT * FROM research_pressure").fetchall()
            for r in rrows:
                self.conn.execute(
                    """INSERT OR IGNORE INTO research_pressure(
                       event_id,set_num,game_num,server,receiver,had_3030,reached_deuce,
                       first_after_server,lead_n,lead_return,trail_n,trail_save,first_3030_in_match,created_at
                       ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    tuple(r[k] for k in (
                        "event_id","set_num","game_num","server","receiver","had_3030","reached_deuce",
                        "first_after_server","lead_n","lead_return","trail_n","trail_save","first_3030_in_match","created_at"
                    )),
                )
            try:
                prows = old.execute("SELECT player,service_games,deuce_games,service_points,points_won,updated_at FROM player_profiles").fetchall()
                for r in prows:
                    self.conn.execute(
                        """INSERT OR REPLACE INTO player_profiles(player,service_games,deuce_games,service_points,points_won,updated_at)
                           VALUES(?,?,?,?,?,?)""",
                        (r["player"],r["service_games"],r["deuce_games"],r["service_points"],r["points_won"],r["updated_at"]),
                    )
                g = old.execute("SELECT service_games,deuce_games,service_points,points_won FROM global_profile WHERE id=1").fetchone()
                if g:
                    self.conn.execute(
                        """UPDATE global_profile SET service_games=?,deuce_games=?,service_points=?,points_won=? WHERE id=1""",
                        (g["service_games"],g["deuce_games"],g["service_points"],g["points_won"]),
                    )
            except sqlite3.Error:
                pass
            self.conn.commit()
            self._pressure_cache = None
            print(f"[V6.1 bootstrap] перенесено research_pressure: {len(rrows)}; hit/miss НЕ переносились")
        except Exception as e:
            print(f"[V6.1 bootstrap] {e}", file=sys.stderr)
        finally:
            try:
                old.close()
            except Exception:
                pass

    def _load_v61_priors(self) -> None:
        p = Path(self.training_db)
        if not p.exists():
            return
        try:
            c = sqlite3.connect(f"file:{p}?mode=ro", uri=True, timeout=2)
            c.row_factory = sqlite3.Row
            for r in c.execute("SELECT player,service_games,deuce_games,service_points,points_won FROM player_profiles"):
                self.legacy_style[str(r["player"])] = (
                    int(r["service_games"]), int(r["deuce_games"]), int(r["service_points"]), int(r["points_won"])
                )
            g = c.execute("SELECT service_games,deuce_games,service_points,points_won FROM global_profile WHERE id=1").fetchone()
            if g:
                self.legacy_style_global = (int(g[0]), int(g[1]), int(g[2]), int(g[3]))

            rows = c.execute(
                """SELECT event_id,set_num,game_num,server_score_state,score_state,status,created_at,id
                   FROM signal_history WHERE status IN ('hit','miss') ORDER BY created_at,id"""
            ).fetchall()
            seen: set[str] = set()
            h = n = 0
            for r in rows:
                state = str(r["server_score_state"] or r["score_state"] or "")
                eid = str(r["event_id"])
                if state != "30:30" or eid in seen:
                    continue
                seen.add(eid)
                if int(r["set_num"]) == STRICT_SET_NUM and STRICT_GAME_MIN <= int(r["game_num"]) <= STRICT_GAME_MAX:
                    n += 1
                    h += int(str(r["status"]) == "hit")
            self.legacy_premium_set1 = (h, n)
            if n:
                print(f"[V6.1 prior] set 1 + first 30:30 + games {STRICT_GAME_MIN}-{STRICT_GAME_MAX}: {h}/{n} = {h/n*100:.1f}%")
        except Exception as e:
            print(f"[V6.1 prior] {e}", file=sys.stderr)
        finally:
            try:
                c.close()
            except Exception:
                pass

    def _combined_style(self, server: str) -> tuple[float, int, float, float]:
        """Returns posterior deuce-style, evidence games, global style, serve-point prior."""
        lg, ld, lp, lw = self.legacy_style.get(server, (0, 0, 0, 0))
        cur = self.player_profile(server)
        n = lg + cur.service_games
        d = ld + cur.deuce_games
        pt = lp + cur.service_points
        pw = lw + cur.points_won

        gg, gd, gp, gw = self.legacy_style_global
        cg = self.global_profile()
        gn = gg + cg.service_games
        gdn = gd + cg.deuce_games
        gpt = gp + cg.service_points
        gpw = gw + cg.points_won
        global_style = gdn / gn if gn else DEFAULT_DEUCE_RATE
        global_point = gpw / gpt if gpt else DEFAULT_SERVER_POINT_P
        style = (d + global_style * STYLE_PRIOR_GAMES) / (n + STYLE_PRIOR_GAMES) if n + STYLE_PRIOR_GAMES else global_style
        point_p = (pw + global_point * 40.0) / (pt + 40.0) if pt else global_point
        return clamp(style, 0.05, 0.60), n, clamp(global_style,0.05,0.60), clamp(point_p,0.35,0.75)

    def _premium_set1_current(self) -> tuple[int, int]:
        r = self.conn.execute(
            """SELECT COALESCE(SUM(reached_deuce),0), COUNT(*) FROM research_pressure
               WHERE had_3030=1 AND first_3030_in_match=1 AND set_num=? AND game_num BETWEEN ? AND ?""",
            (STRICT_SET_NUM, STRICT_GAME_MIN, STRICT_GAME_MAX),
        ).fetchone()
        return int(r[0]), int(r[1])

    def estimate_v6_probability(self, parsed: ParsedMatch, server: str, receiver: str) -> tuple[float, float, float, int]:
        snap = self._snapshot()
        legacy_global, _ = self._legacy_context_rate()
        g = snap["global_"]
        old_h, old_n = self.legacy_global
        eff_old_n = old_n * LEGACY_GLOBAL_WEIGHT
        eff_old_h = old_h * LEGACY_GLOBAL_WEIGHT
        new_n, new_h = g["n3030"], g["h3030"]
        global_direct = (eff_old_h + new_h) / (eff_old_n + new_n) if eff_old_n + new_n else legacy_global

        # Context is now SET 1 only. Set 2 is research-only until it proves itself forward.
        old_ph, old_pn = self.legacy_premium_set1
        new_ph, new_pn = self._premium_set1_current()
        context_direct = self._smooth(
            old_ph * LEGACY_CONTEXT_WEIGHT + new_ph,
            old_pn * LEGACY_CONTEXT_WEIGHT + new_pn,
            global_direct,
            CONTEXT_PRIOR_STRENGTH,
        )

        # Direct 30:30 personal evidence (causal, strongly shrunk).
        sp, sn = self._rate_for_scope(snap["server"].get(server), "h3030", "n3030", global_direct, 45.0)
        rp, rn = self._rate_for_scope(snap["receiver"].get(receiver), "h3030", "n3030", global_direct, 45.0)
        pp, pn = self._rate_for_scope(snap["pair"].get((server, receiver)), "h3030", "n3030", global_direct, 80.0)
        lsh, lsn = self.legacy_server.get(server, (0, 0))
        lrh, lrn = self.legacy_receiver.get(receiver, (0, 0))
        lph, lpn = self.legacy_pair.get((server, receiver), (0, 0))
        if lsn:
            sp = 0.75*sp + 0.25*self._smooth(lsh*0.5, lsn*0.5, global_direct, 45.0)
        if lrn:
            rp = 0.75*rp + 0.25*self._smooth(lrh*0.5, lrn*0.5, global_direct, 45.0)
        if lpn:
            pp = 0.85*pp + 0.15*self._smooth(lph*0.5, lpn*0.5, global_direct, 80.0)
        personal_direct = self._weighted_rate([
            (global_direct,1.0),
            (sp,min(0.65,sn/45.0)),
            (rp,min(0.65,rn/45.0)),
            (pp,min(0.30,pn/80.0)),
        ], global_direct)

        # Server deuce-style: useful as a FUTURE prior, but not as the leaked +20pp backtest.
        style, style_n, global_style, serve_p = self._combined_style(server)
        reliability = min(1.0, style_n / 30.0)
        style_adj = clamp((style - global_style) * 0.20 * reliability,
                          -STYLE_EFFECT_MAX_PP/100.0, STYLE_EFFECT_MAX_PP/100.0)
        direct = clamp(context_direct + 0.35*(personal_direct-global_direct) + style_adj, 0.30, 0.75)

        # Pressure transitions. Instead of dead 0.50 fallbacks, use the server's
        # shrunken point-on-serve prior: first point / save BP ~ serve_p,
        # receiver return from 40:30 ~ 1-serve_p. Real research_pressure overrides it.
        def global_rate(hf: str, nf: str, fallback: float) -> float:
            return self._smooth(g[hf], g[nf], fallback, 35.0)

        gf = global_rate("hfirst","nfirst",serve_p)
        gl = global_rate("hlead","nlead",1.0-serve_p)
        gt = global_rate("htrail","ntrail",serve_p)

        def combine_transition(hf: str, nf: str, gp0: float, k: float) -> tuple[float, int]:
            sa = snap["server"].get(server); ra = snap["receiver"].get(receiver); pa = snap["pair"].get((server,receiver))
            sp2,sn2=self._rate_for_scope(sa,hf,nf,gp0,k)
            rp2,rn2=self._rate_for_scope(ra,hf,nf,gp0,k)
            pp2,pn2=self._rate_for_scope(pa,hf,nf,gp0,k*1.8)
            val=self._weighted_rate([
                (gp0,1.0),(sp2,min(.65,sn2/k)),(rp2,min(.65,rn2/k)),(pp2,min(.25,pn2/(k*1.8)))
            ],gp0)
            return val, int(sn2+rn2+pn2)

        p_first,e1=combine_transition("hfirst","nfirst",gf,30.0)
        p_lead,e2=combine_transition("hlead","nlead",gl,30.0)
        p_trail,e3=combine_transition("htrail","ntrail",gt,30.0)
        transition=clamp(p_first*p_lead + (1.0-p_first)*p_trail,0.20,0.80)
        global_transition_evidence=g["nfirst"]+g["nlead"]+g["ntrail"]
        tw=min(TRANSITION_MAX_WEIGHT, TRANSITION_MAX_WEIGHT*global_transition_evidence/1000.0)
        final=(1.0-tw)*direct + tw*transition
        evidence=int(new_n+e1+e2+e3+style_n)
        self._last_diag = {
            "style": style, "style_n": style_n, "global_style": global_style,
            "style_adj": style_adj, "serve_p": serve_p,
            "context": context_direct, "transition": transition,
            "p_first": p_first, "p_lead": p_lead, "p_trail": p_trail,
        }
        return clamp(final,0.20,0.80),transition,context_direct,evidence

    def build_prediction(self, parsed: ParsedMatch, match_link: str) -> Prediction:
        pred = super().build_prediction(parsed, match_link)
        # V6.1 precision gate: only first set. This is a filter, not an invented probability bonus.
        pred.strict_eligible = bool(pred.strict_eligible and parsed.current_set == STRICT_SET_NUM)
        diag = self._last_diag or {}
        pred.estimated_server_point_p = float(diag.get("serve_p", 0.5))
        style = float(diag.get("style", 0.0))*100.0
        sn = int(diag.get("style_n", 0))
        sa = float(diag.get("style_adj",0.0))*100.0
        if pred.strict_eligible:
            pred.signal_reason = (
                f"V6.1: 1-й сет; первый 30:30 матча; гейм {STRICT_GAME_MIN}–{STRICT_GAME_MAX}; "
                f"стиль подающего {style:.1f}% (n={sn}, поправка {sa:+.1f} п.п.)"
            )
        else:
            pred.signal_reason = "V6.1: research-only (не проходит строгий фильтр)"
        return pred


# ============================== V7 MATCH BRAIN ==============================
# V7 принципиально НЕ ждёт 30:30. Решение принимается в начале/самом начале гейма
# по истории двух фиксированных направлений матча:
#   STATE_P1_SERVE: P1 подаёт / P2 принимает
#   STATE_P2_SERVE: P1 принимает / P2 подаёт
# Игроки в координатах PARI НИКОГДА не меняются местами.

V7_FEATURES = (
    "bias",
    "same_n", "same_deuce", "same_pressure", "same_recent_pressure",
    "same_balance", "same_trend", "same_last_deuce", "same_deuce_recency",
    "same_extreme_rate", "same_loss_streak",
    "match_n", "match_deuce", "match_pressure", "match_recent_pressure", "match_last_deuce",
    "server_service_deuce", "server_service_close",
    "receiver_return_deuce", "receiver_return_close",
    "pair_deuce", "pair_close",
    "legacy_server_deuce", "serve_point_balance",
    "set_progress", "game_progress",
)


def _v7_mean(xs: list[float], default: float = 0.0) -> float:
    return sum(xs) / len(xs) if xs else default


def _v7_terminal_signature(states_server: list[tuple[int, int]], deuce: bool) -> tuple[int, int, int]:
    """pressure 0..3, dominance -3..+3, server_won {-1,0,1}.

    dominance всегда относительно подающего. При deuce намеренно 0: до 40:40
    стороны уже доказали равенство; победитель после deuce для нашей цели не нужен.
    """
    if deuce:
        return 3, 0, 0
    terminal = None
    for a, b in reversed(states_server):
        if a == 40 and b in (0, 15, 30):
            terminal = (a, b)
            break
        if b == 40 and a in (0, 15, 30):
            terminal = (a, b)
            break
    if terminal is None:
        return 1, 0, 0
    a, b = terminal
    if a == 40:
        pressure = {0: 0, 15: 1, 30: 2}.get(b, 1)
        dominance = {0: 3, 15: 2, 30: 1}.get(b, 1)
        return pressure, dominance, 1
    pressure = {0: 0, 15: 1, 30: 2}.get(a, 1)
    dominance = -{0: 3, 15: 2, 30: 1}.get(a, 1)
    return pressure, dominance, -1


class V7Tracker(V61Tracker):
    """V7: match-history ranker + online learner + adaptive abstention policy.

    Обучение идёт НА КАЖДОМ завершённом гейме, а не только на выданных сигналах.
    Попал сигнал -> положительная награда. Промазал -> сильное наказание.
    Промолчал, но deuce случился -> маленький штраф за пропущенную возможность.
    Поэтому модель не может выгодно "спрятаться" и всегда молчать.
    """

    def __init__(self, db_path: str, missing_cycles_to_unknown: int = 15,
                 training_db: str = TRAINING_DB, previous_v6_db: str | None = PREVIOUS_V6_DB):
        self._v7_weights_cache: dict[str, float] | None = None
        super().__init__(db_path, missing_cycles_to_unknown, training_db, previous_v6_db)
        self._init_v7_tables()
        self._init_v7_model()

    def _init_v7_tables(self) -> None:
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS v7_game_facts(
            event_id TEXT NOT NULL,
            set_num INTEGER NOT NULL,
            game_num INTEGER NOT NULL,
            player1 TEXT NOT NULL,
            player2 TEXT NOT NULL,
            server_num INTEGER NOT NULL,
            server TEXT NOT NULL,
            receiver TEXT NOT NULL,
            pari_terminal TEXT NOT NULL,
            path_json TEXT NOT NULL,
            deuce INTEGER NOT NULL,
            had_3030 INTEGER NOT NULL,
            pressure INTEGER NOT NULL,
            dominance INTEGER NOT NULL,
            server_won INTEGER NOT NULL,
            comeback_server INTEGER NOT NULL,
            comeback_receiver INTEGER NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY(event_id,set_num,game_num)
        );
        CREATE INDEX IF NOT EXISTS idx_v7_fact_event ON v7_game_facts(event_id,set_num,game_num);
        CREATE INDEX IF NOT EXISTS idx_v7_fact_server ON v7_game_facts(server);
        CREATE INDEX IF NOT EXISTS idx_v7_fact_receiver ON v7_game_facts(receiver);

        CREATE TABLE IF NOT EXISTS v7_role_profiles(
            player TEXT PRIMARY KEY,
            service_games INTEGER NOT NULL DEFAULT 0,
            service_deuces INTEGER NOT NULL DEFAULT 0,
            service_close INTEGER NOT NULL DEFAULT 0,
            return_games INTEGER NOT NULL DEFAULT 0,
            return_deuces INTEGER NOT NULL DEFAULT 0,
            return_close INTEGER NOT NULL DEFAULT 0,
            updated_at REAL NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS v7_pair_profiles(
            server TEXT NOT NULL,
            receiver TEXT NOT NULL,
            games INTEGER NOT NULL DEFAULT 0,
            deuces INTEGER NOT NULL DEFAULT 0,
            close_games INTEGER NOT NULL DEFAULT 0,
            pressure_sum REAL NOT NULL DEFAULT 0,
            PRIMARY KEY(server,receiver)
        );
        CREATE TABLE IF NOT EXISTS v7_game_snapshots(
            key TEXT PRIMARY KEY,
            event_id TEXT NOT NULL,
            set_num INTEGER NOT NULL,
            game_num INTEGER NOT NULL,
            player1 TEXT NOT NULL,
            player2 TEXT NOT NULL,
            server_num INTEGER NOT NULL,
            server TEXT NOT NULL,
            receiver TEXT NOT NULL,
            feature_json TEXT NOT NULL,
            history_strength REAL NOT NULL,
            online_probability REAL NOT NULL,
            signal_strength REAL NOT NULL,
            threshold REAL NOT NULL,
            entry_score TEXT NOT NULL,
            terminal_score TEXT NOT NULL DEFAULT '',
            signal_issued INTEGER NOT NULL DEFAULT 0,
            resolved INTEGER NOT NULL DEFAULT 0,
            label INTEGER,
            reward REAL,
            created_at REAL NOT NULL,
            resolved_at REAL
        );
        CREATE INDEX IF NOT EXISTS idx_v7_snap_resolved ON v7_game_snapshots(resolved,created_at);
        CREATE TABLE IF NOT EXISTS v7_model_weights(
            feature TEXT PRIMARY KEY,
            weight REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS v7_meta(
            key TEXT PRIMARY KEY,
            value REAL NOT NULL
        );
        """)
        snap_cols = {str(r[1]) for r in self.conn.execute("PRAGMA table_info(v7_game_snapshots)").fetchall()}
        if "terminal_score" not in snap_cols:
            self.conn.execute("ALTER TABLE v7_game_snapshots ADD COLUMN terminal_score TEXT NOT NULL DEFAULT ''")
        self.conn.commit()

    def _meta_get(self, key: str, default: float) -> float:
        r = self.conn.execute("SELECT value FROM v7_meta WHERE key=?", (key,)).fetchone()
        return float(r[0]) if r else float(default)

    def _meta_set(self, key: str, value: float) -> None:
        self.conn.execute(
            "INSERT INTO v7_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, float(value)),
        )

    def _init_v7_model(self) -> None:
        n = int(self.conn.execute("SELECT COUNT(*) FROM v7_model_weights").fetchone()[0])
        if n == 0:
            base = clamp(DEFAULT_DEUCE_RATE, 0.05, 0.45)
            bias = math.log(base / (1.0 - base))
            for f in V7_FEATURES:
                self.conn.execute("INSERT INTO v7_model_weights(feature,weight) VALUES(?,?)", (f, bias if f == "bias" else 0.0))
        if self.conn.execute("SELECT 1 FROM v7_meta WHERE key='threshold'").fetchone() is None:
            self._meta_set('threshold', MIN_SIGNAL_STRENGTH)
        if self.conn.execute("SELECT 1 FROM v7_meta WHERE key='training_count'").fetchone() is None:
            self._meta_set('training_count', 0.0)
        if self.conn.execute("SELECT 1 FROM v7_meta WHERE key='last_policy_count'").fetchone() is None:
            self._meta_set('last_policy_count', 0.0)
        # V7.1 persistent score / pain state. Existing V7 DBs migrate automatically.
        defaults = {
            'score_balance': 0.0,
            'positive_points': 0.0,
            'negative_points': 0.0,
            'pain': 0.0,
            'search_drive': 0.0,
            'recent_precision': 0.0,
            'recent_recall': 0.0,
            'last_reward': 0.0,
            'hit_streak': 0.0,
            'miss_streak': 0.0,
            'missed_deuce_streak': 0.0,
            'seek_policy_version': 0.0,
        }
        for k, v in defaults.items():
            if self.conn.execute("SELECT 1 FROM v7_meta WHERE key=?", (k,)).fetchone() is None:
                self._meta_set(k, v)

        # One-time migration from the old "fear only" policy.  Preserve all learned
        # weights/history/score, but give the controller room to move again.  Search
        # drive is bootstrapped from recent missed-deuce recall instead of being guessed.
        if self._meta_get('seek_policy_version', 0.0) < V7_POLICY_VERSION:
            rows = self.conn.execute("""SELECT signal_issued,label FROM v7_game_snapshots
                WHERE resolved=1 AND label IS NOT NULL ORDER BY resolved_at DESC LIMIT ?""",
                (V7_POLICY_WINDOW,)).fetchall()
            deuces = sum(int(r['label'] or 0) for r in rows)
            caught = sum(1 for r in rows if int(r['signal_issued'] or 0) and int(r['label'] or 0))
            recall = (caught / deuces) if deuces else 0.0
            bootstrap_search = (clamp((V7_TARGET_RECALL - recall) / max(V7_TARGET_RECALL, 1e-6) * 100.0, 0.0, 100.0) if deuces else 0.0)
            self._meta_set('search_drive', max(self._meta_get('search_drive', 0.0), bootstrap_search))
            self._meta_set('recent_recall', recall * 100.0)
            # Old cumulative debt could pin pain=100 forever. Keep fear, but free it
            # from the lifetime score balance; from now on pain is a recent-state EWMA.
            self._meta_set('pain', min(self._meta_get('pain', 0.0), 75.0))
            self._meta_set('threshold', min(self._meta_get('threshold', MIN_SIGNAL_STRENGTH), 78.0))
            self._meta_set('seek_policy_version', V7_POLICY_VERSION)
        self.conn.commit()
        self._v7_weights_cache = None

    def _weights(self) -> dict[str, float]:
        if self._v7_weights_cache is None:
            self._v7_weights_cache = {str(r[0]): float(r[1]) for r in self.conn.execute("SELECT feature,weight FROM v7_model_weights")}
        return self._v7_weights_cache

    @staticmethod
    def _sigmoid(z: float) -> float:
        z = clamp(z, -25.0, 25.0)
        return 1.0 / (1.0 + math.exp(-z))

    def _online_probability(self, features: dict[str, float]) -> float:
        w = self._weights()
        z = sum(w.get(f, 0.0) * float(features.get(f, 0.0)) for f in V7_FEATURES)
        return self._sigmoid(z)

    def _online_update(self, features: dict[str, float], label: int, sample_weight: float) -> None:
        w = dict(self._weights())
        p = self._online_probability(features)
        n = self._meta_get('training_count', 0.0)
        lr = V7_MODEL_LR / math.sqrt(1.0 + n / 200.0)
        err = (float(label) - p) * float(sample_weight)
        for f in V7_FEATURES:
            x = float(features.get(f, 0.0))
            old = w.get(f, 0.0)
            reg = 0.0 if f == 'bias' else V7_MODEL_L2 * old
            new = clamp(old + lr * (err * x - reg), -5.0, 5.0)
            w[f] = new
            self.conn.execute("UPDATE v7_model_weights SET weight=? WHERE feature=?", (new, f))
        self._meta_set('training_count', n + 1.0)
        self._v7_weights_cache = w

    def _fact_from_completed(self, parsed: ParsedMatch, set_num: int, game_num: int) -> dict[str, Any] | None:
        try:
            server_num = get_server_for_game(set_num, game_num, parsed.first_server_match, parsed.max_game_by_set)
        except Exception:
            return None
        raw = _unique_score_states_for_game(parsed.events, set_num, game_num)
        if not raw:
            return None
        rel = raw if server_num == 1 else [(b, a) for a, b in raw]
        deuce = int((set_num, game_num) in parsed.deuce_games or (40, 40) in rel)
        pressure, dominance, server_won = _v7_terminal_signature(rel, bool(deuce))
        had_3030 = int((30, 30) in rel)
        comeback_server = int((0, 30) in rel and (30, 30) in rel and rel.index((0, 30)) < rel.index((30, 30)))
        comeback_receiver = int((30, 0) in rel and (30, 30) in rel and rel.index((30, 0)) < rel.index((30, 30)))
        last = raw[-1]
        server = parsed.player1 if server_num == 1 else parsed.player2
        receiver = parsed.player2 if server_num == 1 else parsed.player1
        return {
            'event_id': str(parsed.event_id), 'set_num': set_num, 'game_num': game_num,
            'player1': parsed.player1, 'player2': parsed.player2,
            'server_num': server_num, 'server': server, 'receiver': receiver,
            'pari_terminal': f"{last[0]}:{last[1]}",
            'path_json': json.dumps([[a,b] for a,b in raw], ensure_ascii=False),
            'deuce': deuce, 'had_3030': had_3030, 'pressure': pressure,
            'dominance': dominance, 'server_won': server_won,
            'comeback_server': comeback_server, 'comeback_receiver': comeback_receiver,
        }

    def ingest_completed_games(self, parsed: ParsedMatch) -> None:
        # Сохраняем старые агрегаты как дополнительный prior, но V7 учится на своей истории всех геймов.
        super().ingest_completed_games(parsed)
        changed = False
        for cg in parsed.completed_games:
            f = self._fact_from_completed(parsed, cg.set_num, cg.game_num)
            if not f:
                continue
            exists = self.conn.execute(
                "SELECT 1 FROM v7_game_facts WHERE event_id=? AND set_num=? AND game_num=?",
                (f['event_id'], f['set_num'], f['game_num'])
            ).fetchone()
            if exists is None:
                self.conn.execute("""INSERT INTO v7_game_facts(
                    event_id,set_num,game_num,player1,player2,server_num,server,receiver,pari_terminal,path_json,
                    deuce,had_3030,pressure,dominance,server_won,comeback_server,comeback_receiver,created_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (f['event_id'],f['set_num'],f['game_num'],f['player1'],f['player2'],f['server_num'],f['server'],f['receiver'],
                     f['pari_terminal'],f['path_json'],f['deuce'],f['had_3030'],f['pressure'],f['dominance'],f['server_won'],
                     f['comeback_server'],f['comeback_receiver'],time.time()))
                close = int(f['pressure'] >= 2)
                self.conn.execute("""INSERT INTO v7_role_profiles(player,service_games,service_deuces,service_close,updated_at)
                    VALUES(?,1,?,?,?) ON CONFLICT(player) DO UPDATE SET
                    service_games=service_games+1, service_deuces=service_deuces+excluded.service_deuces,
                    service_close=service_close+excluded.service_close, updated_at=excluded.updated_at""",
                    (f['server'],f['deuce'],close,time.time()))
                self.conn.execute("""INSERT INTO v7_role_profiles(player,return_games,return_deuces,return_close,updated_at)
                    VALUES(?,1,?,?,?) ON CONFLICT(player) DO UPDATE SET
                    return_games=return_games+1, return_deuces=return_deuces+excluded.return_deuces,
                    return_close=return_close+excluded.return_close, updated_at=excluded.updated_at""",
                    (f['receiver'],f['deuce'],close,time.time()))
                self.conn.execute("""INSERT INTO v7_pair_profiles(server,receiver,games,deuces,close_games,pressure_sum)
                    VALUES(?,?,1,?,?,?) ON CONFLICT(server,receiver) DO UPDATE SET
                    games=games+1,deuces=deuces+excluded.deuces,close_games=close_games+excluded.close_games,
                    pressure_sum=pressure_sum+excluded.pressure_sum""",
                    (f['server'],f['receiver'],f['deuce'],close,float(f['pressure'])))
                changed = True
            self._resolve_v7_snapshot(f['event_id'], f['set_num'], f['game_num'], int(f['deuce']), str(f['pari_terminal']))
        if changed:
            self.conn.commit()

    def _role_rates(self, player: str, role: str, prior_deuce: float) -> tuple[float, float, int]:
        r = self.conn.execute("SELECT * FROM v7_role_profiles WHERE player=?", (player,)).fetchone()
        if not r:
            return prior_deuce, 0.45, 0
        if role == 'service':
            n,d,c = int(r['service_games']),int(r['service_deuces']),int(r['service_close'])
        else:
            n,d,c = int(r['return_games']),int(r['return_deuces']),int(r['return_close'])
        deuce = (d + prior_deuce*8.0)/(n+8.0)
        close = (c + 0.45*6.0)/(n+6.0)
        return clamp(deuce,0.05,0.75),clamp(close,0.05,0.95),n

    def _legacy_server_prior(self, server: str) -> tuple[float, float]:
        ggames,gdeuces,gpoints,gwins = self.legacy_style_global
        base = gdeuces/ggames if ggames else DEFAULT_DEUCE_RATE
        pn = self.legacy_style.get(server)
        if pn:
            n,d,pts,wins = pn
            rate = (d + base*18.0)/(n+18.0)
            pp = (wins + DEFAULT_SERVER_POINT_P*35.0)/(pts+35.0) if pts else DEFAULT_SERVER_POINT_P
            return clamp(rate,0.05,0.65), clamp(pp,0.35,0.80)
        return base, DEFAULT_SERVER_POINT_P

    def _history_features(self, parsed: ParsedMatch, server_num: int) -> tuple[dict[str,float], float, str, str]:
        facts = self.conn.execute(
            "SELECT * FROM v7_game_facts WHERE event_id=? ORDER BY set_num,game_num",
            (str(parsed.event_id),)
        ).fetchall()
        server = parsed.player1 if server_num == 1 else parsed.player2
        receiver = parsed.player2 if server_num == 1 else parsed.player1
        same = [r for r in facts if int(r['server_num']) == server_num]

        prior_deuce, serve_p = self._legacy_server_prior(server)
        srv_deuce,srv_close,srv_n = self._role_rates(server,'service',prior_deuce)
        ret_deuce,ret_close,ret_n = self._role_rates(receiver,'return',prior_deuce)
        pr = self.conn.execute("SELECT games,deuces,close_games,pressure_sum FROM v7_pair_profiles WHERE server=? AND receiver=?", (server,receiver)).fetchone()
        if pr:
            pair_n=int(pr['games']); pair_deuce=(int(pr['deuces'])+prior_deuce*5)/(pair_n+5); pair_close=(int(pr['close_games'])+0.45*4)/(pair_n+4)
        else:
            pair_n=0; pair_deuce=prior_deuce; pair_close=0.45

        def vals(rows, key): return [float(r[key]) for r in rows]
        same_n=len(same); match_n=len(facts)
        same_deuce=(sum(int(r['deuce']) for r in same)+prior_deuce*2.0)/(same_n+2.0)
        same_pressure=_v7_mean([float(r['pressure'])/3.0 for r in same],0.45)
        same_recent=_v7_mean([float(r['pressure'])/3.0 for r in same[-3:]],same_pressure)
        match_deuce=(sum(int(r['deuce']) for r in facts)+prior_deuce*3.0)/(match_n+3.0)
        match_pressure=_v7_mean([float(r['pressure'])/3.0 for r in facts],0.45)
        match_recent=_v7_mean([float(r['pressure'])/3.0 for r in facts[-3:]],match_pressure)
        dom_mean=_v7_mean([float(r['dominance'])/3.0 for r in same],0.0)
        same_balance=clamp(1.0-abs(dom_mean),0.0,1.0)
        prev = same[:-2] if len(same)>2 else []
        prev_p=_v7_mean([float(r['pressure'])/3.0 for r in prev],same_pressure)
        recent2=_v7_mean([float(r['pressure'])/3.0 for r in same[-2:]],same_pressure)
        same_trend=clamp(recent2-prev_p,-1.0,1.0) if same_n>=2 else 0.0
        same_last_deuce=float(int(same[-1]['deuce'])) if same else 0.0
        match_last_deuce=float(int(facts[-1]['deuce'])) if facts else 0.0
        since=8
        for i,r in enumerate(reversed(same)):
            if int(r['deuce']): since=i; break
        same_deuce_recency=1.0/(1.0+since) if same else 0.0
        extreme_rate=_v7_mean([1.0 if abs(int(r['dominance']))>=2 else 0.0 for r in same[-3:]],0.0)
        loss_streak=0
        for r in reversed(same):
            if int(r['server_won']) < 0: loss_streak += 1
            else: break

        point_balance=clamp(1.0-abs(serve_p-0.5)/0.30,0.0,1.0)
        features = {
            'bias':1.0,
            'same_n':min(1.0,same_n/5.0), 'same_deuce':same_deuce, 'same_pressure':same_pressure,
            'same_recent_pressure':same_recent, 'same_balance':same_balance, 'same_trend':same_trend,
            'same_last_deuce':same_last_deuce, 'same_deuce_recency':same_deuce_recency,
            'same_extreme_rate':extreme_rate, 'same_loss_streak':min(1.0,loss_streak/3.0),
            'match_n':min(1.0,match_n/10.0), 'match_deuce':match_deuce, 'match_pressure':match_pressure,
            'match_recent_pressure':match_recent, 'match_last_deuce':match_last_deuce,
            'server_service_deuce':srv_deuce, 'server_service_close':srv_close,
            'receiver_return_deuce':ret_deuce, 'receiver_return_close':ret_close,
            'pair_deuce':pair_deuce, 'pair_close':pair_close,
            'legacy_server_deuce':prior_deuce, 'serve_point_balance':point_balance,
            'set_progress':min(1.0,max(0,parsed.current_set-1)/2.0),
            'game_progress':min(1.0,max(0,parsed.current_game-1)/11.0),
        }

        # Человекочитаемый ranker. Он намеренно НЕ начинается от 47% и не является вероятностью.
        score = 42.0
        score += 22.0*(same_pressure-0.45)
        score += 18.0*(same_deuce-prior_deuce)
        score += 15.0*(same_recent-0.45)
        score += 10.0*(same_balance-0.55)
        score += 8.0*(srv_deuce-prior_deuce)
        score += 8.0*(ret_deuce-prior_deuce)
        score += 5.0*(pair_deuce-prior_deuce)
        score += 8.0*(match_recent-0.45)
        score += 6.0*same_last_deuce
        score += 4.0*same_deuce_recency
        score += 8.0*max(0.0,same_trend) - 4.0*max(0.0,-same_trend)
        score += min(6.0,same_n*2.0)
        score -= 10.0*extreme_rate
        score -= 5.0*min(1.0,loss_streak/2.0)
        if same_n == 0:
            score -= 7.0
        history_strength=clamp(score,0.0,100.0)

        if server_num == 1:
            matchup_state = "P1 подаёт / P2 принимает"
        else:
            matchup_state = "P1 принимает / P2 подаёт"
        last_scores = [str(r['pari_terminal']) for r in same[-4:]]
        history_summary = (
            f"{matchup_state}: {same_n} прошлых геймов этой подачи; "
            f"deuce {sum(int(r['deuce']) for r in same)}/{same_n or 0}; "
            f"pressure {same_pressure*100:.0f}/100; последние: {', '.join(last_scores) if last_scores else 'нет'}"
        )
        return features, history_strength, matchup_state, history_summary

    def _ensure_snapshot(self, parsed: ParsedMatch, server_num: int, features: dict[str,float], history_strength: float,
                         online_p: float, strength: float, threshold: float, server: str, receiver: str) -> None:
        key=self._key(str(parsed.event_id),parsed.current_set,parsed.current_game)
        self.conn.execute("""INSERT OR IGNORE INTO v7_game_snapshots(
            key,event_id,set_num,game_num,player1,player2,server_num,server,receiver,feature_json,
            history_strength,online_probability,signal_strength,threshold,entry_score,created_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (key,str(parsed.event_id),parsed.current_set,parsed.current_game,parsed.player1,parsed.player2,server_num,server,receiver,
             json.dumps(features,ensure_ascii=False,sort_keys=True),history_strength,online_p*100.0,strength,threshold,
             score_label(parsed.raw_score_a,parsed.raw_score_b),time.time()))
        self.conn.commit()

    def _resolve_v7_snapshot(self, event_id: str, set_num: int, game_num: int, label: int, terminal_score: str = '') -> None:
        """Resolve one completed game and update both *fear* and *search*.

        V7.4 deliberately separates two failure modes:
        - false confident signal -> fear rises and the gate tightens a little;
        - missed real deuce -> search drive rises and that positive example trains hard.

        This prevents the old failure mode where a large historical debt pinned pain at
        100 and the only available response was to keep raising the threshold.
        """
        key = self._key(event_id, set_num, game_num)
        row = self.conn.execute("SELECT * FROM v7_game_snapshots WHERE key=? AND resolved=0", (key,)).fetchone()
        if row is None:
            return

        issued = bool(int(row['signal_issued']))
        label = int(bool(label))
        balance_before = self._meta_get('score_balance', 0.0)
        pain_before = self._meta_get('pain', 0.0)
        search_before = self._meta_get('search_drive', 0.0)
        hit_streak = int(self._meta_get('hit_streak', 0.0))
        miss_streak = int(self._meta_get('miss_streak', 0.0))
        missed_streak = int(self._meta_get('missed_deuce_streak', 0.0))
        threshold = self._meta_get('threshold', MIN_SIGNAL_STRENGTH)

        if issued and label:
            reward = V7_HIT_REWARD
            hit_streak += 1; miss_streak = 0; missed_streak = 0
            # A caught deuce confirms that opening the gate was useful.
            sw = 2.10 + 0.35 * clamp(search_before / 100.0, 0.0, 1.0)
            pain_after = clamp(pain_before * 0.84 - (3.0 + min(5.0, hit_streak * 0.8)), 0.0, V7_PAIN_MAX)
            search_after = clamp(search_before - (8.0 + min(8.0, hit_streak * 1.0)), 0.0, V7_SEARCH_MAX)
            threshold -= 0.30 + min(0.45, hit_streak * 0.06)

        elif issued and not label:
            reward = V7_MISS_PENALTY
            miss_streak += 1; hit_streak = 0; missed_streak = 0
            # Still painful. But unlike the old policy, a false signal is not allowed
            # to dominate positive learning by 5-6x at permanent pain=100.
            fear_boost = 1.0 + 0.65 * clamp(pain_before / 100.0, 0.0, 1.0)
            sw = 1.65 * fear_boost
            pain_after = clamp(pain_before * 0.92 + 11.0 + min(12.0, miss_streak * 1.7), 0.0, V7_PAIN_MAX)
            search_after = clamp(search_before - 2.0, 0.0, V7_SEARCH_MAX)
            threshold += 0.25 + min(0.65, pain_after / 160.0)

        elif (not issued) and label:
            reward = V7_MISSED_DEUCE_PENALTY
            missed_streak += 1; hit_streak = 0; miss_streak = 0
            # This is the key SEEK behaviour: a missed real deuce is a hard positive.
            # The model must learn its feature pattern, not merely be told "be braver".
            search_after = clamp(search_before + 10.0 + min(16.0, missed_streak * 1.6), 0.0, V7_SEARCH_MAX)
            search_boost = 1.0 + 1.25 * clamp(search_after / 100.0, 0.0, 1.0)
            sw = 2.70 * search_boost
            pain_after = clamp(pain_before * 0.96 + 4.0 + min(8.0, missed_streak * 0.8), 0.0, V7_PAIN_MAX)
            threshold -= 0.55 + min(1.45, search_after / 65.0)

        else:
            reward = V7_CORRECT_SILENCE_REWARD
            hit_streak = max(0, hit_streak - 1); miss_streak = 0; missed_streak = 0
            # Easy negatives are abundant, so do not let them drown rare deuce examples.
            sw = 0.30
            pain_after = clamp(pain_before * 0.985 - 0.20, 0.0, V7_PAIN_MAX)
            search_after = clamp(search_before - 0.25, 0.0, V7_SEARCH_MAX)

        try:
            features = {k: float(v) for k, v in json.loads(str(row['feature_json'])).items()}
            self._online_update(features, label, sw)
        except Exception as e:
            print(f"[V7.4 SEEK learn] {e}", file=sys.stderr)

        balance_after = balance_before + float(reward)
        self._meta_set('score_balance', balance_after)
        if reward > 0:
            self._meta_set('positive_points', self._meta_get('positive_points', 0.0) + reward)
        elif reward < 0:
            self._meta_set('negative_points', self._meta_get('negative_points', 0.0) + abs(reward))
        self._meta_set('pain', pain_after)
        self._meta_set('search_drive', search_after)
        self._meta_set('last_reward', reward)
        self._meta_set('hit_streak', float(hit_streak))
        self._meta_set('miss_streak', float(miss_streak))
        self._meta_set('missed_deuce_streak', float(missed_streak))
        self._meta_set('threshold', clamp(threshold, V7_THRESHOLD_FLOOR, V7_THRESHOLD_CEIL))

        self.conn.execute(
            "UPDATE v7_game_snapshots SET resolved=1,label=?,reward=?,terminal_score=?,resolved_at=? WHERE key=?",
            (label, float(reward), str(terminal_score or ''), time.time(), key),
        )
        self._adapt_policy_if_needed()
        self.conn.commit()

    def _adapt_policy_if_needed(self) -> None:
        n = int(self._meta_get('training_count', 0))
        last = int(self._meta_get('last_policy_count', 0))
        if n - last < 10:
            return
        rows = self.conn.execute("""SELECT signal_issued,label,reward,signal_strength FROM v7_game_snapshots
            WHERE resolved=1 AND label IS NOT NULL ORDER BY resolved_at DESC LIMIT ?""",
            (V7_POLICY_WINDOW,)).fetchall()
        if len(rows) < 30:
            self._meta_set('last_policy_count', n)
            return

        signals = [r for r in rows if int(r['signal_issued'] or 0)]
        total_deuces = sum(int(r['label'] or 0) for r in rows)
        hits = sum(1 for r in signals if int(r['label'] or 0))
        missed = sum(1 for r in rows if (not int(r['signal_issued'] or 0)) and int(r['label'] or 0))
        precision = hits / len(signals) if signals else 0.0
        recall = hits / total_deuces if total_deuces else 0.0
        self._meta_set('recent_precision', precision * 100.0)
        self._meta_set('recent_recall', recall * 100.0)

        t = self._meta_get('threshold', MIN_SIGNAL_STRENGTH)
        search = self._meta_get('search_drive', 0.0)
        pain = self._meta_get('pain', 0.0)

        # Economics-aware guardrail: precision near 20% must NOT be treated as
        # catastrophic for an ~8.0 market. The old 50% target was what strangled A.
        if len(signals) >= 8 and precision < V7_MIN_ACCEPTABLE_PRECISION:
            t += 0.45 + 0.45 * clamp(pain / 100.0, 0.0, 1.0)
        elif len(signals) >= 8 and precision >= V7_TARGET_PRECISION and recall < V7_TARGET_RECALL:
            # Accuracy is good enough but coverage is poor -> search wider.
            t -= 0.80 + 1.20 * clamp(search / 100.0, 0.0, 1.0)
        elif precision >= V7_MIN_ACCEPTABLE_PRECISION and recall < 0.18:
            t -= 0.55 + 0.85 * clamp(search / 100.0, 0.0, 1.0)

        # Independent anti-cowardice control. Even with few/no signals, repeatedly
        # missing real deuces must open the gate enough to gather informative trials.
        if total_deuces >= 8 and recall < 0.10:
            t -= 0.65 + 0.85 * clamp(search / 100.0, 0.0, 1.0)
        if len(signals) < 4 and missed >= 6:
            t -= 0.75

        self._meta_set('threshold', clamp(t, V7_THRESHOLD_FLOOR, V7_THRESHOLD_CEIL))
        self._meta_set('last_policy_count', n)

    def build_prediction(self, parsed: ParsedMatch, match_link: str) -> Prediction:
        server_num=get_server_for_game(parsed.current_set,parsed.current_game,parsed.first_server_match,parsed.max_game_by_set)
        server=parsed.player1 if server_num==1 else parsed.player2
        receiver=parsed.player2 if server_num==1 else parsed.player1
        features,hscore,state,hsummary=self._history_features(parsed,server_num)
        online_p=self._online_probability(features)
        prior,_=self._legacy_server_prior(server)
        model_rank=clamp(50.0+(online_p-prior)*140.0,0.0,100.0)
        trained=self._meta_get('training_count',0.0)
        # Пока learner не накопил факты, он вообще не имеет права тянуть хороший history-score к базе.
        learner_w=min(0.45,0.45*trained/500.0)
        strength=clamp((1.0-learner_w)*hscore+learner_w*model_rank,0.0,100.0)
        threshold=self._meta_get('threshold',MIN_SIGNAL_STRENGTH)
        pain_state=self._meta_get('pain',0.0)
        search_state=self._meta_get('search_drive',0.0)

        raw_a,raw_b=parsed.raw_score_a,parsed.raw_score_b
        a,b=_int(raw_a),_int(raw_b)
        already=(parsed.current_set,parsed.current_game) in parsed.deuce_games
        market=is_market_available(raw_a,raw_b)
        facts_n=int(self.conn.execute("SELECT COUNT(*) FROM v7_game_facts WHERE event_id=?",(str(parsed.event_id),)).fetchone()[0])
        same_n=int(self.conn.execute("SELECT COUNT(*) FROM v7_game_facts WHERE event_id=? AND server_num=?",(str(parsed.event_id),server_num)).fetchone()[0])
        early=(a,b) in V7_EARLY_SCORES
        eligible=(early and not already and market and facts_n>=V7_MIN_MATCH_GAMES and
                  same_n>=V7_MIN_SAME_ORIENTATION_GAMES and strength>=threshold)

        self._ensure_snapshot(parsed,server_num,features,hscore,online_p,strength,threshold,server,receiver)
        prof=self.player_profile(server); glob=self.global_profile()
        quality='много данных' if facts_n>=8 else 'достаточно данных' if facts_n>=4 else 'мало данных'
        learner_text = (f"online=разогрев {int(trained)}/50" if trained < 50 else f"online={online_p*100:.1f}%")
        reason=(
            f"V7.4 SEEK SIGNAL: {state}; история={hscore:.1f}/100; {learner_text}; "
            f"порог={threshold:.1f}; страх={pain_state:.0f}; поиск={search_state:.0f}; обучено={int(trained)} геймов"
            if eligible else
            f"V7.4 SEEK WATCH: {state}; сила={strength:.1f}/{threshold:.1f}; история={hscore:.1f}; {learner_text}; "
            f"страх={pain_state:.0f}; поиск={search_state:.0f}"
        )
        return Prediction(
            event_id=parsed.event_id,player1=parsed.player1,player2=parsed.player2,
            set_num=parsed.current_set,game_num=parsed.current_game,server=server,
            current_score=score_label(raw_a,raw_b),game_band=game_band(parsed.current_game),
            server_service_games=prof.service_games,server_deuce_games=prof.deuce_games,
            server_service_points=prof.service_points,server_points_won=prof.points_won,
            global_service_games=glob.service_games,posterior_deuce_rate=online_p,
            estimated_server_point_p=0.5,raw_probability_deuce=round(online_p*100.0,1),
            probability_deuce=round(strength,1),calibration_adjustment=round(strength-hscore,1),
            data_quality=quality,confidence=quality,server_score_state=server_score_label(raw_a,raw_b,server_num),
            already_deuce=already,market_available=market,match_link=match_link,
            live_probability_deuce=round(online_p*100.0,1),strict_eligible=eligible,
            transition_probability=round(model_rank,1),context_probability=round(hscore,1),signal_reason=reason,
            signal_strength=round(strength,1),online_probability=round(online_p*100.0,1),
            policy_threshold=round(threshold,1),matchup_state=state,history_summary=hsummary,training_games=int(trained),
        )

    def observe(self, observation: MatchObservation, min_probability: float) -> None:
        event_id=observation.event_id; pred=observation.prediction
        current=(pred.set_num,pred.game_num); self.missing_cycles[event_id]=0
        current_key=self._key(event_id,pred.set_num,pred.game_num)
        if current in observation.deuce_games:
            self._resolve_key(current_key,'hit')
        rows=self.conn.execute("SELECT key,set_num,game_num FROM pending_predictions WHERE event_id=?",(event_id,)).fetchall()
        for row in rows:
            sg=(int(row['set_num']),int(row['game_num']))
            if sg<current:
                self._resolve_key(row['key'],'hit' if sg in observation.deuce_games else 'miss')
        if not pred.strict_eligible or pred.already_deuce or current in observation.deuce_games or not pred.market_available:
            return
        if self.conn.execute("SELECT 1 FROM signal_history WHERE key=?",(current_key,)).fetchone() is not None:
            return
        if self.conn.execute("SELECT 1 FROM pending_predictions WHERE key=?",(current_key,)).fetchone() is not None:
            return
        created=time.time()
        values=(current_key,event_id,pred.player1,pred.player2,pred.set_num,pred.game_num,pred.game_band,pred.server,
                pred.current_score,pred.server_score_state,pred.signal_strength,pred.online_probability,pred.data_quality,
                pred.server_service_games,pred.server_deuce_games,pred.server_service_points,pred.server_points_won,
                pred.global_service_games,created)
        self.conn.execute("""INSERT INTO pending_predictions(
            key,event_id,player1,player2,set_num,game_num,game_band,server,score_state,server_score_state,
            probability,raw_probability,data_quality,server_service_games,server_deuce_games,server_service_points,
            server_points_won,global_service_games,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",values)
        self.conn.execute("""INSERT INTO signal_history(
            key,event_id,player1,player2,set_num,game_num,game_band,server,score_state,server_score_state,
            probability,raw_probability,data_quality,server_service_games,server_deuce_games,server_service_points,
            server_points_won,global_service_games,status,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'pending',?)""",values)
        self.conn.execute("UPDATE v7_game_snapshots SET signal_issued=1 WHERE key=?",(current_key,))
        self.conn.commit()

    def learning_summary(self) -> dict[str, Any]:
        r = self.conn.execute("""SELECT COUNT(*) n,
            COALESCE(SUM(CASE WHEN signal_issued=1 THEN 1 ELSE 0 END),0) sig,
            COALESCE(SUM(CASE WHEN signal_issued=0 AND label=1 THEN 1 ELSE 0 END),0) missed,
            COALESCE(AVG(reward),0) reward, COALESCE(SUM(reward),0) score
            FROM v7_game_snapshots WHERE resolved=1""").fetchone()
        return {
            'trained': int(r['n']),
            'signal_games': int(r['sig']),
            'missed_deuces': int(r['missed']),
            'avg_reward': float(r['reward']),
            'score_balance': float(r['score']),
            'positive_points': self._meta_get('positive_points', 0.0),
            'negative_points': self._meta_get('negative_points', 0.0),
            'pain': self._meta_get('pain', 0.0),
            'last_reward': self._meta_get('last_reward', 0.0),
            'hit_streak': int(self._meta_get('hit_streak', 0.0)),
            'miss_streak': int(self._meta_get('miss_streak', 0.0)),
            'missed_deuce_streak': int(self._meta_get('missed_deuce_streak', 0.0)),
            'threshold': self._meta_get('threshold', MIN_SIGNAL_STRENGTH),
        }


class V73DualTracker(V7Tracker):
    """Run the existing V7.3 learner and an XGBoost model side-by-side.

    Model A (production): existing online logistic + history ranker + adaptive threshold.
    Model B (shadow): XGBoost trained only on already-resolved snapshots. It sees the
    same V7_FEATURES and the same decision gate/threshold, but NEVER changes the live
    signal or Model A's learning/pain policy. This makes forward comparison causal.
    """

    def __init__(self, *args, **kwargs):
        self._xgb_model = None
        self._xgb_train_count = 0
        self._xgb_last_error = ""
        self._xgb_backend_available = False
        super().__init__(*args, **kwargs)
        self._init_v73_dual_schema()
        self._refresh_xgb_model(force=True)

    def _init_v73_dual_schema(self) -> None:
        cols = {str(r[1]) for r in self.conn.execute("PRAGMA table_info(v7_game_snapshots)").fetchall()}
        migrations = {
            "decision_eligible": "INTEGER NOT NULL DEFAULT 0",
            "xgb_ready": "INTEGER NOT NULL DEFAULT 0",
            "xgb_probability": "REAL",
            "xgb_strength": "REAL",
            "xgb_signal_issued": "INTEGER NOT NULL DEFAULT 0",
            # R4: решение B фиксируется один раз в ранней точке гейма и больше не
            # перезаписывается последующими счетами 30:15/40:15 и т.п.
            "xgb_decision_frozen": "INTEGER NOT NULL DEFAULT 0",
            "xgb_reward": "REAL",
        }
        for name, ddl in migrations.items():
            if name not in cols:
                self.conn.execute(f"ALTER TABLE v7_game_snapshots ADD COLUMN {name} {ddl}")
        self.conn.commit()

    @staticmethod
    def _xgb_vector(features: dict[str, float]) -> list[float]:
        return [float(features.get(f, 0.0)) for f in V7_FEATURES]

    def _refresh_xgb_model(self, force: bool = False) -> None:
        try:
            from xgboost import XGBClassifier  # type: ignore
            self._xgb_backend_available = True
        except Exception as e:
            self._xgb_backend_available = False
            self._xgb_last_error = f"xgboost unavailable: {e}"
            return

        total = int(self.conn.execute(
            "SELECT COUNT(*) FROM v7_game_snapshots WHERE resolved=1 AND label IS NOT NULL"
        ).fetchone()[0])
        if total < V73_XGB_MIN_GAMES:
            self._xgb_last_error = f"warmup {total}/{V73_XGB_MIN_GAMES}"
            return
        if (not force) and self._xgb_model is not None and total - self._xgb_train_count < V73_XGB_RETRAIN_EVERY:
            return

        rows = self.conn.execute(
            """SELECT feature_json,label FROM v7_game_snapshots
               WHERE resolved=1 AND label IS NOT NULL
               ORDER BY resolved_at DESC LIMIT ?""",
            (V73_XGB_MAX_TRAIN_GAMES,),
        ).fetchall()
        rows = list(reversed(rows))
        X: list[list[float]] = []
        y: list[int] = []
        for r in rows:
            try:
                f = {k: float(v) for k, v in json.loads(str(r['feature_json'])).items()}
                X.append(self._xgb_vector(f))
                y.append(int(bool(r['label'])))
            except Exception:
                continue
        pos = sum(y)
        neg = len(y) - pos
        if len(y) < V73_XGB_MIN_GAMES or pos < V73_XGB_MIN_POSITIVES or neg < V73_XGB_MIN_POSITIVES:
            self._xgb_last_error = f"classes pos={pos}, neg={neg}"
            return

        # Slight class balancing, capped so rare deuces do not dominate the model.
        scale_pos = clamp(neg / max(1.0, float(pos)), 1.0, 3.0)
        model = XGBClassifier(
            n_estimators=160,
            max_depth=3,
            learning_rate=0.035,
            min_child_weight=5.0,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=3.0,
            reg_alpha=0.10,
            gamma=0.05,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos,
            random_state=42,
            n_jobs=1,
            tree_method="hist",
        )
        try:
            model.fit(X, y, verbose=False)
            self._xgb_model = model
            self._xgb_train_count = total
            self._xgb_last_error = ""
            self._meta_set('xgb_training_count', float(total))
            self.conn.commit()
        except Exception as e:
            self._xgb_last_error = f"fit failed: {e}"

    def _xgb_probability_for(self, features: dict[str, float]) -> float | None:
        if self._xgb_model is None:
            return None
        try:
            p = float(self._xgb_model.predict_proba([self._xgb_vector(features)])[0][1])
            return clamp(p, 0.001, 0.999)
        except Exception as e:
            self._xgb_last_error = f"predict failed: {e}"
            return None

    def build_prediction(self, parsed: ParsedMatch, match_link: str) -> Prediction:
        # Model A performs the production calculation and opens the snapshot.
        pred = super().build_prediction(parsed, match_link)
        key = self._key(str(parsed.event_id), parsed.current_set, parsed.current_game)
        row = self.conn.execute(
            "SELECT feature_json,xgb_decision_frozen,xgb_ready,xgb_probability,xgb_strength,xgb_signal_issued "
            "FROM v7_game_snapshots WHERE key=?", (key,)
        ).fetchone()
        if row is None:
            return pred
        try:
            features = {k: float(v) for k, v in json.loads(str(row['feature_json'])).items()}
        except Exception:
            features = {}

        a = _int(parsed.raw_score_a)
        b = _int(parsed.raw_score_b)
        server_num = get_server_for_game(parsed.current_set, parsed.current_game, parsed.first_server_match, parsed.max_game_by_set)
        facts_n = int(self.conn.execute(
            "SELECT COUNT(*) FROM v7_game_facts WHERE event_id=?", (str(parsed.event_id),)
        ).fetchone()[0])
        same_n = int(self.conn.execute(
            "SELECT COUNT(*) FROM v7_game_facts WHERE event_id=? AND server_num=?", (str(parsed.event_id), server_num)
        ).fetchone()[0])
        base_eligible = bool(
            (a, b) in V7_EARLY_SCORES
            and not pred.already_deuce
            and pred.market_available
            and facts_n >= V7_MIN_MATCH_GAMES
            and same_n >= V7_MIN_SAME_ORIENTATION_GAMES
        )

        xp = self._xgb_probability_for(features)
        xready = xp is not None
        xstrength: float | None = None
        xsignal = False
        if xready and xp is not None:
            prior, _ = self._legacy_server_prior(pred.server)
            xrank = clamp(50.0 + (xp - prior) * 140.0, 0.0, 100.0)
            # Same blend schedule as Model A so the comparison isolates the learner.
            trained = self._meta_get('training_count', 0.0)
            learner_w = min(0.45, 0.45 * trained / 500.0)
            xstrength = clamp((1.0 - learner_w) * pred.context_probability + learner_w * xrank, 0.0, 100.0)
            xsignal = bool(base_eligible and xstrength >= pred.policy_threshold)

        # R4 IMPORTANT: freeze B's decision exactly once while the market is in the
        # early decision window. Previously this UPDATE ran every live cycle and a
        # SIGNAL seen at 0:15 could be overwritten by SILENCE at 30:15. That made
        # real forecasts disappear from the A/B statistics after the game finished.
        frozen = bool(int(row['xgb_decision_frozen'] or 0))
        if (not frozen) and base_eligible and xready and xp is not None and xstrength is not None:
            self.conn.execute(
                """UPDATE v7_game_snapshots
                   SET decision_eligible=1,xgb_ready=1,xgb_probability=?,xgb_strength=?,
                       xgb_signal_issued=?,xgb_decision_frozen=1
                   WHERE key=? AND xgb_decision_frozen=0""",
                (xp * 100.0, xstrength, int(xsignal), key),
            )
            self.conn.commit()
            frozen = True
        elif (not frozen) and base_eligible:
            # Remember that A had a valid early decision opportunity, but do not
            # fabricate a B decision if XGBoost was not ready at that moment.
            self.conn.execute(
                "UPDATE v7_game_snapshots SET decision_eligible=1 WHERE key=?", (key,)
            )
            self.conn.commit()

        # Report the frozen B decision, not a later recalculation.
        frozen_row = self.conn.execute(
            "SELECT xgb_decision_frozen,xgb_ready,xgb_probability,xgb_strength,xgb_signal_issued "
            "FROM v7_game_snapshots WHERE key=?", (key,)
        ).fetchone()
        if frozen_row is not None and int(frozen_row['xgb_decision_frozen'] or 0):
            fp = float(frozen_row['xgb_probability'] or 0.0)
            fs = float(frozen_row['xgb_strength'] or 0.0)
            fsignal = bool(int(frozen_row['xgb_signal_issued'] or 0))
            pred.signal_reason += (
                f"; XGB-shadow={fp:.1f}% / сила {fs:.1f}"
                f" / {'SIGNAL' if fsignal else 'SILENCE'} [FROZEN]"
            )
        else:
            warm = self._xgb_train_count or int(self.conn.execute(
                "SELECT COUNT(*) FROM v7_game_snapshots WHERE resolved=1 AND label IS NOT NULL"
            ).fetchone()[0])
            pred.signal_reason += f"; XGB-shadow=разогрев {warm}/{V73_XGB_MIN_GAMES}"
        return pred

    def _resolve_v7_snapshot(self, event_id: str, set_num: int, game_num: int, label: int, terminal_score: str = '') -> None:
        key = self._key(event_id, set_num, game_num)
        before = self.conn.execute(
            "SELECT resolved,xgb_decision_frozen,xgb_signal_issued FROM v7_game_snapshots WHERE key=?", (key,)
        ).fetchone()
        was_open = bool(before is not None and int(before['resolved']) == 0)
        super()._resolve_v7_snapshot(event_id, set_num, game_num, label, terminal_score)
        if was_open and before is not None and int(before['xgb_decision_frozen'] or 0):
            xr = self._shadow_reward(int(before['xgb_signal_issued'] or 0), int(bool(label)))
            self.conn.execute(
                "UPDATE v7_game_snapshots SET xgb_reward=? WHERE key=?", (float(xr), key)
            )
            self.conn.commit()
        if was_open:
            self._refresh_xgb_model(force=False)

    @staticmethod
    def _shadow_reward(signal: int, label: int) -> float:
        if signal and label:
            return V7_HIT_REWARD
        if signal and not label:
            return V7_MISS_PENALTY
        if (not signal) and label:
            return V7_MISSED_DEUCE_PENALTY
        return V7_CORRECT_SILENCE_REWARD

    def dual_summary(self) -> dict[str, Any]:
        rows = self.conn.execute(
            """SELECT signal_issued,xgb_signal_issued,label,reward,xgb_reward FROM v7_game_snapshots
               WHERE resolved=1 AND xgb_decision_frozen=1 AND xgb_ready=1 AND label IS NOT NULL"""
        ).fetchall()
        live_rows = self.conn.execute(
            """SELECT signal_issued,xgb_signal_issued,resolved FROM v7_game_snapshots
               WHERE xgb_decision_frozen=1 AND xgb_ready=1"""
        ).fetchall()
        def stats(which: str) -> dict[str, Any]:
            vals = [(int(r[which]), int(r['label'])) for r in rows]
            signals = sum(s for s, _ in vals)
            hits = sum(1 for s, y in vals if s and y)
            false = sum(1 for s, y in vals if s and not y)
            missed = sum(1 for s, y in vals if (not s) and y)
            score = sum(self._shadow_reward(s, y) for s, y in vals)
            return {
                'signals': signals,
                'hits': hits,
                'false_signals': false,
                'missed_deuces': missed,
                'precision': (hits / signals * 100.0) if signals else 0.0,
                'score': score,
            }
        return {
            'available': bool(self._xgb_backend_available),
            'ready': self._xgb_model is not None,
            'trained_on': int(self._xgb_train_count),
            'status': self._xgb_last_error or 'ready',
            'comparable_games': len(rows),
            'deuces': sum(int(r['label']) for r in rows),
            'disagreements': sum(int(r['signal_issued']) != int(r['xgb_signal_issued']) for r in rows),
            'a_issued_live': sum(int(r['signal_issued']) for r in live_rows),
            'b_issued_live': sum(int(r['xgb_signal_issued']) for r in live_rows),
            'pending_comparable': sum(1 for r in live_rows if int(r['resolved']) == 0),
            'a': stats('signal_issued'),
            'b': stats('xgb_signal_issued'),
        }

    def learning_summary(self) -> dict[str, Any]:
        out = super().learning_summary()
        out['dual'] = self.dual_summary()
        return out

# ===========================================================================

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
        key=lambda row: float((re.search(r"Сила сигнала V7:\s*([0-9.]+)", row[-1]) or re.search(r"Вероятность 40:40:\s*([0-9.]+)%", row[-1])).group(1))
        if row and isinstance(row[-1], str) and (re.search(r"Сила сигнала V7:\s*([0-9.]+)", row[-1]) or re.search(r"Вероятность 40:40:\s*([0-9.]+)%", row[-1])) else 0.0,
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
    tracker = V73DualTracker(args.db, missing_cycles_to_unknown=args.missing_cycles_to_unknown, training_db=TRAINING_DB)
    stale = tracker.cleanup_stale_pending(args.stale_hours)

    print(f"HTTP backend: {HTTP_BACKEND}")
    print("Источник: PARI live/sportscast")
    print(f"Статистика: {args.db} | старых pending → UNKNOWN: {stale}")
    print(f"V7.4 SEEK DUAL BRAIN [{V73_DUAL_BUILD}]: A=online logistic fear+search (боевой), B=XGBoost (shadow).")
    print("R4: решения B замораживаются при первом раннем прогнозе и больше не стираются следующим счётом.")
    print("P1/P2 всегда остаются в координатах PARI; меняется только признак, кто подаёт.")
    print("Модель учится на КАЖДОМ завершённом гейме: caught deuce +4.0, miss -4.0, missed deuce -0.50, correct silence +0.01.")
    print("Сигнал выдаётся только в начале гейма (0:0/после первого очка/15:15), а не на 30:30.")
    print(f"Стартовый порог силы: {MIN_SIGNAL_STRENGTH:.1f}/100; дальше порог адаптируется сам в диапазоне {V7_THRESHOLD_FLOOR:.0f}-{V7_THRESHOLD_CEIL:.0f}.")
    print("V5/V6 используются только как priors; forward hit/miss V7 считаются с нуля в pari_deuce_v7.sqlite3.")
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
                if isinstance(tracker, V7Tracker):
                    ls = tracker.learning_summary()
                    print(f"V7 обучение: {ls['trained']} геймов | порог {ls['threshold']:.1f}/100 | "
                          f"баланс {ls['score_balance']:+.2f} | боль {ls['pain']:.0f}/100 | "
                          f"пропущено deuce без сигнала: {ls['missed_deuces']} | reward {ls['avg_reward']:+.3f}")
                    dual = ls.get('dual', {})
                    if dual:
                        a = dual.get('a', {}); b = dual.get('b', {})
                        print(f"DUAL: сравнимых {dual.get('comparable_games',0)} | XGB train {dual.get('trained_on',0)} | "
                              f"A {a.get('hits',0)}/{a.get('signals',0)}={a.get('precision',0):.1f}% score {a.get('score',0):+.2f} | "
                              f"B {b.get('hits',0)}/{b.get('signals',0)}={b.get('precision',0):.1f}% score {b.get('score',0):+.2f} | "
                              f"разошлись {dual.get('disagreements',0)}")

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



def v7_self_test() -> None:
    tr=V7Tracker(":memory:",missing_cycles_to_unknown=1,training_db="/__missing_v5__.sqlite3",previous_v6_db=None)
    # synthetic fixed-coordinate match: P1 serves odd games, P2 serves even.
    events=[{"type":1123,"i3":1}]
    # g1 P1 serve -> 40:30 close
    for a,b in [(0,0),(15,0),(30,0),(30,15),(30,30),(40,30)]: events.append({"type":999,"i1":1,"i2":1,"i8":a,"i9":b})
    # g2 P2 serve, PARI coords P1:P2 -> 15:40 easy-ish for server
    events.append({"type":1125,"i1":1,"i2":2,"i8":0,"i9":0})
    for a,b in [(0,0),(0,15),(15,15),(15,30),(15,40)]: events.append({"type":999,"i1":1,"i2":2,"i8":a,"i9":b})
    # g3 P1 serve -> deuce
    events.append({"type":1125,"i1":1,"i2":3,"i8":0,"i9":0})
    for a,b in [(0,0),(15,0),(15,15),(30,15),(30,30),(40,30),(40,40)]: events.append({"type":999,"i1":1,"i2":3,"i8":a,"i9":b})
    # g4 is current, P2 serves; fixed coordinates must remain P1:P2
    events.append({"type":1125,"i1":1,"i2":4,"i8":0,"i9":0})
    parsed=parse_match("v7test","A","B",events); assert parsed is not None
    tr.ingest_completed_games(parsed)
    facts=tr.conn.execute("SELECT server_num,pari_terminal FROM v7_game_facts ORDER BY game_num").fetchall()
    assert [int(x['server_num']) for x in facts]==[1,2,1]
    assert str(facts[1]['pari_terminal'])=='15:40'  # не переворачиваем координаты БК
    pred=tr.build_prediction(parsed,"https://pari.ru/")
    assert pred.matchup_state=='P1 принимает / P2 подаёт'
    assert pred.current_score=='0:0'
    assert 0.0 <= pred.signal_strength <= 100.0
    # snapshot exists and a deuce without signal gives the requested small negative reward.
    key=tr._key('v7test',1,4)
    before_balance = tr._meta_get('score_balance', 0.0)
    tr._resolve_v7_snapshot('v7test',1,4,1,'40:40')
    rr=tr.conn.execute("SELECT reward,resolved,terminal_score FROM v7_game_snapshots WHERE key=?",(key,)).fetchone()
    assert int(rr['resolved'])==1 and float(rr['reward'])==V7_MISSED_DEUCE_PENALTY
    assert str(rr['terminal_score'])=='40:40'
    once_balance = tr._meta_get('score_balance', 0.0)
    assert abs(once_balance - (before_balance + V7_MISSED_DEUCE_PENALTY)) < 1e-12
    tr._resolve_v7_snapshot('v7test',1,4,1,'40:40')
    assert abs(tr._meta_get('score_balance', 0.0) - once_balance) < 1e-12
    tr.close()

def main() -> int:
    args = argparse.Namespace(
        interval=INTERVAL,
        timeout=TIMEOUT,
        min_probability=0.0,
        top=TOP,
        output=OUTPUT,
        db=DB,
        missing_cycles_to_unknown=MISSING_CYCLES_TO_UNKNOWN,
        stale_hours=STALE_HOURS,
    )

    mode = str(RUN_MODE).strip().lower()
    if mode in {"self-test", "selftest", "test"}:
        self_test()
        v7_self_test()
        print("SELF-TEST V7: OK")
        return 0
    if mode == "offline":
        return run_offline(Path(OFFLINE_FILE))
    if mode != "live":
        raise ValueError(f"Неизвестный RUN_MODE={RUN_MODE!r}. Используй: 'live', 'offline' или 'self-test'.")
    return run_live(args)


if __name__ == "__main__":
    raise SystemExit(main())
