from flask import Flask, render_template, jsonify
import json
import os
import sqlite3
import math
import time

app = Flask(__name__)
BASE_DIR = os.path.dirname(__file__)
JSON_PATH = os.path.join(BASE_DIR, 'tennis.json')
DB_PATH = os.path.join(BASE_DIR, 'pari_deuce_v7.sqlite3')
MIN_BEST_SIGNALS = 50

APP_UI_VERSION = 'V7.4-SEEK-DUAL-STATS-R5-2026-09-23'
APP_FILE_NAME = os.path.basename(__file__)
APP_FILE_PATH = os.path.abspath(__file__)
print(f'[ADMIN UI] {APP_UI_VERSION} | {APP_FILE_PATH}')

HIT_REWARD = 4.0
MISS_PENALTY = -4.0
MISSED_DEUCE_PENALTY = -0.50
CORRECT_SILENCE_REWARD = 0.01


def _pct(hits, total):
    return round(hits / total * 100.0, 1) if total else 0.0


def _wilson_lower(hits, total, z=1.96):
    if total <= 0:
        return 0.0
    p = hits / total
    den = 1.0 + z*z/total
    centre = p + z*z/(2*total)
    adj = z * math.sqrt((p*(1-p) + z*z/(4*total))/total)
    return (centre - adj) / den * 100.0


def _table_exists(conn, name):
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def _columns(conn, table):
    if not _table_exists(conn, table):
        return set()
    return {str(r[1]) for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _read_dimension(conn, dimension):
    if not _table_exists(conn, 'segment_stats'):
        return []
    rows = conn.execute(
        '''SELECT label,total,hits,misses,unknown,probability_sum
           FROM segment_stats WHERE dimension=? ORDER BY total DESC''',
        (dimension,),
    ).fetchall()
    result = []
    for r in rows:
        total = int(r['total'])
        hits = int(r['hits'])
        result.append({
            'label': str(r['label']), 'checked': total, 'hits': hits,
            'misses': int(r['misses']), 'unknown': int(r['unknown']),
            'rate': _pct(hits, total),
            'model_avg': round(float(r['probability_sum']) / total, 1) if total else 0.0,
            'wilson_lower': round(_wilson_lower(hits, total), 1),
        })
    return result


def _reward_for(signal, label):
    if signal and label:
        return HIT_REWARD
    if signal and not label:
        return MISS_PENALTY
    if (not signal) and label:
        return MISSED_DEUCE_PENALTY
    return CORRECT_SILENCE_REWARD


def _model_stats(rows, signal_key, reward_key=None):
    issued = sum(int(r[signal_key] or 0) for r in rows)
    pending_signals = sum(1 for r in rows if int(r[signal_key] or 0) and not int(r['resolved'] or 0))
    settled = [r for r in rows if int(r['resolved'] or 0) and r['label'] is not None]
    settled_signals = [r for r in settled if int(r[signal_key] or 0)]
    hits = sum(1 for r in settled_signals if int(r['label']) == 1)
    false_signals = sum(1 for r in settled_signals if int(r['label']) == 0)
    missed = sum(1 for r in settled if not int(r[signal_key] or 0) and int(r['label']) == 1)
    rewards = []
    for r in settled:
        val = None
        if reward_key and reward_key in r.keys() and r[reward_key] is not None:
            val = float(r[reward_key])
        if val is None:
            val = _reward_for(int(r[signal_key] or 0), int(r['label']))
        rewards.append(val)
    score = sum(rewards)
    positive = sum(x for x in rewards if x > 0)
    negative = sum(-x for x in rewards if x < 0)
    last_reward = rewards[0] if rewards else 0.0
    return {
        'issued': issued,
        'pending_signals': pending_signals,
        'checked_signals': len(settled_signals),
        'resolved_games': len(settled),
        'hits': hits,
        'false_signals': false_signals,
        'precision': round(hits / len(settled_signals) * 100.0, 1) if settled_signals else 0.0,
        'missed_deuces': missed,
        'score': round(score, 2),
        'positive': round(positive, 2),
        'negative': round(negative, 2),
        'avg_reward': round(score / len(settled), 3) if settled else 0.0,
        'last_reward': round(last_reward, 2),
    }


def _recent_decisions(conn, snap_cols, limit=80):
    if not snap_cols:
        return []
    xgb_cols = {'xgb_ready','xgb_probability','xgb_strength','xgb_signal_issued'}
    has_xgb = xgb_cols.issubset(snap_cols)
    has_frozen = 'xgb_decision_frozen' in snap_cols
    has_xreward = 'xgb_reward' in snap_cols
    has_terminal = 'terminal_score' in snap_cols
    fields = [
        'key','event_id','set_num','game_num','player1','player2','server','entry_score',
        'signal_strength','threshold','signal_issued','resolved','label','reward','created_at','resolved_at'
    ]
    if has_terminal:
        fields.append('terminal_score')
    if has_xgb:
        fields += ['xgb_ready','xgb_probability','xgb_strength','xgb_signal_issued']
    if has_frozen:
        fields.append('xgb_decision_frozen')
    if has_xreward:
        fields.append('xgb_reward')
    order_expr = 'COALESCE(resolved_at,created_at) DESC'
    rows = conn.execute(
        f"SELECT {','.join(fields)} FROM v7_game_snapshots ORDER BY {order_expr} LIMIT ?", (limit,)
    ).fetchall()
    out = []
    for r in rows:
        label = None if r['label'] is None else int(r['label'])
        resolved = bool(int(r['resolved'] or 0))
        a_signal = bool(int(r['signal_issued'] or 0))
        a_reward = None if not resolved else (float(r['reward']) if r['reward'] is not None else _reward_for(a_signal, label or 0))
        xready = bool(int(r['xgb_ready'] or 0)) if has_xgb else False
        xfrozen = bool(int(r['xgb_decision_frozen'] or 0)) if has_frozen else xready
        b_signal = bool(int(r['xgb_signal_issued'] or 0)) if has_xgb and xfrozen else False
        b_reward = None
        if resolved and xfrozen and label is not None:
            if has_xreward and r['xgb_reward'] is not None:
                b_reward = float(r['xgb_reward'])
            else:
                b_reward = _reward_for(b_signal, label)
        ts = float(r['resolved_at'] or r['created_at'] or 0.0)
        out.append({
            'key': str(r['key']),
            'time': time.strftime('%H:%M:%S', time.localtime(ts)) if ts else '—',
            'match': f"{r['player1']} — {r['player2']}",
            'set_num': int(r['set_num']), 'game_num': int(r['game_num']),
            'server': str(r['server']), 'entry_score': str(r['entry_score'] or '—'),
            'terminal_score': str(r['terminal_score'] or '') if has_terminal else '',
            'resolved': resolved, 'label': label,
            'a_strength': round(float(r['signal_strength'] or 0.0), 1),
            'a_threshold': round(float(r['threshold'] or 0.0), 1),
            'a_signal': a_signal, 'a_reward': None if a_reward is None else round(a_reward, 2),
            'b_ready': xready, 'b_frozen': xfrozen,
            'b_probability': None if not has_xgb or r['xgb_probability'] is None else round(float(r['xgb_probability']), 1),
            'b_strength': None if not has_xgb or r['xgb_strength'] is None else round(float(r['xgb_strength']), 1),
            'b_signal': b_signal, 'b_reward': None if b_reward is None else round(b_reward, 2),
        })
    return out


def read_stats():
    empty_model = {
        'issued':0,'pending_signals':0,'checked_signals':0,'resolved_games':0,'hits':0,
        'false_signals':0,'precision':0.0,'missed_deuces':0,'score':0.0,'positive':0.0,
        'negative':0.0,'avg_reward':0.0,'last_reward':0.0,
    }
    empty = {
        'total':0,'hits':0,'misses':0,'unknown':0,'rate':0.0,'pending':0,
        'calibration_error':None,'best':[],'probability':[],'raw_probability':[],'scores':[],
        'server_scores':[],'sets':[],'game_bands':[],'servers':[],'data_quality':[],
        'trained_games':0,'policy_threshold':76.0,'missed_deuces':0,'avg_reward':0.0,'signal_games':0,
        'score_balance':0.0,'positive_points':0.0,'negative_points':0.0,'pain':0.0,'search_drive':0.0,'recent_precision':0.0,'recent_recall':0.0,'last_reward':0.0,
        'hit_streak':0,'miss_streak':0,'missed_deuce_streak':0,'current_avg_strength':0.0,'current_max_strength':0.0,
        'model_a':dict(empty_model),'model_b':dict(empty_model),
        'dual_model':{'ready':False,'trained_on':0,'comparable_games':0,'pending_comparable':0,'deuces':0,'disagreements':0},
        'recent_decisions':[], 'stats_note':'',
    }
    if not os.path.exists(DB_PATH):
        empty['stats_note'] = f'БД не найдена: {DB_PATH}'
        return empty

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH, timeout=2)
        conn.row_factory = sqlite3.Row

        if _table_exists(conn, 'segment_stats'):
            overall = conn.execute("SELECT total,hits,misses,unknown,probability_sum FROM segment_stats WHERE dimension='overall' AND label='Все сигналы'").fetchone()
        else:
            overall = None
        total = int(overall['total']) if overall else 0
        hits = int(overall['hits']) if overall else 0
        misses = int(overall['misses']) if overall else 0
        unknown = int(overall['unknown']) if overall else 0
        pending = int(conn.execute('SELECT COUNT(*) FROM pending_predictions').fetchone()[0]) if _table_exists(conn,'pending_predictions') else 0

        probability = _read_dimension(conn, 'probability')
        raw_probability = _read_dimension(conn, 'raw_probability')
        scores = _read_dimension(conn, 'score')
        server_scores = _read_dimension(conn, 'score_server')
        sets = _read_dimension(conn, 'set')
        game_bands = _read_dimension(conn, 'game_band')
        servers = _read_dimension(conn, 'server')
        data_quality = _read_dimension(conn, 'data_quality')

        cal_rows = [x for x in probability if x['checked'] > 0]
        denom = sum(x['checked'] for x in cal_rows)
        calibration_error = round(sum(abs(x['rate']-x['model_avg'])*x['checked'] for x in cal_rows)/denom,1) if denom else None

        best=[]
        for title, rows in (('Счёт относительно подающего',server_scores if server_scores else scores),('Сет',sets),('Номер гейма',game_bands)):
            eligible=[x for x in rows if x['checked']>=MIN_BEST_SIGNALS]
            if eligible:
                top=max(eligible,key=lambda x:(x['wilson_lower'],x['rate'],x['checked']))
                best.append({'group':title,'label':top['label'],'rate':top['rate'],'checked':top['checked']})

        trained_games=signal_games=missed_deuces=0
        policy_threshold=76.0
        avg_reward=score_balance=positive_points=negative_points=pain=search_drive=recent_precision=recent_recall=last_reward=0.0
        hit_streak=miss_streak=missed_deuce_streak=0
        current_avg_strength=current_max_strength=0.0
        model_a=dict(empty_model); model_b=dict(empty_model)
        dual={'ready':False,'trained_on':0,'comparable_games':0,'pending_comparable':0,'deuces':0,'disagreements':0}
        recent=[]
        stats_note=''

        snap_cols=_columns(conn,'v7_game_snapshots')
        if snap_cols:
            lr=conn.execute("""SELECT COUNT(*) n,
                COALESCE(SUM(CASE WHEN signal_issued=1 THEN 1 ELSE 0 END),0) sig,
                COALESCE(SUM(CASE WHEN signal_issued=0 AND label=1 THEN 1 ELSE 0 END),0) missed,
                COALESCE(AVG(reward),0) reward,COALESCE(SUM(reward),0) score,
                COALESCE(SUM(CASE WHEN reward>0 THEN reward ELSE 0 END),0) positive,
                COALESCE(SUM(CASE WHEN reward<0 THEN -reward ELSE 0 END),0) negative
                FROM v7_game_snapshots WHERE resolved=1""").fetchone()
            trained_games=int(lr['n']); signal_games=int(lr['sig']); missed_deuces=int(lr['missed'])
            avg_reward=round(float(lr['reward']),3); score_balance=round(float(lr['score']),2)
            positive_points=round(float(lr['positive']),2); negative_points=round(float(lr['negative']),2)
            last=conn.execute("SELECT reward FROM v7_game_snapshots WHERE resolved=1 ORDER BY resolved_at DESC LIMIT 1").fetchone()
            if last and last[0] is not None: last_reward=round(float(last[0]),2)
            strengths=conn.execute("SELECT COALESCE(AVG(signal_strength),0),COALESCE(MAX(signal_strength),0) FROM v7_game_snapshots WHERE resolved=0").fetchone()
            current_avg_strength=round(float(strengths[0]),1); current_max_strength=round(float(strengths[1]),1)

            # A: every real A decision, including pending signals. This increments immediately.
            a_rows=conn.execute("SELECT resolved,label,signal_issued,reward FROM v7_game_snapshots ORDER BY COALESCE(resolved_at,created_at) DESC").fetchall()
            model_a=_model_stats(a_rows,'signal_issued','reward')

            xgb_needed={'xgb_ready','xgb_signal_issued'}
            if xgb_needed.issubset(snap_cols):
                frozen_col='xgb_decision_frozen' if 'xgb_decision_frozen' in snap_cols else None
                where = "xgb_ready=1"
                if frozen_col:
                    where += " AND xgb_decision_frozen=1"
                xreward_select=',xgb_reward' if 'xgb_reward' in snap_cols else ''
                b_rows=conn.execute(f"SELECT resolved,label,xgb_signal_issued{xreward_select} FROM v7_game_snapshots WHERE {where} ORDER BY COALESCE(resolved_at,created_at) DESC").fetchall()
                model_b=_model_stats(b_rows,'xgb_signal_issued','xgb_reward' if 'xgb_reward' in snap_cols else None)
                settled=[r for r in b_rows if int(r['resolved'] or 0) and r['label'] is not None]
                dual['comparable_games']=len(settled)
                dual['pending_comparable']=sum(1 for r in b_rows if not int(r['resolved'] or 0))
                dual['deuces']=sum(int(r['label']) for r in settled)
                # disagreements require A column too
                cmp_where = where + " AND resolved=1 AND label IS NOT NULL"
                cmp_rows=conn.execute(f"SELECT signal_issued,xgb_signal_issued,label FROM v7_game_snapshots WHERE {cmp_where}").fetchall()
                dual['disagreements']=sum(int(r['signal_issued'])!=int(r['xgb_signal_issued']) for r in cmp_rows)
            recent=_recent_decisions(conn,snap_cols,80)

        if _table_exists(conn,'v7_meta'):
            meta={str(r['key']):float(r['value']) for r in conn.execute('SELECT key,value FROM v7_meta').fetchall()}
            policy_threshold=round(float(meta.get('threshold',76.0)),1)
            pain=round(float(meta.get('pain',0.0)),1)
            search_drive=round(float(meta.get('search_drive',0.0)),1)
            recent_precision=round(float(meta.get('recent_precision',0.0)),1)
            recent_recall=round(float(meta.get('recent_recall',0.0)),1)
            hit_streak=int(meta.get('hit_streak',0)); miss_streak=int(meta.get('miss_streak',0)); missed_deuce_streak=int(meta.get('missed_deuce_streak',0))
            dual['trained_on']=int(meta.get('xgb_training_count',0)); dual['ready']=dual['trained_on']>=80
        if 'xgb_decision_frozen' not in snap_cols:
            stats_note='Анализатор старой сборки: нет xgb_decision_frozen. Замени pari_deuce_analyzer_v7_3_dual.py на R5, иначе B-прогноз может стираться следующим счётом.'

        return {
            'total':total,'hits':hits,'misses':misses,'unknown':unknown,'rate':_pct(hits,total),'pending':pending,
            'calibration_error':calibration_error,'best':best,'probability':probability,'raw_probability':raw_probability,
            'scores':scores,'server_scores':server_scores,'sets':sets,'game_bands':game_bands,'servers':servers[:20],
            'data_quality':data_quality,'trained_games':trained_games,'policy_threshold':policy_threshold,
            'missed_deuces':missed_deuces,'avg_reward':avg_reward,'signal_games':signal_games,'score_balance':score_balance,
            'positive_points':positive_points,'negative_points':negative_points,'pain':pain,'search_drive':search_drive,
            'recent_precision':recent_precision,'recent_recall':recent_recall,'last_reward':last_reward,
            'hit_streak':hit_streak,'miss_streak':miss_streak,'missed_deuce_streak':missed_deuce_streak,
            'current_avg_strength':current_avg_strength,'current_max_strength':current_max_strength,
            'model_a':model_a,'model_b':model_b,'dual_model':dual,'recent_decisions':recent,'stats_note':stats_note,
        }
    except sqlite3.Error as e:
        out=dict(empty); out['error']=str(e); return out
    finally:
        if conn is not None: conn.close()


STATS_WIDGET = r'''
<style>
#modelStatsWrap{margin:10px 0 14px;padding:12px;border:1px solid #2b3645;border-radius:12px;background:#111820;color:#e7edf5;font-family:Arial,sans-serif}
.stats-head{margin-bottom:10px;padding:9px 11px;background:#13202b;border:1px solid #29445b;border-radius:9px}
.overall-row,.lane-cards{display:flex;gap:7px;flex-wrap:wrap;align-items:stretch}
.ms-card{background:#17212b;border:1px solid #2a3948;border-radius:9px;padding:7px 9px;min-width:92px;flex:1 1 92px}
.ms-label{font-size:10px;color:#8fa2b5;margin-bottom:3px;white-space:nowrap}.ms-value{font-size:16px;font-weight:800}
.ms-hit .ms-value{color:#33d17a}.ms-miss .ms-value{color:#ff6b6b}.ms-rate .ms-value{color:#67b7ff}.ms-pending .ms-value{color:#f7c948}.ms-score .ms-value{color:#f7c948}.ms-positive .ms-value{color:#33d17a}.ms-negative .ms-value{color:#ff6b6b}.ms-pain .ms-value{color:#ff9f43}
.model-lane{margin-top:10px;padding:10px;border:1px solid #2f4355;border-radius:11px;background:#101923}.model-a{border-color:#36516e}.model-b{border-color:#4a5f3d}
.lane-head{display:flex;gap:10px;justify-content:space-between;align-items:center;flex-wrap:wrap;margin-bottom:7px}.lane-title{font-size:15px;font-weight:800}.lane-sub{font-size:11px;color:#8fa2b5}
.good{color:#33d17a}.bad{color:#ff6b6b}.warn{color:#f7c948}.blue{color:#67b7ff}
#modelStatsBtn{margin-top:10px;background:#23364a;color:#dbe9f7;border:1px solid #36516e;border-radius:8px;padding:8px 12px;cursor:pointer}
#modelStatsDetails{display:none;margin-top:12px;border-top:1px solid #2b3645;padding-top:10px;font-size:12px;overflow-x:auto}.ms-section{margin:12px 0}.ms-section h4{margin:0 0 6px;font-size:13px;color:#dbe9f7}
.ms-row{display:grid;grid-template-columns:minmax(120px,1.4fr) 80px 70px 80px 80px 70px;gap:8px;padding:5px 0;border-bottom:1px solid #1d2a36;align-items:center}
.ab-table{width:100%;border-collapse:collapse;min-width:1180px;font-size:11px}.ab-table th,.ab-table td{padding:6px 7px;border-bottom:1px solid #1d2a36;text-align:left;vertical-align:middle}.ab-table th{position:sticky;top:0;background:#17212b;color:#a9bacb}.ab-table tr.pending{background:#171b20}.pill{display:inline-block;padding:2px 6px;border-radius:6px;border:1px solid #33475a;font-weight:700}.pill-signal{color:#33d17a;border-color:#2e6a4c}.pill-silent{color:#a7b3bf}.reward-pos{color:#33d17a;font-weight:800}.reward-neg{color:#ff6b6b;font-weight:800}.reward-wait{color:#f7c948}.muted{color:#8fa2b5}
#bestConditions{margin-top:10px;display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:8px}.best-card{background:#151f29;border:1px solid #2c3d4d;border-radius:9px;padding:9px 11px}.best-title{font-size:11px;color:#8fa2b5}.best-name{font-size:15px;font-weight:700;margin:3px 0}.best-meta{font-size:12px;color:#c8d3df}
.notice{margin-top:8px;padding:8px 10px;border:1px solid #6e5a27;background:#211d12;border-radius:8px;color:#ffd166;display:none}
@media(max-width:900px){.ms-card{min-width:110px}.ms-row{grid-template-columns:minmax(90px,1.3fr) 62px 62px 70px 70px 62px;font-size:10px}}
</style>
<div id="modelStatsWrap">
  <div class="stats-head">
    <div style="display:flex;gap:12px;align-items:center;justify-content:space-between;flex-wrap:wrap;margin-bottom:5px">
      <b style="font-size:15px">V7.3 DUAL BRAIN · FULL STATS R4</b>
      <span style="padding:4px 8px;border-radius:7px;background:#26384a;border:1px solid #41617e;color:#8fd3ff;font-weight:800">ВЕРСИЯ: __APP_UI_VERSION__</span>
    </div>
    <div style="color:#ffd166;font-weight:700">ФАЙЛ: __APP_FILE_NAME__</div>
    <div style="color:#8295a8;font-size:10px;word-break:break-all">Путь: __APP_FILE_PATH__</div>
    <div style="margin-top:5px">Счётчик <b>«Выдано»</b> растёт сразу при появлении прогноза. <b>Reward/штраф</b> начисляется только после завершения гейма.</div>
  </div>
  <div id="statsNotice" class="notice"></div>

  <div class="overall-row">
    <div class="ms-card"><div class="ms-label">Проверено сигналов</div><div id="msTotal" class="ms-value">0</div></div>
    <div class="ms-card ms-hit"><div class="ms-label">✅ Сработало</div><div id="msHits" class="ms-value">0</div></div>
    <div class="ms-card ms-miss"><div class="ms-label">❌ Не сработало</div><div id="msMisses" class="ms-value">0</div></div>
    <div class="ms-card"><div class="ms-label">⚪ Не определено</div><div id="msUnknown" class="ms-value">0</div></div>
    <div class="ms-card ms-rate"><div class="ms-label">🎯 Проходимость</div><div id="msRate" class="ms-value">0%</div></div>
    <div class="ms-card ms-pending"><div class="ms-label">⏳ Pending A</div><div id="msPending" class="ms-value">0</div></div>
    <div class="ms-card"><div class="ms-label">🧠 Обучено геймов</div><div id="msTrained" class="ms-value">0</div></div>
    <div class="ms-card"><div class="ms-label">🎚 Порог A</div><div id="msThreshold" class="ms-value">0</div></div>
    <div class="ms-card"><div class="ms-label">👋 Пропущено deuce</div><div id="msMissedDeuce" class="ms-value">0</div></div>
    <div class="ms-card"><div class="ms-label">Reward средний</div><div id="msReward" class="ms-value">0</div></div>
    <div class="ms-card ms-score"><div class="ms-label">🏆 Баланс A</div><div id="msBalance" class="ms-value">0</div></div>
    <div class="ms-card ms-positive"><div class="ms-label">💚 Заработано</div><div id="msPositive" class="ms-value">0</div></div>
    <div class="ms-card ms-negative"><div class="ms-label">💥 Штрафы</div><div id="msNegative" class="ms-value">0</div></div>
    <div class="ms-card"><div class="ms-label">Последний reward</div><div id="msLastReward" class="ms-value">0</div></div>
    <div class="ms-card ms-pain"><div class="ms-label">🔥 Pain A</div><div id="msPain" class="ms-value">0</div></div>
    <div class="ms-card"><div class="ms-label">Средняя сила live</div><div id="msAvgStrength" class="ms-value">0</div></div>
    <div class="ms-card"><div class="ms-label">Макс. сила live</div><div id="msMaxStrength" class="ms-value">0</div></div>
  </div>

  <div class="model-lane model-a">
    <div class="lane-head"><div><span class="lane-title">A · Online logistic SEEK</span> <span class="lane-sub">боевой сигнал</span></div><div class="lane-sub">порог <b id="aThreshold">0</b> · страх <b id="aPain">0</b> · поиск <b id="aSearch">0</b></div></div>
    <div class="lane-cards">
      <div class="ms-card"><div class="ms-label">📣 Выдано сразу</div><div id="aIssued" class="ms-value">0</div></div>
      <div class="ms-card ms-pending"><div class="ms-label">⏳ Ждут результата</div><div id="aPending" class="ms-value">0</div></div>
      <div class="ms-card"><div class="ms-label">⚖️ Проверено сигналов</div><div id="aChecked" class="ms-value">0</div></div>
      <div class="ms-card ms-hit"><div class="ms-label">✅ Поймано 40:40</div><div id="aHits" class="ms-value">0</div></div>
      <div class="ms-card ms-miss"><div class="ms-label">❌ Ложные</div><div id="aFalse" class="ms-value">0</div></div>
      <div class="ms-card ms-rate"><div class="ms-label">🎯 Precision</div><div id="aPrecision" class="ms-value">0%</div></div>
      <div class="ms-card ms-rate"><div class="ms-label">🔎 Recent precision</div><div id="aRecentPrecision" class="ms-value">0%</div></div>
      <div class="ms-card ms-rate"><div class="ms-label">🕸 Recent recall</div><div id="aRecentRecall" class="ms-value">0%</div></div>
      <div class="ms-card"><div class="ms-label">👋 Пропущено deuce</div><div id="aMissed" class="ms-value">0</div></div>
      <div class="ms-card ms-score"><div class="ms-label">🏆 Score</div><div id="aScore" class="ms-value">0</div></div>
      <div class="ms-card ms-positive"><div class="ms-label">💚 +Reward</div><div id="aPositive" class="ms-value">0</div></div>
      <div class="ms-card ms-negative"><div class="ms-label">💥 -Penalty</div><div id="aNegative" class="ms-value">0</div></div>
      <div class="ms-card"><div class="ms-label">Последний reward</div><div id="aLast" class="ms-value">0</div></div>
      <div class="ms-card"><div class="ms-label">Reward / гейм</div><div id="aAvg" class="ms-value">0</div></div>
    </div>
  </div>

  <div class="model-lane model-b">
    <div class="lane-head"><div><span class="lane-title">B · XGBoost</span> <span class="lane-sub">shadow, решение фиксируется и не стирается</span></div><div class="lane-sub">статус <b id="bStatus">—</b> · train <b id="bTrained">0</b></div></div>
    <div class="lane-cards">
      <div class="ms-card"><div class="ms-label">📣 Выдано сразу</div><div id="bIssued" class="ms-value">0</div></div>
      <div class="ms-card ms-pending"><div class="ms-label">⏳ Ждут результата</div><div id="bPending" class="ms-value">0</div></div>
      <div class="ms-card"><div class="ms-label">⚖️ Проверено сигналов</div><div id="bChecked" class="ms-value">0</div></div>
      <div class="ms-card ms-hit"><div class="ms-label">✅ Поймано 40:40</div><div id="bHits" class="ms-value">0</div></div>
      <div class="ms-card ms-miss"><div class="ms-label">❌ Ложные</div><div id="bFalse" class="ms-value">0</div></div>
      <div class="ms-card ms-rate"><div class="ms-label">🎯 Precision</div><div id="bPrecision" class="ms-value">0%</div></div>
      <div class="ms-card"><div class="ms-label">👋 Пропущено deuce</div><div id="bMissed" class="ms-value">0</div></div>
      <div class="ms-card ms-score"><div class="ms-label">🏆 Score</div><div id="bScore" class="ms-value">0</div></div>
      <div class="ms-card ms-positive"><div class="ms-label">💚 +Reward</div><div id="bPositive" class="ms-value">0</div></div>
      <div class="ms-card ms-negative"><div class="ms-label">💥 -Penalty</div><div id="bNegative" class="ms-value">0</div></div>
      <div class="ms-card"><div class="ms-label">Последний reward</div><div id="bLast" class="ms-value">0</div></div>
      <div class="ms-card"><div class="ms-label">Reward / гейм</div><div id="bAvg" class="ms-value">0</div></div>
    </div>
  </div>

  <div style="margin-top:8px" class="lane-sub">⚔️ Сравнимых завершённых: <b id="dualCompared">0</b> · ждут результата: <b id="dualPending">0</b> · модели разошлись: <b id="dualDisagreements">0</b></div>
  <button id="modelStatsBtn" type="button">Подробная статистика A + B</button>
  <div id="bestConditions"></div>
  <div id="modelStatsDetails"></div>
</div>
<script>
(function(){
  const esc=(s)=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));
  const score=v=>{v=Number(v||0);return (v>=0?'+':'')+v.toFixed(2)};
  const reward=v=>{if(v===null||v===undefined)return '<span class="reward-wait">⏳</span>';v=Number(v);return `<span class="${v>=0?'reward-pos':'reward-neg'}">${v>=0?'+':''}${v.toFixed(2)}</span>`};
  const decision=v=>v?'<span class="pill pill-signal">SIGNAL</span>':'<span class="pill pill-silent">SILENCE</span>';
  const fact=r=>!r.resolved?'<span class="reward-wait">⏳ идёт</span>':(r.label===1?'<span class="good">40:40 ✅</span>':'<span class="muted">без 40:40</span>');
  function rowsBlock(title,rows){if(!rows||!rows.length)return '';return `<div class="ms-section"><h4>${esc(title)}</h4><div class="ms-row"><b>Условие</b><b>Проверено</b><b>Прошло</b><b>Факт</b><b>Среднее</b><b>Неопр.</b></div>`+rows.map(x=>`<div class="ms-row"><span>${esc(x.label)}</span><span>${x.checked}</span><span>${x.hits}</span><span>${x.rate}%</span><span>${x.model_avg}%</span><span>${x.unknown}</span></div>`).join('')+'</div>'}
  function decisionsBlock(rows){
    if(!rows||!rows.length)return '<div class="ms-section"><h4>Последние решения A/B</h4><div class="muted">Пока нет снимков геймов.</div></div>';
    return `<div class="ms-section"><h4>Последние решения A/B — прогноз → факт → reward/штраф</h4><table class="ab-table"><thead><tr><th>Время</th><th>Матч / гейм</th><th>Старт → итог</th><th>A сила / порог</th><th>A решение</th><th>A reward</th><th>B P / сила</th><th>B решение</th><th>B reward</th><th>Факт</th></tr></thead><tbody>`+
      rows.map(r=>`<tr class="${r.resolved?'':'pending'}"><td>${esc(r.time)}</td><td><b>${esc(r.match)}</b><br><span class="muted">сет ${r.set_num}, гейм ${r.game_num} · ${esc(r.server)}</span></td><td>${esc(r.entry_score)} → ${r.resolved?esc(r.terminal_score||'конец'):'…'}</td><td>${Number(r.a_strength||0).toFixed(1)} / ${Number(r.a_threshold||0).toFixed(1)}</td><td>${decision(r.a_signal)}</td><td>${reward(r.a_reward)}</td><td>${r.b_frozen?`${r.b_probability??'—'}% / ${r.b_strength??'—'}`:'<span class="muted">не зафиксировано</span>'}</td><td>${r.b_frozen?decision(r.b_signal):'<span class="muted">—</span>'}</td><td>${r.b_frozen?reward(r.b_reward):'<span class="muted">—</span>'}</td><td>${fact(r)}</td></tr>`).join('')+`</tbody></table></div>`;
  }
  const set=(id,v)=>{const e=document.getElementById(id);if(e)e.textContent=v};
  function fillModel(prefix,m){set(prefix+'Issued',m.issued||0);set(prefix+'Pending',m.pending_signals||0);set(prefix+'Checked',m.checked_signals||0);set(prefix+'Hits',m.hits||0);set(prefix+'False',m.false_signals||0);set(prefix+'Precision',Number(m.precision||0).toFixed(1)+'%');set(prefix+'Missed',m.missed_deuces||0);set(prefix+'Score',score(m.score));set(prefix+'Positive','+'+Number(m.positive||0).toFixed(2));set(prefix+'Negative','-'+Number(m.negative||0).toFixed(2));set(prefix+'Last',score(m.last_reward));set(prefix+'Avg',Number(m.avg_reward||0).toFixed(3));}
  async function loadModelStats(){
    try{
      const r=await fetch('/api/stats?_='+Date.now(),{cache:'no-store'});const s=await r.json();
      set('msTotal',s.total??0);set('msHits',s.hits??0);set('msMisses',s.misses??0);set('msUnknown',s.unknown??0);set('msRate',Number(s.rate||0).toFixed(1)+'%');set('msPending',s.pending??0);set('msTrained',s.trained_games??0);set('msThreshold',Number(s.policy_threshold||0).toFixed(1)+'/100');set('msMissedDeuce',s.missed_deuces??0);set('msReward',Number(s.avg_reward||0).toFixed(3));set('msBalance',score(s.score_balance));set('msPositive','+'+Number(s.positive_points||0).toFixed(2));set('msNegative','-'+Number(s.negative_points||0).toFixed(2));set('msLastReward',score(s.last_reward));set('msPain',Number(s.pain||0).toFixed(0)+'/100');set('msAvgStrength',Number(s.current_avg_strength||0).toFixed(1)+'/100');set('msMaxStrength',Number(s.current_max_strength||0).toFixed(1)+'/100');
      fillModel('a',s.model_a||{});fillModel('b',s.model_b||{});set('aThreshold',Number(s.policy_threshold||0).toFixed(1)+'/100');set('aPain',Number(s.pain||0).toFixed(0)+'/100');set('aSearch',Number(s.search_drive||0).toFixed(0)+'/100');set('aRecentPrecision',Number(s.recent_precision||0).toFixed(1)+'%');set('aRecentRecall',Number(s.recent_recall||0).toFixed(1)+'%');
      const dm=s.dual_model||{};set('bTrained',dm.trained_on||0);set('bStatus',dm.ready?'ГОТОВА':`РАЗОГРЕВ ${dm.trained_on||0}/80`);set('dualCompared',dm.comparable_games||0);set('dualPending',dm.pending_comparable||0);set('dualDisagreements',dm.disagreements||0);
      const note=document.getElementById('statsNotice');if(s.stats_note){note.textContent=s.stats_note;note.style.display='block'}else{note.style.display='none'}
      const best=document.getElementById('bestConditions');best.innerHTML=(s.best||[]).length?(s.best||[]).map(x=>`<div class="best-card"><div class="best-title">V7 условие · ${esc(x.group)}</div><div class="best-name">${esc(x.label)}</div><div class="best-meta">Проходимость <b>${x.rate}%</b> · Проверено <b>${x.checked}</b></div></div>`).join(''):'';
      const d=document.getElementById('modelStatsDetails');d.innerHTML=decisionsBlock(s.recent_decisions)+rowsBlock('A: по диапазону силы сигнала',s.probability)+rowsBlock('A: по online-вероятности deuce',s.raw_probability)+rowsBlock('A: по счёту относительно подающего',s.server_scores)+rowsBlock('A: по обычному счёту в момент сигнала',s.scores)+rowsBlock('A: по сетам',s.sets)+rowsBlock('A: по номеру гейма',s.game_bands)+rowsBlock('A: по подающим',s.servers)+rowsBlock('A: по объёму данных',s.data_quality);
    }catch(e){console.log('stats error',e)}
  }
  document.getElementById('modelStatsBtn').addEventListener('click',()=>{const d=document.getElementById('modelStatsDetails');d.style.display=d.style.display==='block'?'none':'block'});
  loadModelStats();setInterval(loadModelStats,2000);
})();
</script>
'''


@app.after_request
def disable_browser_cache(response):
    response.headers['Cache-Control']='no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma']='no-cache'
    response.headers['Expires']='0'
    return response


@app.route('/')
def index():
    html=render_template('index.html')
    widget=(STATS_WIDGET.replace('__APP_UI_VERSION__',APP_UI_VERSION).replace('__APP_FILE_NAME__',APP_FILE_NAME).replace('__APP_FILE_PATH__',APP_FILE_PATH))
    return html.replace('</body>',widget+'\n</body>',1) if '</body>' in html else html+widget


@app.route('/api/matches')
def get_matches():
    if not os.path.exists(JSON_PATH): return jsonify([])
    try:
        with open(JSON_PATH,'r',encoding='utf-8') as f: data=json.load(f)
        return jsonify(data if isinstance(data,list) else [])
    except Exception as e:
        return jsonify({'error':str(e)}),500


@app.route('/api/version')
def get_version():
    return jsonify({'ui_version':APP_UI_VERSION,'file_name':APP_FILE_NAME,'file_path':APP_FILE_PATH,'db_path':DB_PATH})


@app.route('/api/stats')
def get_stats():
    return jsonify(read_stats())


if __name__=='__main__':
    app.run(debug=False,host='0.0.0.0',port=5002)
