from flask import Flask, jsonify, render_template_string
import json
import os
import sqlite3

app = Flask(__name__)
BASE_DIR = os.path.dirname(__file__)
JSON_PATH = os.path.join(BASE_DIR, "exact_game_live.json")
DB_PATH = os.path.join(BASE_DIR, "pari_exact_game_stats.sqlite3")
OUTCOMES = ("40:0", "40:15", "40:30", "+:40", "0:40", "15:40", "30:40", "40:+")


def _pct(a, b):
    return round(a / b * 100.0, 1) if b else 0.0


def read_stats():
    empty = {
        "checked": 0, "top1_hits": 0, "top2_hits": 0, "unknown": 0, "pending": 0,
        "top1_accuracy": 0.0, "top2_accuracy": 0.0, "brier": None, "log_loss": None,
        "outcomes": [], "recent": [],
    }
    if not os.path.exists(DB_PATH):
        return empty
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH, timeout=2)
        conn.row_factory = sqlite3.Row
        overall = conn.execute("SELECT * FROM overall_stats WHERE id=1").fetchone()
        if overall is None:
            return empty
        checked = int(overall["checked"])
        top1_hits = int(overall["top1_hits"])
        top2_hits = int(overall["top2_hits"])
        unknown = int(overall["unknown"])
        pending = int(conn.execute("SELECT COUNT(*) FROM pending_predictions").fetchone()[0])

        rows = conn.execute(
            "SELECT outcome,resolved,actual_count,predicted_sum,top_pick_count,top_pick_hits FROM outcome_stats"
        ).fetchall()
        by_name = {str(r["outcome"]): r for r in rows}
        outcomes = []
        for name in OUTCOMES:
            r = by_name.get(name)
            if r is None:
                continue
            resolved = int(r["resolved"])
            actual = int(r["actual_count"])
            top_count = int(r["top_pick_count"])
            top_hits = int(r["top_pick_hits"])
            model_avg = float(r["predicted_sum"]) / resolved * 100.0 if resolved else 0.0
            actual_rate = actual / resolved * 100.0 if resolved else 0.0
            outcomes.append({
                "outcome": name,
                "checked": resolved,
                "actual_count": actual,
                "actual_rate": round(actual_rate, 1),
                "model_avg": round(model_avg, 1),
                "difference": round(actual_rate - model_avg, 1) if resolved else 0.0,
                "top_pick_count": top_count,
                "top_pick_hits": top_hits,
                "top_pick_accuracy": _pct(top_hits, top_count),
            })

        recent_rows = conn.execute(
            """SELECT player1,player2,set_num,game_num,server,top1,actual_outcome,top1_hit,top2_hit,created_at
               FROM signal_history WHERE status='resolved' ORDER BY id DESC LIMIT 30"""
        ).fetchall()
        recent = [{
            "match": f"{r['player1']} — {r['player2']}",
            "set": int(r["set_num"]),
            "game": int(r["game_num"]),
            "server": str(r["server"]),
            "forecast": str(r["top1"]),
            "actual": str(r["actual_outcome"]),
            "hit": bool(r["top1_hit"]),
            "top2_hit": bool(r["top2_hit"]),
        } for r in recent_rows]

        return {
            "checked": checked,
            "top1_hits": top1_hits,
            "top2_hits": top2_hits,
            "unknown": unknown,
            "pending": pending,
            "top1_accuracy": round(_pct(top1_hits, checked), 1),
            "top2_accuracy": round(_pct(top2_hits, checked), 1),
            "brier": round(float(overall["brier_sum"]) / checked, 4) if checked else None,
            "log_loss": round(float(overall["logloss_sum"]) / checked, 3) if checked else None,
            "outcomes": outcomes,
            "recent": recent,
        }
    except sqlite3.Error as e:
        out = dict(empty)
        out["error"] = str(e)
        return out
    finally:
        if conn is not None:
            conn.close()


PAGE = r'''
<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PARI — Как закончится гейм</title>
<style>
:root{color-scheme:dark;--bg:#0b1118;--card:#121b25;--card2:#172330;--line:#26384a;--text:#eaf2fb;--muted:#8fa4b8;--green:#38d98a;--blue:#69b8ff;--yellow:#f2c94c;--red:#ff6b78}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:Arial,sans-serif}.wrap{max-width:1450px;margin:auto;padding:18px}.nav{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px}.nav a{color:#dcecff;text-decoration:none;padding:9px 13px;background:#17283a;border:1px solid #2f4b67;border-radius:9px}.nav a.active{border-color:#69b8ff;background:#1a3045}.title{font-size:25px;font-weight:800;margin:2px 0 5px}.sub{color:var(--muted);font-size:13px;margin-bottom:14px}.stats{display:flex;gap:9px;flex-wrap:wrap;margin:14px 0}.stat{min-width:130px;background:var(--card);border:1px solid var(--line);border-radius:11px;padding:9px 12px}.stat small{display:block;color:var(--muted);font-size:11px;margin-bottom:3px}.stat b{font-size:20px}.good{color:var(--green)}.blue{color:var(--blue)}.warn{color:var(--yellow)}
.section{margin-top:18px}.section h2{font-size:17px;margin:0 0 10px}.matches{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:12px}.match{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:13px}.matchhead{display:flex;justify-content:space-between;gap:10px;align-items:flex-start}.players{font-size:16px;font-weight:700}.meta{font-size:12px;color:var(--muted);margin-top:4px}.pari{color:#7fc1ff;text-decoration:none;font-size:12px;white-space:nowrap}.topline{margin:10px 0;padding:8px 10px;background:#15283a;border-radius:8px;border:1px solid #2b4a66}.topline b{color:var(--green)}.grid8{display:grid;grid-template-columns:repeat(4,1fr);gap:7px}.out{background:var(--card2);border:1px solid #2a3d50;border-radius:8px;padding:8px;text-align:center}.out.top{border-color:#4f9bd6;background:#183149}.out .name{font-size:13px;color:#bdd0e2}.out .prob{font-size:18px;font-weight:800;margin-top:2px}.small{font-size:11px;color:var(--muted);margin-top:8px}.panel{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px;overflow:auto}.btn{background:#20364b;color:#e8f3ff;border:1px solid #365779;border-radius:8px;padding:8px 11px;cursor:pointer}.hidden{display:none}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:7px 8px;border-bottom:1px solid #213142;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{color:#9fb4c8}.hit{color:var(--green)}.miss{color:var(--red)}
@media(max-width:650px){.wrap{padding:10px}.matches{grid-template-columns:1fr}.grid8{grid-template-columns:repeat(2,1fr)}.matchhead{display:block}.pari{display:inline-block;margin-top:7px}}
</style>
</head>
<body><div class="wrap">
<div class="nav">
  <a href="http://127.0.0.1:5000/">40:40 / деусы · :5000</a>
  <a class="active" href="http://127.0.0.1:5001/">Как закончится гейм · :5001</a>
</div>
<div class="title">🎾 Как закончится гейм</div>
<div class="sub">8 исходов. Прогноз фиксируется только в начале гейма при 0:0. Если скрипт пропустил начало гейма — этот гейм в статистику не засчитывается.</div>
<div class="stats">
 <div class="stat"><small>Проверено прогнозов</small><b id="checked">0</b></div>
 <div class="stat"><small>🎯 Угадан точный исход</small><b class="good" id="top1Hits">0</b></div>
 <div class="stat"><small>Точность TOP-1</small><b class="blue" id="top1Acc">0.0%</b></div>
 <div class="stat"><small>Попадание в TOP-2</small><b class="blue" id="top2Acc">0.0%</b></div>
 <div class="stat"><small>⚪ Не удалось определить</small><b id="unknown">0</b></div>
 <div class="stat"><small>⏳ Сейчас в ожидании</small><b class="warn" id="pending">0</b></div>
</div>
<div class="section"><h2>Живые прогнозы</h2><div id="matches" class="matches"></div></div>
<div class="section"><button id="statsBtn" class="btn">Подробная статистика</button></div>
<div id="details" class="section hidden">
 <div class="panel"><h2>Калибровка по каждому исходу</h2><div class="sub">«Факт» — как часто исход реально случался. «Модель» — какую вероятность модель в среднем ему давала.</div><div id="outcomeTable"></div></div>
 <div class="panel" style="margin-top:12px"><h2>Последние проверенные прогнозы</h2><div id="recentTable"></div></div>
</div>
</div>
<script>
const order=['40:0','40:15','40:30','+:40','0:40','15:40','30:40','40:+'];
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));
function renderMatches(list){
 const root=document.getElementById('matches');
 if(!list||!list.length){root.innerHTML='<div class="panel sub">Сейчас нет прогнозов, зафиксированных на 0:0. Скрипт ждёт начало новых геймов.</div>';return;}
 root.innerHTML=list.map(m=>{
   const probs=m.probabilities||{};
   const boxes=order.map(o=>`<div class="out ${o===m.top1?'top':''}"><div class="name">${esc(o)}</div><div class="prob">${Number(probs[o]||0).toFixed(1)}%</div></div>`).join('');
   return `<div class="match"><div class="matchhead"><div><div class="players">${esc(m.player1)} — ${esc(m.player2)}</div><div class="meta">Сет ${m.set_num}, гейм ${m.game_num} · подаёт: ${esc(m.server)} · текущий счёт ${esc(m.current_score)}</div></div><a class="pari" target="_blank" href="${esc(m.match_link)}">Открыть матч PARI ↗</a></div><div class="topline">Наиболее вероятный исход: <b>${esc(m.top1)} · ${Number(m.top1_probability).toFixed(1)}%</b></div><div class="grid8">${boxes}</div><div class="small">Оценка выигрыша очка подающим: ${(Number(m.p_server)*100).toFixed(1)}% · ${esc(m.data_quality)} · TOP-2: ${esc((m.top2||[]).join(', '))}</div></div>`;
 }).join('');
}
async function loadMatches(){try{const r=await fetch('/api/matches?_='+Date.now(),{cache:'no-store'});renderMatches(await r.json());}catch(e){}}
function renderOutcomeTable(rows){
 document.getElementById('outcomeTable').innerHTML='<table><thead><tr><th>Исход</th><th>Проверено</th><th>Случилось</th><th>Факт</th><th>Модель</th><th>Разница</th><th>Был TOP-1</th><th>TOP-1 угадан</th></tr></thead><tbody>'+ (rows||[]).map(x=>`<tr><td><b>${esc(x.outcome)}</b></td><td>${x.checked}</td><td>${x.actual_count}</td><td>${x.actual_rate}%</td><td>${x.model_avg}%</td><td>${x.difference>0?'+':''}${x.difference} п.п.</td><td>${x.top_pick_count}</td><td>${x.top_pick_accuracy}%</td></tr>`).join('')+'</tbody></table>';
}
function renderRecent(rows){
 document.getElementById('recentTable').innerHTML='<table><thead><tr><th>Матч</th><th>Сет / гейм</th><th>Подающий</th><th>Прогноз</th><th>Факт</th><th>Результат</th></tr></thead><tbody>'+ (rows||[]).map(x=>`<tr><td>${esc(x.match)}</td><td>${x.set} / ${x.game}</td><td>${esc(x.server)}</td><td>${esc(x.forecast)}</td><td>${esc(x.actual)}</td><td class="${x.hit?'hit':'miss'}">${x.hit?'✅ точно':'❌'}${!x.hit&&x.top2_hit?' · TOP-2':''}</td></tr>`).join('')+'</tbody></table>';
}
async function loadStats(){try{const r=await fetch('/api/stats?_='+Date.now(),{cache:'no-store'}),s=await r.json();document.getElementById('checked').textContent=s.checked||0;document.getElementById('top1Hits').textContent=s.top1_hits||0;document.getElementById('top1Acc').textContent=Number(s.top1_accuracy||0).toFixed(1)+'%';document.getElementById('top2Acc').textContent=Number(s.top2_accuracy||0).toFixed(1)+'%';document.getElementById('unknown').textContent=s.unknown||0;document.getElementById('pending').textContent=s.pending||0;renderOutcomeTable(s.outcomes);renderRecent(s.recent);}catch(e){}}
document.getElementById('statsBtn').addEventListener('click',()=>document.getElementById('details').classList.toggle('hidden'));
loadMatches();loadStats();setInterval(loadMatches,2500);setInterval(loadStats,3000);
</script></body></html>
'''


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/api/matches")
def api_matches():
    if not os.path.exists(JSON_PATH):
        return jsonify([])
    try:
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data if isinstance(data, list) else [])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats")
def api_stats():
    return jsonify(read_stats())


if __name__ == "__main__":
    print("ЗАПУЩЕН ФАЙЛ:", __file__)
    print("ПОРТ: 5001")
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5001
    )