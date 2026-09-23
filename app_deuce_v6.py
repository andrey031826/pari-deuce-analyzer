from flask import Flask, render_template, jsonify
import json
import os
import sqlite3
import math

app = Flask(__name__)
BASE_DIR = os.path.dirname(__file__)
JSON_PATH = os.path.join(BASE_DIR, 'tennis.json')
DB_PATH = os.path.join(BASE_DIR, 'pari_deuce_v6.sqlite3')
MIN_BEST_SIGNALS = 50


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


def _read_dimension(conn, dimension):
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
            'label': str(r['label']),
            'checked': total,
            'hits': hits,
            'misses': int(r['misses']),
            'unknown': int(r['unknown']),
            'rate': _pct(hits, total),
            'model_avg': round(float(r['probability_sum']) / total, 1) if total else 0.0,
            'wilson_lower': round(_wilson_lower(hits, total), 1),
        })
    return result


def read_stats():
    empty = {
        'total': 0, 'hits': 0, 'misses': 0, 'unknown': 0, 'rate': 0.0, 'pending': 0,
        'calibration_error': None, 'best': [], 'probability': [], 'raw_probability': [], 'scores': [],
        'server_scores': [], 'sets': [], 'game_bands': [], 'servers': [], 'data_quality': []
    }
    if not os.path.exists(DB_PATH):
        return empty

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH, timeout=2)
        conn.row_factory = sqlite3.Row

        overall = conn.execute(
            '''SELECT total,hits,misses,unknown,probability_sum
               FROM segment_stats WHERE dimension='overall' AND label='Все сигналы' '''
        ).fetchone()
        total = int(overall['total']) if overall else 0
        hits = int(overall['hits']) if overall else 0
        misses = int(overall['misses']) if overall else 0
        unknown = int(overall['unknown']) if overall else 0
        pending = int(conn.execute('SELECT COUNT(*) FROM pending_predictions').fetchone()[0])

        probability = _read_dimension(conn, 'probability')
        raw_probability = _read_dimension(conn, 'raw_probability')
        scores = _read_dimension(conn, 'score')
        server_scores = _read_dimension(conn, 'score_server')
        sets = _read_dimension(conn, 'set')
        game_bands = _read_dimension(conn, 'game_band')
        servers = _read_dimension(conn, 'server')
        data_quality = _read_dimension(conn, 'data_quality')

        # Средняя абсолютная ошибка калибровки в процентных пунктах.
        cal_rows = [x for x in probability if x['checked'] > 0]
        denom = sum(x['checked'] for x in cal_rows)
        calibration_error = None
        if denom:
            calibration_error = round(
                sum(abs(x['rate'] - x['model_avg']) * x['checked'] for x in cal_rows) / denom,
                1,
            )

        # Один лучший достаточно проверенный сегмент из каждой понятной группы.
        best = []
        for title, rows in (
            ('Счёт относительно подающего', server_scores if server_scores else scores),
            ('Сет', sets),
            ('Номер гейма', game_bands),
        ):
            eligible = [x for x in rows if x['checked'] >= MIN_BEST_SIGNALS]
            if eligible:
                top = max(eligible, key=lambda x: (x['wilson_lower'], x['rate'], x['checked']))
                best.append({
                    'group': title,
                    'label': top['label'],
                    'rate': top['rate'],
                    'checked': top['checked'],
                })

        return {
            'total': total,
            'hits': hits,
            'misses': misses,
            'unknown': unknown,
            'rate': _pct(hits, total),
            'pending': pending,
            'calibration_error': calibration_error,
            'best': best,
            'probability': probability,
            'raw_probability': raw_probability,
            'scores': scores,
            'server_scores': server_scores,
            'sets': sets,
            'game_bands': game_bands,
            'servers': servers[:20],
            'data_quality': data_quality,
        }
    except sqlite3.Error as e:
        out = dict(empty)
        out['error'] = str(e)
        return out
    finally:
        if conn is not None:
            conn.close()


STATS_WIDGET = r'''
<style>
#modelStatsWrap{margin:10px 0 14px;padding:12px;border:1px solid #2b3645;border-radius:12px;background:#111820;color:#e7edf5;font-family:Arial,sans-serif}
#modelStatsBar{display:flex;gap:10px;flex-wrap:wrap;align-items:stretch}
.ms-card{background:#17212b;border:1px solid #2a3948;border-radius:9px;padding:8px 11px;min-width:108px}
.ms-label{font-size:11px;color:#8fa2b5;margin-bottom:3px}.ms-value{font-size:18px;font-weight:700}
.ms-hit .ms-value{color:#33d17a}.ms-miss .ms-value{color:#ff6b6b}.ms-rate .ms-value{color:#67b7ff}.ms-pending .ms-value{color:#f7c948}.ms-unknown .ms-value{color:#b7bec8}
#modelStatsBtn{margin-left:auto;background:#23364a;color:#dbe9f7;border:1px solid #36516e;border-radius:8px;padding:8px 12px;cursor:pointer}
#bestConditions{margin-top:12px;display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:8px}
.best-card{background:#151f29;border:1px solid #2c3d4d;border-radius:9px;padding:9px 11px}.best-title{font-size:11px;color:#8fa2b5}.best-name{font-size:15px;font-weight:700;margin:3px 0}.best-meta{font-size:12px;color:#c8d3df}
#modelStatsDetails{display:none;margin-top:12px;border-top:1px solid #2b3645;padding-top:10px;font-size:12px}.ms-section{margin:12px 0}.ms-section h4{margin:0 0 6px;font-size:13px;color:#dbe9f7}
.ms-row{display:grid;grid-template-columns:minmax(100px,1.4fr) 85px 80px 90px 90px 85px;gap:8px;padding:5px 0;border-bottom:1px solid #1d2a36;align-items:center}
.ms-muted{color:#8fa2b5;font-size:12px;margin-top:7px}
@media(max-width:760px){.ms-card{min-width:92px}.ms-row{grid-template-columns:minmax(80px,1.3fr) 62px 62px 70px 70px 62px;font-size:10px}}
</style>
<div id="modelStatsWrap">
  <div style="margin-bottom:10px;padding:9px 11px;background:#13202b;border:1px solid #29445b;border-radius:9px">
    <b>V6 STRICT</b> · статистика считается с нуля только по V6.<br>
    Новый ставочный сигнал: только <b>первый 30:30 матча</b>, только <b>геймы 4–6</b>, и только если оценка V6 ≥ 50.5%.
    Старые V5 исходы используются только как обучающий prior и в эту проходимость не входят.
  </div>
  <div id="modelStatsBar">
    <div class="ms-card"><div class="ms-label">Проверено</div><div id="msTotal" class="ms-value">0</div></div>
    <div class="ms-card ms-hit"><div class="ms-label">✅ Сработало</div><div id="msHits" class="ms-value">0</div></div>
    <div class="ms-card ms-miss"><div class="ms-label">❌ Не сработало</div><div id="msMisses" class="ms-value">0</div></div>
    <div class="ms-card ms-unknown"><div class="ms-label">⚪ Не определено</div><div id="msUnknown" class="ms-value">0</div></div>
    <div class="ms-card ms-rate"><div class="ms-label">🎯 Проходимость</div><div id="msRate" class="ms-value">0.0%</div></div>
    <div class="ms-card ms-pending"><div class="ms-label">⏳ В ожидании</div><div id="msPending" class="ms-value">0</div></div>
    <div class="ms-card"><div class="ms-label">Среднее отклонение V6</div><div id="msCalibration" class="ms-value">—</div></div>
    <button id="modelStatsBtn" type="button">Подробная статистика</button>
  </div>
  <div id="bestConditions"></div>
  <div id="modelStatsDetails"></div>
</div>
<script>
(function(){
  const esc=(s)=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));
  function rowsBlock(title, rows){
    if(!rows || !rows.length) return '';
    return `<div class="ms-section"><h4>${esc(title)}</h4>`+
      '<div class="ms-row"><b>Условие</b><b>Проверено</b><b>Прошло</b><b>Факт</b><b>Модель</b><b>Неопр.</b></div>'+
      rows.map(x=>`<div class="ms-row"><span>${esc(x.label)}</span><span>${x.checked}</span><span>${x.hits}</span><span>${x.rate}%</span><span>${x.model_avg}%</span><span>${x.unknown}</span></div>`).join('')+
      '</div>';
  }
  async function loadModelStats(){
    try{
      const r=await fetch('/api/stats?_='+Date.now(),{cache:'no-store'});
      const s=await r.json();
      document.getElementById('msTotal').textContent=s.total ?? 0;
      document.getElementById('msHits').textContent=s.hits ?? 0;
      document.getElementById('msMisses').textContent=s.misses ?? 0;
      document.getElementById('msUnknown').textContent=s.unknown ?? 0;
      document.getElementById('msRate').textContent=(s.rate ?? 0).toFixed(1)+'%';
      document.getElementById('msPending').textContent=s.pending ?? 0;
      document.getElementById('msCalibration').textContent=s.calibration_error==null?'—':s.calibration_error.toFixed(1)+' п.п.';

      const best=document.getElementById('bestConditions');
      if((s.best||[]).length){
        best.innerHTML=(s.best||[]).map(x=>
          `<div class="best-card"><div class="best-title">V6 условие · ${esc(x.group)}</div><div class="best-name">${esc(x.label)}</div><div class="best-meta">Проходимость <b>${x.rate}%</b> · Проверено <b>${x.checked}</b> сигналов</div></div>`
        ).join('');
      }else{
        best.innerHTML='<div class="ms-muted">Лучшие условия появятся после того, как по сегменту накопится хотя бы 50 проверенных сигналов.</div>';
      }

      const d=document.getElementById('modelStatsDetails');
      d.innerHTML =
        rowsBlock('По диапазону выданной вероятности', s.probability) +
        rowsBlock('По сырому диапазону модели (новая калибровка)', s.raw_probability) +
        rowsBlock('По счёту относительно подающего (первое число — подающий)', s.server_scores) +
        rowsBlock('По обычному счёту в момент сигнала', s.scores) +
        rowsBlock('По сетам', s.sets) +
        rowsBlock('По номеру гейма', s.game_bands) +
        rowsBlock('По подающим (топ-20 по числу проверок)', s.servers) +
        rowsBlock('По объёму данных', s.data_quality);
      if(!d.innerHTML) d.innerHTML='<div>Пока нет завершённых сигналов.</div>';
    }catch(e){ console.log('stats error',e); }
  }
  document.getElementById('modelStatsBtn').addEventListener('click',()=>{
    const d=document.getElementById('modelStatsDetails');
    d.style.display=d.style.display==='block'?'none':'block';
  });
  loadModelStats();
  setInterval(loadModelStats,3000);
})();
</script>
'''


@app.route('/')
def index():
    html = render_template('index.html')
    if '</body>' in html:
        html = html.replace('</body>', STATS_WIDGET + '\n</body>', 1)
    else:
        html += STATS_WIDGET
    return html


@app.route('/api/matches')
def get_matches():
    if not os.path.exists(JSON_PATH):
        return jsonify([])
    try:
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data if isinstance(data, list) else [])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats')
def get_stats():
    return jsonify(read_stats())


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
