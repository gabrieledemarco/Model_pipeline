"""
15M signal resolution experiment.

Compares the strategy evaluated at 1H vs 15M base timeframe,
keeping everything else invariant:
  - Same composite formula (9 components, same weights)
  - Same ±3 threshold and session 08-21 filter
  - Same walk-forward scheme (6m train / 2m OOS / 23 windows)
  - Same ML gate (binary LightGBM, P≥0.50)

At 15M:
  - s_1h becomes an HTF signal (shifted +1H, forward-filled to 15M)
  - s_15m becomes the same-TF entry signal (no shift, computed directly)
  - Entry/exit uses 15M ATR → tighter stops, finer granularity
  - Up to 4× more signal bars per window

Strategies compared:
  A. Baseline 1H    : reference (no ML gate, 1H bars)
  B. ML Gate 1H     : binary P≥0.50, 1H bars (best config from previous analysis)
  C. Baseline 15M   : same composite, evaluated at 15M
  D. ML Gate 15M    : binary P≥0.50, 15M bars

Output → reports/report_15m.html
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.strategy.data_fetcher import fetch_extended_data, generate_oi, generate_funding
from src.strategy.indicators   import add_indicators
from src.strategy.signals      import build_signal_matrix, build_signal_matrix_15m
from src.strategy.optimizer    import apply_filters, ScenarioConfig
from src.strategy.engine       import run_backtest, INIT_CAP
from src.strategy.ml_features  import build_feature_matrix
from src.strategy.ml_gate      import (
    walk_forward_binary_gate,
    run_gated_backtest,
)

SESSION_CFG = ScenarioConfig(
    "Session 08-21",
    session_hours=(8, 21),
    long_threshold=3.0,
    short_threshold=-3.0,
)


# ─────────────────────────────────────────────────────────────────────────────

def kpi_row(name: str, bt: dict, extra: dict = None) -> dict:
    k = bt["kpis"]
    r = {
        "name":          name,
        "total_return":  round(k.get("total_return",  0) * 100, 2),
        "max_dd":        round(k.get("max_drawdown",  0) * 100, 2),
        "sharpe":        round(k.get("sharpe",        0), 3),
        "calmar":        round(k.get("calmar",        0), 3),
        "win_rate":      round(k.get("win_rate",      0) * 100, 1),
        "profit_factor": round(k.get("profit_factor", 0), 2),
        "n_trades":      k.get("n_trades", 0),
        "expectancy":    round(k.get("expectancy",    0), 0),
        "final_equity":  round(k.get("final_equity",  0), 0),
        "equity":        bt["equity"].tolist(),
        "drawdown":      bt["drawdown"].tolist(),
        "index":         [str(t) for t in bt["equity"].index],
    }
    if extra:
        r.update(extra)
    return r


# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n══ 15M Signal Resolution Experiment ═════════════════════════════")

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("\n[1/7] Loading data …")
    raw = fetch_extended_data(start_year=2022, start_month=1,
                              fetch_1m=False, fetch_flow=True)
    tf_ind = {}
    for tf in ["1W", "1D", "4H", "1H", "15M"]:
        df = raw.get(tf, pd.DataFrame())
        tf_ind[tf] = add_indicators(df) if not df.empty and len(df) > 20 else df

    df_1h  = tf_ind["1H"]
    df_15m = tf_ind["15M"]
    oi_df  = generate_oi(tf_ind["1D"]["close"])
    funding = generate_funding(tf_ind["1D"]["close"])

    print(f"  1H bars : {len(df_1h):,}   |   15M bars: {len(df_15m):,}")

    # ── 2. Signals at both resolutions ────────────────────────────────────────
    print("\n[2/7] Building signals …")
    raw_sig_1h  = build_signal_matrix(
        tf_data=tf_ind, oi_df=oi_df, funding=funding,
        premium_1h=None, df_15m=df_15m, df_1m=None,
    )
    raw_sig_15m = build_signal_matrix_15m(
        tf_data=tf_ind, oi_df=oi_df, funding=funding, premium_1h=None,
    )
    sig_1h  = apply_filters(raw_sig_1h,  SESSION_CFG)
    sig_15m = apply_filters(raw_sig_15m, SESSION_CFG)

    n_1h  = int((sig_1h["signal"]  != 0).sum())
    n_15m = int((sig_15m["signal"] != 0).sum())
    print(f"  1H  active signal bars: {n_1h:,}")
    print(f"  15M active signal bars: {n_15m:,}")

    # ── 3. A. Baseline 1H ─────────────────────────────────────────────────────
    print("\n[3/7] A. Baseline 1H …")
    bt_a = run_backtest(df_1h, sig_1h)
    row_a = kpi_row("A. Baseline 1H", bt_a,
                    extra={"n_sig": n_1h, "base_tf": "1H"})
    print(f"  → Return={row_a['total_return']:+.1f}%  DD={row_a['max_dd']:.1f}%  "
          f"Trades={row_a['n_trades']}  Calmar={row_a['calmar']:.3f}")

    # ── 4. B. ML Gate 1H P≥0.50 ──────────────────────────────────────────────
    print("\n[4/7] B. ML Gate 1H P≥0.50 …")
    feat_1h = build_feature_matrix(tf_ind, sig_1h, base_tf="1H")
    print(f"  Feature matrix: {feat_1h.shape[0]:,} × {feat_1h.shape[1]}")
    gate_1h = walk_forward_binary_gate(
        df_1h, sig_1h, feat_1h, gate_threshold=0.50,
        use_feat_sel=True, verbose=True,
    )
    bt_b = run_gated_backtest(df_1h, sig_1h, gate_1h)
    n_b  = int((gate_1h.gated_signal != 0).sum())
    row_b = kpi_row("B. ML Gate 1H P≥0.50", bt_b,
                    extra={"n_sig": n_b, "filter_rate": round((1 - n_b / n_1h) * 100, 1),
                           "base_tf": "1H",
                           "window_stats": gate_1h.window_stats,
                           "importances": gate_1h.importances.head(15).to_dict("records")
                                          if gate_1h.importances is not None else []})
    print(f"  → Return={row_b['total_return']:+.1f}%  DD={row_b['max_dd']:.1f}%  "
          f"Trades={row_b['n_trades']}  Calmar={row_b['calmar']:.3f}  "
          f"Filtered={row_b['filter_rate']:.0f}%")

    # ── 5. C. Baseline 15M ───────────────────────────────────────────────────
    print("\n[5/7] C. Baseline 15M …")
    bt_c = run_backtest(df_15m, sig_15m)
    row_c = kpi_row("C. Baseline 15M", bt_c,
                    extra={"n_sig": n_15m, "base_tf": "15M"})
    print(f"  → Return={row_c['total_return']:+.1f}%  DD={row_c['max_dd']:.1f}%  "
          f"Trades={row_c['n_trades']}  Calmar={row_c['calmar']:.3f}")

    # ── 6. D. ML Gate 15M P≥0.50 ─────────────────────────────────────────────
    print("\n[6/7] D. ML Gate 15M P≥0.50 …")
    feat_15m = build_feature_matrix(tf_ind, sig_15m, base_tf="15M")
    print(f"  Feature matrix: {feat_15m.shape[0]:,} × {feat_15m.shape[1]}")
    # walk_forward_binary_gate accepts any OHLCV DataFrame as first arg
    gate_15m = walk_forward_binary_gate(
        df_15m, sig_15m, feat_15m, gate_threshold=0.50,
        use_feat_sel=True, verbose=True,
    )
    bt_d = run_gated_backtest(df_15m, sig_15m, gate_15m)
    n_d  = int((gate_15m.gated_signal != 0).sum())
    row_d = kpi_row("D. ML Gate 15M P≥0.50", bt_d,
                    extra={"n_sig": n_d, "filter_rate": round((1 - n_d / n_15m) * 100, 1),
                           "base_tf": "15M",
                           "window_stats": gate_15m.window_stats,
                           "importances": gate_15m.importances.head(15).to_dict("records")
                                          if gate_15m.importances is not None else []})
    print(f"  → Return={row_d['total_return']:+.1f}%  DD={row_d['max_dd']:.1f}%  "
          f"Trades={row_d['n_trades']}  Calmar={row_d['calmar']:.3f}  "
          f"Filtered={row_d['filter_rate']:.0f}%")

    results = [row_a, row_b, row_c, row_d]

    # ── 7. HTML ───────────────────────────────────────────────────────────────
    print("\n[7/7] Generating report …")
    html = _build_html(results)
    out  = Path("reports/report_15m.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"  Saved → {out}")
    print("\n══ Done ══════════════════════════════════════════════════════════")


# ─────────────────────────────────────────────────────────────────────────────
# HTML builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_html(results: list) -> str:

    COLORS = {
        "A. Baseline 1H":        "#8b949e",
        "B. ML Gate 1H P≥0.50":  "#81c784",
        "C. Baseline 15M":       "#64b5f6",
        "D. ML Gate 15M P≥0.50": "#ffb74d",
    }

    def _cc(v, good_high=True):
        cls = ("pos" if v > 0 else "neg") if good_high else ("neg" if v < 0 else "pos")
        return f'<td class="{cls}">{v}</td>'

    def _summary_rows():
        rows = ""
        for r in results:
            color = COLORS.get(r["name"], "#aaa")
            rows += f"""
            <tr>
              <td><span class="dot" style="background:{color}"></span>{r['name']}</td>
              <td>{r['n_trades']}</td>
              {_cc(r['total_return'])}
              {_cc(r['max_dd'], False)}
              {_cc(r['calmar'])}
              {_cc(r['sharpe'])}
              {_cc(r['win_rate'])}
              {_cc(r['profit_factor'])}
              <td>{r.get('filter_rate', '—')}</td>
            </tr>"""
        return rows

    data = {
        "results": [{
            "name":    r["name"],
            "equity":  r["equity"],
            "dd":      r["drawdown"],
            "index":   r["index"],
            "color":   COLORS.get(r["name"], "#aaa"),
        } for r in results],
        "window_stats": {
            r["name"]: r.get("window_stats", [])
            for r in results if r.get("window_stats")
        },
        "importances": {
            r["name"]: r.get("importances", [])
            for r in results if r.get("importances")
        },
    }
    data_json = json.dumps(data, default=str)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>15M Resolution Experiment — BTCUSDT</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0d1117; color: #c9d1d9; font-family: 'Segoe UI', sans-serif; padding: 24px; }}
  h1  {{ color: #58a6ff; margin-bottom: 6px; font-size: 1.6rem; }}
  h2  {{ color: #8b949e; font-size: 1.05rem; margin: 28px 0 10px; border-bottom: 1px solid #21262d; padding-bottom: 6px; }}
  .subtitle {{ color: #8b949e; font-size: 0.88rem; margin-bottom: 24px; }}
  .card {{ background: #161b22; border: 1px solid #21262d; border-radius: 10px; padding: 20px; margin-bottom: 20px; }}
  canvas {{ max-height: 340px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
  th  {{ background: #21262d; color: #8b949e; padding: 8px 10px; text-align: right; font-weight: 600; }}
  th:first-child {{ text-align: left; }}
  td  {{ padding: 7px 10px; text-align: right; border-bottom: 1px solid #21262d; }}
  td:first-child {{ text-align: left; }}
  tr:hover td {{ background: #1c2128; }}
  .pos {{ color: #3fb950; }} .neg {{ color: #f85149; }}
  .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .note {{ font-size: 0.78rem; color: #6e7681; margin-bottom: 10px; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 600; margin-left: 6px; }}
  .b1h  {{ background: #1f3a1f; color: #81c784; }}
  .b15m {{ background: #1a2d3d; color: #64b5f6; }}
</style>
</head>
<body>

<h1>15M Signal Resolution Experiment — BTCUSDT Perpetual Futures</h1>
<p class="subtitle">
  Composite ±3, Session 08–21 UTC &nbsp;|&nbsp;
  Walk-forward 6m/2m, 23 windows (2022–2026) &nbsp;|&nbsp;
  <span class="badge b1h">1H</span> same composite evaluated at 1H granularity &nbsp;|&nbsp;
  <span class="badge b15m">15M</span> s_1h becomes HTF (+1H shift), s_15m becomes same-TF
</p>

<script>
const DATA = {data_json};
</script>

<!-- ── Summary table ──────────────────────────────────────────────────────── -->
<h2>1. Performance Summary</h2>
<div class="card">
<table>
  <thead>
    <tr>
      <th>Strategy</th><th>Trades</th><th>Return%</th><th>Max DD%</th>
      <th>Calmar</th><th>Sharpe</th><th>Win%</th><th>Profit Factor</th>
      <th>Filtered%</th>
    </tr>
  </thead>
  <tbody>{_summary_rows()}</tbody>
</table>
</div>

<!-- ── Equity curves ──────────────────────────────────────────────────────── -->
<h2>2. Equity Curves</h2>
<div class="card"><canvas id="equity_chart"></canvas></div>

<!-- ── Drawdown ───────────────────────────────────────────────────────────── -->
<h2>3. Drawdown</h2>
<div class="card"><canvas id="dd_chart"></canvas></div>

<!-- ── Walk-forward window stats ─────────────────────────────────────────── -->
<h2>4. Walk-Forward Window Stats</h2>
<div class="grid-2">
  <div class="card" id="wf_1h">
    <p style="font-weight:600;color:#81c784;margin-bottom:10px">B. ML Gate 1H — per-window val_acc</p>
    <canvas id="wf_chart_1h"></canvas>
  </div>
  <div class="card" id="wf_15m">
    <p style="font-weight:600;color:#ffb74d;margin-bottom:10px">D. ML Gate 15M — per-window val_acc</p>
    <canvas id="wf_chart_15m"></canvas>
  </div>
</div>

<!-- ── Feature importance ─────────────────────────────────────────────────── -->
<h2>5. Top Feature Importances</h2>
<div class="grid-2">
  <div class="card">
    <p style="font-weight:600;color:#81c784;margin-bottom:10px">B. ML Gate 1H</p>
    <canvas id="imp_chart_1h" style="max-height:280px"></canvas>
  </div>
  <div class="card">
    <p style="font-weight:600;color:#ffb74d;margin-bottom:10px">D. ML Gate 15M</p>
    <canvas id="imp_chart_15m" style="max-height:280px"></canvas>
  </div>
</div>

<script>
// ── Shared x-axis (use 1H equity index as reference for display) ────────────
const refResult = DATA.results.find(r => r.name.includes("Baseline 1H")) || DATA.results[0];
const labels = refResult.index.map((s, i) => i % Math.floor(refResult.index.length / 12) === 0 ? s.slice(0, 10) : '');

// ── Equity chart ─────────────────────────────────────────────────────────────
new Chart(document.getElementById('equity_chart').getContext('2d'), {{
  type: 'line',
  data: {{
    labels: refResult.index,
    datasets: DATA.results.map(r => ({{
      label: r.name,
      data:  r.equity,
      borderColor: r.color,
      borderWidth: 1.8,
      pointRadius: 0,
      fill: false,
    }})),
  }},
  options: {{
    responsive: true, animation: false,
    plugins: {{ legend: {{ labels: {{ color: '#8b949e' }} }} }},
    scales: {{
      x: {{ ticks: {{ color: '#8b949e', maxTicksLimit: 12 }}, grid: {{ color: '#21262d' }} }},
      y: {{ ticks: {{ color: '#8b949e', callback: v => '$' + v.toLocaleString() }}, grid: {{ color: '#21262d' }} }},
    }},
  }},
}});

// ── Drawdown chart ───────────────────────────────────────────────────────────
new Chart(document.getElementById('dd_chart').getContext('2d'), {{
  type: 'line',
  data: {{
    labels: refResult.index,
    datasets: DATA.results.map(r => ({{
      label: r.name,
      data:  r.dd.map(v => v * 100),
      borderColor: r.color,
      borderWidth: 1.5,
      pointRadius: 0,
      fill: false,
    }})),
  }},
  options: {{
    responsive: true, animation: false,
    plugins: {{ legend: {{ labels: {{ color: '#8b949e' }} }} }},
    scales: {{
      x: {{ ticks: {{ color: '#8b949e', maxTicksLimit: 12 }}, grid: {{ color: '#21262d' }} }},
      y: {{ ticks: {{ color: '#8b949e', callback: v => v.toFixed(0) + '%' }}, grid: {{ color: '#21262d' }} }},
    }},
  }},
}});

// ── Walk-forward val_acc charts ──────────────────────────────────────────────
function makeWfChart(canvasId, name, color) {{
  const ws = DATA.window_stats[name];
  if (!ws || !ws.length) return;
  const labels = ws.map(w => w.oos_start ? w.oos_start.slice(0, 7) : 'W' + w.window);
  const vals   = ws.map(w => w.val_acc);
  const ntrades= ws.map(w => w.n_train);
  new Chart(document.getElementById(canvasId).getContext('2d'), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [
        {{ label: 'val_acc %', data: vals, backgroundColor: color + 'cc', yAxisID: 'y' }},
      ],
    }},
    options: {{
      responsive: true, animation: false,
      plugins: {{ legend: {{ labels: {{ color: '#8b949e' }} }} }},
      scales: {{
        x: {{ ticks: {{ color: '#8b949e', maxRotation: 45 }}, grid: {{ display: false }} }},
        y: {{ min: 30, max: 80, ticks: {{ color: '#8b949e', callback: v => v + '%' }}, grid: {{ color: '#21262d' }} }},
      }},
    }},
  }});
}}
makeWfChart('wf_chart_1h',  'B. ML Gate 1H P≥00.50',  '#81c784');
makeWfChart('wf_chart_15m', 'D. ML Gate 15M P≥00.50', '#ffb74d');

// ── Feature importance charts ────────────────────────────────────────────────
function makeImpChart(canvasId, name, color) {{
  const imp = DATA.importances[name];
  if (!imp || !imp.length) return;
  const feats = imp.map(r => r.feature);
  const vals  = imp.map(r => r.importance);
  new Chart(document.getElementById(canvasId).getContext('2d'), {{
    type: 'bar',
    data: {{
      labels: feats,
      datasets: [{{ data: vals, backgroundColor: color + 'cc', borderWidth: 0 }}],
    }},
    options: {{
      indexAxis: 'y', responsive: true, animation: false,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }},
        y: {{ ticks: {{ color: '#8b949e', font: {{ size: 10 }} }}, grid: {{ display: false }} }},
      }},
    }},
  }});
}}
makeImpChart('imp_chart_1h',  'B. ML Gate 1H P≥00.50',  '#81c784');
makeImpChart('imp_chart_15m', 'D. ML Gate 15M P≥00.50', '#ffb74d');
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
