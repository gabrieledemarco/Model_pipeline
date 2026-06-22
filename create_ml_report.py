"""
ML gate experiment: train a LightGBM classifier to filter composite signals
based on predicted trade profitability.

Pipeline
────────
1. Load data (from cache) + build indicators
2. Build bias-free signal matrix (Session 08-21 filter)
3. Build feature matrix (ml_features.build_feature_matrix)
4. Walk-forward ML gate (ml_gate.walk_forward_ml_gate)
   - 6m train / 2m OOS / 23 windows
   - Label: did this trade make money?
   - Predict: P(profitable | features at entry bar)
5. Compare three equity curves:
   a. Baseline: composite ±5 + session filter
   b. ML-gated (thr=0.55): composite signal allowed only if ML agrees
   c. ML-gated (thr=0.60): stricter gate
6. Generate HTML report → reports/ml_report.html
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.strategy.data_fetcher   import fetch_extended_data, generate_oi, generate_funding
from src.strategy.indicators     import add_indicators
from src.strategy.signals        import build_signal_matrix
from src.strategy.optimizer      import apply_filters, ScenarioConfig
from src.strategy.engine         import run_backtest, INIT_CAP
from src.strategy.ml_features    import build_feature_matrix
from src.strategy.ml_gate        import walk_forward_ml_gate, run_gated_backtest

SESSION_CFG = ScenarioConfig("Session 08-21", session_hours=(8, 21), long_threshold=3.0, short_threshold=-3.0)


def kpi_row(name: str, bt: dict) -> dict:
    k = bt["kpis"]
    return {
        "name":          name,
        "total_return":  round(k.get("total_return", 0) * 100, 2),
        "max_dd":        round(k.get("max_drawdown", 0) * 100, 2),
        "sharpe":        round(k.get("sharpe", 0), 3),
        "win_rate":      round(k.get("win_rate", 0) * 100, 1),
        "profit_factor": round(k.get("profit_factor", 0), 2),
        "n_trades":      k.get("n_trades", 0),
        "expectancy":    round(k.get("expectancy", 0), 0),
        "final_equity":  round(k.get("final_equity", 0), 0),
        "equity":        bt["equity"].tolist(),
        "drawdown":      bt["drawdown"].tolist(),
        "index":         [str(t) for t in bt["equity"].index],
    }


def main():
    print("\n══ ML Gate Report ═════════════════════════════════════════════════")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/5] Loading data …")
    raw = fetch_extended_data(start_year=2022, start_month=1, fetch_1m=False)
    tf_ind = {}
    for tf in ["1W", "1D", "4H", "1H", "15M"]:
        df = raw.get(tf, pd.DataFrame())
        tf_ind[tf] = add_indicators(df) if not df.empty and len(df) > 20 else df

    df_1h   = tf_ind["1H"]
    df_15m  = tf_ind["15M"]
    oi_df   = generate_oi(tf_ind["1D"]["close"])   if not tf_ind["1D"].empty else pd.DataFrame()
    funding = generate_funding(tf_ind["1D"]["close"]) if not tf_ind["1D"].empty else pd.Series(dtype=float)

    # ── 2. Build baseline signals (threshold ±3, session 08-21) ──────────────
    print("\n[2/5] Building signal matrix …")
    raw_signals = build_signal_matrix(
        tf_data    = tf_ind,
        oi_df      = oi_df,
        funding    = funding,
        premium_1h = None,
        df_15m     = df_15m,
        df_1m      = None,
    )
    signals = apply_filters(raw_signals, SESSION_CFG)

    n_sig = int((signals["signal"] != 0).sum())
    print(f"  Signals: {n_sig} bars  "
          f"(long={int((signals['signal']==1).sum())}  "
          f"short={int((signals['signal']==-1).sum())})")

    # ── 3. Baseline backtest ──────────────────────────────────────────────────
    print("\n[3/5] Baseline backtest …")
    base_bt  = run_backtest(df_1h, signals)
    baseline = kpi_row("Baseline (composite ±3)", base_bt)
    print(f"  Return={baseline['total_return']:.1f}%  "
          f"DD={baseline['max_dd']:.1f}%  "
          f"Win={baseline['win_rate']:.0f}%  "
          f"Trades={baseline['n_trades']}")

    # ── 4. Feature matrix ─────────────────────────────────────────────────────
    print("\n[4/5] Building feature matrix …")
    feat_df = build_feature_matrix(tf_ind, signals)
    print(f"  Features: {len(feat_df.columns)} columns × {len(feat_df)} rows")

    # ── 5. Walk-forward ML gate ───────────────────────────────────────────────
    print("\n[5/5] Walk-forward ML gate training …")
    gate_results = {}

    for thr in [0.50, 0.55, 0.60]:
        print(f"\n  Threshold P≥{thr:.2f}:")
        gate = walk_forward_ml_gate(
            df_1h, signals, feat_df,
            gate_threshold=thr, verbose=True,
        )
        gated_bt = run_gated_backtest(df_1h, signals, gate)
        r = kpi_row(f"ML Gate P≥{thr:.2f}", gated_bt)
        n_allowed = int((gate.gated_signal != 0).sum())
        filter_rate = (1 - n_allowed / n_sig) * 100 if n_sig else 0
        r["n_allowed"]   = n_allowed
        r["filter_rate"] = round(filter_rate, 1)
        r["window_stats"] = gate.window_stats
        r["importances"]  = gate.importances.head(20).to_dict("records") if gate.importances is not None else []
        gate_results[f"{thr:.2f}"] = r
        print(f"  → Return={r['total_return']:.1f}%  DD={r['max_dd']:.1f}%  "
              f"Win={r['win_rate']:.0f}%  Trades={r['n_trades']}  "
              f"Filtered={filter_rate:.0f}%")

    # ── Generate report ───────────────────────────────────────────────────────
    print("\nGenerating HTML report …")
    html = _build_html(baseline, gate_results)
    out  = Path("reports/ml_report.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"  Saved → {out}")
    print("\n══ Done ══════════════════════════════════════════════════════════")


# ─────────────────────────────────────────────────────────────────────────────
# HTML report
# ─────────────────────────────────────────────────────────────────────────────

def _color(val, metric):
    if metric in ("total_return", "win_rate", "profit_factor", "expectancy"):
        return "pos" if val > 0 else "neg"
    if metric == "max_dd":
        return "pos" if val > -15 else "neg"
    return ""


def _build_html(baseline: dict, gate_results: dict) -> str:
    colours = {"0.50": "#FF9800", "0.55": "#2196F3", "0.60": "#4CAF50"}
    thr_keys = list(gate_results.keys())

    # ── Summary metrics ───────────────────────────────────────────────────────
    best_key = max(thr_keys, key=lambda k: gate_results[k]["total_return"])
    best = gate_results[best_key]

    # Downsampled index for charts
    idx = baseline["index"]
    idx_series = pd.DatetimeIndex(idx)
    step = max(1, len(idx) // 800)
    chart_idx = json.dumps([str(idx_series[i].date()) for i in range(0, len(idx), step)])

    def ds(r, col="equity"):
        return json.dumps([round(r[col][j], 2) for j in range(0, len(r[col]), step)])

    def imp_chart_data(imp_list, n=15):
        top = imp_list[:n]
        labels = json.dumps([x["feature"] for x in top])
        vals   = json.dumps([round(x["importance"], 1) for x in top])
        return labels, vals

    # Feature importance for first gate threshold
    first_key = thr_keys[1] if len(thr_keys) > 1 else thr_keys[0]
    imp_labels, imp_vals = imp_chart_data(gate_results[first_key]["importances"])

    # Window stats table for first gate
    win_rows = ""
    for w in gate_results[first_key]["window_stats"]:
        pct = round(w["n_gated"] / w["n_sig_oos"] * 100, 0) if w["n_sig_oos"] else 0
        win_rows += (
            f"<tr><td>{w['window']}</td>"
            f"<td>{w['oos_start']}</td><td>{w['oos_end']}</td>"
            f"<td>{w['n_train_tr']}</td>"
            f"<td class='pos'>{w['n_pos_tr']}</td>"
            f"<td class='neg'>{w['n_neg_tr']}</td>"
            f"<td>{w['train_acc']}%</td>"
            f"<td>{w['n_sig_oos']}</td>"
            f"<td>{w['n_gated']} ({pct:.0f}%)</td></tr>\n"
        )

    # Main comparison table
    def cmp_row(r):
        cells = ""
        for col, sfx in [("total_return","%"),("max_dd","%"),("sharpe",""),
                          ("win_rate","%"),("profit_factor",""),("n_trades",""),("expectancy","$")]:
            v = r[col]
            c = _color(v, col)
            cells += f"<td class='{c}'>{v}{sfx}</td>"
        extra = f"<td>{r.get('n_allowed','—')}</td><td>{r.get('filter_rate','—')}%</td>"
        return f"<tr><td>{r['name']}</td>{cells}{extra}</tr>\n"

    cmp_table = cmp_row(baseline)
    for k, r in gate_results.items():
        cmp_table += cmp_row(r)

    # Equity datasets
    eq_ds = f"""{{
  label: "Baseline",
  data: {ds(baseline)},
  borderColor: "#8b949e",
  borderWidth: 2,
  borderDash: [6,3],
  pointRadius: 0,
  fill: false,
  tension: 0.1,
}}"""
    for k, r in gate_results.items():
        col = colours.get(k, "#ffffff")
        eq_ds += f""",
{{
  label: "{r['name']}",
  data: {ds(r)},
  borderColor: "{col}",
  borderWidth: 2,
  pointRadius: 0,
  fill: false,
  tension: 0.1,
}}"""

    dd_ds = f"""{{
  label: "Baseline DD",
  data: {ds(baseline, "drawdown")},
  borderColor: "#8b949e",
  borderWidth: 1.5,
  borderDash: [6,3],
  pointRadius: 0,
  fill: false,
  tension: 0.1,
}}"""
    for k, r in gate_results.items():
        col = colours.get(k, "#ffffff")
        dd_ds += f""",
{{
  label: "{r['name']} DD",
  data: {ds(r, "drawdown")},
  borderColor: "{col}",
  borderWidth: 1.5,
  pointRadius: 0,
  fill: "-1",
  backgroundColor: "{col}11",
  tension: 0.1,
}}"""

    base_ret = baseline["total_return"]
    best_ret  = best["total_return"]
    best_filt = best.get("filter_rate", 0)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ML Gate Report — BTCUSDT Strategy</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{font-family:'Segoe UI',Arial,sans-serif;background:#0d1117;color:#c9d1d9;font-size:14px;}}
h1{{text-align:center;padding:24px;font-size:22px;color:#58a6ff;border-bottom:1px solid #30363d;}}
h2{{color:#79c0ff;font-size:16px;margin:20px 0 10px;padding-left:4px;border-left:3px solid #388bfd;}}
h3{{color:#8b949e;font-size:13px;margin:10px 0 6px;}}
.section{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:20px;margin:16px;}}
.highlight-box{{background:#1c2333;border:1px solid #388bfd;border-radius:6px;padding:12px 16px;margin-bottom:14px;}}
.highlight-box p{{line-height:1.8;color:#c9d1d9;}}
.highlight-box strong{{color:#58a6ff;}}
table{{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px;}}
th{{background:#1c2333;color:#8b949e;padding:8px 10px;text-align:right;font-weight:600;position:sticky;top:0;}}
th:first-child{{text-align:left;}}
td{{padding:7px 10px;border-bottom:1px solid #21262d;text-align:right;}}
td:first-child{{text-align:left;color:#c9d1d9;}}
tr:hover td{{background:#1c2333;}}
.pos{{color:#3fb950;}} .neg{{color:#f85149;}}
.chart-container{{position:relative;height:340px;margin:10px 0;}}
.chart-sm{{position:relative;height:260px;margin:6px 0;}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px;}}
.stat-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:12px 0;}}
.stat-card{{background:#1c2333;border:1px solid #30363d;border-radius:6px;padding:12px;text-align:center;}}
.stat-card .val{{font-size:20px;font-weight:700;margin:4px 0;}}
.stat-card .lbl{{font-size:11px;color:#8b949e;}}
.badge{{display:inline-block;background:#1c2333;border:1px solid #388bfd;color:#58a6ff;
        border-radius:4px;padding:2px 8px;font-size:11px;margin-right:6px;}}
.warn{{color:#e3b341;font-size:12px;margin-top:8px;}}
</style>
</head>
<body>
<h1>BTCUSDT — ML Gate Report</h1>

<!-- Summary -->
<div class="section">
<h2>Overview</h2>
<div class="highlight-box">
<p>
LightGBM gate trained to predict trade profitability from multi-timeframe features.
Session 08-21 UTC. Composite threshold <strong>±3</strong> (best from threshold sweep).
Walk-forward: <strong>{TRAIN_MONTHS}m train / {OOS_MONTHS}m OOS</strong>,
sliding by {OOS_MONTHS}m → {len(gate_results[first_key]['window_stats'])} windows.
<br>
Baseline return: <strong class="{'pos' if base_ret>0 else 'neg'}">{base_ret:+.1f}%</strong>.
Best ML gate (P≥{best_key}): <strong class="{'pos' if best_ret>0 else 'neg'}">{best_ret:+.1f}%</strong>
filtering out <strong>{best_filt:.0f}%</strong> of signals.
</p>
<p class="warn">
⚠ Gate only acts on OOS windows. Bars without ML coverage retain composite signal (first {TRAIN_MONTHS} months bootstrapping cost).
</p>
</div>

<div class="stat-grid">
  <div class="stat-card">
    <div class="lbl">Baseline Return</div>
    <div class="val {'pos' if base_ret>0 else 'neg'}">{base_ret:+.1f}%</div>
    <div class="lbl">composite ±3</div>
  </div>
  <div class="stat-card">
    <div class="lbl">Best Gate Return</div>
    <div class="val {'pos' if best_ret>0 else 'neg'}">{best_ret:+.1f}%</div>
    <div class="lbl">P≥{best_key}</div>
  </div>
  <div class="stat-card">
    <div class="lbl">Best Gate DD</div>
    <div class="val {'pos' if best['max_dd']>-20 else 'neg'}">{best['max_dd']:.1f}%</div>
    <div class="lbl">P≥{best_key}</div>
  </div>
  <div class="stat-card">
    <div class="lbl">Best Win Rate</div>
    <div class="val {'pos' if best['win_rate']>50 else 'neg'}">{best['win_rate']:.0f}%</div>
    <div class="lbl">P≥{best_key}</div>
  </div>
</div>
</div>

<!-- Equity curves -->
<div class="section">
<h2>Equity Curves — Baseline vs ML Gate</h2>
<div class="chart-container"><canvas id="c_equity"></canvas></div>
<h2>Drawdown</h2>
<div class="chart-container" style="height:200px"><canvas id="c_dd"></canvas></div>
</div>

<!-- Comparison table -->
<div class="section">
<h2>Performance Comparison</h2>
<div style="overflow-x:auto"><table>
<tr><th>Strategy</th><th>Return%</th><th>Max DD%</th><th>Sharpe</th>
    <th>Win%</th><th>PF</th><th>Trades</th><th>Expect($)</th>
    <th># Allowed</th><th>Filter%</th></tr>
{cmp_table}</table></div>
</div>

<!-- Feature importance -->
<div class="section">
<div class="grid2">
  <div>
    <h2><span class="badge">ML</span>Top 15 Features (avg across windows)</h2>
    <div class="chart-sm"><canvas id="c_imp"></canvas></div>
  </div>
  <div>
    <h2><span class="badge">ML</span>Walk-forward Window Stats (P≥{first_key})</h2>
    <div style="overflow-x:auto;max-height:320px;"><table>
    <tr><th>#</th><th>OOS Start</th><th>OOS End</th>
        <th>Train</th><th class="pos">+</th><th class="neg">-</th>
        <th>TrainAcc</th><th>Signals</th><th>Gated</th></tr>
    {win_rows}
    </table></div>
  </div>
</div>
</div>

<script>
const IDX = {chart_idx};
const OPT = {{
  responsive:true, maintainAspectRatio:false, animation:{{duration:0}},
  plugins:{{legend:{{labels:{{color:'#8b949e',font:{{size:11}}}}}}}},
  scales:{{
    x:{{ticks:{{color:'#8b949e',maxTicksLimit:12,font:{{size:10}}}},grid:{{color:'#21262d'}}}},
    y:{{ticks:{{color:'#8b949e',font:{{size:10}},callback:v=>v.toLocaleString()}},grid:{{color:'#21262d'}}}}
  }}
}};

new Chart(document.getElementById('c_equity'),{{
  type:'line', data:{{labels:IDX, datasets:[{eq_ds}]}}, options:OPT
}});

new Chart(document.getElementById('c_dd'),{{
  type:'line', data:{{labels:IDX, datasets:[{dd_ds}]}},
  options:{{...OPT, scales:{{...OPT.scales,
    y:{{...OPT.scales.y, ticks:{{...OPT.scales.y.ticks, callback:v=>(v*100).toFixed(1)+'%'}}}}
  }}}}
}});

new Chart(document.getElementById('c_imp'),{{
  type:'bar',
  data:{{labels:{imp_labels}, datasets:[{{
    label:'Avg importance', data:{imp_vals},
    backgroundColor:'#388bfd88', borderColor:'#388bfd', borderWidth:1,
  }}]}},
  options:{{
    responsive:true, maintainAspectRatio:false, animation:{{duration:0}},
    indexAxis:'y',
    plugins:{{legend:{{display:false}}}},
    scales:{{
      x:{{ticks:{{color:'#8b949e',font:{{size:10}}}},grid:{{color:'#21262d'}}}},
      y:{{ticks:{{color:'#8b949e',font:{{size:10}}}},grid:{{color:'#21262d'}}}}
    }}
  }}
}});
</script>
</body>
</html>"""


# Import constant for HTML
from src.strategy.ml_gate import TRAIN_MONTHS, OOS_MONTHS

if __name__ == "__main__":
    main()
