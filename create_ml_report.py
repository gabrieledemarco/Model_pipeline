"""
ML gate report — v2 (trade-level) + v3 (bar-level).

Strategies compared (composite ±3, session 08-21):
  A. Baseline            : no ML gate
  B. Binary P≥0.50 (v2) : trade-level, early stopping + feat selection
  C. Binary P≥0.55 (v2) : trade-level, early stopping + feat selection
  D. Regression ret≥0%   : trade-level, predict trade return %
  E. Bar-level P≥0.55    : ALL 1H bars as training (~6k vs ~200 trades)
  F. Bar-level P≥0.60    : more selective bar-level gate

Walk-forward: 6m train / 2m OOS / 23 windows.
Output → reports/ml_report.html
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
from src.strategy.ml_gate        import (
    walk_forward_binary_gate,
    walk_forward_regression_gate,
    walk_forward_bar_level_gate,
    run_gated_backtest,
    TRAIN_MONTHS, OOS_MONTHS,
)

SESSION_CFG = ScenarioConfig(
    "Session 08-21",
    session_hours=(8, 21),
    long_threshold=3.0,
    short_threshold=-3.0,
)


def kpi_row(name: str, bt: dict, extra: dict = None) -> dict:
    k = bt["kpis"]
    r = {
        "name":          name,
        "total_return":  round(k.get("total_return", 0) * 100, 2),
        "max_dd":        round(k.get("max_drawdown", 0) * 100, 2),
        "sharpe":        round(k.get("sharpe", 0), 3),
        "calmar":        round(k.get("calmar", 0), 3),
        "win_rate":      round(k.get("win_rate", 0) * 100, 1),
        "profit_factor": round(k.get("profit_factor", 0), 2),
        "n_trades":      k.get("n_trades", 0),
        "expectancy":    round(k.get("expectancy", 0), 0),
        "final_equity":  round(k.get("final_equity", 0), 0),
        "equity":        bt["equity"].tolist(),
        "drawdown":      bt["drawdown"].tolist(),
        "index":         [str(t) for t in bt["equity"].index],
    }
    if extra:
        r.update(extra)
    return r


# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n══ ML Gate Report (v2 trade-level + v3 bar-level) ════════════════")

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("\n[1/7] Loading data …")
    raw = fetch_extended_data(start_year=2022, start_month=1, fetch_1m=False, fetch_flow=True)
    tf_ind = {}
    for tf in ["1W", "1D", "4H", "1H", "15M"]:
        df = raw.get(tf, pd.DataFrame())
        tf_ind[tf] = add_indicators(df) if not df.empty and len(df) > 20 else df

    df_1h   = tf_ind["1H"]
    df_15m  = tf_ind["15M"]
    oi_df   = generate_oi(tf_ind["1D"]["close"])
    funding = generate_funding(tf_ind["1D"]["close"])

    # ── 2. Signals ────────────────────────────────────────────────────────────
    print("\n[2/7] Building signals (±3, session 08-21) …")
    raw_sig = build_signal_matrix(
        tf_data=tf_ind, oi_df=oi_df, funding=funding,
        premium_1h=None, df_15m=df_15m, df_1m=None,
    )
    signals = apply_filters(raw_sig, SESSION_CFG)
    n_sig = int((signals["signal"] != 0).sum())
    print(f"  Active signal bars: {n_sig:,}")

    # ── 3. Baseline ───────────────────────────────────────────────────────────
    print("\n[3/7] Baseline backtest …")
    base_bt  = run_backtest(df_1h, signals)
    baseline = kpi_row("A. Baseline (no gate)", base_bt,
                       extra={"n_allowed": n_sig, "filter_rate": 0.0,
                              "window_stats": [], "importances": []})
    print(f"  Return={baseline['total_return']:+.1f}%  "
          f"DD={baseline['max_dd']:.1f}%  Win={baseline['win_rate']:.0f}%  "
          f"Trades={baseline['n_trades']}")

    # ── 4. Feature matrix ─────────────────────────────────────────────────────
    print("\n[4/7] Feature matrix …")
    feat_df = build_feature_matrix(tf_ind, signals)
    print(f"  Shape: {feat_df.shape[0]:,} rows × {feat_df.shape[1]} features")

    results = [baseline]

    # ── 5. Binary gate v2 (early stopping + feature selection) ───────────────
    print("\n[5/7] Binary gate v2 (early stopping + feat selection) …")
    for thr, label in [(0.50, "B. Binary P≥0.50"), (0.55, "C. Binary P≥0.55")]:
        print(f"\n  {label}:")
        gate = walk_forward_binary_gate(
            df_1h, signals, feat_df,
            gate_threshold=thr, use_feat_sel=True, verbose=True,
        )
        bt = run_gated_backtest(df_1h, signals, gate)
        n_allowed   = int((gate.gated_signal != 0).sum())
        filter_rate = (1 - n_allowed / n_sig) * 100 if n_sig else 0
        r = kpi_row(label, bt, extra={
            "n_allowed":   n_allowed,
            "filter_rate": round(filter_rate, 1),
            "window_stats": gate.window_stats,
            "importances":  gate.importances.head(20).to_dict("records") if gate.importances is not None else [],
        })
        results.append(r)
        print(f"  → Return={r['total_return']:+.1f}%  DD={r['max_dd']:.1f}%  "
              f"Win={r['win_rate']:.0f}%  Trades={r['n_trades']}  "
              f"Filtered={filter_rate:.0f}%")

    # ── 6. Regression gate v2 ─────────────────────────────────────────────────
    print("\n[6/7] Regression gate v2 …")
    print("\n  D. Regression ret≥0%:")
    gate = walk_forward_regression_gate(
        df_1h, signals, feat_df,
        min_return_pct=0.0, use_feat_sel=True, verbose=True,
    )
    bt = run_gated_backtest(df_1h, signals, gate)
    n_allowed   = int((gate.gated_signal != 0).sum())
    filter_rate = (1 - n_allowed / n_sig) * 100 if n_sig else 0
    r = kpi_row("D. Regression ret≥0%", bt, extra={
        "n_allowed":   n_allowed,
        "filter_rate": round(filter_rate, 1),
        "window_stats": gate.window_stats,
        "importances":  gate.importances.head(20).to_dict("records") if gate.importances is not None else [],
        "gate_type": "regression",
    })
    results.append(r)
    print(f"  → Return={r['total_return']:+.1f}%  DD={r['max_dd']:.1f}%  "
          f"Win={r['win_rate']:.0f}%  Trades={r['n_trades']}  "
          f"Filtered={filter_rate:.0f}%")

    # ── 7. Bar-level direction gate v3 ────────────────────────────────────────
    print("\n[7/7] Bar-level direction gate v3 (~6k training bars per window) …")
    for thr, label in [(0.55, "E. Bar-level P≥0.55"), (0.60, "F. Bar-level P≥0.60")]:
        print(f"\n  {label}:")
        gate = walk_forward_bar_level_gate(
            df_1h, signals, feat_df,
            gate_threshold=thr, forward_bars=4,
            use_feat_sel=True, verbose=True,
        )
        bt = run_gated_backtest(df_1h, signals, gate)
        n_allowed   = int((gate.gated_signal != 0).sum())
        filter_rate = (1 - n_allowed / n_sig) * 100 if n_sig else 0
        r = kpi_row(label, bt, extra={
            "n_allowed":   n_allowed,
            "filter_rate": round(filter_rate, 1),
            "window_stats": gate.window_stats,
            "importances":  gate.importances.head(20).to_dict("records") if gate.importances is not None else [],
            "gate_type": "bar_level",
        })
        results.append(r)
        print(f"  → Return={r['total_return']:+.1f}%  DD={r['max_dd']:.1f}%  "
              f"Win={r['win_rate']:.0f}%  Trades={r['n_trades']}  "
              f"Filtered={filter_rate:.0f}%")

    # ── HTML report ───────────────────────────────────────────────────────────
    print("\nGenerating report …")
    html = _build_html(results)
    out  = Path("reports/ml_report.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"  Saved → {out}")
    print("\n══ Done ══════════════════════════════════════════════════════════")


# ─────────────────────────────────────────────────────────────────────────────
# HTML builder
# ─────────────────────────────────────────────────────────────────────────────

def _c(val, metric):
    if metric in ("total_return","win_rate","profit_factor","calmar","expectancy"):
        return "pos" if val > 0 else "neg"
    if metric == "max_dd":
        return "pos" if val > -15 else "neg"
    return ""


def _build_html(results: list) -> str:
    COLOURS = ["#8b949e","#FF9800","#2196F3","#4CAF50","#E91E63","#9C27B0","#00BCD4"]

    baseline = results[0]
    best     = max(results[1:], key=lambda r: r["total_return"])

    # Downsample
    idx     = pd.DatetimeIndex(baseline["index"])
    step    = max(1, len(idx) // 800)
    ch_idx  = json.dumps([str(idx[j].date()) for j in range(0, len(idx), step)])

    def ds(r, col="equity"):
        return json.dumps([round(r[col][j], 2) for j in range(0, len(r[col]), step)])

    eq_datasets = []
    dd_datasets = []
    for i, r in enumerate(results):
        col  = COLOURS[i % len(COLOURS)]
        lbl  = r["name"].replace('"', "'")
        bw   = "2.5" if i == 0 else "1.8"
        dash = "[6,3]" if i == 0 else "[]"
        eq_datasets.append(f"""{{
  label:"{lbl}", data:{ds(r)},
  borderColor:"{col}", borderWidth:{bw}, borderDash:{dash},
  pointRadius:0, fill:false, tension:0.1
}}""")
        dd_datasets.append(f"""{{
  label:"{lbl}", data:{ds(r,"drawdown")},
  borderColor:"{col}", borderWidth:{bw}, borderDash:{dash},
  pointRadius:0, fill:false, tension:0.1
}}""")

    eq_ds = "[" + ",\n".join(eq_datasets) + "]"
    dd_ds = "[" + ",\n".join(dd_datasets) + "]"

    # Comparison table
    def cmp_row(r):
        row = f"<tr><td>{r['name']}</td>"
        for col, sfx in [("total_return","%"),("max_dd","%"),("calmar",""),
                          ("win_rate","%"),("profit_factor",""),
                          ("n_trades",""),("expectancy","$")]:
            v = r[col]; c = _c(v, col)
            row += f"<td class='{c}'>{v}{sfx}</td>"
        n_allowed   = r.get("n_allowed","—")
        filter_rate = r.get("filter_rate","—")
        row += f"<td>{n_allowed}</td><td>{filter_rate}%</td></tr>\n"
        return row

    cmp_rows = "".join(cmp_row(r) for r in results)

    # Feature importance — Binary v2 P≥0.55 (index 2)
    ml_res   = results[2] if len(results) > 2 else results[1]
    imp_list = ml_res.get("importances", [])[:15]
    imp_labels = json.dumps([x["feature"] for x in imp_list])
    imp_vals   = json.dumps([round(x["importance"], 1) for x in imp_list])

    # Feature importance — first bar-level result (index 4, "E.")
    bar_res   = next((r for r in results if r.get("gate_type") == "bar_level"), results[-1])
    bimp_list = bar_res.get("importances", [])[:15]
    bimp_labels = json.dumps([x["feature"] for x in bimp_list])
    bimp_vals   = json.dumps([round(x["importance"], 1) for x in bimp_list])

    # Window stats — Binary v2 P≥0.55 (index 2)
    ws_res   = results[2] if len(results) > 2 else results[1]
    win_rows = ""
    for w in ws_res.get("window_stats", []):
        pct = round(w["n_gated"] / w["n_sig_oos"] * 100, 0) if w.get("n_sig_oos") else 0
        win_rows += (
            f"<tr><td>{w['window']}</td>"
            f"<td>{w['oos_start']}</td><td>{w['oos_end']}</td>"
            f"<td>{w['n_train']}</td><td>{w['n_pos']}</td><td>{w['n_neg']}</td>"
            f"<td class='{'pos' if w.get('val_acc',0)>55 else 'neg'}'>{w.get('val_acc','—')}%</td>"
            f"<td>{w.get('best_iter','—')}</td><td>{w.get('n_feat','—')}</td>"
            f"<td>{w['n_sig_oos']} → {w['n_gated']} ({pct:.0f}%)</td></tr>\n"
        )

    # Window stats — Regression (index 3, "D.")
    reg_ws_res   = results[3] if len(results) > 3 else results[0]
    reg_win_rows = ""
    for w in reg_ws_res.get("window_stats", []):
        pct = round(w["n_gated"] / w["n_sig_oos"] * 100, 0) if w.get("n_sig_oos") else 0
        reg_win_rows += (
            f"<tr><td>{w['window']}</td>"
            f"<td>{w['oos_start']}</td><td>{w['oos_end']}</td>"
            f"<td>{w['n_train']}</td>"
            f"<td>{w.get('train_mean_ret','—')}%</td>"
            f"<td>{w.get('val_rmse','—')}%</td>"
            f"<td>{w.get('best_iter','—')}</td><td>{w.get('n_feat','—')}</td>"
            f"<td>{w['n_sig_oos']} → {w['n_gated']} ({pct:.0f}%)</td></tr>\n"
        )

    # Window stats — Bar-level P≥0.55 (index 4, "E.")
    bar_ws_rows = ""
    for w in bar_res.get("window_stats", []):
        pct = round(w["n_gated"] / w["n_sig_oos"] * 100, 0) if w.get("n_sig_oos") else 0
        bar_ws_rows += (
            f"<tr><td>{w['window']}</td>"
            f"<td>{w['oos_start']}</td><td>{w['oos_end']}</td>"
            f"<td>{w.get('n_train_bars','—')}</td>"
            f"<td>{w.get('n_up','—')}</td><td>{w.get('n_dn','—')}</td>"
            f"<td class='{'pos' if w.get('val_acc',0)>53 else 'neg'}'>{w.get('val_acc','—')}%</td>"
            f"<td>{w.get('best_iter','—')}</td><td>{w.get('n_feat','—')}</td>"
            f"<td>{w['n_sig_oos']} → {w['n_gated']} ({pct:.0f}%)</td></tr>\n"
        )

    base_ret   = baseline["total_return"]
    best_ret   = best["total_return"]
    n_ws       = len(ws_res.get("window_stats", []))
    n_bar_ws   = len(bar_res.get("window_stats", []))
    bar_name   = bar_res.get("name", "Bar-level")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ML Gate — BTCUSDT</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{font-family:'Segoe UI',Arial,sans-serif;background:#0d1117;color:#c9d1d9;font-size:14px;}}
h1{{text-align:center;padding:24px;font-size:22px;color:#58a6ff;border-bottom:1px solid #30363d;}}
h2{{color:#79c0ff;font-size:16px;margin:20px 0 10px;padding-left:4px;border-left:3px solid #388bfd;}}
h3{{color:#8b949e;font-size:13px;margin:10px 0 6px;}}
.section{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:20px;margin:16px;}}
.box{{background:#1c2333;border:1px solid #388bfd;border-radius:6px;padding:12px 16px;margin-bottom:14px;line-height:1.8;}}
.box strong{{color:#58a6ff;}}
table{{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px;}}
th{{background:#1c2333;color:#8b949e;padding:8px 10px;text-align:right;font-weight:600;white-space:nowrap;}}
th:first-child{{text-align:left;}}
td{{padding:6px 10px;border-bottom:1px solid #21262d;text-align:right;white-space:nowrap;}}
td:first-child{{text-align:left;color:#c9d1d9;}}
tr:hover td{{background:#1c2333;}}
.pos{{color:#3fb950;}} .neg{{color:#f85149;}}
.chart-xl{{position:relative;height:360px;margin:10px 0;}}
.chart-md{{position:relative;height:250px;margin:6px 0;}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px;}}
.stat-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:12px 0;}}
.card{{background:#1c2333;border:1px solid #30363d;border-radius:6px;padding:12px;text-align:center;}}
.card .v{{font-size:20px;font-weight:700;margin:4px 0;}}
.card .l{{font-size:11px;color:#8b949e;}}
.badge{{display:inline-block;background:#1c2333;border:1px solid #388bfd;
        color:#58a6ff;border-radius:4px;padding:2px 8px;font-size:11px;margin-right:4px;}}
.scroll{{overflow-x:auto;max-height:360px;overflow-y:auto;}}
</style>
</head>
<body>
<h1>BTCUSDT — ML Gate (v2 trade-level · v3 bar-level)</h1>

<!-- Summary -->
<div class="section">
<h2>Overview</h2>
<div class="box">
Baseline (composite ±3, session 08-21): <strong class="{'pos' if base_ret>0 else 'neg'}">{base_ret:+.1f}%</strong>.
Best ML gate: <strong class="{'pos' if best_ret>0 else 'neg'}">{best_ret:+.1f}%</strong> — <strong>{best['name']}</strong>.
<br>Walk-forward: <strong>{TRAIN_MONTHS}m train / {OOS_MONTHS}m OOS</strong>.
<strong>v2 trade-level</strong>: early stopping + feature selection on ~200 trade labels/window.
<strong>v3 bar-level</strong>: trains on ALL 1H bars (~6k/window), target = 4H forward direction,
direction-aware gate (LONG: P(up)≥thr · SHORT: P(up)≤1−thr).
</div>
<div class="stat-grid">
  <div class="card"><div class="l">Baseline Return</div><div class="v {'pos' if base_ret>0 else 'neg'}">{base_ret:+.1f}%</div><div class="l">composite ±3</div></div>
  <div class="card"><div class="l">Best ML Return</div><div class="v {'pos' if best_ret>0 else 'neg'}">{best_ret:+.1f}%</div><div class="l">{best['name'][:20]}</div></div>
  <div class="card"><div class="l">Best ML DD</div><div class="v {'pos' if best['max_dd']>-20 else 'neg'}">{best['max_dd']:.1f}%</div><div class="l">{best['name'][:20]}</div></div>
  <div class="card"><div class="l">Best Win Rate</div><div class="v {'pos' if best['win_rate']>50 else 'neg'}">{best['win_rate']:.0f}%</div><div class="l">{best['name'][:20]}</div></div>
</div>
</div>

<!-- Equity -->
<div class="section">
<h2>Equity Curves</h2>
<div class="chart-xl"><canvas id="ceq"></canvas></div>
<h2 style="margin-top:16px">Drawdown</h2>
<div class="chart-md"><canvas id="cdd"></canvas></div>
</div>

<!-- Comparison table -->
<div class="section">
<h2>Performance Summary</h2>
<div class="scroll"><table>
<tr><th>Strategy</th><th>Return%</th><th>Max DD%</th><th>Calmar</th>
    <th>Win%</th><th>PF</th><th>Trades</th><th>Expect($)</th>
    <th>Allowed</th><th>Filtered%</th></tr>
{cmp_rows}</table></div>
</div>

<!-- Feature importance -->
<div class="section">
<h2>Feature Importance (avg OOS importance, top-15)</h2>
<div class="grid2">
  <div>
    <h3><span class="badge">v2 Binary</span>P≥0.55 (trade-level)</h3>
    <div class="chart-md"><canvas id="cimp_b"></canvas></div>
  </div>
  <div>
    <h3><span class="badge">v3 Bar-level</span>P≥0.55 (ALL bars)</h3>
    <div class="chart-md"><canvas id="cimp_bar"></canvas></div>
  </div>
</div>
</div>

<!-- Window stats: binary v2 -->
<div class="section">
<h2><span class="badge">v2 Binary</span>Walk-forward Window Stats — Binary P≥0.55 (trade-level)</h2>
<div class="scroll"><table>
<tr><th>#</th><th>OOS Start</th><th>OOS End</th>
    <th>Train Trades</th><th>Win+</th><th>Loss-</th>
    <th>Val Acc</th><th>Best Iter</th><th># Feats</th><th>Signals → Gated</th></tr>
{win_rows}</table></div>
</div>

<!-- Window stats: regression v2 -->
<div class="section">
<h2><span class="badge">v2 Regression</span>Walk-forward Window Stats — Regression ret≥0% (trade-level)</h2>
<div class="scroll"><table>
<tr><th>#</th><th>OOS Start</th><th>OOS End</th>
    <th>Train Trades</th><th>μ Train Ret</th><th>Val RMSE</th>
    <th>Best Iter</th><th># Feats</th><th>Signals → Gated</th></tr>
{reg_win_rows}</table></div>
</div>

<!-- Window stats: bar-level v3 -->
<div class="section">
<h2><span class="badge">v3 Bar-level</span>Walk-forward Window Stats — {bar_name} (ALL 1H bars)</h2>
<p style="color:#8b949e;font-size:12px;margin-bottom:8px">
  Training target: y=1 if close[T+4h]&gt;close[T]. ~6,000 bars/window vs ~200 trades.
  Gate is direction-aware: LONG passes if P(up)≥thr; SHORT passes if P(up)≤1−thr.
</p>
<div class="scroll"><table>
<tr><th>#</th><th>OOS Start</th><th>OOS End</th>
    <th>Train Bars</th><th>↑ Up</th><th>↓ Dn</th>
    <th>Val Acc</th><th>Best Iter</th><th># Feats</th><th>Signals → Gated</th></tr>
{bar_ws_rows}</table></div>
</div>

<script>
const IDX={ch_idx};
const OPT={{responsive:true,maintainAspectRatio:false,animation:{{duration:0}},
  plugins:{{legend:{{labels:{{color:'#8b949e',font:{{size:11}}}}}}}},
  scales:{{
    x:{{ticks:{{color:'#8b949e',maxTicksLimit:12,font:{{size:10}}}},grid:{{color:'#21262d'}}}},
    y:{{ticks:{{color:'#8b949e',font:{{size:10}},callback:v=>v.toLocaleString()}},grid:{{color:'#21262d'}}}}
  }}
}};
new Chart(document.getElementById('ceq'),{{type:'line',data:{{labels:IDX,datasets:{eq_ds}}},options:OPT}});
new Chart(document.getElementById('cdd'),{{type:'line',
  data:{{labels:IDX,datasets:{dd_ds}}},
  options:{{...OPT,scales:{{...OPT.scales,y:{{...OPT.scales.y,
    ticks:{{...OPT.scales.y.ticks,callback:v=>(v*100).toFixed(1)+'%'}}}}}}}}
}});

function impChart(id, labels, vals){{
  new Chart(document.getElementById(id),{{type:'bar',
    data:{{labels,datasets:[{{label:'importance',data:vals,
      backgroundColor:'#388bfd88',borderColor:'#388bfd',borderWidth:1}}]}},
    options:{{responsive:true,maintainAspectRatio:false,animation:{{duration:0}},
      indexAxis:'y',plugins:{{legend:{{display:false}}}},
      scales:{{x:{{ticks:{{color:'#8b949e',font:{{size:10}}}},grid:{{color:'#21262d'}}}},
               y:{{ticks:{{color:'#8b949e',font:{{size:9}}}},grid:{{color:'#21262d'}}}}}}
    }}
  }});
}}
impChart('cimp_b',{imp_labels},{imp_vals});
impChart('cimp_bar',{bimp_labels},{bimp_vals});
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
