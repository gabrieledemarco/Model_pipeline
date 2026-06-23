"""
Threshold optimization + volatility regime filter.

1. Runs the walk-forward binary gate ONCE to get OOS probability predictions.
2. Sweeps ML thresholds [0.44 … 0.60] without re-training.
3. Crosses each threshold with 3 vol-regime configs:
     A. No vol filter
     B. Block extreme-high vol (rvol_ratio > 2.0)
     C. Block both extremes  (0.5 ≤ rvol_ratio ≤ 2.0)
4. Also benchmarks: vol filter alone (no ML), baseline (no gate).

Output → reports/threshold_report.html
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
from src.strategy.signals      import build_signal_matrix
from src.strategy.optimizer    import apply_filters, ScenarioConfig
from src.strategy.engine       import run_backtest, INIT_CAP
from src.strategy.ml_features  import build_feature_matrix
from src.strategy.ml_gate      import walk_forward_binary_gate, TRAIN_MONTHS, OOS_MONTHS

SESSION_CFG = ScenarioConfig(
    "Session 08-21",
    session_hours=(8, 21),
    long_threshold=3.0,
    short_threshold=-3.0,
)

THRESHOLDS = [0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60]

VOL_CONFIGS = [
    ("no_filter",     "No vol filter",              0.00, 99.0),
    ("high_vol_off",  "Block high-vol (ratio>2.0)", 0.00,  2.0),
    ("regime_filter", "Regime filter (0.5–2.0)",    0.50,  2.0),
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rvol_ratio(df_1h: pd.DataFrame) -> pd.Series:
    if "rvol_ratio" in df_1h.columns:
        return df_1h["rvol_ratio"]
    log_ret  = df_1h["log_ret"] if "log_ret" in df_1h.columns else np.log(df_1h["close"]).diff()
    rvol_20  = log_ret.rolling(20,  min_periods=1).std() * np.sqrt(8760)
    rvol_120 = rvol_20.rolling(120, min_periods=10).mean()
    return (rvol_20 / rvol_120.replace(0, np.nan)).fillna(1.0)


def _vol_mask(df_1h: pd.DataFrame, low: float, high: float) -> pd.Series:
    ratio = _rvol_ratio(df_1h)
    return (ratio >= low) & (ratio <= high)


def _gate_and_bt(df_1h, signals, base_signal, all_probs, threshold,
                 vol_mask=None) -> tuple[dict, int, float]:
    """Apply ML threshold + optional vol mask, run backtest."""
    n_sig    = int((base_signal != 0).sum())
    has_prob = all_probs.notna()

    allow = (base_signal != 0) & has_prob & (all_probs >= threshold)
    if vol_mask is not None:
        allow = allow & vol_mask.reindex(base_signal.index, fill_value=False)

    gated     = pd.Series(0, index=base_signal.index)
    gated[allow] = base_signal[allow]

    sig_copy            = signals.copy()
    sig_copy["signal"]  = gated
    bt                  = run_backtest(df_1h, sig_copy)
    n_allowed           = int((gated != 0).sum())
    filter_rate         = (1 - n_allowed / n_sig) * 100 if n_sig else 0.0
    return bt, n_allowed, filter_rate


def kpi(name: str, bt: dict, threshold: float | None, vol_key: str,
        n_allowed: int, filter_rate: float) -> dict:
    k = bt["kpis"]
    return {
        "name":          name,
        "threshold":     threshold,
        "vol_key":       vol_key,
        "total_return":  round(k.get("total_return",  0) * 100, 2),
        "max_dd":        round(k.get("max_drawdown",  0) * 100, 2),
        "sharpe":        round(k.get("sharpe",        0), 3),
        "calmar":        round(k.get("calmar",        0), 3),
        "win_rate":      round(k.get("win_rate",      0) * 100, 1),
        "profit_factor": round(k.get("profit_factor", 0), 2),
        "n_trades":      k.get("n_trades", 0),
        "expectancy":    round(k.get("expectancy",    0), 0),
        "n_allowed":     n_allowed,
        "filter_rate":   round(filter_rate, 1),
        "equity":        bt["equity"].tolist(),
        "drawdown":      bt["drawdown"].tolist(),
        "index":         [str(t) for t in bt["equity"].index],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n══ Threshold Opt + Vol Regime ════════════════════════════════════")

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("\n[1/5] Loading data …")
    raw = fetch_extended_data(start_year=2022, start_month=1,
                              fetch_1m=False, fetch_flow=True)
    tf_ind = {}
    for tf in ["1W", "1D", "4H", "1H", "15M"]:
        df = raw.get(tf, pd.DataFrame())
        tf_ind[tf] = add_indicators(df) if not df.empty and len(df) > 20 else df

    df_1h   = tf_ind["1H"]
    df_15m  = tf_ind.get("15M", pd.DataFrame())
    oi_df   = generate_oi(tf_ind["1D"]["close"])
    funding = generate_funding(tf_ind["1D"]["close"])

    # ── 2. Signals ────────────────────────────────────────────────────────────
    print("\n[2/5] Building signals (±3, session 08-21) …")
    raw_sig = build_signal_matrix(
        tf_data=tf_ind, oi_df=oi_df, funding=funding,
        premium_1h=None, df_15m=df_15m, df_1m=None,
    )
    signals     = apply_filters(raw_sig, SESSION_CFG)
    base_signal = signals["signal"].copy()
    n_sig       = int((base_signal != 0).sum())
    print(f"  Active signal bars: {n_sig:,}")

    # ── 3. Baseline + vol-only benchmarks ────────────────────────────────────
    print("\n[3/5] Baseline & vol-filter-only benchmarks …")
    base_bt  = run_backtest(df_1h, signals)
    baseline = kpi("Baseline (no gate)", base_bt, None, "baseline",
                   n_sig, 0.0)
    print(f"  Baseline: ret={baseline['total_return']:+.1f}%  "
          f"DD={baseline['max_dd']:.1f}%  calmar={baseline['calmar']:.3f}")

    ratio_series = _rvol_ratio(df_1h)
    vol_only_results = []
    for vk, vl, vlo, vhi in VOL_CONFIGS:
        if vk == "no_filter":
            continue
        vm = _vol_mask(df_1h, vlo, vhi)
        n_vol_ok   = int(vm.sum())
        gated      = base_signal.where(vm, 0)
        sig_copy   = signals.copy()
        sig_copy["signal"] = gated
        bt_vo      = run_backtest(df_1h, sig_copy)
        n_allowed  = int((gated != 0).sum())
        fr         = (1 - n_allowed / n_sig) * 100 if n_sig else 0
        r          = kpi(f"Vol-only: {vl}", bt_vo, None, vk, n_allowed, fr)
        r["vol_label"] = vl
        vol_only_results.append(r)
        print(f"  {vl}: ret={r['total_return']:+.1f}%  "
              f"DD={r['max_dd']:.1f}%  bars_in_regime={n_vol_ok:,}")

    # ── 4. Walk-forward gate (single training pass) ───────────────────────────
    print("\n[4/5] Walk-forward binary gate (single pass, P≥0.50) …")
    feat_df = build_feature_matrix(tf_ind, signals)
    print(f"  Feature matrix: {feat_df.shape[0]:,} rows × {feat_df.shape[1]} features")

    gate    = walk_forward_binary_gate(
        df_1h, signals, feat_df,
        gate_threshold=0.50, use_feat_sel=True, verbose=True,
    )
    all_probs = gate.oos_pred
    print(f"  OOS bars with predictions: {all_probs.notna().sum():,}")

    # ── 5. Threshold × vol regime sweep ───────────────────────────────────────
    print("\n[5/5] Threshold × vol-regime sweep …")

    sweep = []
    vol_masks_map = {}
    for vk, vl, vlo, vhi in VOL_CONFIGS:
        vol_masks_map[vk] = _vol_mask(df_1h, vlo, vhi) if vk != "no_filter" else None

    for vk, vl, vlo, vhi in VOL_CONFIGS:
        vm = vol_masks_map[vk]
        print(f"\n  [{vl}]")
        for thr in THRESHOLDS:
            bt, n_allowed, fr = _gate_and_bt(
                df_1h, signals, base_signal, all_probs, thr, vm)
            name = f"P≥{thr:.2f} + {vl}"
            r    = kpi(name, bt, thr, vk, n_allowed, fr)
            r["vol_label"] = vl
            sweep.append(r)
            print(f"    P≥{thr:.2f}: ret={r['total_return']:+.1f}%  "
                  f"DD={r['max_dd']:.1f}%  calmar={r['calmar']:.3f}  "
                  f"trades={r['n_trades']}  filtered={fr:.0f}%")

    best_calmar = max(sweep, key=lambda r: r["calmar"])
    best_return = max(sweep, key=lambda r: r["total_return"])
    best_sharpe = max(sweep, key=lambda r: r["sharpe"])

    print(f"\n  Best Calmar : {best_calmar['name']}  → {best_calmar['calmar']:.3f}")
    print(f"  Best Return : {best_return['name']}  → {best_return['total_return']:+.1f}%")
    print(f"  Best Sharpe : {best_sharpe['name']}  → {best_sharpe['sharpe']:.3f}")

    # ── HTML ──────────────────────────────────────────────────────────────────
    print("\nGenerating report …")
    html = _build_html(
        baseline, vol_only_results, sweep, gate,
        ratio_series, best_calmar, best_return, best_sharpe,
    )
    out = Path("reports/threshold_report.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"  Saved → {out}")
    print("\n══ Done ══════════════════════════════════════════════════════════")


# ─────────────────────────────────────────────────────────────────────────────
# HTML
# ─────────────────────────────────────────────────────────────────────────────

def _c(val, metric):
    if metric in ("total_return", "win_rate", "profit_factor", "calmar",
                  "sharpe", "expectancy"):
        return "pos" if val > 0 else "neg"
    if metric == "max_dd":
        return "pos" if val > -20 else "neg"
    return ""


def _build_html(baseline, vol_only, sweep, gate,
                ratio_series, best_calmar, best_return, best_sharpe) -> str:

    VOL_COLOURS = {
        "no_filter":     "#2196F3",
        "high_vol_off":  "#FF9800",
        "regime_filter": "#4CAF50",
    }
    VOL_NAMES = {
        "no_filter":     "No vol filter",
        "high_vol_off":  "Block high-vol",
        "regime_filter": "Regime filter (0.5–2.0)",
    }

    # ── Threshold sweep charts ────────────────────────────────────────────────
    thr_labels = json.dumps([f"{t:.2f}" for t in THRESHOLDS])

    def _series_for_vol(vol_key, metric):
        vals = []
        for thr in THRESHOLDS:
            row = next((r for r in sweep
                        if r["vol_key"] == vol_key and
                           abs(r["threshold"] - thr) < 0.001), None)
            vals.append(round(row[metric], 2) if row else 0)
        return json.dumps(vals)

    def _sweep_dataset(vol_key, metric, dash="[]"):
        col = VOL_COLOURS[vol_key]
        return (f'{{label:"{VOL_NAMES[vol_key]}", '
                f'data:{_series_for_vol(vol_key, metric)}, '
                f'borderColor:"{col}", backgroundColor:"{col}33", '
                f'borderWidth:2, borderDash:{dash}, pointRadius:4, '
                f'fill:false, tension:0.3}}')

    ret_datasets  = "[" + ",".join(_sweep_dataset(vk, "total_return") for vk in VOL_COLOURS) + "]"
    cal_datasets  = "[" + ",".join(_sweep_dataset(vk, "calmar") for vk in VOL_COLOURS) + "]"
    dd_datasets   = "[" + ",".join(_sweep_dataset(vk, "max_dd") for vk in VOL_COLOURS) + "]"
    sr_datasets   = "[" + ",".join(_sweep_dataset(vk, "sharpe") for vk in VOL_COLOURS) + "]"

    # ── Equity curves: top-5 by calmar + baseline ─────────────────────────────
    top5 = sorted(sweep, key=lambda r: r["calmar"], reverse=True)[:5]
    EQ_COLOURS = ["#8b949e", "#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]

    idx      = pd.DatetimeIndex(baseline["index"])
    step     = max(1, len(idx) // 800)
    ch_idx   = json.dumps([str(idx[j].date()) for j in range(0, len(idx), step)])

    def ds(r, col="equity"):
        vals = r[col]
        return json.dumps([round(vals[j], 2) for j in range(0, len(vals), step)])

    eq_curves = [f'{{label:"{baseline["name"]}", data:{ds(baseline)}, '
                 f'borderColor:"{EQ_COLOURS[0]}", borderWidth:2.5, '
                 f'borderDash:[6,3], pointRadius:0, fill:false, tension:0.1}}']
    for i, r in enumerate(top5):
        col = EQ_COLOURS[(i + 1) % len(EQ_COLOURS)]
        lbl = r["name"].replace('"', "'")
        eq_curves.append(
            f'{{label:"{lbl}", data:{ds(r)}, '
            f'borderColor:"{col}", borderWidth:1.8, pointRadius:0, '
            f'fill:false, tension:0.1}}')
    eq_ds = "[" + ",".join(eq_curves) + "]"

    dd_curves = [f'{{label:"{baseline["name"]}", data:{ds(baseline,"drawdown")}, '
                 f'borderColor:"{EQ_COLOURS[0]}", borderWidth:2.5, '
                 f'borderDash:[6,3], pointRadius:0, fill:false, tension:0.1}}']
    for i, r in enumerate(top5):
        col = EQ_COLOURS[(i + 1) % len(EQ_COLOURS)]
        lbl = r["name"].replace('"', "'")
        dd_curves.append(
            f'{{label:"{lbl}", data:{ds(r,"drawdown")}, '
            f'borderColor:"{col}", borderWidth:1.8, pointRadius:0, '
            f'fill:false, tension:0.1}}')
    dd_ds = "[" + ",".join(dd_curves) + "]"

    # ── Vol regime distribution ───────────────────────────────────────────────
    r30 = ratio_series.resample("30D").mean().dropna()
    vol_idx    = json.dumps([str(d.date()) for d in r30.index])
    vol_vals   = json.dumps([round(v, 3) for v in r30.values])
    n_normal   = int(((ratio_series >= 0.5) & (ratio_series <= 2.0)).sum())
    n_total    = len(ratio_series)
    n_high_vol = int((ratio_series > 2.0).sum())
    n_low_vol  = int((ratio_series < 0.5).sum())
    pct_normal = round(n_normal / n_total * 100, 1)

    # ── Tables ────────────────────────────────────────────────────────────────
    def _trow(r, highlight=False):
        hl = " style='background:#1c3a22;'" if highlight else ""
        row  = f"<tr{hl}><td>{r['name']}</td>"
        for m, sfx in [("total_return", "%"), ("max_dd", "%"), ("calmar", ""),
                       ("sharpe", ""), ("win_rate", "%"), ("profit_factor", ""),
                       ("n_trades", ""), ("filter_rate", "%")]:
            v = r[m]; c = _c(v, m)
            row += f"<td class='{c}'>{v}{sfx}</td>"
        row += "</tr>\n"
        return row

    # full sweep table sorted by calmar
    all_rows = sorted(sweep, key=lambda r: r["calmar"], reverse=True)
    sweep_table = "".join(_trow(r, r is best_calmar) for r in all_rows)

    # vol-only table
    vol_only_rows = (
        _trow(baseline) +
        "".join(_trow(r) for r in vol_only)
    )

    # optimal summary
    base_ret    = baseline["total_return"]
    base_calmar = baseline["calmar"]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Threshold Opt + Vol Regime — BTCUSDT</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{font-family:'Segoe UI',Arial,sans-serif;background:#0d1117;color:#c9d1d9;font-size:14px;}}
h1{{text-align:center;padding:24px;font-size:22px;color:#58a6ff;border-bottom:1px solid #30363d;}}
h2{{color:#79c0ff;font-size:16px;margin:20px 0 10px;padding-left:4px;border-left:3px solid #388bfd;}}
.section{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:20px;margin:16px;}}
.box{{background:#1c2333;border:1px solid #388bfd;border-radius:6px;padding:12px 16px;margin-bottom:14px;line-height:1.8;}}
.box strong{{color:#58a6ff;}}
table{{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px;}}
th{{background:#1c2333;color:#8b949e;padding:8px 10px;text-align:right;font-weight:600;white-space:nowrap;}}
th:first-child{{text-align:left;}}
td{{padding:6px 10px;border-bottom:1px solid #21262d;text-align:right;white-space:nowrap;}}
td:first-child{{text-align:left;color:#c9d1d9;max-width:320px;overflow:hidden;text-overflow:ellipsis;}}
tr:hover td{{background:#1c2333;}}
.pos{{color:#3fb950;}} .neg{{color:#f85149;}}
.chart-xl{{position:relative;height:360px;margin:10px 0;}}
.chart-md{{position:relative;height:240px;margin:6px 0;}}
.chart-sm{{position:relative;height:180px;margin:6px 0;}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px;}}
.grid4{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:12px 0;}}
.card{{background:#1c2333;border:1px solid #30363d;border-radius:6px;padding:12px;text-align:center;}}
.card .v{{font-size:20px;font-weight:700;margin:4px 0;}}
.card .l{{font-size:11px;color:#8b949e;}}
.badge{{display:inline-block;background:#1c2333;border:1px solid #388bfd;
        color:#58a6ff;border-radius:4px;padding:2px 8px;font-size:11px;margin-right:4px;}}
.scroll{{overflow-x:auto;max-height:440px;overflow-y:auto;}}
.opt-row{{background:#0f2922!important;}}
</style>
</head>
<body>
<h1>BTCUSDT — Threshold Optimisation + Volatility Regime Filter</h1>

<!-- Summary -->
<div class="section">
<h2>Overview</h2>
<div class="box">
Baseline (composite ±3, session 08-21): <strong class="{'pos' if base_ret>0 else 'neg'}">{base_ret:+.1f}%</strong>
| Calmar: <strong>{base_calmar:.3f}</strong><br>
Best Calmar: <strong class="pos">{best_calmar['calmar']:.3f}</strong>
→ <strong>{best_calmar['name']}</strong>
(ret <span class="{'pos' if best_calmar['total_return']>0 else 'neg'}">{best_calmar['total_return']:+.1f}%</span>
DD <span class="{'pos' if best_calmar['max_dd']>-20 else 'neg'}">{best_calmar['max_dd']:.1f}%</span>
{best_calmar['n_trades']} trades)<br>
Best Return: <strong class="{'pos' if best_return['total_return']>0 else 'neg'}">{best_return['total_return']:+.1f}%</strong>
→ <strong>{best_return['name']}</strong><br>
Best Sharpe: <strong class="{'pos' if best_sharpe['sharpe']>0 else 'neg'}">{best_sharpe['sharpe']:.3f}</strong>
→ <strong>{best_sharpe['name']}</strong>
</div>
<div class="grid4">
  <div class="card"><div class="l">Baseline Return</div>
    <div class="v {'pos' if base_ret>0 else 'neg'}">{base_ret:+.1f}%</div>
    <div class="l">no filter, ±3 composite</div></div>
  <div class="card"><div class="l">Best Calmar Return</div>
    <div class="v {'pos' if best_calmar['total_return']>0 else 'neg'}">{best_calmar['total_return']:+.1f}%</div>
    <div class="l">P≥{best_calmar['threshold']:.2f} + {VOL_NAMES[best_calmar['vol_key']]}</div></div>
  <div class="card"><div class="l">Best Calmar DD</div>
    <div class="v {'pos' if best_calmar['max_dd']>-20 else 'neg'}">{best_calmar['max_dd']:.1f}%</div>
    <div class="l">calmar = {best_calmar['calmar']:.3f}</div></div>
  <div class="card"><div class="l">Vol Regime Normal %</div>
    <div class="v pos">{pct_normal}%</div>
    <div class="l">bars in [0.5, 2.0] rvol_ratio</div></div>
</div>
</div>

<!-- Threshold sweep charts -->
<div class="section">
<h2>Threshold Sweep — Metric vs P-threshold</h2>
<div class="grid2">
  <div><h3 style="color:#8b949e;font-size:12px;margin-bottom:4px">Total Return %</h3>
    <div class="chart-sm"><canvas id="cret"></canvas></div></div>
  <div><h3 style="color:#8b949e;font-size:12px;margin-bottom:4px">Calmar Ratio</h3>
    <div class="chart-sm"><canvas id="ccal"></canvas></div></div>
  <div><h3 style="color:#8b949e;font-size:12px;margin-bottom:4px">Max Drawdown %</h3>
    <div class="chart-sm"><canvas id="cdd"></canvas></div></div>
  <div><h3 style="color:#8b949e;font-size:12px;margin-bottom:4px">Sharpe Ratio</h3>
    <div class="chart-sm"><canvas id="csr"></canvas></div></div>
</div>
</div>

<!-- Equity curves -->
<div class="section">
<h2>Equity Curves — Top-5 by Calmar + Baseline</h2>
<div class="chart-xl"><canvas id="ceq"></canvas></div>
<h2 style="margin-top:16px">Drawdown</h2>
<div class="chart-md"><canvas id="cdd2"></canvas></div>
</div>

<!-- Vol regime filter effect -->
<div class="section">
<h2>Volatility Regime Filter — Standalone Effect</h2>
<p style="color:#8b949e;font-size:12px;margin-bottom:10px">
  Vol filter applied to raw signals without ML gate. Shows baseline contribution of regime filtering.
  Regime filter covers <strong class="pos">{pct_normal}%</strong> of 1H bars
  ({n_high_vol:,} high-vol blocked, {n_low_vol:,} low-vol blocked, {n_normal:,} normal).
</p>
<div class="scroll"><table>
<tr><th>Config</th><th>Return%</th><th>Max DD%</th><th>Calmar</th>
    <th>Win%</th><th>PF</th><th>Trades</th><th>Filtered%</th></tr>
{vol_only_rows}
</table></div>
<div class="chart-sm" style="margin-top:12px">
  <canvas id="cvol"></canvas>
</div>
</div>

<!-- Full sweep table -->
<div class="section">
<h2>Full Sweep — Sorted by Calmar (highlighted row = best)</h2>
<div class="scroll"><table>
<tr><th>Strategy</th><th>Return%</th><th>Max DD%</th><th>Calmar</th>
    <th>Sharpe</th><th>Win%</th><th>PF</th><th>Trades</th><th>Filtered%</th></tr>
{sweep_table}
</table></div>
</div>

<script>
const IDX = {ch_idx};
const VOL_IDX = {vol_idx};
const BASE_RET = {base_ret};

const SWEEP_OPT = {{
  responsive:true, maintainAspectRatio:false, animation:{{duration:0}},
  plugins:{{legend:{{labels:{{color:'#8b949e',font:{{size:10}}}}}}}},
  scales:{{
    x:{{ticks:{{color:'#8b949e',maxTicksLimit:10,font:{{size:10}}}},grid:{{color:'#21262d'}}}},
    y:{{ticks:{{color:'#8b949e',font:{{size:10}}}},grid:{{color:'#21262d'}}}}
  }}
}};
const EQ_OPT = {{
  responsive:true, maintainAspectRatio:false, animation:{{duration:0}},
  plugins:{{legend:{{labels:{{color:'#8b949e',font:{{size:11}}}}}}}},
  scales:{{
    x:{{ticks:{{color:'#8b949e',maxTicksLimit:12,font:{{size:10}}}},grid:{{color:'#21262d'}}}},
    y:{{ticks:{{color:'#8b949e',font:{{size:10}},callback:v=>v.toLocaleString()}},grid:{{color:'#21262d'}}}}
  }}
}};

// threshold sweep charts
const THR = {thr_labels};
function mkSweep(id, datasets, extra={{}}) {{
  new Chart(document.getElementById(id), {{
    type:'line', data:{{labels:THR, datasets}},
    options:{{...SWEEP_OPT, ...extra}}
  }});
}}
mkSweep('cret', {ret_datasets}, {{
  plugins:{{...SWEEP_OPT.plugins,
    annotation:{{annotations:{{base:{{type:'line',yMin:BASE_RET,yMax:BASE_RET,
      borderColor:'#8b949e',borderWidth:1,borderDash:[4,4]}}}}}}}}
}});
mkSweep('ccal', {cal_datasets});
mkSweep('cdd',  {dd_datasets});
mkSweep('csr',  {sr_datasets});

// equity / drawdown
new Chart(document.getElementById('ceq'), {{
  type:'line', data:{{labels:IDX, datasets:{eq_ds}}}, options:EQ_OPT
}});
new Chart(document.getElementById('cdd2'), {{
  type:'line', data:{{labels:IDX, datasets:{dd_ds}}},
  options:{{...EQ_OPT, scales:{{...EQ_OPT.scales,
    y:{{...EQ_OPT.scales.y,
      ticks:{{...EQ_OPT.scales.y.ticks, callback:v=>(v*100).toFixed(1)+'%'}}}}}}}}
}});

// vol ratio over time
new Chart(document.getElementById('cvol'), {{
  type:'line',
  data:{{labels:VOL_IDX, datasets:[{{
    label:'rvol_ratio (30D avg)', data:{vol_vals},
    borderColor:'#FF9800', borderWidth:1.5, pointRadius:0, fill:false, tension:0.3
  }}]}},
  options:{{responsive:true, maintainAspectRatio:false, animation:{{duration:0}},
    plugins:{{legend:{{labels:{{color:'#8b949e',font:{{size:10}}}},
      annotation:{{annotations:{{
        lo:{{type:'line',yMin:0.5,yMax:0.5,borderColor:'#4CAF50',borderWidth:1,borderDash:[3,3]}},
        hi:{{type:'line',yMin:2.0,yMax:2.0,borderColor:'#f85149',borderWidth:1,borderDash:[3,3]}}
      }}}}}}}},
    scales:{{
      x:{{ticks:{{color:'#8b949e',maxTicksLimit:12,font:{{size:10}}}},grid:{{color:'#21262d'}}}},
      y:{{ticks:{{color:'#8b949e',font:{{size:10}}}},grid:{{color:'#21262d'}}}}
    }}
  }}
}});
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
