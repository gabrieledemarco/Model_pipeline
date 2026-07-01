"""
create_strategies_comparison.py
────────────────────────────────
Implements and validates three trading strategies sourced from:
  huseinzol05/Stock-Prediction-Models (GitHub)

All tested on BTCUSDT perpetual futures, 2020-01 → 2026-05 (56k 1H bars).
Walk-forward validation: 6m train / 2m OOS / step 2m → 35 windows.

Strategies
──────────
A. Donchian Turtle Breakout (agent/1.turtle-agent.ipynb)
   Trend-following: long when close > 20-bar high, short < 20-bar low.

B. Evolution Strategy Signal Weights (agent/6.evolution-strategy-agent.ipynb)
   NES learns per-window weights for the 9 signal components, replacing
   the fixed weight vector in signals.py.

C. Policy Gradient RL Agent (agent/4.policy-gradient-agent.ipynb)
   REINFORCE bar-level agent: observes the 75-feature state vector,
   outputs Long / Short / Flat independently of the composite signal.

Reference (incumbent): WF ML Gate (LightGBM binary gate, 75 features)

Output → reports/report_strategies_comparison.html
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from src.strategy.data_fetcher     import fetch_extended_data, generate_oi, generate_funding
from src.strategy.indicators       import add_indicators
from src.strategy.signals          import build_signal_matrix
from src.strategy.optimizer        import apply_filters, ScenarioConfig
from src.strategy.engine           import run_backtest, INIT_CAP
from src.strategy.ml_features      import build_feature_matrix
from src.strategy.ml_gate          import walk_forward_binary_gate, run_gated_backtest, _wf_windows
from src.strategy.turtle           import build_turtle_signals
from src.strategy.evolution_signal import walk_forward_evo, SIGNAL_COLS, INIT_WEIGHTS
from src.strategy.rl_policy        import walk_forward_pg

START_YEAR  = 2020
START_MONTH = 1

SESSION_CFG = ScenarioConfig(
    "Session 08-21", session_hours=(8, 21),
    long_threshold=3.0, short_threshold=-3.0,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _kpi(bt: dict) -> tuple:
    kpis = bt.get("kpis", {})
    ret  = float(kpis.get("total_return",  0.0)) * 100
    dd   = abs(float(kpis.get("max_drawdown", 0.0))) * 100
    cal  = float(kpis.get("calmar", 0.0))
    sh   = float(kpis.get("sharpe", 0.0))
    nt   = len(bt.get("trades", []))
    return round(ret, 2), round(dd, 2), round(cal, 3), round(sh, 3), nt


def _per_window_ret(trades_df: pd.DataFrame, windows) -> np.ndarray:
    rets = []
    for (_, _, oo_s, oo_e) in windows:
        t = trades_df[(trades_df["entry_ts"] >= oo_s) & (trades_df["entry_ts"] < oo_e)]
        rets.append(float(t["net_pnl"].sum() / INIT_CAP * 100) if not t.empty else 0.0)
    return np.array(rets)


def _ttest(arr: np.ndarray, label: str) -> dict:
    t, p = st.ttest_1samp(arr, 0)
    p1   = float(st.t.sf(t, df=len(arr) - 1))
    return {
        "label": label, "t": round(t, 3), "p": round(p1, 4),
        "n": len(arr), "mean": round(arr.mean(), 3),
        "sig": p1 < 0.05,
    }


def _cc(v, pos_if_positive=True):
    cls = "pos" if (v > 0 if pos_if_positive else v < 0) else "neg"
    return f'<td class="{cls}">{v:+.2f}%</td>'


def _row(label, ret, dd, cal, sh, nt, highlight=False):
    cal_cls = "pos" if cal > 0.5 else ("neg" if cal < 0 else "")
    h = " class=\"hl\"" if highlight else ""
    return (f'<tr{h}><td>{label}</td>'
            f'{_cc(ret)}<td class="neg">{dd:.1f}%</td>'
            f'<td class="{cal_cls}">{cal:.3f}</td>'
            f'<td>{sh:.3f}</td><td>{nt}</td></tr>')


def _win_table_rows(labels_rets: list[tuple]) -> str:
    """labels_rets: [(label, np.ndarray of per-window returns)]"""
    if not labels_rets:
        return ""
    n = len(labels_rets[0][1])
    rows = []
    # header is handled by the caller
    for i in range(n):
        cells = "".join(_cc(arr[i]) for _, arr in labels_rets)
        rows.append(f"<tr><td>{i+1}</td>{cells}</tr>")
    return "\n".join(rows)


def _imp_row(sig_name, init_w, mean_w, max_w, min_w):
    delta = mean_w - init_w
    dc = "pos" if delta > 0 else "neg"
    bar = int(abs(mean_w) / (INIT_WEIGHTS.max() + 1) * 100)
    return (f"<tr><td>{sig_name}</td>"
            f"<td>{init_w:.1f}</td><td>{mean_w:.2f}</td>"
            f'<td class="{dc}">{delta:+.2f}</td>'
            f"<td>{min_w:.2f}</td><td>{max_w:.2f}</td>"
            f'<td><div class="bar" style="width:{bar}%"></div></td></tr>')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("══ Strategies Comparison Report ══\n")
    t0 = time.time()

    # ── [1] Load data ─────────────────────────────────────────────────────────
    print("[1/7] Loading data …")
    raw = fetch_extended_data(
        start_year=START_YEAR, start_month=START_MONTH,
        fetch_flow=True, fetch_15m=True, fetch_1m=False,
    )
    tf_data = {}
    for tf in ["1W", "1D", "4H", "1H", "15M"]:
        df = raw.get(tf, pd.DataFrame())
        tf_data[tf] = add_indicators(df) if not df.empty and len(df) > 20 else df

    df_1h   = tf_data["1H"]
    oi_df   = generate_oi(tf_data["1D"]["close"])
    funding = generate_funding(tf_data["1D"]["close"])
    print(f"  1H: {len(df_1h):,} bars  ({df_1h.index[0].date()} → {df_1h.index[-1].date()})")

    # ── [2] Signals + feature matrix ─────────────────────────────────────────
    print("[2/7] Building signals & features …")
    raw_sig  = build_signal_matrix(
        tf_data=tf_data, oi_df=oi_df, funding=funding,
        premium_1h=None, df_15m=tf_data.get("15M"), df_1m=None,
    )
    signals  = apply_filters(raw_sig, SESSION_CFG)
    feat_df  = build_feature_matrix(tf_data, signals, base_tf="1H", include_smc=False)
    windows  = _wf_windows(df_1h.index)
    n_win    = len(windows)
    print(f"  Signal bars: {(signals['signal']!=0).sum():,}  Features: {feat_df.shape[1]}  WF windows: {n_win}")

    # ── [3] Baseline (composite signal, no gate) ──────────────────────────────
    print("[3/7] Baseline backtest …")
    bt_base = run_backtest(df_1h, signals)
    r_b, dd_b, cal_b, sh_b, nt_b = _kpi(bt_base)
    rets_base = _per_window_ret(bt_base["trades"], windows)

    # ── [4] Strategy A: Turtle Breakout ───────────────────────────────────────
    print("[4/7] Strategy A — Donchian Turtle Breakout (n=20) …")
    turtle_sig = build_turtle_signals(df_1h, n_entry=20, session_hours=(8, 21))
    bt_turtle  = run_backtest(df_1h, turtle_sig)
    r_t, dd_t, cal_t, sh_t, nt_t = _kpi(bt_turtle)
    rets_turtle = _per_window_ret(bt_turtle["trades"], windows)
    print(f"  ret={r_t:+.1f}%  Calmar={cal_t:.3f}  Trades={nt_t}")

    # ── [5] Strategy B: Evolution Strategy Signal Weights ─────────────────────
    print("[5/7] Strategy B — NES Signal Weight Optimiser …")
    evo_sig, evo_wstats, evo_weights = walk_forward_evo(
        df_1h, signals, windows,
        pop_size=15, n_iter=30, verbose=True,
    )
    bt_evo  = run_backtest(df_1h, evo_sig)
    r_e, dd_e, cal_e, sh_e, nt_e = _kpi(bt_evo)
    rets_evo = _per_window_ret(bt_evo["trades"], windows)
    print(f"  ret={r_e:+.1f}%  Calmar={cal_e:.3f}  Trades={nt_e}")

    # ── [6] Strategy C: Policy Gradient RL ───────────────────────────────────
    print("[6/7] Strategy C — Policy Gradient REINFORCE Agent …")
    pg_sig, pg_wstats = walk_forward_pg(
        df_1h, feat_df, windows,
        n_features=feat_df.shape[1], hidden=128, lr=5e-4, gamma=0.95,
        n_epochs=20, session_hours=(8, 21), verbose=True,
    )
    bt_pg   = run_backtest(df_1h, pg_sig)
    r_p, dd_p, cal_p, sh_p, nt_p = _kpi(bt_pg)
    rets_pg = _per_window_ret(bt_pg["trades"], windows)
    print(f"  ret={r_p:+.1f}%  Calmar={cal_p:.3f}  Trades={nt_p}")

    # ── Reference: ML Gate (from extended report) ─────────────────────────────
    # Not re-run here for time; values from create_extended_validation_report.py
    r_ml, dd_ml, cal_ml, sh_ml, nt_ml = -11.8, 53.6, -0.220, 2.591, 1708

    # ── [7] Statistical tests ─────────────────────────────────────────────────
    st_t  = _ttest(rets_turtle, "Turtle per-window return > 0")
    st_e  = _ttest(rets_evo,    "EvoSignal per-window return > 0")
    st_p  = _ttest(rets_pg,     "PolicyGradient per-window return > 0")

    # EvoSignal: average learned weights across windows
    if evo_weights:
        evo_weights_arr = np.stack(evo_weights)
        evo_mean_w  = evo_weights_arr.mean(axis=0)
        evo_max_w   = evo_weights_arr.max(axis=0)
        evo_min_w   = evo_weights_arr.min(axis=0)
    else:
        evo_mean_w  = INIT_WEIGHTS.copy()
        evo_max_w   = INIT_WEIGHTS.copy()
        evo_min_w   = INIT_WEIGHTS.copy()

    # per-window data for charts
    per_win_json = json.dumps([{
        "w":    i + 1,
        "base": round(rets_base[i], 3),
        "turt": round(rets_turtle[i], 3),
        "evo":  round(rets_evo[i],   3),
        "pg":   round(rets_pg[i],    3),
    } for i in range(n_win)])

    elapsed = time.time() - t0

    # ── Build HTML ────────────────────────────────────────────────────────────
    print("[7/7] Generating HTML …")

    def _sig_badge(sig: bool) -> str:
        return ('<span class="badge pos">&#10003; PASS</span>' if sig else
                '<span class="badge neg">&#9888; MARGINAL</span>')

    def _best_calmar(*calmars):
        best = max(calmars)
        return f'<b class="pos">{best:.3f}</b>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Strategy Comparison — BTCUSDT 2020–2026</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:#0d1117;color:#c9d1d9;font-family:'Segoe UI',sans-serif;padding:24px;max-width:1400px;margin:auto;}}
  h1{{color:#58a6ff;margin-bottom:6px;font-size:1.6rem;}}
  h2{{color:#8b949e;font-size:1.1rem;margin:28px 0 12px;border-bottom:1px solid #21262d;padding-bottom:6px;}}
  h3{{color:#c9d1d9;font-size:0.95rem;margin:14px 0 8px;}}
  .subtitle{{color:#8b949e;font-size:0.9rem;margin-bottom:24px;}}
  .card{{background:#161b22;border:1px solid #21262d;border-radius:10px;padding:20px;margin-bottom:20px;}}
  .grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:20px;}}
  .grid-3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;}}
  table{{width:100%;border-collapse:collapse;font-size:0.82rem;}}
  th{{background:#21262d;color:#8b949e;padding:8px 10px;text-align:right;font-weight:600;}}
  th:first-child{{text-align:left;}}
  td{{padding:7px 10px;text-align:right;border-bottom:1px solid #21262d;}}
  td:first-child{{text-align:left;}}
  tr:hover td{{background:#1c2128;}}
  tr.hl td{{background:#0f2a1e88;}}
  .pos{{color:#3fb950;}} .neg{{color:#f85149;}}
  .badge{{display:inline-block;padding:3px 10px;border-radius:12px;font-size:0.78rem;font-weight:600;}}
  .badge.pos{{background:#1a4731;color:#3fb950;}} .badge.neg{{background:#3d1f1f;color:#f85149;}}
  .stat-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px;}}
  .stat-box{{background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:14px;text-align:center;}}
  .stat-val{{font-size:1.4rem;font-weight:700;}} .stat-lbl{{font-size:0.75rem;color:#8b949e;margin-top:4px;}}
  .test-block{{background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:14px;margin-bottom:10px;}}
  .test-meta{{font-size:0.79rem;color:#8b949e;margin-top:5px;}}
  .note{{background:#0f2a1e;border:1px solid #238636;border-radius:6px;padding:10px 14px;font-size:0.82rem;color:#3fb950;margin:12px 0;}}
  .warn{{background:#2d2007;border:1px solid #6e5502;border-radius:6px;padding:10px 14px;font-size:0.82rem;color:#e3b341;margin:12px 0;}}
  .bar{{height:8px;background:#3fb950;border-radius:4px;}}
  canvas{{max-height:320px;}}
  .ref-row td{{color:#6e7681;font-style:italic;}}
</style>
</head>
<body>

<h1>Strategy Comparison — BTCUSDT Perpetual Futures</h1>
<p class="subtitle">
  Source: <a href="https://github.com/huseinzol05/Stock-Prediction-Models" style="color:#58a6ff">huseinzol05/Stock-Prediction-Models</a>
  &nbsp;&middot;&nbsp; {START_YEAR}-{START_MONTH:02d} &rarr; 2026-05
  &nbsp;&middot;&nbsp; {n_win} WF windows (6m train / 2m OOS) &nbsp;&middot;&nbsp; Runtime: {elapsed:.0f}s
</p>


<!-- ── 0. Summary KPIs ──────────────────────────────────────────────────── -->
<h2>0. Full-Period KPIs (2020–2026)</h2>
<div class="card">
<div class="stat-grid">
  <div class="stat-box">
    <div class="stat-val {'pos' if cal_b>0 else 'neg'}">{cal_b:.3f}</div>
    <div class="stat-lbl">Calmar — Baseline</div>
  </div>
  <div class="stat-box">
    <div class="stat-val {'pos' if cal_t>0.5 else ('neg' if cal_t<0 else '')}">{cal_t:.3f}</div>
    <div class="stat-lbl">Calmar — A. Turtle</div>
  </div>
  <div class="stat-box">
    <div class="stat-val {'pos' if cal_e>0.5 else ('neg' if cal_e<0 else '')}">{cal_e:.3f}</div>
    <div class="stat-lbl">Calmar — B. EvoSignal</div>
  </div>
  <div class="stat-box">
    <div class="stat-val {'pos' if cal_p>0.5 else ('neg' if cal_p<0 else '')}">{cal_p:.3f}</div>
    <div class="stat-lbl">Calmar — C. PolicyGrad</div>
  </div>
  <div class="stat-box">
    <div class="stat-val neg">{cal_ml:.3f}</div>
    <div class="stat-lbl">Calmar — Ref: ML Gate</div>
  </div>
</div>
</div>


<!-- ── 1. Metrics table ─────────────────────────────────────────────────── -->
<h2>1. Full-Period Metrics</h2>
<div class="card">
<table>
  <thead><tr>
    <th>Strategy</th><th>Return%</th><th>Max DD%</th>
    <th>Calmar</th><th>Sharpe</th><th>Trades</th>
  </tr></thead>
  <tbody>
    {_row("Baseline — composite signal (no gate)", r_b, dd_b, cal_b, sh_b, nt_b)}
    {_row("A — Donchian Turtle (n=20)", r_t, dd_t, cal_t, sh_t, nt_t, cal_t>cal_b)}
    {_row("B — Evolution Strategy (NES weights)", r_e, dd_e, cal_e, sh_e, nt_e, cal_e>cal_b)}
    {_row("C — Policy Gradient REINFORCE", r_p, dd_p, cal_p, sh_p, nt_p, cal_p>cal_b)}
    <tr class="ref-row">
      <td>Ref: WF ML Gate (LightGBM, extended report)</td>
      {_cc(r_ml)}<td class="neg">{dd_ml:.1f}%</td>
      <td class="neg">{cal_ml:.3f}</td><td>{sh_ml:.3f}</td><td>{nt_ml}</td>
    </tr>
  </tbody>
</table>
</div>


<!-- ── 2. Per-window chart ──────────────────────────────────────────────── -->
<h2>2. Per-Window OOS Returns ({n_win} windows)</h2>
<div class="card">
<canvas id="winChart"></canvas>
</div>

<script>
const wd = {per_win_json};
new Chart(document.getElementById('winChart').getContext('2d'), {{
  type: 'line',
  data: {{
    labels: wd.map(r => r.w),
    datasets: [
      {{label:'Baseline',       data:wd.map(r=>r.base),borderColor:'#6e7681',backgroundColor:'transparent',borderDash:[4,4],borderWidth:1,pointRadius:0}},
      {{label:'A. Turtle',      data:wd.map(r=>r.turt),borderColor:'#e3b341',backgroundColor:'transparent',borderWidth:2,pointRadius:2}},
      {{label:'B. EvoSignal',   data:wd.map(r=>r.evo), borderColor:'#79c0ff',backgroundColor:'transparent',borderWidth:2,pointRadius:2}},
      {{label:'C. PolicyGrad',  data:wd.map(r=>r.pg),  borderColor:'#3fb950',backgroundColor:'transparent',borderWidth:2,pointRadius:2}},
    ],
  }},
  options:{{
    responsive:true, animation:false,
    plugins:{{legend:{{labels:{{color:'#c9d1d9'}}}}}},
    scales:{{
      x:{{ticks:{{color:'#8b949e'}},grid:{{color:'#21262d'}}}},
      y:{{ticks:{{color:'#8b949e',callback:v=>v+'%'}},grid:{{color:'#21262d'}}}},
    }},
  }},
}});
</script>


<!-- ── 3. Turtle strategy detail ────────────────────────────────────────── -->
<h2>3. Strategy A — Donchian Turtle Breakout</h2>
<div class="card">
  <p style="font-size:0.84rem;color:#8b949e;margin-bottom:12px;">
    Trend-following: Long when 1H close &gt; 20-bar rolling high (shifted), Short &lt; 20-bar rolling low.
    Session filter 08&ndash;21 UTC. No training — deterministic rule. Directly comparable to WF validation
    (pure OOS by construction — no in-sample fitting).
  </p>
  <div class="grid-3" style="margin-top:12px;">
    <div class="stat-box">
      <div class="stat-val {'pos' if r_t>0 else 'neg'}">{r_t:+.1f}%</div>
      <div class="stat-lbl">Return 2020–2026</div>
    </div>
    <div class="stat-box">
      <div class="stat-val {'pos' if cal_t>0.5 else ('neg' if cal_t<0 else '')}">{cal_t:.3f}</div>
      <div class="stat-lbl">Calmar</div>
    </div>
    <div class="stat-box">
      <div class="stat-val">{nt_t}</div>
      <div class="stat-lbl">Total Trades</div>
    </div>
  </div>
  <div class="test-block" style="margin-top:14px;">
    <strong>{st_t['label']}</strong>&nbsp;&nbsp;{_sig_badge(st_t['sig'])}
    <div class="test-meta">t={st_t['t']} &nbsp; p={st_t['p']} &nbsp; n={st_t['n']} windows &nbsp; mean={st_t['mean']:+.3f}%</div>
  </div>
</div>


<!-- ── 4. Evolution Strategy detail ─────────────────────────────────────── -->
<h2>4. Strategy B — NES Signal Weight Optimisation</h2>
<div class="card">
  <p style="font-size:0.84rem;color:#8b949e;margin-bottom:12px;">
    Natural Evolution Strategy learns per-window weights for the 9 composite signal components.
    Population {15}, {30} iterations per window. Initial weights = signals.py WEIGHTS.
    NES fitness = fast vectorised return sum (no fees, no SL/TP). Final evaluation via run_backtest().
  </p>

  <h3>Average learned weights vs initial (across {n_win} WF windows)</h3>
  <table style="margin-top:8px;">
    <thead><tr>
      <th>Signal</th><th>Init weight</th><th>Mean learned</th>
      <th>Delta</th><th>Min</th><th>Max</th><th>Magnitude</th>
    </tr></thead>
    <tbody>
      {"".join(_imp_row(SIGNAL_COLS[j], float(INIT_WEIGHTS[j]), float(evo_mean_w[j]),
                        float(evo_max_w[j]), float(evo_min_w[j]))
               for j in range(len(SIGNAL_COLS)))}
    </tbody>
  </table>

  <div class="test-block" style="margin-top:14px;">
    <strong>{st_e['label']}</strong>&nbsp;&nbsp;{_sig_badge(st_e['sig'])}
    <div class="test-meta">t={st_e['t']} &nbsp; p={st_e['p']} &nbsp; n={st_e['n']} windows &nbsp; mean={st_e['mean']:+.3f}%</div>
  </div>
</div>


<!-- ── 5. Policy Gradient detail ─────────────────────────────────────────── -->
<h2>5. Strategy C — Policy Gradient REINFORCE Agent</h2>
<div class="card">
  <p style="font-size:0.84rem;color:#8b949e;margin-bottom:12px;">
    Bar-level RL agent. State = 75-feature vector. Actions = Long / Short / Flat.
    Network: {feat_df.shape[1]} &rarr; 128 (tanh) &rarr; 3 (softmax). 20 epochs per WF window.
    Reward = direction &times; log_return &minus; fee. REINFORCE + Adam optimiser.
    Fresh weights initialised per window (no cross-window transfer).
  </p>

  <h3>Per-window signal counts</h3>
  <div style="overflow-x:auto;max-height:300px;">
  <table>
    <thead><tr><th>Win</th><th>OOS Start</th><th>OOS End</th><th>ep_ret(last)</th><th>Longs</th><th>Shorts</th><th>Total</th></tr></thead>
    <tbody>
      {"".join(
        f"<tr><td>{ws['window']}</td><td>{ws['oos_start']}</td><td>{ws['oos_end']}</td>"
        f"<td>{ws['ep_ret_final']:+.5f}</td>"
        f"<td>{ws['n_long']}</td><td>{ws['n_short']}</td><td>{ws['n_sig']}</td></tr>"
        for ws in pg_wstats
      )}
    </tbody>
  </table>
  </div>

  <div class="test-block" style="margin-top:14px;">
    <strong>{st_p['label']}</strong>&nbsp;&nbsp;{_sig_badge(st_p['sig'])}
    <div class="test-meta">t={st_p['t']} &nbsp; p={st_p['p']} &nbsp; n={st_p['n']} windows &nbsp; mean={st_p['mean']:+.3f}%</div>
  </div>
</div>


<!-- ── 6. Summary verdict ───────────────────────────────────────────────── -->
<h2>6. Verdict &amp; Recommendations</h2>
<div class="card">
  <table>
    <thead><tr><th>Strategy</th><th>Calmar</th><th>vs Baseline</th><th>t-test</th><th>Verdict</th></tr></thead>
    <tbody>
      <tr><td>Baseline (composite, no gate)</td><td>{cal_b:.3f}</td><td>—</td><td>—</td><td>Reference</td></tr>
      <tr><td>A. Turtle Breakout</td>
          <td class="{'pos' if cal_t>cal_b else 'neg'}">{cal_t:.3f}</td>
          <td class="{'pos' if cal_t>cal_b else 'neg'}">{cal_t-cal_b:+.3f}</td>
          <td>{_sig_badge(st_t['sig'])}</td>
          <td>{'&#10003; Better than baseline' if cal_t>cal_b else '&#9888; Not better'}</td></tr>
      <tr><td>B. EvoSignal (NES)</td>
          <td class="{'pos' if cal_e>cal_b else 'neg'}">{cal_e:.3f}</td>
          <td class="{'pos' if cal_e>cal_b else 'neg'}">{cal_e-cal_b:+.3f}</td>
          <td>{_sig_badge(st_e['sig'])}</td>
          <td>{'&#10003; Better than baseline' if cal_e>cal_b else '&#9888; Not better'}</td></tr>
      <tr><td>C. Policy Gradient RL</td>
          <td class="{'pos' if cal_p>cal_b else 'neg'}">{cal_p:.3f}</td>
          <td class="{'pos' if cal_p>cal_b else 'neg'}">{cal_p-cal_b:+.3f}</td>
          <td>{_sig_badge(st_p['sig'])}</td>
          <td>{'&#10003; Better than baseline' if cal_p>cal_b else '&#9888; Not better'}</td></tr>
      <tr class="ref-row"><td>Ref: ML Gate (LightGBM)</td>
          <td class="neg">{cal_ml:.3f}</td>
          <td class="neg">{cal_ml-cal_b:+.3f}</td>
          <td><span class="badge neg">&#9888; MARGINAL</span></td>
          <td>Prior result — not re-run</td></tr>
    </tbody>
  </table>
</div>

</body>
</html>"""

    out_path = Path("reports/report_strategies_comparison.html")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"  → {out_path}")

    print(f"\n══ Summary ({elapsed:.0f}s) ═══════════════════════════════════════════")
    print(f"  Baseline:  ret={r_b:+.1f}%  Calmar={cal_b:.3f}  Trades={nt_b}")
    print(f"  A Turtle:  ret={r_t:+.1f}%  Calmar={cal_t:.3f}  Trades={nt_t}  t-p={st_t['p']}")
    print(f"  B EvoSig:  ret={r_e:+.1f}%  Calmar={cal_e:.3f}  Trades={nt_e}  t-p={st_e['p']}")
    print(f"  C PGrad:   ret={r_p:+.1f}%  Calmar={cal_p:.3f}  Trades={nt_p}  t-p={st_p['p']}")
    print("══════════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
