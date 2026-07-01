"""
create_hmm_gate_report.py
─────────────────────────
Compares the WF binary gate with and without HMM regime features.

Config A : 75 baseline features (same as extended report)
Config B : 75 + 5 HMM features (hmm_state, hmm_prob_{bear,side,bull}, hmm_duration)

HMM is fitted PER WINDOW on training bars only (GaussianHMM, 3 states).
States are labelled semantically (0=bear, 1=side, 2=bull) by mean log-return.

Data: 2020-01 → 2026-05 (~57k 1H bars, 35 WF windows)
Output: reports/report_hmm_gate.html
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from src.strategy.data_fetcher  import fetch_extended_data, generate_oi, generate_funding
from src.strategy.indicators    import add_indicators
from src.strategy.signals       import build_signal_matrix
from src.strategy.optimizer     import apply_filters, ScenarioConfig
from src.strategy.engine        import run_backtest, INIT_CAP
from src.strategy.ml_features   import build_feature_matrix
from src.strategy.ml_gate       import (
    walk_forward_binary_gate,
    run_gated_backtest,
    _wf_windows,
    TOP_N_FEATURES,
)
import scipy.stats as st

START_YEAR  = 2020
START_MONTH = 1

SESSION_CFG = ScenarioConfig(
    "Session 08-21",
    session_hours=(8, 21),
    long_threshold=3.0,
    short_threshold=-3.0,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _kpi(bt: dict) -> tuple:
    """Return (ret%, dd%, calmar, sharpe, n_trades) from backtest result."""
    kpis = bt.get("kpis", {})
    eq   = bt["equity"]
    ret  = float(kpis.get("total_return", 0.0)) * 100
    dd   = abs(float(kpis.get("max_drawdown", 0.0))) * 100
    cal  = float(kpis.get("calmar", 0.0))
    sh   = float(kpis.get("sharpe", 0.0))
    nt   = len(bt.get("trades", []))
    return round(ret, 2), round(dd, 2), round(cal, 3), round(sh, 3), nt


def _per_window_stats(bt_a, bt_b, windows, df_1h):
    """Build per-window comparison table."""
    rows = []
    trades_a = bt_a["trades"]
    trades_b = bt_b["trades"]
    for (_, _, oo_s, oo_e) in windows:
        ta = trades_a[(trades_a["entry_ts"] >= oo_s) & (trades_a["entry_ts"] < oo_e)]
        tb = trades_b[(trades_b["entry_ts"] >= oo_s) & (trades_b["entry_ts"] < oo_e)]
        ret_a = float(ta["net_pnl"].sum() / INIT_CAP * 100) if not ta.empty else 0.0
        ret_b = float(tb["net_pnl"].sum() / INIT_CAP * 100) if not tb.empty else 0.0
        rows.append({
            "oos_start": str(oo_s.date()),
            "oos_end":   str(oo_e.date()),
            "ret_a":     round(ret_a, 3),
            "ret_b":     round(ret_b, 3),
            "delta":     round(ret_b - ret_a, 3),
            "n_a":       len(ta),
            "n_b":       len(tb),
        })
    return rows


def _cc(v, higher_better=True):
    cls = "pos" if (v > 0 if higher_better else v < 0) else "neg"
    return f'<td class="{cls}">{v:+.2f}%</td>'


def _win_rows(rows):
    out = []
    for r in rows:
        d  = r["delta"]
        dc = "pos" if d > 0 else "neg"
        out.append(
            f"<tr>"
            f"<td>{r['oos_start']}</td><td>{r['oos_end']}</td>"
            f"{_cc(r['ret_a'])}{_cc(r['ret_b'])}"
            f'<td class="{dc}">{d:+.3f}%</td>'
            f"<td>{r['n_a']}</td><td>{r['n_b']}</td>"
            f"</tr>"
        )
    return "\n".join(out)


def _imp_rows(imp_df, n=25):
    out = []
    max_imp = float(imp_df["importance"].max()) or 1.0
    for _, row in imp_df.head(n).iterrows():
        bar_w = int(row["importance"] / max_imp * 100)
        is_hmm = row["feature"].startswith("hmm_")
        cls = ' class="hmm-feat"' if is_hmm else ""
        out.append(
            f'<tr{cls}>'
            f'<td>{row["feature"]}</td>'
            f'<td><div class="bar" style="width:{bar_w}%;"></div></td>'
            f'<td>{row["importance"]:.1f}</td>'
            f'</tr>'
        )
    return "\n".join(out)


def _sig_win_rows(window_stats_a, window_stats_b):
    """Per-window regime distribution from window_stats."""
    out = []
    for wa, wb in zip(window_stats_a, window_stats_b):
        out.append(
            f"<tr>"
            f"<td>{wa['oos_start']}</td>"
            f"<td>{wa['val_acc']}%</td><td>{wa['n_gated']}/{wa['n_sig_oos']}</td>"
            f"<td>{wb['val_acc']}%</td><td>{wb['n_gated']}/{wb['n_sig_oos']}</td>"
            f"</tr>"
        )
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("══ HMM Gate Comparison Report ═══\n")

    # ── [1] Load data ─────────────────────────────────────────────────────────
    print("[1/6] Loading data from 2020-01 …")
    raw = fetch_extended_data(
        start_year=START_YEAR, start_month=START_MONTH,
        fetch_flow=True, fetch_15m=True, fetch_1m=False,
    )
    tf_data = {}
    for tf in ["1W", "1D", "4H", "1H", "15M"]:
        df = raw.get(tf, pd.DataFrame())
        tf_data[tf] = add_indicators(df) if not df.empty and len(df) > 20 else df

    df_1h   = tf_data["1H"]
    df_15m  = tf_data["15M"]
    oi_df   = generate_oi(tf_data["1D"]["close"])
    funding = generate_funding(tf_data["1D"]["close"])
    print(f"  1H: {len(df_1h):,} bars  ({df_1h.index[0].date()} → {df_1h.index[-1].date()})")

    # ── [2] Signals + feature matrix ─────────────────────────────────────────
    print("[2/6] Building signals & features …")
    raw_sig = build_signal_matrix(
        tf_data=tf_data, oi_df=oi_df, funding=funding,
        premium_1h=None, df_15m=df_15m, df_1m=None,
    )
    signals = apply_filters(raw_sig, SESSION_CFG)
    feat_df = build_feature_matrix(tf_data, signals, base_tf="1H", include_smc=False)
    print(f"  Signal bars: {(signals['signal']!=0).sum():,}  Features: {feat_df.shape[1]}")

    # ── [3] Baseline backtest (no gate) ─────────────────────────────────────
    bt_base = run_backtest(df_1h, signals)
    r_b, dd_b, cal_b, sh_b, nt_b = _kpi(bt_base)

    # ── [4] Config A: WF gate without HMM ───────────────────────────────────
    print("[3/6] Config A — WF gate without HMM …")
    res_a = walk_forward_binary_gate(
        df_1h, signals, feat_df,
        gate_threshold=0.50, use_feat_sel=True, verbose=True, hmm_df=None,
    )
    bt_a = run_gated_backtest(df_1h, signals, res_a)
    r_a, dd_a, cal_a, sh_a, nt_a = _kpi(bt_a)

    # ── [5] Config B: WF gate WITH HMM ───────────────────────────────────────
    print("[4/6] Config B — WF gate WITH HMM (per-window fit) …")
    res_b = walk_forward_binary_gate(
        df_1h, signals, feat_df,
        gate_threshold=0.50, use_feat_sel=True, verbose=True, hmm_df=df_1h,
    )
    bt_b = run_gated_backtest(df_1h, signals, res_b)
    r_b2, dd_b2, cal_b2, sh_b2, nt_b2 = _kpi(bt_b)

    # ── [6] Per-window stats ──────────────────────────────────────────────────
    print("[5/6] Computing per-window comparison …")
    windows = _wf_windows(df_1h.index)
    per_win = _per_window_stats(bt_a, bt_b, windows, df_1h)
    n_win   = len(per_win)
    n_better = sum(1 for r in per_win if r["delta"] > 0)

    rets_a = np.array([r["ret_a"] for r in per_win])
    rets_b = np.array([r["ret_b"] for r in per_win])
    delta  = rets_b - rets_a
    delta_mean = float(delta.mean())
    delta_std  = float(delta.std(ddof=1))
    t_stat = delta_mean / (delta_std / np.sqrt(n_win)) if delta_std > 0 else 0.0
    p_val = float(st.t.sf(t_stat, df=n_win - 1))

    per_win_json = json.dumps(per_win)

    # ── [7] HTML ──────────────────────────────────────────────────────────────
    print("[6/6] Generating HTML …")
    n_hmm_better_cls = "pos" if n_better > n_win // 2 else "neg"
    cal_a_cls = "pos" if cal_a > 0.5 else "neg"
    cal_b_cls = "pos" if cal_b2 > 0.5 else "neg"
    p_cls     = "pos" if p_val < 0.05 else "neg"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>HMM Gate — BTCUSDT 2020–2026</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:#0d1117;color:#c9d1d9;font-family:'Segoe UI',sans-serif;padding:24px;}}
  h1{{color:#58a6ff;margin-bottom:6px;font-size:1.6rem;}}
  h2{{color:#8b949e;font-size:1.1rem;margin:28px 0 12px;border-bottom:1px solid #21262d;padding-bottom:6px;}}
  h3{{color:#c9d1d9;font-size:0.95rem;margin:16px 0 8px;}}
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
  .pos{{color:#3fb950;}} .neg{{color:#f85149;}}
  .badge{{display:inline-block;padding:3px 10px;border-radius:12px;font-size:0.78rem;font-weight:600;}}
  .badge.pos{{background:#1a4731;color:#3fb950;}} .badge.neg{{background:#3d1f1f;color:#f85149;}}
  .stat-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:14px;}}
  .stat-box{{background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:14px;text-align:center;}}
  .stat-val{{font-size:1.4rem;font-weight:700;}} .stat-lbl{{font-size:0.75rem;color:#8b949e;margin-top:4px;}}
  .note{{background:#0f2a1e;border:1px solid #238636;border-radius:6px;padding:10px 14px;margin:12px 0;font-size:0.82rem;color:#3fb950;}}
  .warn{{background:#2d2007;border:1px solid #6e5502;border-radius:6px;padding:10px 14px;margin:12px 0;font-size:0.82rem;color:#e3b341;}}
  .bar{{height:8px;background:#3fb950;border-radius:4px;}}
  .hmm-feat td:first-child{{color:#79c0ff;font-weight:600;}}
  canvas{{max-height:300px;}}
</style>
</head>
<body>

<h1>HMM Regime Features — BTCUSDT WF ML Gate</h1>
<p class="subtitle">
  {START_YEAR}-{START_MONTH:02d} &rarr; 2026-05 &nbsp;&middot;&nbsp;
  {n_win} WF windows &nbsp;&middot;&nbsp; 75 vs 80 features &nbsp;&middot;&nbsp;
  GaussianHMM 3-state (bear / sideways / bull)
</p>


<!-- ── 0. Summary KPIs ──────────────────────────────────────────────────── -->
<h2>0. Full-Period KPIs (2020–2026)</h2>
<div class="card">
<div class="stat-grid">
  <div class="stat-box">
    <div class="stat-val {'pos' if r_b>=0 else 'neg'}">{r_b:+.1f}%</div>
    <div class="stat-lbl">Baseline Return</div>
  </div>
  <div class="stat-box">
    <div class="stat-val {cal_a_cls}">{cal_a:.3f}</div>
    <div class="stat-lbl">Calmar — Config A (no HMM)</div>
  </div>
  <div class="stat-box">
    <div class="stat-val {cal_b_cls}">{cal_b2:.3f}</div>
    <div class="stat-lbl">Calmar — Config B (HMM)</div>
  </div>
  <div class="stat-box">
    <div class="stat-val {n_hmm_better_cls}">{n_better}/{n_win}</div>
    <div class="stat-lbl">Windows HMM &gt; no-HMM</div>
  </div>
  <div class="stat-box">
    <div class="stat-val {p_cls}">p={p_val:.3f}</div>
    <div class="stat-lbl">Paired t-test (one-tailed)</div>
  </div>
</div>
</div>


<!-- ── 1. Side-by-side metrics ──────────────────────────────────────────── -->
<h2>1. Config A vs Config B — Full Sample</h2>
<div class="card">
<table>
  <thead><tr>
    <th>Config</th><th>Return%</th><th>Max DD%</th>
    <th>Calmar</th><th>Sharpe</th><th>Trades</th>
  </tr></thead>
  <tbody>
    <tr><td>Baseline (no gate)</td>
      {_cc(r_b)}{_cc(-dd_b,False)}<td>{cal_b:.3f}</td><td>{sh_b:.3f}</td><td>{nt_b}</td></tr>
    <tr><td><b>A — WF Gate (no HMM)</b></td>
      {_cc(r_a)}{_cc(-dd_a,False)}<td class="{cal_a_cls}">{cal_a:.3f}</td><td>{sh_a:.3f}</td><td>{nt_a}</td></tr>
    <tr><td><b>B — WF Gate + HMM</b></td>
      {_cc(r_b2)}{_cc(-dd_b2,False)}<td class="{cal_b_cls}">{cal_b2:.3f}</td><td>{sh_b2:.3f}</td><td>{nt_b2}</td></tr>
  </tbody>
</table>
</div>


<!-- ── 2. Per-window comparison ─────────────────────────────────────────── -->
<h2>2. Per-Window OOS Returns — A vs B ({n_win} windows, HMM wins {n_better}/{n_win})</h2>
<div class="card">
<div class="grid-2">
<div style="overflow-x:auto;max-height:480px;">
<table>
  <thead><tr>
    <th>Start</th><th>End</th>
    <th>A ret%</th><th>B ret%</th><th>Delta%</th>
    <th>Trades A</th><th>Trades B</th>
  </tr></thead>
  <tbody>{_win_rows(per_win)}</tbody>
</table>
</div>
<div>
  <canvas id="winChart"></canvas>
</div>
</div>
</div>

<script>
const winData = {per_win_json};
new Chart(document.getElementById('winChart').getContext('2d'), {{
  type: 'bar',
  data: {{
    labels: winData.map(r => r.oos_start),
    datasets: [
      {{
        label: 'A (no HMM)',
        data: winData.map(r => r.ret_a),
        backgroundColor: winData.map(r => r.ret_a >= 0 ? '#3fb95044' : '#f8514944'),
        borderColor:     winData.map(r => r.ret_a >= 0 ? '#3fb950' : '#f85149'),
        borderWidth: 1,
      }},
      {{
        label: 'B (HMM)',
        data: winData.map(r => r.ret_b),
        backgroundColor: winData.map(r => r.ret_b >= 0 ? '#58a6ff44' : '#ff7b7244'),
        borderColor:     winData.map(r => r.ret_b >= 0 ? '#58a6ff' : '#ff7b72'),
        borderWidth: 1,
      }},
    ],
  }},
  options: {{
    responsive: true, animation: false,
    plugins: {{ legend: {{ labels: {{ color: '#c9d1d9' }} }} }},
    scales: {{
      x: {{ ticks: {{ color: '#8b949e', maxRotation: 45, font: {{ size: 9 }} }}, grid: {{ display: false }} }},
      y: {{ ticks: {{ color: '#8b949e', callback: v => v+'%' }}, grid: {{ color: '#21262d' }} }},
    }},
  }},
}});
</script>


<!-- ── 3. Window-level gate comparison ──────────────────────────────────── -->
<h2>3. Gate Pass Rate — A vs B per Window</h2>
<div class="card" style="overflow-x:auto;max-height:480px;">
<table>
  <thead><tr>
    <th>OOS Start</th>
    <th>A val_acc</th><th>A gated/signals</th>
    <th>B val_acc</th><th>B gated/signals</th>
  </tr></thead>
  <tbody>{_sig_win_rows(res_a.window_stats, res_b.window_stats)}</tbody>
</table>
</div>


<!-- ── 4. Feature importance (Config B) ─────────────────────────────────── -->
<h2>4. Feature Importance — Config B (top-25, HMM features highlighted in blue)</h2>
<div class="card grid-2">
<div>
<table>
  <thead><tr><th>Feature</th><th>Importance (bar)</th><th>Score</th></tr></thead>
  <tbody>{_imp_rows(res_b.importances, n=25)}</tbody>
</table>
</div>
<div>
<canvas id="impChart"></canvas>
</div>
</div>

<script>
const impData = {json.dumps(res_b.importances.head(25).to_dict(orient="records"))};
new Chart(document.getElementById('impChart').getContext('2d'), {{
  type: 'bar',
  data: {{
    labels: impData.map(r => r.feature),
    datasets: [{{
      label: 'Importance',
      data: impData.map(r => r.importance),
      backgroundColor: impData.map(r => r.feature.startsWith('hmm_') ? '#79c0ff88' : '#3fb95066'),
      borderColor:     impData.map(r => r.feature.startsWith('hmm_') ? '#79c0ff' : '#3fb950'),
      borderWidth: 1,
    }}],
  }},
  options: {{
    indexAxis: 'y',
    responsive: true, animation: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }},
      y: {{ ticks: {{ color: '#c9d1d9', font: {{ size: 10 }} }}, grid: {{ display: false }} }},
    }},
  }},
}});
</script>


<!-- ── 5. Interpretation ─────────────────────────────────────────────────── -->
<h2>5. Interpretation</h2>
<div class="card">
  <h3>HMM regime state distribution</h3>
  <p style="font-size:0.85rem;color:#8b949e;margin-bottom:10px;">
    3-state GaussianHMM fit per window on [log_return, log_vol_24h].
    States ordered by mean log-return: 0=bear, 1=sideways, 2=bull.
    Re-trained every 2 months on the 6-month training window — no lookahead.
  </p>

  <h3 style="margin-top:14px;">What to look for</h3>
  <ul style="font-size:0.83rem;color:#8b949e;padding-left:18px;line-height:1.7;">
    <li><b>hmm_state</b> or <b>hmm_prob_*</b> appear in top-20 → regime is genuinely predictive after warmup</li>
    <li>B Calmar &gt; A Calmar → HMM features help the gate differentiate profitable from unprofitable windows</li>
    <li>If B &asymp; A → adding regime context doesn't add signal beyond what 75 existing features already capture</li>
    <li>If B &lt; A → HMM features consume feature-selection budget without contributing signal (noise)</li>
  </ul>

  <h3 style="margin-top:14px;">Statistical test</h3>
  <p style="font-size:0.83rem;color:#8b949e;">
    Paired one-tailed t-test: H&#8320; = per-window(B) &le; per-window(A).
    t={t_stat:.3f} &nbsp; p={p_val:.4f} &nbsp; n={n_win} windows.
    {"<span class='pos'>&#10003; PASS &mdash; HMM significantly improves per-window returns (p&lt;0.05)</span>" if p_val < 0.05 else
     "<span class='neg'>&#9888; MARGINAL &mdash; HMM delta not statistically significant at &alpha;=0.05</span>"}
  </p>
</div>

</body>
</html>"""

    out_path = Path("reports/report_hmm_gate.html")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"  → {out_path}")

    print(f"\n══ Results ════════════════════════════════════════════════════")
    print(f"  Baseline:     ret={r_b:+.1f}%  Calmar={cal_b:.3f}  Sharpe={sh_b:.3f}  Trades={nt_b}")
    print(f"  A (no HMM):   ret={r_a:+.1f}%  Calmar={cal_a:.3f}  Sharpe={sh_a:.3f}  Trades={nt_a}")
    print(f"  B (HMM):      ret={r_b2:+.1f}%  Calmar={cal_b2:.3f}  Sharpe={sh_b2:.3f}  Trades={nt_b2}")
    print(f"  HMM wins {n_better}/{n_win} windows  paired-t p={p_val:.4f}")
    if cal_b2 > cal_a:
        print(f"  ✓ HMM improves Calmar: {cal_a:.3f} → {cal_b2:.3f} (+{cal_b2-cal_a:.3f})")
    else:
        print(f"  ✗ HMM does not improve Calmar: {cal_a:.3f} → {cal_b2:.3f} ({cal_b2-cal_a:+.3f})")
    print("══════════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
