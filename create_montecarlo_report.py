"""
create_montecarlo_report.py
────────────────────────────
Monte Carlo simulation on actual OOS trades.

Strategies tested
─────────────────
  1. Wyckoff WF-optimised    (1H, n_range=24) — OOS stitched trades
  2. Wyckoff WF-opt + Vol    (vol_target=0.20) — OOS stitched trades
  3. Baseline composite      (fixed vol 0.20)  — full-period trades (reference)

Simulation procedure
────────────────────
For each strategy's trade list (net_pnl per trade):
  1. Randomly permute trade order  N_SIM = 10 000 times
  2. Compute equity curve for each permutation
  3. Record: final equity, total return %, max drawdown %, Calmar ratio
  4. Report distribution: percentile table + fan chart + histograms

This answers:
  • How much of the result is due to trade ordering (luck)?
  • What is the realistic range of outcomes?
  • What is P(ruin), P(positive), P(DD > 20%)  under this edge?

Output → reports/report_montecarlo.html

Usage
─────
  python create_montecarlo_report.py
"""
from __future__ import annotations

import base64
import io
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors

from src.strategy.data_fetcher  import fetch_extended_data, generate_oi, generate_funding
from src.strategy.indicators    import add_indicators
from src.strategy.signals       import build_signal_matrix
from src.strategy.optimizer     import apply_filters, ScenarioConfig
from src.strategy.engine        import run_backtest, INIT_CAP
from src.strategy.wyckoff       import build_wyckoff_signals

START_YEAR  = 2020
START_MONTH = 1
N_SIM       = 10_000
BATCH       = 1_000        # sims processed at once (memory control)

_BASELINE_CFG = ScenarioConfig(
    "Session 08-21", session_hours=(8, 21),
    long_threshold=3.0, short_threshold=-3.0,
)


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward helpers (from wyckoff report)
# ─────────────────────────────────────────────────────────────────────────────

def _wf_windows(index: pd.DatetimeIndex,
                train_m: int = 6, oos_m: int = 2, step_m: int = 2) -> list:
    start, end = index[0], index[-1]
    windows, cur = [], start
    while True:
        tr_end = cur + pd.DateOffset(months=train_m)
        oo_s   = tr_end
        oo_e   = oo_s + pd.DateOffset(months=oos_m)
        if oo_e > end:
            break
        windows.append((cur, tr_end, oo_s, oo_e))
        cur = cur + pd.DateOffset(months=step_m)
    return windows


_SL_GRID  = [0.5, 1.0, 1.5, 2.0, 2.5]
_TP1_GRID = [1.0, 1.5, 2.0, 3.0]


def _best_is_params(df_is, sig_is, vol_target=None, min_trades=5):
    best_sl, best_tp1, best_metric = 2.0, 2.0, -np.inf
    any_valid = False
    for sl in _SL_GRID:
        for tp1 in _TP1_GRID:
            try:
                bt = run_backtest(df_is, sig_is,
                                  atr_sl_override=sl, atr_tp1_override=tp1,
                                  vol_target=vol_target)
            except Exception:
                continue
            trd = bt.get("trades", pd.DataFrame())
            if not isinstance(trd, pd.DataFrame) or len(trd) < min_trades:
                continue
            cal    = float(bt["kpis"].get("calmar", 0.0))
            ret    = float(bt["kpis"].get("total_return", 0.0))
            metric = cal if cal > 0 else ret
            any_valid = True
            if metric > best_metric:
                best_metric = metric
                best_sl, best_tp1 = sl, tp1
    return (best_sl, best_tp1) if any_valid else (2.0, 2.0)


def collect_oos_trades(df_1h, signals, windows, vol_target=None) -> np.ndarray:
    """Run WF and collect all OOS trade net_pnl values."""
    all_pnl: list[float] = []
    for (tr_s, tr_e, oo_s, oo_e) in windows:
        df_is  = df_1h.loc[(df_1h.index >= tr_s) & (df_1h.index < tr_e)]
        sig_is = signals.loc[(signals.index >= tr_s) & (signals.index < tr_e)]
        df_oos  = df_1h.loc[(df_1h.index >= oo_s) & (df_1h.index < oo_e)]
        sig_oos = signals.loc[(signals.index >= oo_s) & (signals.index < oo_e)]
        if len(df_is) < 50 or len(df_oos) < 5:
            continue
        sl, tp1 = _best_is_params(df_is, sig_is, vol_target=vol_target)
        try:
            bt = run_backtest(df_oos, sig_oos,
                              atr_sl_override=sl, atr_tp1_override=tp1,
                              vol_target=vol_target)
        except Exception:
            continue
        trd = bt.get("trades", pd.DataFrame())
        if isinstance(trd, pd.DataFrame) and not trd.empty:
            all_pnl.extend(trd["net_pnl"].tolist())
    return np.array(all_pnl, dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo core (vectorised, batched)
# ─────────────────────────────────────────────────────────────────────────────

def monte_carlo(trade_pnls: np.ndarray,
                initial_capital: float = INIT_CAP,
                n_sim: int = N_SIM,
                batch: int = BATCH,
                ) -> dict:
    """
    Randomly permute *trade_pnls* n_sim times.

    Returns dict with keys:
      total_ret   : array (n_sim,) — final total return %
      max_dd      : array (n_sim,) — max drawdown %
      calmar      : array (n_sim,)
      final_eq    : array (n_sim,)
      pct_equity  : array (7, n_trades+1) — equity percentile bands [5,10,25,50,75,90,95]
      actual_eq   : array (n_trades+1)    — equity from original trade order
    """
    n = len(trade_pnls)
    rng = np.random.default_rng(seed=42)

    all_ret = np.empty(n_sim, dtype=float)
    all_dd  = np.empty(n_sim, dtype=float)
    all_cal = np.empty(n_sim, dtype=float)
    all_feq = np.empty(n_sim, dtype=float)

    # Accumulate sample paths for percentile fan (first 2000 sims)
    pct_paths: list[np.ndarray] = []
    pct_sample = min(n_sim, 2000)

    for start in range(0, n_sim, batch):
        bs = min(batch, n_sim - start)
        # Random permutation indices: (bs, n)
        idx = rng.permuted(np.tile(np.arange(n), (bs, 1)), axis=1)
        pnl_mat = trade_pnls[idx]           # (bs, n)

        # Equity curves: (bs, n+1)
        cum = np.cumsum(pnl_mat, axis=1)
        eq  = np.empty((bs, n + 1), dtype=float)
        eq[:, 0]  = initial_capital
        eq[:, 1:] = initial_capital + cum

        all_feq[start:start+bs] = eq[:, -1]
        all_ret[start:start+bs] = (eq[:, -1] / initial_capital - 1.0) * 100

        peak = np.maximum.accumulate(eq, axis=1)
        dd   = (eq - peak) / peak
        all_dd[start:start+bs]  = -dd.min(axis=1) * 100

        stored = start + bs
        if stored <= pct_sample:
            pct_paths.append(eq)

    all_cal = np.where(all_dd > 0.001, all_ret / all_dd, 0.0)

    # Compute equity percentile bands from stored paths
    all_paths = np.vstack(pct_paths)                # (pct_sample, n+1)
    pct_equity = np.percentile(all_paths, [5, 10, 25, 50, 75, 90, 95], axis=0)

    # Actual equity curve (original trade order)
    actual_eq = np.full(n + 1, initial_capital)
    for i, p in enumerate(trade_pnls):
        actual_eq[i + 1] = actual_eq[i] + p

    return {
        "total_ret":  all_ret,
        "max_dd":     all_dd,
        "calmar":     all_cal,
        "final_eq":   all_feq,
        "pct_equity": pct_equity,
        "actual_eq":  actual_eq,
        "n_trades":   n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib chart helpers
# ─────────────────────────────────────────────────────────────────────────────

_BG   = "#0f1117"
_CARD = "#12151f"
_GRID = "#1e2130"
_TEXT = "#e0e0e0"
_SUB  = "#90a4ae"


def _style_ax(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(_CARD)
    ax.tick_params(colors=_SUB, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(_GRID)
    ax.grid(color=_GRID, linewidth=0.5, linestyle="--")
    if title:   ax.set_title(title,   color=_TEXT, fontsize=10, pad=8)
    if xlabel:  ax.set_xlabel(xlabel, color=_SUB,  fontsize=8)
    if ylabel:  ax.set_ylabel(ylabel, color=_SUB,  fontsize=8)


def _to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"


def _plot_fan(mc: dict, label: str, color: str) -> str:
    n = mc["n_trades"]
    x = np.arange(n + 1)
    p = mc["pct_equity"]          # (7, n+1)
    act = mc["actual_eq"]
    cap = act[0]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor(_BG)

    alpha_fills = [0.12, 0.20, 0.35]
    labels_fill = ["5–95 %", "10–90 %", "25–75 %"]
    pairs       = [(0, 6), (1, 5), (2, 4)]
    for (lo, hi), alpha, lbl in zip(pairs, alpha_fills, labels_fill):
        ax.fill_between(x, p[lo], p[hi], alpha=alpha, color=color, label=lbl)
    ax.plot(x, p[3], color=color, linewidth=2.0, label="Median (p50)")
    ax.plot(x, act, color="white", linewidth=1.5, linestyle="--", label="Actual OOS")
    ax.axhline(cap, color=_SUB, linewidth=0.7, linestyle=":")

    _style_ax(ax, f"Equity Fan — {label}", "Trade #", "Equity (USDT)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v/1000:.0f}k"))
    ax.legend(facecolor=_CARD, edgecolor=_GRID, labelcolor=_TEXT,
              fontsize=8, loc="upper left")
    fig.tight_layout(pad=1.2)
    return _to_b64(fig)


def _plot_hist(data: np.ndarray, actual: float, label: str,
               xlabel: str, color: str, vline_label: str = "Actual") -> str:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor(_BG)

    data_range = data.max() - data.min()
    bins = 80 if data_range > 0.01 else 1
    ax.hist(data, bins=bins, color=color, alpha=0.75, edgecolor="none")
    ax.axvline(actual, color="white", linewidth=1.8, linestyle="--",
               label=f"{vline_label}: {actual:.1f}")
    pct5  = np.percentile(data, 5)
    pct95 = np.percentile(data, 95)
    ax.axvline(pct5,  color="#f44336", linewidth=1.0, linestyle=":",
               label=f"p5: {pct5:.1f}")
    ax.axvline(pct95, color="#4caf50", linewidth=1.0, linestyle=":",
               label=f"p95: {pct95:.1f}")

    _style_ax(ax, f"{label} — {xlabel} distribution", xlabel, "Frequency")
    ax.legend(facecolor=_CARD, edgecolor=_GRID, labelcolor=_TEXT, fontsize=7)
    fig.tight_layout(pad=1.2)
    return _to_b64(fig)


def _plot_scatter(mc: dict, label: str, color: str) -> str:
    ret = mc["total_ret"]
    dd  = mc["max_dd"]
    cal = mc["calmar"]
    actual_ret = (mc["actual_eq"][-1] / mc["actual_eq"][0] - 1.0) * 100
    peak = np.maximum.accumulate(mc["actual_eq"])
    actual_dd  = abs(((mc["actual_eq"] - peak) / peak).min()) * 100

    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor(_BG)

    # Colour by Calmar
    vmin, vmax = np.percentile(cal, 5), np.percentile(cal, 95)
    sc = ax.scatter(dd, ret, c=cal, cmap="RdYlGn", s=2, alpha=0.35,
                    vmin=vmin, vmax=vmax, rasterized=True)
    cb = fig.colorbar(sc, ax=ax, shrink=0.85)
    cb.ax.tick_params(colors=_SUB, labelsize=7)
    cb.set_label("Calmar", color=_SUB, fontsize=8)

    ax.scatter(actual_dd, actual_ret, color="white", s=80, zorder=5,
               marker="*", label="Actual OOS")
    ax.axhline(0, color=_GRID, linewidth=0.7)

    _style_ax(ax, f"Return vs Max-DD scatter — {label}", "Max Drawdown %", "Total Return %")
    ax.legend(facecolor=_CARD, edgecolor=_GRID, labelcolor=_TEXT, fontsize=8)
    fig.tight_layout(pad=1.2)
    return _to_b64(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Statistics summary
# ─────────────────────────────────────────────────────────────────────────────

def _stats(mc: dict) -> dict:
    ret = mc["total_ret"]
    dd  = mc["max_dd"]
    cal = mc["calmar"]
    act_eq  = mc["actual_eq"]
    act_ret = (act_eq[-1] / act_eq[0] - 1.0) * 100
    peak    = np.maximum.accumulate(act_eq)
    act_dd  = abs(((act_eq - peak) / peak).min()) * 100
    act_cal = act_ret / act_dd if act_dd > 0.001 else 0.0

    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    return {
        "n_trades":   mc["n_trades"],
        "n_sim":      len(ret),
        "actual_ret": round(float(act_ret), 2),
        "actual_dd":  round(float(act_dd),  2),
        "actual_cal": round(float(act_cal), 3),
        "p_positive": round(float((ret > 0).mean() * 100), 1),
        "p_ruin":     round(float((dd > 30).mean() * 100), 1),
        "p_dd_lt20":  round(float((dd < 20).mean() * 100), 1),
        "exp_ret":    round(float(ret.mean()), 2),
        "exp_dd":     round(float(dd.mean()),  2),
        "exp_cal":    round(float(cal.mean()), 3),
        "ret_pcts":   {p: round(float(np.percentile(ret, p)), 2) for p in pcts},
        "dd_pcts":    {p: round(float(np.percentile(dd,  p)), 2) for p in pcts},
        "cal_pcts":   {p: round(float(np.percentile(cal, p)), 3) for p in pcts},
    }


# ─────────────────────────────────────────────────────────────────────────────
# HTML builder
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
body{font-family:system-ui,sans-serif;background:#0f1117;color:#e0e0e0;margin:0;padding:24px}
h1{font-size:1.6rem;color:#fff;border-bottom:2px solid #2196f3;padding-bottom:8px}
h2{font-size:1.1rem;color:#90caf9;margin-top:36px}
h3{font-size:.95rem;color:#b0bec5;margin-top:18px}
p.meta{color:#546e7a;font-size:.8rem;margin-top:4px}
table{border-collapse:collapse;width:100%;margin-top:10px;font-size:.82rem}
th{background:#1e2130;color:#90caf9;padding:7px 10px;text-align:center;
   border-bottom:2px solid #2196f3}
td{padding:5px 10px;border-bottom:1px solid #1e2130;text-align:center}
td:first-child{text-align:left}
tr:hover td{background:#1a1e2e}
.pos{color:#4caf50;font-weight:600}
.neg{color:#f44336;font-weight:600}
.warn{color:#ff9800;font-weight:600}
.card{background:#12151f;border:1px solid #1e2130;border-radius:8px;
      padding:16px;margin-top:16px}
.strat-section{border-top:2px solid #2196f3;margin-top:48px;padding-top:12px}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}
.charts-3{display:grid;grid-template-columns:2fr 1fr 1fr;gap:12px;margin-top:12px}
img{width:100%;border-radius:6px}
.kpi-row{display:flex;gap:20px;flex-wrap:wrap;margin:12px 0}
.kpi{background:#12151f;border:1px solid #1e2130;border-radius:6px;
     padding:10px 16px;min-width:120px}
.kpi-label{color:#546e7a;font-size:.75rem;text-transform:uppercase}
.kpi-val{font-size:1.3rem;font-weight:700;margin-top:4px}
.scroll{overflow-x:auto}
"""


def _pnl_cls(v: float) -> str:
    return "pos" if v > 0 else ("neg" if v < 0 else "")


def _kpi_box(label: str, val: str, cls: str = "") -> str:
    return (f'<div class="kpi"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-val {cls}">{val}</div></div>')


def _pct_table(s: dict) -> str:
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    hdr  = "".join(f"<th>p{p}</th>" for p in pcts)

    def _row(label, key, fmt=":.1f", neg_bad=True):
        cells = ""
        for p in pcts:
            v = s[key][p]
            if neg_bad:
                cls = _pnl_cls(v) if label == "Return %" else ""
            else:
                cls = ""
            cells += f'<td class="{cls}">{v:{fmt[1:]}}</td>'
        return f"<tr><td>{label}</td>{cells}</tr>"

    return (
        '<div class="scroll"><table>'
        f'<thead><tr><th>Metric</th>{hdr}</tr></thead>'
        '<tbody>'
        + _row("Return %",     "ret_pcts", ":.1f")
        + _row("Max DD %",     "dd_pcts",  ":.1f")
        + _row("Calmar",       "cal_pcts", ":.3f")
        + '</tbody></table></div>'
    )


def _summary_table(strats: list[dict]) -> str:
    def _row(s):
        st = s["stats"]
        rc = _pnl_cls(st["actual_ret"])
        cc = "pos" if st["actual_cal"] > 0.5 else ("warn" if st["actual_cal"] > 0 else "neg")
        return (
            f'<tr>'
            f'<td>{s["label"]}</td>'
            f'<td>{st["n_trades"]}</td>'
            f'<td class="{rc}">{st["actual_ret"]:+.1f}%</td>'
            f'<td class="neg">{st["actual_dd"]:.1f}%</td>'
            f'<td class="{cc}">{st["actual_cal"]:.3f}</td>'
            f'<td>{st["exp_ret"]:+.1f}%</td>'
            f'<td class="neg">{st["exp_dd"]:.1f}%</td>'
            f'<td class="{_pnl_cls(st["exp_cal"])}">{st["exp_cal"]:.3f}</td>'
            f'<td>{st["p_positive"]:.0f}%</td>'
            f'<td class="neg">{st["p_ruin"]:.0f}%</td>'
            f'<td>{st["p_dd_lt20"]:.0f}%</td>'
            f'</tr>'
        )
    rows = "".join(_row(s) for s in strats)
    return (
        '<div class="scroll"><table><thead><tr>'
        '<th>Strategy</th><th>N Trades</th>'
        '<th>Actual Return</th><th>Actual DD</th><th>Actual Calmar</th>'
        '<th>MC Mean Return</th><th>MC Mean DD</th><th>MC Mean Calmar</th>'
        '<th>P(+Return)</th><th>P(DD&gt;30%)</th><th>P(DD&lt;20%)</th>'
        '</tr></thead>'
        f'<tbody>{rows}</tbody></table></div>'
    )


def build_html(strats: list[dict], elapsed: float) -> str:
    summary = _summary_table(strats)

    sections = ""
    for s in strats:
        st  = s["stats"]
        rc  = _pnl_cls(st["actual_ret"])
        cc  = "pos" if st["actual_cal"] > 0.5 else "warn"
        pct = _pct_table(st)

        sections += f"""
<div class="strat-section">
<h2>{s["label"]}</h2>
<p class="meta">
  {st["n_trades"]} OOS trades · {st["n_sim"]:,} Monte Carlo simulations ·
  Actual: ret={st["actual_ret"]:+.1f}%  DD={st["actual_dd"]:.1f}%  Calmar={st["actual_cal"]:.3f}
</p>

<div class="kpi-row">
  {_kpi_box("Actual Return",   f'{st["actual_ret"]:+.1f}%', rc)}
  {_kpi_box("Actual Max DD",   f'{st["actual_dd"]:.1f}%',   "neg")}
  {_kpi_box("Actual Calmar",   f'{st["actual_cal"]:.3f}',   cc)}
  {_kpi_box("MC Mean Return",  f'{st["exp_ret"]:+.1f}%',    _pnl_cls(st["exp_ret"]))}
  {_kpi_box("MC Mean DD",      f'{st["exp_dd"]:.1f}%',      "neg")}
  {_kpi_box("P(+Return)",      f'{st["p_positive"]:.0f}%',  "pos" if st["p_positive"]>50 else "neg")}
  {_kpi_box("P(DD>30%)",       f'{st["p_ruin"]:.0f}%',      "pos" if st["p_ruin"]<10 else "neg")}
  {_kpi_box("P(DD<20%)",       f'{st["p_dd_lt20"]:.0f}%',   "pos" if st["p_dd_lt20"]>70 else "warn")}
</div>

<div class="charts-3">
  <img src="{s["fan_img"]}"  alt="equity fan">
  <img src="{s["ret_img"]}"  alt="return dist">
  <img src="{s["dd_img"]}"   alt="dd dist">
</div>
<div class="charts">
  <img src="{s["scat_img"]}" alt="scatter">
  <div class="card">
    <h3>Percentile Distribution</h3>
    {pct}
    <p class="meta" style="margin-top:8px">
      p5 return = {st["ret_pcts"][5]:+.1f}% · p50 = {st["ret_pcts"][50]:+.1f}% · p95 = {st["ret_pcts"][95]:+.1f}%<br>
      p5 DD = {st["dd_pcts"][5]:.1f}% · p50 DD = {st["dd_pcts"][50]:.1f}% · p95 DD = {st["dd_pcts"][95]:.1f}%
    </p>
  </div>
</div>
</div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Monte Carlo — BTCUSDT Wyckoff</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Monte Carlo Simulation — BTCUSDT OOS Trades</h1>
<p class="meta">
{N_SIM:,} random permutations per strategy · Initial capital: {INIT_CAP:,.0f} USDT ·
Period: 2020-2026 · WF: 6m IS / 2m OOS / step 2m · Generated in {elapsed:.0f}s
</p>
<p class="meta">
Each simulation randomly reorders the actual OOS trade sequence.
This isolates positive expectancy (constant across permutations) from
sequencing luck (which changes). The actual result vs. the MC distribution
shows whether the realised outcome is typical or an outlier.
</p>

<h2>Cross-Strategy Summary</h2>
{summary}

{sections}

<h2>Interpretation Guide</h2>
<div class="card">
<p><strong>Fan chart</strong>: shaded bands show 5-95%, 10-90%, 25-75% equity ranges
across all 10,000 simulations. The white dashed line is the actual realised equity path
(original trade ordering). If it's near the median (p50), luck played little role.</p>
<p><strong>P(+Return)</strong>: fraction of simulations ending in profit. If &gt;50%, the
strategy has positive expectancy regardless of trade ordering.</p>
<p><strong>P(DD&gt;30%)</strong>: probability of hitting a 30% drawdown at some point.
Low values mean the strategy is resilient even under adverse orderings.</p>
<p><strong>Scatter (Return vs Max-DD)</strong>: each dot is one simulation, coloured by
Calmar (green = high, red = low). The star ★ marks the actual OOS outcome.
A star in the dense green region means the result is typical; an outlier means luck
(good or bad) contributed significantly.</p>
</div>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("══ Monte Carlo Report ══\n")
    t0 = time.time()

    # ── [1] Load data ─────────────────────────────────────────────────────────
    print("[1/4] Loading data …")
    raw = fetch_extended_data(
        start_year=START_YEAR, start_month=START_MONTH,
        fetch_flow=True, fetch_15m=False, fetch_1m=False,
    )
    tf_data: dict = {}
    for tf in ["1W", "1D", "4H", "1H"]:
        df = raw.get(tf, pd.DataFrame())
        tf_data[tf] = add_indicators(df) if not df.empty and len(df) > 20 else df
    df_1h = tf_data["1H"]
    df_1d = tf_data["1D"]
    print(f"  1H: {len(df_1h):,} bars  "
          f"({df_1h.index[0].date()} → {df_1h.index[-1].date()})")

    windows = _wf_windows(df_1h.index)

    # ── [2] Collect OOS trades for each strategy ───────────────────────────────
    print("\n[2/4] Collecting OOS trades …")

    # A. Wyckoff WF-opt (no vol)
    print("  A: Wyckoff WF-opt (1H, n_range=24) …")
    wy_sig = build_wyckoff_signals(df_1h, n_range=24, session_hours=(8, 21))
    pnl_wy = collect_oos_trades(df_1h, wy_sig, windows, vol_target=None)
    print(f"     {len(pnl_wy)} OOS trades  "
          f"cumulative P&L={pnl_wy.sum():+,.0f} USDT")

    # B. Wyckoff WF-opt + vol sizing
    print("  B: Wyckoff WF-opt + Vol 0.20 …")
    pnl_wv = collect_oos_trades(df_1h, wy_sig, windows, vol_target=0.20)
    print(f"     {len(pnl_wv)} OOS trades  "
          f"cumulative P&L={pnl_wv.sum():+,.0f} USDT")

    # C. Baseline composite + vol sizing (full-period, reference)
    print("  C: Baseline composite + Vol 0.20 (full period) …")
    oi_df   = generate_oi(df_1d["close"])
    funding = generate_funding(df_1d["close"])
    raw_sig = build_signal_matrix(
        tf_data=tf_data, oi_df=oi_df, funding=funding,
        premium_1h=None, df_15m=None, df_1m=None,
    )
    base_sig = apply_filters(raw_sig, _BASELINE_CFG)
    bt_base  = run_backtest(df_1h, base_sig, vol_target=0.20)
    trd_base = bt_base.get("trades", pd.DataFrame())
    pnl_base = (trd_base["net_pnl"].to_numpy(float)
                if isinstance(trd_base, pd.DataFrame) and not trd_base.empty
                else np.array([]))
    print(f"     {len(pnl_base)} trades  "
          f"cumulative P&L={pnl_base.sum():+,.0f} USDT")

    # ── [3] Monte Carlo ───────────────────────────────────────────────────────
    print(f"\n[3/4] Running Monte Carlo ({N_SIM:,} sims × 3 strategies) …")

    strat_inputs = [
        ("Wyckoff WF-opt (1H)",            pnl_wy,   "#4caf50"),
        ("Wyckoff WF-opt + Vol 0.20 (1H)", pnl_wv,   "#2196f3"),
        ("Baseline + Vol 0.20 (full period)", pnl_base, "#ff9800"),
    ]

    strats_out: list[dict] = []
    for label, pnls, color in strat_inputs:
        if len(pnls) < 5:
            print(f"  [{label}] too few trades — skipping")
            continue
        print(f"  [{label}] {len(pnls)} trades …", end=" ", flush=True)
        t1 = time.time()
        mc = monte_carlo(pnls, n_sim=N_SIM)
        print(f"done in {time.time()-t1:.1f}s")

        st = _stats(mc)
        print(f"    actual ret={st['actual_ret']:+.1f}%  DD={st['actual_dd']:.1f}%  "
              f"Cal={st['actual_cal']:.3f}  "
              f"P(+)={st['p_positive']:.0f}%  P(ruin)={st['p_ruin']:.0f}%")

        fan_img  = _plot_fan(mc, label, color)
        ret_img  = _plot_hist(mc["total_ret"], st["actual_ret"],
                              label, "Total Return %", color)
        dd_img   = _plot_hist(mc["max_dd"],    st["actual_dd"],
                              label, "Max Drawdown %", color,
                              vline_label="Actual DD")
        scat_img = _plot_scatter(mc, label, color)

        strats_out.append({
            "label":    label,
            "stats":    st,
            "fan_img":  fan_img,
            "ret_img":  ret_img,
            "dd_img":   dd_img,
            "scat_img": scat_img,
        })

    # ── [4] HTML ──────────────────────────────────────────────────────────────
    print("\n[4/4] Generating HTML …")
    elapsed = time.time() - t0
    html    = build_html(strats_out, elapsed)

    out = Path("reports/report_montecarlo.html")
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"  → {out}  ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
