"""
Comparison: Baseline (8 signals) vs Extended (+ s_15m + s_1m).

Baseline  : 1W, 1D, 4H, 1H, OI-basis, funding, volume, cyclicality
Extended  : baseline + 15m precision entry + 1m microstructure

Uses cached parquet data. Runs backtest + WFO for each variant, then
prints a side-by-side table and saves a PDF comparison report.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).parent))

from src.strategy.data_fetcher import (fetch_extended_data, fetch_real_funding,
                                        fetch_real_oi, generate_oi)
from src.strategy.indicators   import add_indicators
from src.strategy.signals      import build_signal_matrix, WEIGHTS, LONG_THRESH, SHORT_THRESH
from src.strategy.engine       import run_backtest, INIT_CAP, _compute_kpis
from src.strategy.optimizer    import SCENARIOS, apply_filters
from src.strategy.walk_forward import run_walk_forward

BG    = "#0d1117"; PANEL = "#161b22"; BORDER = "#30363d"
WHITE = "#e6edf3"; GRAY  = "#8b949e"; GREEN  = "#2ea043"
RED   = "#f85149"; BLUE  = "#58a6ff"; ORANGE = "#f0883e"; GOLD = "#e3b341"

plt.style.use("dark_background")

SCENARIO  = "Session 08-21"
SAVE_PATH = Path("reports/ablation_15m_1m_report.pdf")
SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data from cache …")
tf_data = fetch_extended_data(fetch_15m=True, fetch_1m=True)
for tf in tf_data:
    if not tf_data[tf].empty:
        tf_data[tf] = add_indicators(tf_data[tf])

df_1h  = tf_data["1H"]
df_15m = tf_data.get("15M", pd.DataFrame())
df_1m  = tf_data.get("1M",  pd.DataFrame())

print(f"  1H : {len(df_1h):,} bars  [{df_1h.index[0].date()} → {df_1h.index[-1].date()}]")
if not df_15m.empty:
    print(f"  15M: {len(df_15m):,} bars  [{df_15m.index[0].date()} → {df_15m.index[-1].date()}]")
if not df_1m.empty:
    print(f"  1M : {len(df_1m):,} bars  [{df_1m.index[0].date()} → {df_1m.index[-1].date()}]")

funding, is_real = fetch_real_funding(tf_data["1H"], tf_data["1D"])
print(f"  Funding: {'real' if is_real else 'synthetic'}")

premium_1h, oi_is_real = fetch_real_oi(df_1h)
oi_df = generate_oi(tf_data["1D"]["close"])
print(f"  OI/Basis: {'real_basis' if oi_is_real else 'synthetic'}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Build signal matrices
# ─────────────────────────────────────────────────────────────────────────────
print("Building signal matrices …")
sig_baseline = build_signal_matrix(
    tf_data, oi_df, funding,
    premium_1h = premium_1h if oi_is_real else None,
    df_15m     = None,
    df_1m      = None,
)
sig_extended = build_signal_matrix(
    tf_data, oi_df, funding,
    premium_1h = premium_1h if oi_is_real else None,
    df_15m     = df_15m if not df_15m.empty else None,
    df_1m      = df_1m  if not df_1m.empty  else None,
)
print("  Done.\n")

# Quick signal count diff
for label, sig in [("Baseline", sig_baseline), ("Extended", sig_extended)]:
    nl = int((sig["signal"] ==  1).sum())
    ns = int((sig["signal"] == -1).sum())
    nf = int((sig["signal"] ==  0).sum())
    print(f"  [{label}] long={nl:,}  short={ns:,}  flat={nf:,}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Backtest + WFO for each variant
# ─────────────────────────────────────────────────────────────────────────────
def run_variant(signals: pd.DataFrame, label: str) -> dict:
    print(f"─── {label} ───")
    filtered = apply_filters(signals, SCENARIOS[SCENARIO])
    common   = df_1h.index.intersection(filtered.index)
    bt       = run_backtest(df_1h.reindex(common), filtered.reindex(common), INIT_CAP)
    kpis     = _compute_kpis(bt["equity"], bt["drawdown"], bt["trades"], INIT_CAP)
    print(f"  Backtest → Return {kpis['total_return']*100:+.1f}%  "
          f"Sharpe {kpis['sharpe']:.3f}  DD {kpis['max_drawdown']*100:.1f}%  "
          f"Trades {kpis['n_trades']}")
    wf = run_walk_forward(df_1h, signals, scenario_name=SCENARIO)
    print(f"  WFO     → {wf['pct_profitable']:.0f}% profitable  "
          f"OOS Sharpe {wf['full_kpis'].get('sharpe', 0):.3f}")
    return {
        "label":  label,
        "kpis":   kpis,
        "wf":     wf,
        "eq":     bt["equity"],
        "dd":     bt["drawdown"],
        "trades": bt["trades"],
    }

r_base = run_variant(sig_baseline, "Baseline (8 signals)")
print()
r_ext  = run_variant(sig_extended, "Extended (+15m +1m)")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Comparison table
# ─────────────────────────────────────────────────────────────────────────────
def _k(r, key, scale=1, fmt="+.2f"):
    v = r["kpis"].get(key, 0) * scale
    return f"{v:{fmt}}"

def _wk(r, key, scale=1, fmt="+.2f"):
    v = r["wf"]["full_kpis"].get(key, 0) * scale
    return f"{v:{fmt}}"

rows = [
    ("Total Return",      _k(r_base,"total_return",100)+"%", _k(r_ext,"total_return",100)+"%"),
    ("Sharpe Ratio",      _k(r_base,"sharpe",1,".3f"),       _k(r_ext,"sharpe",1,".3f")),
    ("Max Drawdown",      _k(r_base,"max_drawdown",100)+"%", _k(r_ext,"max_drawdown",100)+"%"),
    ("Calmar Ratio",      _k(r_base,"calmar",1,".3f"),       _k(r_ext,"calmar",1,".3f")),
    ("Win Rate",          _k(r_base,"win_rate",100,".1f")+"%",_k(r_ext,"win_rate",100,".1f")+"%"),
    ("Profit Factor",     _k(r_base,"profit_factor",1,".2f"),_k(r_ext,"profit_factor",1,".2f")),
    ("# Trades",          str(r_base["kpis"]["n_trades"]),   str(r_ext["kpis"]["n_trades"])),
    ("── Walk-Forward ──","──────────","──────────"),
    ("WFO Windows",       str(r_base["wf"]["n_windows"]),    str(r_ext["wf"]["n_windows"])),
    ("% Profitable Win",  f"{r_base['wf']['pct_profitable']:.0f}%", f"{r_ext['wf']['pct_profitable']:.0f}%"),
    ("Median OOS Ret",    f"{r_base['wf']['median_oos_ret']:+.2f}%",f"{r_ext['wf']['median_oos_ret']:+.2f}%"),
    ("OOS Sharpe",        _wk(r_base,"sharpe",1,".3f"),      _wk(r_ext,"sharpe",1,".3f")),
    ("OOS Max DD",        _wk(r_base,"max_drawdown",100)+"%",_wk(r_ext,"max_drawdown",100)+"%"),
    ("OOS Total Return",  _wk(r_base,"total_return",100)+"%",_wk(r_ext,"total_return",100)+"%"),
    ("Consistency",       f"{r_base['wf']['consistency']:+.3f}",f"{r_ext['wf']['consistency']:+.3f}"),
]

print(f"\n{'Metric':<28} {'Baseline (8 sig)':>18} {'Extended (+15m+1m)':>20}")
print("─" * 70)
for name, bval, eval_ in rows:
    print(f"  {name:<26} {bval:>18} {eval_:>20}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 5. PDF report
# ─────────────────────────────────────────────────────────────────────────────
def _styled_ax(ax, title=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=GRAY, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.grid(True, alpha=0.10, color=GRAY, linestyle="--")
    if title:
        ax.set_title(title, color=WHITE, fontsize=8, pad=4)


print("Generating PDF …")
with PdfPages(SAVE_PATH) as pdf:

    # ── Page 1: Equity + drawdown overlay ────────────────────────────────────
    fig = plt.figure(figsize=(8.27, 11.69), facecolor=BG)
    fig.suptitle("15m + 1m Signal Layers — Baseline vs Extended",
                 color=WHITE, fontsize=13, y=0.97)

    gs = gridspec.GridSpec(3, 1, hspace=0.42, left=0.08, right=0.96,
                           top=0.92, bottom=0.06)

    ax_eq = fig.add_subplot(gs[0])
    _styled_ax(ax_eq, "Full-Dataset Equity Curve")
    ax_eq.plot(r_base["eq"].index, r_base["eq"].values,
               color=BLUE,   lw=1.4, label="Baseline (8 signals)")
    ax_eq.plot(r_ext["eq"].index,  r_ext["eq"].values,
               color=ORANGE, lw=1.4, label="Extended (+15m +1m)", ls="--")
    ax_eq.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax_eq.legend(fontsize=8, labelcolor=WHITE, framealpha=0.2)

    ax_dd = fig.add_subplot(gs[1])
    _styled_ax(ax_dd, "Drawdown")
    ax_dd.fill_between(r_base["dd"].index, r_base["dd"].values * 100, 0,
                       color=BLUE,   alpha=0.3, label="Baseline")
    ax_dd.fill_between(r_ext["dd"].index,  r_ext["dd"].values * 100, 0,
                       color=ORANGE, alpha=0.3, label="Extended")
    ax_dd.plot(r_base["dd"].index, r_base["dd"].values * 100, color=BLUE,   lw=0.8)
    ax_dd.plot(r_ext["dd"].index,  r_ext["dd"].values * 100, color=ORANGE, lw=0.8, ls="--")
    ax_dd.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax_dd.legend(fontsize=8, labelcolor=WHITE, framealpha=0.2)

    ax_oos = fig.add_subplot(gs[2])
    _styled_ax(ax_oos, "OOS Walk-Forward Equity")
    oos_b = r_base["wf"]["combined_equity"]
    oos_e = r_ext["wf"]["combined_equity"]
    if not oos_b.empty:
        ax_oos.plot(oos_b.index, oos_b.values, color=BLUE,   lw=1.4, label="Baseline")
    if not oos_e.empty:
        ax_oos.plot(oos_e.index, oos_e.values, color=ORANGE, lw=1.4, label="Extended", ls="--")
    ax_oos.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax_oos.legend(fontsize=8, labelcolor=WHITE, framealpha=0.2)

    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)

    # ── Page 2: KPI table + OOS window bars + Sharpe bars ────────────────────
    fig2 = plt.figure(figsize=(8.27, 11.69), facecolor=BG)
    fig2.suptitle("15m + 1m Ablation — KPI & Walk-Forward Comparison",
                  color=WHITE, fontsize=13, y=0.97)

    gs2 = gridspec.GridSpec(3, 2, hspace=0.48, wspace=0.38,
                            left=0.08, right=0.96, top=0.90, bottom=0.06)

    # KPI table
    ax_tbl = fig2.add_subplot(gs2[0, :])
    ax_tbl.set_facecolor(PANEL)
    ax_tbl.axis("off")
    ax_tbl.set_title("Key Performance Indicators", color=WHITE, fontsize=9, pad=6)
    tbl_rows   = [r for r in rows if "──" not in r[0]]
    col_labels = ["Metric", "Baseline (8 sig)", "Extended (+15m+1m)"]
    cell_text  = [[r[0], r[1], r[2]] for r in tbl_rows]
    tbl = ax_tbl.table(cellText=cell_text, colLabels=col_labels,
                       cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    for (ri, ci), cell in tbl.get_celld().items():
        cell.set_facecolor(BG if ri == 0 else PANEL)
        cell.set_edgecolor(BORDER)
        cell.set_text_props(color=WHITE if ri == 0 else GRAY)

    # OOS return bars per window
    for col_idx, (r, label, color) in enumerate([
        (r_base, "Baseline", BLUE),
        (r_ext,  "Extended", ORANGE),
    ]):
        ax = fig2.add_subplot(gs2[1, col_idx])
        _styled_ax(ax, f"OOS Ret/Window — {label}")
        wdf = r["wf"]["windows"]
        if not wdf.empty:
            rets = wdf["OOS Return (%)"].values
            colors_ = [GREEN if v >= 0 else RED for v in rets]
            ax.bar(range(len(rets)), rets, color=colors_, alpha=0.75, width=0.7)
            ax.axhline(0, color=WHITE, lw=0.7, ls="--")
            ax.axhline(float(np.median(rets)), color=GOLD, lw=1, ls=":",
                       label=f"Median {np.median(rets):+.1f}%")
            ax.legend(fontsize=7, labelcolor=GRAY, framealpha=0.2)
            ax.set_xlabel("Window", color=GRAY, fontsize=7)
            ax.set_ylabel("OOS Ret %", color=GRAY, fontsize=7)

    # Sharpe per window: both overlaid
    ax_sh = fig2.add_subplot(gs2[2, :])
    _styled_ax(ax_sh, "OOS Sharpe per Window: Baseline vs Extended")
    wdf_b = r_base["wf"]["windows"]
    wdf_e = r_ext["wf"]["windows"]
    if not wdf_b.empty and not wdf_e.empty:
        n = max(len(wdf_b), len(wdf_e))
        xs = np.arange(n)
        w  = 0.35
        sb = wdf_b["Sharpe"].values
        se = wdf_e["Sharpe"].values
        ax_sh.bar(xs[:len(sb)] - w/2, sb, width=w, color=BLUE,   alpha=0.7, label="Baseline")
        ax_sh.bar(xs[:len(se)] + w/2, se, width=w, color=ORANGE, alpha=0.7, label="Extended")
        ax_sh.axhline(0, color=WHITE, lw=0.7, ls="--")
        ax_sh.axhline(1, color=GREEN, lw=0.7, ls=":", alpha=0.5, label="Sharpe=1")
        ax_sh.legend(fontsize=8, labelcolor=WHITE, framealpha=0.2)
        ax_sh.set_xlabel("Window #", color=GRAY, fontsize=7)
        ax_sh.set_ylabel("Sharpe", color=GRAY, fontsize=7)

    # Page 3: signal score distribution comparison
    fig3 = plt.figure(figsize=(8.27, 11.69), facecolor=BG)
    fig3.suptitle("Signal Score Distribution: Baseline vs Extended",
                  color=WHITE, fontsize=13, y=0.97)

    gs3 = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35,
                            left=0.08, right=0.96, top=0.90, bottom=0.06)

    # Composite score distribution
    ax_comp = fig3.add_subplot(gs3[0, :])
    _styled_ax(ax_comp, "Composite Score Distribution")
    ax_comp.hist(sig_baseline["composite"], bins=60, color=BLUE,   alpha=0.5,
                 label="Baseline", density=True)
    ax_comp.hist(sig_extended["composite"], bins=60, color=ORANGE, alpha=0.5,
                 label="Extended", density=True)
    ax_comp.axvline(5,  color=GREEN, lw=1, ls="--", label="Long thresh (+5)")
    ax_comp.axvline(-5, color=RED,   lw=1, ls="--", label="Short thresh (−5)")
    ax_comp.legend(fontsize=8, labelcolor=WHITE, framealpha=0.2)
    ax_comp.set_xlabel("Composite Score", color=GRAY, fontsize=7)
    ax_comp.set_ylabel("Density", color=GRAY, fontsize=7)

    # s_15m and s_1m contribution
    ax_15m = fig3.add_subplot(gs3[1, 0])
    _styled_ax(ax_15m, "s_15m Distribution (15-min signal)")
    ax_15m.hist(sig_extended["s_15m"], bins=9, color=BLUE, alpha=0.7, edgecolor=BORDER)
    ax_15m.set_xlabel("Score", color=GRAY, fontsize=7)
    ax_15m.set_ylabel("Count", color=GRAY, fontsize=7)
    for v, label in [(-2,"−2"),(-1,"−1"),(0,"0"),(1,"+1"),(2,"+2")]:
        cnt = int((sig_extended["s_15m"] == v).sum())
        if cnt:
            ax_15m.text(v, cnt, f"{cnt:,}", ha="center", va="bottom",
                        color=WHITE, fontsize=6)

    ax_1m = fig3.add_subplot(gs3[1, 1])
    _styled_ax(ax_1m, "s_1m Distribution (1-min signal)")
    ax_1m.hist(sig_extended["s_1m"], bins=5, color=ORANGE, alpha=0.7, edgecolor=BORDER)
    ax_1m.set_xlabel("Score", color=GRAY, fontsize=7)
    ax_1m.set_ylabel("Count", color=GRAY, fontsize=7)
    for v, label in [(-1,"−1"),(0,"0"),(1,"+1")]:
        cnt = int((sig_extended["s_1m"] == v).sum())
        if cnt:
            ax_1m.text(v, cnt, f"{cnt:,}", ha="center", va="bottom",
                       color=WHITE, fontsize=6)

    pdf.savefig(fig2, facecolor=BG)
    pdf.savefig(fig3, facecolor=BG)
    plt.close(fig2)
    plt.close(fig3)

print(f"  PDF saved → {SAVE_PATH}")
print("\nDone.")
