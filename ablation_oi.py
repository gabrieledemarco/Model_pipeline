"""
OI Ablation Test — compares strategy performance with s_oi weight=2 vs weight=0.

Uses cached parquet data (no re-download). Runs:
  1. Full signal matrix (baseline, s_oi weight=2)
  2. Ablated signal matrix (s_oi weight=0)

For each: baseline backtest + walk-forward (6m/2m/2m).
Prints side-by-side comparison table and saves a concise PDF.
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

from src.strategy.data_fetcher   import fetch_extended_data, fetch_real_funding
from src.strategy.indicators     import add_indicators
from src.strategy.signals        import build_signal_matrix, WEIGHTS, LONG_THRESH, SHORT_THRESH
from src.strategy.engine         import run_backtest, INIT_CAP, _compute_kpis
from src.strategy.optimizer      import SCENARIOS, apply_filters
from src.strategy.walk_forward   import run_walk_forward

# ── palette ──────────────────────────────────────────────────────────────────
BG    = "#0d1117"; PANEL = "#161b22"; BORDER = "#30363d"
WHITE = "#e6edf3"; GRAY  = "#8b949e"; GREEN  = "#2ea043"
RED   = "#f85149"; BLUE  = "#58a6ff"; ORANGE = "#f0883e"; GOLD = "#e3b341"

plt.style.use("dark_background")

SCENARIO = "Session 08-21"
SAVE_PATH = Path("reports/ablation_oi_report.pdf")
SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data (uses cache)
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data from cache …")
tf_data = fetch_extended_data()
df_1h   = tf_data["1H"]
df_1d   = tf_data["1D"]

for tf, df in tf_data.items():
    tf_data[tf] = add_indicators(df)

funding_raw, is_real = fetch_real_funding(tf_data["1H"], tf_data["1D"])
print(f"  Funding: {'real' if is_real else 'synthetic'}")

# Synthetic OI (same as pipeline)
from src.strategy.data_fetcher import generate_oi
oi_df = generate_oi(tf_data["1D"]["close"])

df_1h = tf_data["1H"]   # now has indicators
print(f"  Bars: {len(df_1h):,}  [{df_1h.index[0].date()} → {df_1h.index[-1].date()}]")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Build signal matrices — baseline vs ablated
# ─────────────────────────────────────────────────────────────────────────────
def build_signals_with_oi_weight(w: int) -> pd.DataFrame:
    custom_weights = dict(WEIGHTS)
    custom_weights["s_oi"] = w
    sig = build_signal_matrix(tf_data, oi_df, funding_raw)
    # Recompute composite with custom weight
    sig["composite"] = sum(sig[k] * custom_weights[k] for k in custom_weights)
    sig["signal"] = np.where(
        sig["composite"] >= LONG_THRESH, 1,
        np.where(sig["composite"] <= SHORT_THRESH, -1, 0),
    ).astype(int)
    sig["strong"] = (sig["composite"].abs() >= 8.0).astype(int)
    return sig

print("Building signal matrices …")
sig_base   = build_signals_with_oi_weight(2)   # original
sig_ablated = build_signals_with_oi_weight(0)  # OI removed
print("  Done.\n")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Backtest both
# ─────────────────────────────────────────────────────────────────────────────
def run_scenario(signals: pd.DataFrame, label: str) -> dict:
    print(f"─── {label} ───")
    filtered = apply_filters(signals, SCENARIOS[SCENARIO])
    common   = df_1h.index.intersection(filtered.index)
    bt       = run_backtest(df_1h.reindex(common), filtered.reindex(common), INIT_CAP)
    eq       = bt["equity"]
    dd       = bt["drawdown"]
    trades   = bt["trades"]
    kpis     = _compute_kpis(eq, dd, trades, INIT_CAP)
    print(f"  Backtest → Return {kpis['total_return']*100:+.1f}%  "
          f"Sharpe {kpis['sharpe']:.3f}  DD {kpis['max_drawdown']*100:.1f}%  "
          f"Trades {kpis['n_trades']}")
    wf = run_walk_forward(df_1h, signals, scenario_name=SCENARIO)
    return {"label": label, "kpis": kpis, "wf": wf, "eq": eq, "dd": dd, "trades": trades}

r_base    = run_scenario(sig_base,    "Baseline (s_oi weight=2)")
print()
r_ablated = run_scenario(sig_ablated, "Ablated  (s_oi weight=0)")
print()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Comparison table
# ─────────────────────────────────────────────────────────────────────────────
rows = [
    ("Total Return (%)",       f"{r_base['kpis']['total_return']*100:+.2f}%",
                                f"{r_ablated['kpis']['total_return']*100:+.2f}%"),
    ("Sharpe Ratio",           f"{r_base['kpis']['sharpe']:.3f}",
                                f"{r_ablated['kpis']['sharpe']:.3f}"),
    ("Max Drawdown (%)",       f"{r_base['kpis']['max_drawdown']*100:.2f}%",
                                f"{r_ablated['kpis']['max_drawdown']*100:.2f}%"),
    ("Calmar Ratio",           f"{r_base['kpis'].get('calmar',0):.3f}",
                                f"{r_ablated['kpis'].get('calmar',0):.3f}"),
    ("Win Rate (%)",           f"{r_base['kpis']['win_rate']*100:.1f}%",
                                f"{r_ablated['kpis']['win_rate']*100:.1f}%"),
    ("Profit Factor",          f"{r_base['kpis']['profit_factor']:.2f}",
                                f"{r_ablated['kpis']['profit_factor']:.2f}"),
    ("# Trades",               f"{r_base['kpis']['n_trades']}",
                                f"{r_ablated['kpis']['n_trades']}"),
    ("──── Walk-Forward ────", "──────────", "──────────"),
    ("WFO Windows",            f"{r_base['wf']['n_windows']}",
                                f"{r_ablated['wf']['n_windows']}"),
    ("% Profitable Windows",   f"{r_base['wf']['pct_profitable']:.0f}%",
                                f"{r_ablated['wf']['pct_profitable']:.0f}%"),
    ("Median OOS Return (%)",  f"{r_base['wf']['median_oos_ret']:+.2f}%",
                                f"{r_ablated['wf']['median_oos_ret']:+.2f}%"),
    ("OOS Sharpe (combined)",  f"{r_base['wf']['full_kpis'].get('sharpe',0):.3f}",
                                f"{r_ablated['wf']['full_kpis'].get('sharpe',0):.3f}"),
    ("OOS Max DD (%)",         f"{r_base['wf']['full_kpis'].get('max_drawdown',0)*100:.2f}%",
                                f"{r_ablated['wf']['full_kpis'].get('max_drawdown',0)*100:.2f}%"),
    ("OOS Total Return (%)",   f"{r_base['wf']['full_kpis'].get('total_return',0)*100:+.2f}%",
                                f"{r_ablated['wf']['full_kpis'].get('total_return',0)*100:+.2f}%"),
    ("Consistency",            f"{r_base['wf']['consistency']:+.3f}",
                                f"{r_ablated['wf']['consistency']:+.3f}"),
]

print(f"\n{'Metric':<30} {'Baseline (OI=2)':>18} {'Ablated (OI=0)':>18}")
print("─" * 70)
for name, base_val, abl_val in rows:
    print(f"  {name:<28} {base_val:>18} {abl_val:>18}")
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

    # ── Page 1: Equity curves side-by-side ──────────────────────────────────
    fig = plt.figure(figsize=(8.27, 11.69), facecolor=BG)
    fig.suptitle("OI Signal Ablation Test — BTCUSDT Strategy",
                 color=WHITE, fontsize=13, y=0.97)

    gs = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.35,
                           left=0.08, right=0.96, top=0.91, bottom=0.06)

    # Equity comparison (top row, full width)
    ax_eq = fig.add_subplot(gs[0, :])
    _styled_ax(ax_eq, "Full-Dataset Equity Curve: Baseline vs Ablated")
    eq_b = r_base["eq"]
    eq_a = r_ablated["eq"]
    ax_eq.plot(eq_b.index, eq_b.values, color=BLUE,   lw=1.4, label="Baseline (OI=2)")
    ax_eq.plot(eq_a.index, eq_a.values, color=ORANGE, lw=1.4, label="Ablated  (OI=0)", ls="--")
    ax_eq.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax_eq.legend(fontsize=8, labelcolor=WHITE, framealpha=0.2)

    # Drawdown comparison
    ax_dd = fig.add_subplot(gs[1, :])
    _styled_ax(ax_dd, "Drawdown: Baseline vs Ablated")
    dd_b = r_base["dd"]
    dd_a = r_ablated["dd"]
    ax_dd.fill_between(dd_b.index, dd_b.values * 100, 0,
                       color=BLUE, alpha=0.3, label="Baseline")
    ax_dd.fill_between(dd_a.index, dd_a.values * 100, 0,
                       color=ORANGE, alpha=0.3, label="Ablated", linestyle="--")
    ax_dd.plot(dd_b.index, dd_b.values * 100, color=BLUE,   lw=0.8)
    ax_dd.plot(dd_a.index, dd_a.values * 100, color=ORANGE, lw=0.8, ls="--")
    ax_dd.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax_dd.legend(fontsize=8, labelcolor=WHITE, framealpha=0.2)

    # OOS equity comparison (WFO)
    ax_oos = fig.add_subplot(gs[2, :])
    _styled_ax(ax_oos, "OOS Walk-Forward Equity: Baseline vs Ablated")
    oos_b = r_base["wf"]["combined_equity"]
    oos_a = r_ablated["wf"]["combined_equity"]
    if not oos_b.empty:
        ax_oos.plot(oos_b.index, oos_b.values, color=BLUE,   lw=1.4, label="Baseline (OI=2)")
    if not oos_a.empty:
        ax_oos.plot(oos_a.index, oos_a.values, color=ORANGE, lw=1.4, label="Ablated  (OI=0)", ls="--")
    ax_oos.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax_oos.legend(fontsize=8, labelcolor=WHITE, framealpha=0.2)

    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)

    # ── Page 2: KPI comparison table + OOS return bars ──────────────────────
    fig2 = plt.figure(figsize=(8.27, 11.69), facecolor=BG)
    fig2.suptitle("OI Ablation — KPI & Walk-Forward Comparison",
                  color=WHITE, fontsize=13, y=0.97)

    gs2 = gridspec.GridSpec(3, 2, hspace=0.48, wspace=0.38,
                            left=0.08, right=0.96, top=0.90, bottom=0.06)

    # KPI table
    ax_tbl = fig2.add_subplot(gs2[0, :])
    ax_tbl.set_facecolor(PANEL)
    ax_tbl.axis("off")
    ax_tbl.set_title("Key Performance Indicators", color=WHITE, fontsize=9, pad=6)

    tbl_rows  = [r for r in rows if "────" not in r[0]]
    col_labels = ["Metric", "Baseline (OI=2)", "Ablated (OI=0)"]
    cell_text  = [[r[0], r[1], r[2]] for r in tbl_rows]
    tbl = ax_tbl.table(cellText=cell_text, colLabels=col_labels,
                       cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    for (row_i, col_i), cell in tbl.get_celld().items():
        cell.set_facecolor(BG if row_i == 0 else PANEL)
        cell.set_edgecolor(BORDER)
        cell.set_text_props(color=WHITE if row_i == 0 else GRAY)
        if row_i > 0 and col_i > 0:
            txt = cell.get_text().get_text()
            base_txt = cell_text[row_i - 1][1] if col_i == 1 else cell_text[row_i - 1][2]
            try:
                v = float(base_txt.replace("%","").replace("+","").replace("$","").replace(",",""))
                cell.get_text().set_color(GREEN if v > 0 else RED)
            except ValueError:
                pass

    # OOS per-window return bars: baseline
    ax_wb = fig2.add_subplot(gs2[1, 0])
    _styled_ax(ax_wb, "OOS Return/Window — Baseline (OI=2)")
    wdf_b = r_base["wf"]["windows"]
    if not wdf_b.empty:
        rets = wdf_b["OOS Return (%)"].values
        colors_ = [GREEN if r >= 0 else RED for r in rets]
        ax_wb.bar(range(len(rets)), rets, color=colors_, alpha=0.75, width=0.7)
        ax_wb.axhline(0, color=WHITE, lw=0.7, ls="--")
        ax_wb.axhline(float(np.median(rets)), color=GOLD, lw=1, ls=":",
                      label=f"Median {np.median(rets):+.1f}%")
        ax_wb.legend(fontsize=7, labelcolor=GRAY, framealpha=0.2)
        ax_wb.set_xlabel("Window", color=GRAY, fontsize=7)
        ax_wb.set_ylabel("OOS Ret %", color=GRAY, fontsize=7)

    # OOS per-window return bars: ablated
    ax_wa = fig2.add_subplot(gs2[1, 1])
    _styled_ax(ax_wa, "OOS Return/Window — Ablated (OI=0)")
    wdf_a = r_ablated["wf"]["windows"]
    if not wdf_a.empty:
        rets = wdf_a["OOS Return (%)"].values
        colors_ = [GREEN if r >= 0 else RED for r in rets]
        ax_wa.bar(range(len(rets)), rets, color=colors_, alpha=0.75, width=0.7)
        ax_wa.axhline(0, color=WHITE, lw=0.7, ls="--")
        ax_wa.axhline(float(np.median(rets)), color=GOLD, lw=1, ls=":",
                      label=f"Median {np.median(rets):+.1f}%")
        ax_wa.legend(fontsize=7, labelcolor=GRAY, framealpha=0.2)
        ax_wa.set_xlabel("Window", color=GRAY, fontsize=7)
        ax_wa.set_ylabel("OOS Ret %", color=GRAY, fontsize=7)

    # Sharpe per window: both overlaid
    ax_sh = fig2.add_subplot(gs2[2, :])
    _styled_ax(ax_sh, "OOS Sharpe per Window: Baseline vs Ablated")
    if not wdf_b.empty and not wdf_a.empty:
        xs = np.arange(max(len(wdf_b), len(wdf_a)))
        w = 0.35
        sb = wdf_b["Sharpe"].values
        sa = wdf_a["Sharpe"].values[:len(sb)]
        ax_sh.bar(xs[:len(sb)] - w/2, sb, width=w, color=BLUE,   alpha=0.7, label="Baseline (OI=2)")
        ax_sh.bar(xs[:len(sa)] + w/2, sa, width=w, color=ORANGE, alpha=0.7, label="Ablated  (OI=0)")
        ax_sh.axhline(0, color=WHITE, lw=0.7, ls="--")
        ax_sh.axhline(1, color=GREEN, lw=0.7, ls=":", alpha=0.5)
        ax_sh.legend(fontsize=8, labelcolor=WHITE, framealpha=0.2)
        ax_sh.set_xlabel("Window #", color=GRAY, fontsize=7)
        ax_sh.set_ylabel("Sharpe", color=GRAY, fontsize=7)

    pdf.savefig(fig2, facecolor=BG)
    plt.close(fig2)

print(f"  PDF saved → {SAVE_PATH}")
print("\nDone.")
