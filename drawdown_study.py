"""
Drawdown reduction study — 4 risk-management techniques vs baseline.

Techniques tested (combinable):
  1. Notional cap       – max_notional_pct ∈ {0.10, 0.20}
  2. Volatility target  – vol_target ∈ {0.15, 0.20}
  3. Circuit breaker    – dd_halt_pct ∈ {0.15, 0.20}
  4. Strong-signal only – min_score = 8.0

All variants use the extended signal matrix (+ 15m + 1m).
Backtest only (no WFO) for speed; one WFO run for the best combo.
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
from src.strategy.signals      import build_signal_matrix
from src.strategy.engine       import run_backtest, INIT_CAP, _compute_kpis
from src.strategy.optimizer    import SCENARIOS, apply_filters
from src.strategy.walk_forward import run_walk_forward

BG    = "#0d1117"; PANEL = "#161b22"; BORDER = "#30363d"
WHITE = "#e6edf3"; GRAY  = "#8b949e"; GREEN  = "#2ea043"
RED   = "#f85149"; BLUE  = "#58a6ff"; ORANGE = "#f0883e"; GOLD = "#e3b341"
PURPLE= "#bc8cff"

plt.style.use("dark_background")
SCENARIO  = "Session 08-21"
SAVE_PATH = Path("reports/drawdown_study.pdf")
SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data + signals (cached)
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data …")
tf_data = fetch_extended_data(fetch_15m=True, fetch_1m=True)
for tf in tf_data:
    if not tf_data[tf].empty:
        tf_data[tf] = add_indicators(tf_data[tf])

df_1h  = tf_data["1H"]
df_15m = tf_data.get("15M", pd.DataFrame())
df_1m  = tf_data.get("1M",  pd.DataFrame())

funding, _    = fetch_real_funding(tf_data["1H"], tf_data["1D"])
premium_1h, oi_is_real = fetch_real_oi(df_1h)
oi_df = generate_oi(tf_data["1D"]["close"])

print("Building signal matrix …")
signals = build_signal_matrix(
    tf_data, oi_df, funding,
    premium_1h = premium_1h if oi_is_real else None,
    df_15m     = df_15m if not df_15m.empty else None,
    df_1m      = df_1m  if not df_1m.empty  else None,
)
sig_filt = apply_filters(signals, SCENARIOS[SCENARIO])
common   = df_1h.index.intersection(sig_filt.index)
df_bt    = df_1h.reindex(common)
sf_bt    = sig_filt.reindex(common)
print(f"  {len(df_bt):,} bars  [{df_bt.index[0].date()} → {df_bt.index[-1].date()}]\n")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Grid of variants
# ─────────────────────────────────────────────────────────────────────────────
VARIANTS = [
    # (label,                 max_notional, vol_target, dd_halt, min_score)
    ("Baseline",              1.00,  None,  None,  None),
    ("Notional cap 20%",      0.20,  None,  None,  None),
    ("Notional cap 10%",      0.10,  None,  None,  None),
    ("Vol target 20%",        1.00,  0.20,  None,  None),
    ("Vol target 15%",        1.00,  0.15,  None,  None),
    ("Circuit breaker 15%",   1.00,  None,  0.15,  None),
    ("Circuit breaker 20%",   1.00,  None,  0.20,  None),
    ("Strong signals (≥8)",   1.00,  None,  None,  8.0 ),
    ("Cap20 + Vol20",         0.20,  0.20,  None,  None),
    ("Cap20 + CB15",          0.20,  None,  0.15,  None),
    ("Cap20 + Vol20 + CB15",  0.20,  0.20,  0.15,  None),
    ("Cap10 + Vol15 + CB15",  0.10,  0.15,  0.15,  None),
    ("Cap20+Vol20+CB15+Str",  0.20,  0.20,  0.15,  8.0 ),
]

results = []
equity_store: dict[str, pd.Series] = {}

print(f"{'Variant':<28} {'Return':>9} {'Sharpe':>8} {'MaxDD':>8} {'Calmar':>8} {'Trades':>7}")
print("─" * 75)

for label, notional, vol_t, dd_h, score in VARIANTS:
    bt = run_backtest(
        df_bt, sf_bt, INIT_CAP,
        max_notional_pct = notional,
        vol_target       = vol_t,
        dd_halt_pct      = dd_h,
        min_score        = score,
    )
    k = bt["kpis"]
    equity_store[label] = bt["equity"]
    results.append({
        "label":       label,
        "return":      k["total_return"],
        "sharpe":      k["sharpe"],
        "max_dd":      k["max_drawdown"],
        "calmar":      k["calmar"],
        "win_rate":    k["win_rate"],
        "n_trades":    k["n_trades"],
        "equity":      bt["equity"],
        "drawdown":    bt["drawdown"],
        "max_notional":notional,
        "vol_target":  vol_t,
        "dd_halt":     dd_h,
        "min_score":   score,
    })
    print(f"  {label:<26} {k['total_return']*100:>+8.1f}%  "
          f"{k['sharpe']:>7.2f}  {k['max_drawdown']*100:>+7.1f}%  "
          f"{k['calmar']:>7.2f}  {k['n_trades']:>6}")

df_res = pd.DataFrame(results)
print()

# Best combo = best Calmar among multi-tech combos (index ≥ 8)
best_combo = df_res[df_res["label"].str.contains(r"\+")].sort_values("calmar", ascending=False).iloc[0]
best_label = best_combo["label"]
print(f"Best combo by Calmar: '{best_label}'  "
      f"(Calmar={best_combo['calmar']:.2f}  DD={best_combo['max_dd']*100:.1f}%)\n")


# ─────────────────────────────────────────────────────────────────────────────
# 3. WFO on best combo vs baseline
# ─────────────────────────────────────────────────────────────────────────────
print("Running WFO on baseline + best combo …")

def _wfo(max_notional=1.0, vol_target=None, dd_halt=None, min_score=None):
    return run_walk_forward(
        df_1h, signals, scenario_name=SCENARIO,
        backtest_kwargs=dict(
            max_notional_pct=max_notional,
            vol_target=vol_target,
            dd_halt_pct=dd_halt,
            min_score=min_score,
        ),
    )

wf_base = _wfo()
wf_best = _wfo(
    max_notional = best_combo["max_notional"],
    vol_target   = best_combo["vol_target"],
    dd_halt      = best_combo["dd_halt"],
    min_score    = best_combo["min_score"],
)

print(f"\n  Baseline WFO  → {wf_base['pct_profitable']:.0f}% profitable  "
      f"OOS Sharpe {wf_base['full_kpis']['sharpe']:.3f}  "
      f"OOS DD {wf_base['full_kpis']['max_drawdown']*100:.1f}%")
print(f"  Best combo WFO→ {wf_best['pct_profitable']:.0f}% profitable  "
      f"OOS Sharpe {wf_best['full_kpis']['sharpe']:.3f}  "
      f"OOS DD {wf_best['full_kpis']['max_drawdown']*100:.1f}%\n")


# ─────────────────────────────────────────────────────────────────────────────
# 4. PDF report
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = [BLUE, ORANGE, GREEN, RED, GOLD, PURPLE,
           "#ff7eb6", "#00d2ff", "#a8ff3e", "#ffb347", "#f8e71c", "#c471ed", "#12c2e9"]

def _ax(ax, title=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=GRAY, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.grid(True, alpha=0.10, color=GRAY, linestyle="--")
    if title:
        ax.set_title(title, color=WHITE, fontsize=8, pad=4)


print("Generating PDF …")
with PdfPages(SAVE_PATH) as pdf:

    # ── Page 1: KPI comparison bar charts ─────────────────────────────────────
    fig = plt.figure(figsize=(8.27, 11.69), facecolor=BG)
    fig.suptitle("Drawdown Reduction Study — BTCUSDT Strategy",
                 color=WHITE, fontsize=13, y=0.97)

    gs = gridspec.GridSpec(3, 2, hspace=0.55, wspace=0.38,
                           left=0.12, right=0.96, top=0.91, bottom=0.06)

    labels = [r["label"] for r in results]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]

    def _bar_chart(ax, values, title, fmt="{:.1f}", color_fn=None):
        _ax(ax, title)
        c = [GREEN if v >= 0 else RED for v in values] if color_fn == "sign" else colors
        bars = ax.barh(labels, values, color=c, alpha=0.8, height=0.65)
        for bar, v in zip(bars, values):
            ax.text(v + (max(values) - min(values)) * 0.01, bar.get_y() + bar.get_height() / 2,
                    fmt.format(v), va="center", color=WHITE, fontsize=6)
        ax.tick_params(axis="y", labelsize=6)
        ax.invert_yaxis()

    _bar_chart(fig.add_subplot(gs[0, 0]),
               [r["return"] * 100 for r in results], "Total Return (%)", "{:+.1f}%", "sign")
    _bar_chart(fig.add_subplot(gs[0, 1]),
               [r["sharpe"] for r in results], "Sharpe Ratio", "{:.2f}")
    _bar_chart(fig.add_subplot(gs[1, 0]),
               [r["max_dd"] * 100 for r in results], "Max Drawdown (%)", "{:+.1f}%", "sign")
    _bar_chart(fig.add_subplot(gs[1, 1]),
               [r["calmar"] for r in results], "Calmar Ratio", "{:.2f}")
    _bar_chart(fig.add_subplot(gs[2, 0]),
               [r["win_rate"] * 100 for r in results], "Win Rate (%)", "{:.1f}%")
    _bar_chart(fig.add_subplot(gs[2, 1]),
               [r["n_trades"] for r in results], "# Trades", "{:.0f}")

    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)

    # ── Page 2: Equity curves overlay (key variants) ──────────────────────────
    KEY = ["Baseline", "Notional cap 20%", "Vol target 20%",
           "Circuit breaker 15%", best_label]
    KEY = [k for k in KEY if k in equity_store]

    fig2 = plt.figure(figsize=(8.27, 11.69), facecolor=BG)
    fig2.suptitle("Equity Curves — Key Variants", color=WHITE, fontsize=13, y=0.97)
    gs2 = gridspec.GridSpec(2, 1, hspace=0.40, left=0.08, right=0.96,
                             top=0.92, bottom=0.06)

    ax_eq = fig2.add_subplot(gs2[0])
    _ax(ax_eq, "Equity Curve")
    for i, k in enumerate(KEY):
        lw = 2.0 if k == "Baseline" or k == best_label else 1.0
        ls = "-" if k == "Baseline" else "--" if k == best_label else ":"
        ax_eq.plot(equity_store[k].index, equity_store[k].values,
                   color=PALETTE[i], lw=lw, ls=ls, label=k, alpha=0.85)
    ax_eq.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax_eq.legend(fontsize=7, labelcolor=WHITE, framealpha=0.2, loc="upper left")

    # Drawdown overlay
    ax_dd = fig2.add_subplot(gs2[1])
    _ax(ax_dd, "Drawdown")
    for i, k in enumerate(KEY):
        eq = equity_store[k]
        dd = (eq - eq.cummax()) / eq.cummax() * 100
        lw = 1.8 if k == "Baseline" or k == best_label else 1.0
        ax_dd.plot(eq.index, dd.values, color=PALETTE[i], lw=lw, label=k, alpha=0.8)
    ax_dd.axhline(0, color=WHITE, lw=0.5, ls="--")
    ax_dd.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax_dd.legend(fontsize=7, labelcolor=WHITE, framealpha=0.2, loc="lower left")

    pdf.savefig(fig2, facecolor=BG)
    plt.close(fig2)

    # ── Page 3: WFO comparison baseline vs best combo ─────────────────────────
    fig3 = plt.figure(figsize=(8.27, 11.69), facecolor=BG)
    fig3.suptitle(f"WFO — Baseline vs '{best_label}'",
                  color=WHITE, fontsize=12, y=0.97)
    gs3 = gridspec.GridSpec(3, 2, hspace=0.50, wspace=0.38,
                            left=0.08, right=0.96, top=0.91, bottom=0.06)

    # OOS equity
    ax_oos = fig3.add_subplot(gs3[0, :])
    _ax(ax_oos, "OOS Walk-Forward Equity")
    for r, label, color in [(wf_base, "Baseline", BLUE), (wf_best, best_label, ORANGE)]:
        oos = r["combined_equity"]
        if not oos.empty:
            ax_oos.plot(oos.index, oos.values, color=color, lw=1.4, label=label)
    ax_oos.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax_oos.legend(fontsize=8, labelcolor=WHITE, framealpha=0.2)

    # OOS return bars per window
    for col, (r, label, color) in enumerate([(wf_base, "Baseline", BLUE),
                                              (wf_best, best_label, ORANGE)]):
        ax = fig3.add_subplot(gs3[1, col])
        _ax(ax, f"OOS Ret/Window — {label[:20]}")
        wdf = r["windows"]
        if not wdf.empty:
            rets = wdf["OOS Return (%)"].values
            ax.bar(range(len(rets)), rets,
                   color=[GREEN if v >= 0 else RED for v in rets], alpha=0.75, width=0.7)
            ax.axhline(0, color=WHITE, lw=0.7, ls="--")
            ax.axhline(float(np.median(rets)), color=GOLD, lw=1, ls=":",
                       label=f"Median {np.median(rets):+.1f}%")
            ax.legend(fontsize=7, labelcolor=GRAY, framealpha=0.2)
            ax.set_xlabel("Window", color=GRAY, fontsize=7)

    # Summary table
    ax_tbl = fig3.add_subplot(gs3[2, :])
    ax_tbl.set_facecolor(PANEL); ax_tbl.axis("off")
    base_k = wf_base["full_kpis"]; best_k = wf_best["full_kpis"]
    tbl_data = [
        ["IS Total Return",       f"{df_res.loc[0,'return']*100:+.1f}%",
                                  f"{best_combo['return']*100:+.1f}%"],
        ["IS Sharpe",             f"{df_res.loc[0,'sharpe']:.3f}",
                                  f"{best_combo['sharpe']:.3f}"],
        ["IS Max Drawdown",       f"{df_res.loc[0,'max_dd']*100:+.1f}%",
                                  f"{best_combo['max_dd']*100:+.1f}%"],
        ["IS Calmar",             f"{df_res.loc[0,'calmar']:.3f}",
                                  f"{best_combo['calmar']:.3f}"],
        ["OOS Total Return",      f"{base_k['total_return']*100:+.1f}%",
                                  f"{best_k['total_return']*100:+.1f}%"],
        ["OOS Sharpe",            f"{base_k['sharpe']:.3f}",
                                  f"{best_k['sharpe']:.3f}"],
        ["OOS Max Drawdown",      f"{base_k['max_drawdown']*100:+.1f}%",
                                  f"{best_k['max_drawdown']*100:+.1f}%"],
        ["WFO % Profitable",      f"{wf_base['pct_profitable']:.0f}%",
                                  f"{wf_best['pct_profitable']:.0f}%"],
        ["Median OOS Ret",        f"{wf_base['median_oos_ret']:+.2f}%",
                                  f"{wf_best['median_oos_ret']:+.2f}%"],
        ["Consistency",           f"{wf_base['consistency']:+.3f}",
                                  f"{wf_best['consistency']:+.3f}"],
    ]
    tbl = ax_tbl.table(cellText=tbl_data,
                       colLabels=["Metric", "Baseline", f"Best: {best_label[:25]}"],
                       cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(7)
    for (ri, ci), cell in tbl.get_celld().items():
        cell.set_facecolor(BG if ri == 0 else PANEL)
        cell.set_edgecolor(BORDER)
        cell.set_text_props(color=WHITE if ri == 0 else GRAY)

    pdf.savefig(fig3, facecolor=BG)
    plt.close(fig3)

print(f"  PDF saved → {SAVE_PATH}")
print("\nDone.")
