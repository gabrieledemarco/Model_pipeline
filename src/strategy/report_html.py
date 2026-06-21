"""
Self-contained HTML report generator for the BTCUSDT strategy.

Produces a single .html file with:
  - Dark theme (same palette as PDF/charts)
  - All charts embedded as inline base64 PNG
  - Interactive tables
  - Sections: Summary, Backtest, Walk-Forward, Monte Carlo, Scenarios, Leverage
"""
from __future__ import annotations

import base64
import io
import datetime
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Palette ──────────────────────────────────────────────────────────────────
BG     = "#0d1117"
PANEL  = "#161b22"
BORDER = "#30363d"
WHITE  = "#e6edf3"
GRAY   = "#8b949e"
GREEN  = "#2ea043"
RED    = "#f85149"
GOLD   = "#e3b341"
BLUE   = "#58a6ff"
PURPLE = "#d2a8ff"
ORANGE = "#f0883e"
TEAL   = "#39d353"

plt.style.use("dark_background")

CHARTS_DIR = Path("reports/charts")
SAVE_DIR   = Path("reports")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=GRAY, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.grid(True, alpha=0.10, color=GRAY, linestyle="--")
    if title:  ax.set_title(title,  color=WHITE, fontsize=9, pad=4)
    if xlabel: ax.set_xlabel(xlabel, color=GRAY,  fontsize=7)
    if ylabel: ax.set_ylabel(ylabel, color=GRAY,  fontsize=7)


def _fig_to_b64(fig, dpi=130) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _png_to_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def _img_tag(b64: str, width="100%") -> str:
    return f'<img src="data:image/png;base64,{b64}" style="width:{width};border-radius:6px">'


# ─────────────────────────────────────────────────────────────────────────────
# Chart generators
# ─────────────────────────────────────────────────────────────────────────────

def _chart_equity_drawdown(bt: dict) -> str:
    eq = bt["equity"];  dd = bt["drawdown"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), facecolor=BG,
                                    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
    _ax(ax1, "Equity Curve")
    ax1.plot(eq.index, eq.values, color=BLUE, lw=1.5)
    ax1.fill_between(eq.index, bt["init_cap"], eq.values,
                     where=(eq.values >= bt["init_cap"]), color=GREEN, alpha=0.12)
    ax1.fill_between(eq.index, bt["init_cap"], eq.values,
                     where=(eq.values < bt["init_cap"]), color=RED, alpha=0.15)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"${v:,.0f}"))
    ax1.set_xticklabels([])

    _ax(ax2, "", ylabel="Drawdown %")
    ax2.fill_between(dd.index, dd.values * 100, 0, color=RED, alpha=0.55)
    ax2.plot(dd.index, dd.values * 100, color=RED, lw=0.8)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.0f}%"))
    fig.patch.set_facecolor(BG)
    return _fig_to_b64(fig)


def _chart_wfo_equity(wf: dict) -> str:
    eq     = wf["combined_equity"]
    win_df = wf["windows"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), facecolor=BG,
                                    gridspec_kw={"height_ratios": [2, 1], "hspace": 0.15})
    _ax(ax1, "Walk-Forward — Chained OOS Equity")
    if not eq.empty:
        ax1.plot(eq.index, eq.values, color=TEAL, lw=1.6)
        ax1.fill_between(eq.index, eq.iloc[0], eq.values,
                         where=(eq.values >= eq.iloc[0]), color=GREEN, alpha=0.12)
        ax1.fill_between(eq.index, eq.iloc[0], eq.values,
                         where=(eq.values < eq.iloc[0]), color=RED, alpha=0.15)
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"${v:,.0f}"))
        for _, row in win_df.iterrows():
            ax1.axvspan(pd.Timestamp(row["Train End"]),
                        pd.Timestamp(row["OOS End"]), color=BLUE, alpha=0.04)

    _ax(ax2, "OOS Return per Window (%)")
    if not win_df.empty:
        rets = win_df["OOS Return (%)"].values
        cols = [GREEN if r >= 0 else RED for r in rets]
        xs   = np.arange(len(rets))
        ax2.bar(xs, rets, color=cols, alpha=0.8, width=0.7)
        ax2.axhline(0, color=WHITE, lw=0.7, ls="--")
        ax2.axhline(float(np.median(rets)), color=GOLD, lw=1, ls=":",
                    label=f"Median {np.median(rets):+.1f}%")
        ax2.set_xticks(xs)
        labels = [f"W{int(w)}" for w in win_df.index]
        ax2.set_xticklabels(labels, fontsize=6, color=GRAY)
        ax2.legend(fontsize=7, labelcolor=GRAY, framealpha=0.2)
    fig.patch.set_facecolor(BG)
    return _fig_to_b64(fig)


def _chart_wfo_stats(wf: dict) -> str:
    win_df = wf["windows"]
    trades = wf.get("all_oos_trades", pd.DataFrame())
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), facecolor=BG)
    fig.subplots_adjust(wspace=0.35, left=0.06, right=0.97)

    # Return dist
    _ax(axes[0], "OOS Return Distribution")
    if not win_df.empty:
        rets = win_df["OOS Return (%)"].values
        axes[0].hist(rets, bins=min(12,len(rets)), color=BLUE, alpha=0.75, edgecolor=BORDER)
        axes[0].axvline(float(np.median(rets)), color=GOLD, lw=1.5, ls="--",
                        label=f"Median {np.median(rets):+.1f}%")
        axes[0].axvline(0, color=RED, lw=0.8, ls=":")
        axes[0].legend(fontsize=7, labelcolor=GRAY, framealpha=0.2)

    # Sharpe per window
    _ax(axes[1], "OOS Sharpe per Window")
    if not win_df.empty:
        sharpes = win_df["Sharpe"].values
        cols = [GREEN if s >= 0 else RED for s in sharpes]
        axes[1].bar(range(len(sharpes)), sharpes, color=cols, alpha=0.75, width=0.7)
        axes[1].axhline(0, color=WHITE, lw=0.7, ls="--")
        axes[1].axhline(1.0, color=GREEN, lw=0.7, ls=":", alpha=0.5)

    # Trade PnL dist
    _ax(axes[2], "OOS Trade PnL Distribution")
    if not trades.empty and "net_pnl" in trades.columns:
        pnls = trades["net_pnl"].values
        axes[2].hist(pnls[pnls > 0], bins=25, color=GREEN, alpha=0.65, label=f"Win ({(pnls>0).sum()})")
        axes[2].hist(pnls[pnls <= 0], bins=25, color=RED,   alpha=0.65, label=f"Loss ({(pnls<=0).sum()})")
        axes[2].axvline(0, color=WHITE, lw=0.7)
        axes[2].legend(fontsize=7, labelcolor=GRAY, framealpha=0.2)

    fig.patch.set_facecolor(BG)
    return _fig_to_b64(fig)


def _chart_mc(mc_result: dict) -> str:
    sims  = mc_result.get("sim_equity", pd.DataFrame())
    orig  = mc_result.get("original_equity")
    final = mc_result.get("final_returns", np.array([]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), facecolor=BG)
    fig.subplots_adjust(wspace=0.3, left=0.06, right=0.97)

    _ax(ax1, "Monte Carlo — Equity Paths (1000 sims)")
    if not sims.empty:
        # paths are indexed by trade number (0..n), use integer index for x-axis
        xs = np.arange(len(sims))
        for col in sims.columns[:300]:
            ax1.plot(xs, sims[col].values, color=BLUE, lw=0.3, alpha=0.12)
        p5  = sims.quantile(0.05, axis=1).values
        p50 = sims.quantile(0.50, axis=1).values
        p95 = sims.quantile(0.95, axis=1).values
        ax1.plot(xs, p5,  color=RED,   lw=1.2, label="p5")
        ax1.plot(xs, p50, color=WHITE, lw=1.5, label="p50")
        ax1.plot(xs, p95, color=GREEN, lw=1.2, label="p95")
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"${v:,.0f}"))
        ax1.set_xlabel("Trade #", color=GRAY, fontsize=7)
        ax1.legend(fontsize=7, labelcolor=GRAY, framealpha=0.2)

    _ax(ax2, "Final Return Distribution")
    if len(final) > 0:
        ax2.hist(final * 100, bins=50, color=BLUE, alpha=0.75, edgecolor=BORDER)
        ax2.axvline(float(np.percentile(final, 5)) * 100,  color=RED,   lw=1.2, ls="--", label="p5")
        ax2.axvline(float(np.percentile(final, 50)) * 100, color=WHITE, lw=1.5, ls="--", label="p50")
        ax2.axvline(float(np.percentile(final, 95)) * 100, color=GREEN, lw=1.2, ls="--", label="p95")
        ax2.axvline(0, color=GOLD, lw=1, ls=":")
        ax2.set_xlabel("Final Return (%)", color=GRAY, fontsize=7)
        ax2.legend(fontsize=7, labelcolor=GRAY, framealpha=0.2)

    fig.patch.set_facecolor(BG)
    return _fig_to_b64(fig)


def _chart_scenarios(scenario_kpis: dict) -> str:
    names   = list(scenario_kpis.keys())
    sharpes = [scenario_kpis[n].get("sharpe", 0) for n in names]
    returns = [scenario_kpis[n].get("total_return", 0) * 100 for n in names]
    dds     = [abs(scenario_kpis[n].get("max_drawdown", 0)) * 100 for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), facecolor=BG)
    fig.subplots_adjust(wspace=0.38, left=0.06, right=0.97)
    xs = np.arange(len(names))
    w  = 0.6

    _ax(axes[0], "Sharpe Ratio by Scenario")
    axes[0].bar(xs, sharpes, color=[GREEN if s > 0 else RED for s in sharpes], width=w, alpha=0.8)
    axes[0].set_xticks(xs); axes[0].set_xticklabels(names, rotation=30, ha="right", fontsize=6, color=GRAY)

    _ax(axes[1], "Total Return (%) by Scenario")
    axes[1].bar(xs, returns, color=[GREEN if r > 0 else RED for r in returns], width=w, alpha=0.8)
    axes[1].set_xticks(xs); axes[1].set_xticklabels(names, rotation=30, ha="right", fontsize=6, color=GRAY)
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.0f}%"))

    _ax(axes[2], "Max Drawdown (%) by Scenario")
    axes[2].bar(xs, dds, color=RED, width=w, alpha=0.7)
    axes[2].set_xticks(xs); axes[2].set_xticklabels(names, rotation=30, ha="right", fontsize=6, color=GRAY)
    axes[2].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.0f}%"))

    fig.patch.set_facecolor(BG)
    return _fig_to_b64(fig)


def _chart_leverage(lev_df) -> str:
    if lev_df is None or (hasattr(lev_df, "empty") and lev_df.empty):
        return ""
    df = lev_df if isinstance(lev_df, pd.DataFrame) else pd.DataFrame()
    if df.empty:
        return ""

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), facecolor=BG)
    fig.subplots_adjust(wspace=0.35, left=0.06, right=0.97)

    sharpe_col = "Sharpe" if "Sharpe" in df.columns else None
    ret_col    = "Total Ret (%)" if "Total Ret (%)" in df.columns else None

    for ax, col, title in [
        (axes[0], sharpe_col, "Sharpe by Config"),
        (axes[1], ret_col,    "Return (%) by Config"),
    ]:
        _ax(ax, title)
        if col and col in df.columns:
            vals   = df[col].values[:24]  # cap at 24 bars
            labels = [f"{r.get('Method','')[:3]}/R{r.get('Pct (%)','')}/L{r.get('Leverage','')}"
                      for _, r in df.head(24).iterrows()]
            cols_  = [GREEN if v > 0 else RED for v in vals]
            ax.bar(range(len(vals)), vals, color=cols_, alpha=0.8, width=0.7)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=5, color=GRAY)

    fig.patch.set_facecolor(BG)
    return _fig_to_b64(fig)


# ─────────────────────────────────────────────────────────────────────────────
# HTML building blocks
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
:root {
  --bg:#0d1117; --panel:#161b22; --border:#30363d;
  --white:#e6edf3; --gray:#8b949e;
  --green:#2ea043; --red:#f85149; --blue:#58a6ff;
  --gold:#e3b341; --orange:#f0883e; --purple:#d2a8ff;
}
* { box-sizing:border-box; margin:0; padding:0; }
body { background:var(--bg); color:var(--white); font-family:'Segoe UI',system-ui,sans-serif;
       font-size:13px; line-height:1.5; }
/* Nav */
nav { position:sticky; top:0; background:#010409; border-bottom:1px solid var(--border);
      padding:10px 24px; display:flex; gap:20px; align-items:center; z-index:100; flex-wrap:wrap; }
nav .brand { font-weight:700; font-size:15px; color:var(--blue); margin-right:12px; }
nav a { color:var(--gray); text-decoration:none; font-size:12px; padding:3px 8px;
        border-radius:4px; transition:background 0.15s; }
nav a:hover { background:var(--panel); color:var(--white); }
/* Layout */
.container { max-width:1400px; margin:0 auto; padding:24px 20px; }
section { margin-bottom:48px; }
h2 { font-size:18px; color:var(--white); border-left:3px solid var(--blue);
     padding-left:10px; margin-bottom:18px; }
h3 { font-size:13px; color:var(--gray); margin-bottom:10px; text-transform:uppercase;
     letter-spacing:.05em; }
/* Cards */
.cards { display:flex; flex-wrap:wrap; gap:12px; margin-bottom:20px; }
.card { background:var(--panel); border:1px solid var(--border); border-radius:8px;
        padding:14px 20px; flex:1; min-width:130px; text-align:center; }
.card .val { font-size:22px; font-weight:700; margin-bottom:4px; }
.card .lbl { font-size:11px; color:var(--gray); }
.green { color:var(--green); }
.red   { color:var(--red);   }
.blue  { color:var(--blue);  }
.gold  { color:var(--gold);  }
/* Tables */
.tbl-wrap { overflow-x:auto; border-radius:8px; border:1px solid var(--border); }
table { width:100%; border-collapse:collapse; font-size:12px; }
thead tr { background:#010409; }
th { padding:8px 12px; color:var(--gray); font-weight:600; text-align:right;
     border-bottom:1px solid var(--border); white-space:nowrap; }
th:first-child { text-align:left; }
td { padding:7px 12px; border-bottom:1px solid var(--border); text-align:right;
     color:var(--white); }
td:first-child { text-align:left; color:var(--gray); }
tr:last-child td { border-bottom:none; }
tr:nth-child(even) { background:rgba(255,255,255,.02); }
.pos { color:var(--green); } .neg { color:var(--red); }
/* Chart */
.chart { margin-bottom:16px; }
.chart img { max-width:100%; border-radius:6px; }
/* Badge */
.badge { display:inline-block; padding:2px 8px; border-radius:12px; font-size:11px;
         font-weight:600; }
.badge-green { background:rgba(46,160,67,.2); color:var(--green); border:1px solid var(--green); }
.badge-blue  { background:rgba(88,166,255,.15); color:var(--blue); border:1px solid var(--blue); }
/* Two-col */
.two-col { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
@media(max-width:800px){ .two-col{ grid-template-columns:1fr; } }
/* Footer */
footer { border-top:1px solid var(--border); padding:20px; text-align:center;
         color:var(--gray); font-size:11px; margin-top:40px; }
"""


def _kpi_card(label: str, value: str, color_class: str = "blue") -> str:
    return f'<div class="card"><div class="val {color_class}">{value}</div><div class="lbl">{label}</div></div>'


def _color_val(v: float, fmt: str = "{:.2f}", pos_good=True) -> str:
    cls = "pos" if (v > 0) == pos_good else "neg"
    return f'<span class="{cls}">{fmt.format(v)}</span>'


def _table(headers: list, rows: list, col_colors: dict = None) -> str:
    """Render an HTML table. col_colors = {col_idx: fn(value)->css_class}"""
    ths = "".join(f"<th>{h}</th>" for h in headers)
    body = ""
    for row in rows:
        tds = ""
        for i, cell in enumerate(row):
            cls = ""
            if col_colors and i in col_colors:
                try:
                    cls = f' class="{col_colors[i](float(str(cell).replace("%","").replace("+","").replace("$","").replace(",","") or 0))}"'
                except Exception:
                    pass
            tds += f"<td{cls}>{cell}</td>"
        body += f"<tr>{tds}</tr>"
    return f'<div class="tbl-wrap"><table><thead><tr>{ths}</tr></thead><tbody>{body}</tbody></table></div>'


def _signed(v: float, pct=False, decimals=2) -> str:
    s = f"+{v:.{decimals}f}" if v >= 0 else f"{v:.{decimals}f}"
    return s + ("%" if pct else "")


def _color_signed(v: float, pct=False, decimals=2, pos_good=True) -> str:
    cls = "pos" if (v >= 0) == pos_good else "neg"
    return f'<span class="{cls}">{_signed(v, pct, decimals)}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# Section builders
# ─────────────────────────────────────────────────────────────────────────────

def _section_summary(kpis: dict, oi_source: str, n_bars: int,
                     date_range: str, scenario: str) -> str:
    tr   = kpis["total_return"] * 100
    sh   = kpis["sharpe"]
    dd   = kpis["max_drawdown"] * 100
    cal  = kpis.get("calmar", 0)
    wr   = kpis["win_rate"] * 100
    pf   = kpis["profit_factor"]
    nt   = kpis["n_trades"]
    fin  = kpis["final_equity"]

    oi_badge = (f'<span class="badge badge-green">✓ Real basis (premiumIndex)</span>'
                if "real" in oi_source else
                f'<span class="badge badge-blue">⚠ Synthetic OI</span>')

    cards = "".join([
        _kpi_card("Total Return",   f"{tr:+.1f}%",  "green" if tr>0 else "red"),
        _kpi_card("Sharpe Ratio",   f"{sh:.3f}",    "blue"),
        _kpi_card("Max Drawdown",   f"{dd:.1f}%",   "red"),
        _kpi_card("Calmar Ratio",   f"{cal:.2f}",   "blue"),
        _kpi_card("Win Rate",       f"{wr:.1f}%",   "green" if wr>50 else "red"),
        _kpi_card("Profit Factor",  f"{pf:.2f}",    "green" if pf>1 else "red"),
        _kpi_card("# Trades",       f"{nt:,}",      "gold"),
        _kpi_card("Final Equity",   f"${fin:,.0f}", "blue"),
    ])

    return f"""
<section id="summary">
  <h2>Executive Summary</h2>
  <p style="color:var(--gray);margin-bottom:16px">
    {date_range} &nbsp;·&nbsp; <strong style="color:var(--white)">{n_bars:,}</strong> 1H bars
    &nbsp;·&nbsp; Scenario: <strong style="color:var(--blue)">{scenario}</strong>
    &nbsp;·&nbsp; OI signal: {oi_badge}
  </p>
  <div class="cards">{cards}</div>
</section>"""


def _section_backtest(bt: dict, kpis: dict) -> str:
    eq_b64 = _chart_equity_drawdown(bt)

    # Trade table (last 20)
    trades = bt["trades"]
    tbl_html = ""
    if not trades.empty:
        show = trades.tail(20)[["entry_ts","exit_ts","direction","entry_price",
                                 "exit_price","net_pnl","duration_h"]].copy()
        show["entry_ts"] = show["entry_ts"].dt.strftime("%Y-%m-%d %H:%M")
        show["exit_ts"]  = show["exit_ts"].dt.strftime("%Y-%m-%d %H:%M")
        show["entry_price"] = show["entry_price"].map("${:,.0f}".format)
        show["exit_price"]  = show["exit_price"].map("${:,.0f}".format)
        show["duration_h"]  = show["duration_h"].map("{:.0f}h".format)
        rows = []
        for _, r in show.iterrows():
            pnl   = r["net_pnl"]
            pnl_s = f'<span class="{"pos" if pnl>0 else "neg"}">${pnl:+,.2f}</span>'
            dir_s = f'<span class="{"pos" if r["direction"]=="long" else "red"}">{r["direction"]}</span>'
            rows.append([r["entry_ts"], r["exit_ts"], dir_s,
                         r["entry_price"], r["exit_price"], pnl_s, r["duration_h"]])
        tbl_html = _table(["Entry","Exit","Dir","Entry $","Exit $","Net PnL","Duration"], rows)

    # Stats summary
    avg_w = kpis.get("avg_win",  0)
    avg_l = kpis.get("avg_loss", 0)
    exp   = kpis.get("expectancy", 0)
    sortino = kpis.get("sortino", 0)
    dur   = kpis.get("avg_duration_h", 0)

    stat_rows = [
        ["Total Return",    f"{kpis['total_return']*100:+.2f}%"],
        ["Sharpe Ratio",    f"{kpis['sharpe']:.3f}"],
        ["Sortino Ratio",   f"{sortino:.3f}"],
        ["Calmar Ratio",    f"{kpis.get('calmar',0):.3f}"],
        ["Max Drawdown",    f"{kpis['max_drawdown']*100:.2f}%"],
        ["Win Rate",        f"{kpis['win_rate']*100:.1f}%"],
        ["Profit Factor",   f"{kpis['profit_factor']:.2f}"],
        ["# Trades",        f"{kpis['n_trades']:,}"],
        ["Avg Win",         f"${avg_w:+,.2f}"],
        ["Avg Loss",        f"${avg_l:+,.2f}"],
        ["Expectancy",      f"${exp:+,.2f}"],
        ["Avg Duration",    f"{dur:.0f}h"],
    ]
    stat_tbl = _table(["Metric","Value"], stat_rows)

    # Existing PNG charts
    existing = {
        "C1_equity_curve": "Equity & Drawdown Detail",
        "C2_trade_distribution": "Trade Distribution",
        "C3_mae_mfe": "MAE vs MFE",
        "C7_monthly_pnl_heatmap": "Monthly PnL Heatmap",
        "C6_rolling_sharpe": "Rolling Sharpe",
    }
    png_grid = ""
    for fname, title in existing.items():
        p = CHARTS_DIR / f"{fname}.png"
        if p.exists():
            png_grid += f'<div><h3>{title}</h3>{_img_tag(_png_to_b64(p))}</div>'

    return f"""
<section id="backtest">
  <h2>Backtest Performance — Full Dataset (In-Sample)</h2>
  <div class="two-col" style="margin-bottom:16px">
    <div>{stat_tbl}</div>
    <div class="chart">{_img_tag(eq_b64)}</div>
  </div>
  {png_grid}
  <h3 style="margin-top:20px">Last 20 Trades</h3>
  {tbl_html}
</section>"""


def _section_signals() -> str:
    pairs = [
        ("A1_signal_score_distribution", "Score Distribution"),
        ("A2_alpha_decay",               "Alpha Decay"),
        ("A3_signal_component_correlation", "Component Correlation"),
        ("A4_score_vs_fwd_return",       "Score vs Fwd Return"),
        ("A5_score_quantile_returns",    "Quantile Returns"),
        ("B1_market_overview",           "Market Overview"),
        ("B2_oi_analysis",               "OI / Basis Analysis"),
        ("B3_cyclicality",               "Cyclicality"),
    ]
    html = ""
    for fname, title in pairs:
        p = CHARTS_DIR / f"{fname}.png"
        if p.exists():
            html += f'<div><h3>{title}</h3>{_img_tag(_png_to_b64(p))}</div>'

    return f"""
<section id="signals">
  <h2>Signal Analysis</h2>
  {html}
</section>"""


def _section_wfo(wf: dict) -> str:
    win_df = wf["windows"]
    kpis   = wf.get("full_kpis", {})

    cards = "".join([
        _kpi_card("Windows",          f"{wf['n_windows']}",                      "blue"),
        _kpi_card("Profitable",       f"{wf['pct_profitable']:.0f}%",            "green" if wf['pct_profitable']>60 else "red"),
        _kpi_card("Median OOS Ret",   f"{wf['median_oos_ret']:+.2f}%",           "green" if wf['median_oos_ret']>0 else "red"),
        _kpi_card("OOS Sharpe",       f"{kpis.get('sharpe',0):.3f}",             "blue"),
        _kpi_card("OOS Max DD",       f"{kpis.get('max_drawdown',0)*100:.1f}%",  "red"),
        _kpi_card("OOS Return",       f"{kpis.get('total_return',0)*100:+.1f}%", "green" if kpis.get('total_return',0)>0 else "red"),
        _kpi_card("Consistency",      f"{wf['consistency']:+.3f}",               "blue"),
    ])

    eq_b64   = _chart_wfo_equity(wf)
    stat_b64 = _chart_wfo_stats(wf)

    # Per-window table
    tbl_rows = []
    if not win_df.empty:
        for w, row in win_df.iterrows():
            ret = row["OOS Return (%)"]
            sh  = row["Sharpe"]
            dd  = row["Max DD (%)"]
            tbl_rows.append([
                f"W{int(w)}",
                str(row["Train End"]),
                str(row["OOS End"]),
                _color_signed(ret, pct=True, decimals=1),
                f'{row["# Trades"]}',
                f'{row["Win Rate (%)"]}%',
                f'{row["Profit Factor"]:.2f}',
                _color_signed(sh,  decimals=3),
                _color_signed(dd,  pct=True, decimals=2, pos_good=False),
            ])
    win_tbl = _table(["Win","Train End","OOS End","OOS Ret","Trades","WR","PF","Sharpe","Max DD"], tbl_rows)

    return f"""
<section id="wfo">
  <h2>Walk-Forward Validation (6m train · 2m OOS · 2m step)</h2>
  <div class="cards">{cards}</div>
  <div class="chart">{_img_tag(eq_b64)}</div>
  <div class="chart">{_img_tag(stat_b64)}</div>
  <h3>Per-Window KPIs</h3>
  {win_tbl}
</section>"""


def _section_mc(mc_result: dict, label="In-Sample", oi_label="") -> str:
    if not mc_result:
        return ""
    final   = mc_result.get("final_returns", np.array([]))
    p_profit = float((final > 0).mean() * 100) if len(final) else 0
    p_ruin   = float((final < -0.5).mean() * 100) if len(final) else 0

    cards = "".join([
        _kpi_card("# Simulations",  f"{len(final):,}",                 "blue"),
        _kpi_card("P(Profit)",      f"{p_profit:.1f}%",                "green" if p_profit>80 else "red"),
        _kpi_card("p5 Return",      f"{np.percentile(final,5)*100:+.1f}%" if len(final) else "—", "blue"),
        _kpi_card("p50 Return",     f"{np.percentile(final,50)*100:+.1f}%" if len(final) else "—", "blue"),
        _kpi_card("p95 Return",     f"{np.percentile(final,95)*100:+.1f}%" if len(final) else "—", "green"),
        _kpi_card("P(Ruin >50%DD)", f"{p_ruin:.1f}%",                 "red" if p_ruin>5 else "green"),
    ])

    mc_b64 = _chart_mc(mc_result)
    return f"""
<section id="mc-{label.lower().replace(' ','-')}">
  <h2>Monte Carlo — {label} {oi_label}</h2>
  <div class="cards">{cards}</div>
  <div class="chart">{_img_tag(mc_b64)}</div>
</section>"""


def _section_scenarios(scenario_kpis: dict) -> str:
    sc_b64 = _chart_scenarios(scenario_kpis)
    rows = []
    for name, k in scenario_kpis.items():
        rows.append([
            name,
            _color_signed(k.get("total_return",0)*100, pct=True, decimals=1),
            f"{k.get('sharpe',0):.3f}",
            f"{k.get('sortino',0):.3f}",
            f"{k.get('calmar',0):.3f}",
            _color_signed(k.get("max_drawdown",0)*100, pct=True, decimals=1, pos_good=False),
            f"{k.get('win_rate',0)*100:.1f}%",
            f"{k.get('profit_factor',0):.2f}",
            f"{k.get('n_trades',0):,}",
        ])
    tbl = _table(["Scenario","Return","Sharpe","Sortino","Calmar","Max DD","WR","PF","Trades"], rows)
    return f"""
<section id="scenarios">
  <h2>Scenario Comparison</h2>
  <div class="chart">{_img_tag(sc_b64)}</div>
  {tbl}
</section>"""


def _section_leverage(lev_df) -> str:
    if lev_df is None or (hasattr(lev_df, "empty") and lev_df.empty):
        return ""
    lev_b64 = _chart_leverage(lev_df)
    df = lev_df if isinstance(lev_df, pd.DataFrame) else pd.DataFrame()
    rows = []
    show_cols = ["Scenario","Method","Pct (%)","Leverage","Total Ret (%)","Sharpe","Max DD (%)","Win Rate (%)","Profit Factor"]
    show_cols = [c for c in show_cols if c in df.columns]
    for _, r in df.iterrows():
        row = []
        for c in show_cols:
            v = r[c]
            if c in ("Total Ret (%)","Max DD (%)"):
                try:
                    vf = float(v)
                    pos_good = c != "Max DD (%)"
                    row.append(_color_signed(vf, pct=True, decimals=1, pos_good=pos_good))
                except Exception:
                    row.append(str(v))
            else:
                row.append(str(v))
        rows.append(row)
    tbl = _table(show_cols, rows)
    return f"""
<section id="leverage">
  <h2>Leverage × Sizing Grid</h2>
  <div class="chart">{_img_tag(lev_b64)}</div>
  {tbl}
</section>"""


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate_html(
    bt_result:        dict,
    kpis:             dict,
    wf_result:        dict,
    mc_insample:      dict,
    mc_oos:           dict,
    scenario_kpis:    dict,
    lev_result:       dict,
    df_1h:            pd.DataFrame,
    oi_source:        str = "real_basis",
    scenario_name:    str = "Session 08-21",
    out_path:         Path = Path("reports/BTCUSDT_Strategy_Report.html"),
) -> Path:
    """
    Build and write the self-contained HTML report.
    Returns the output path.
    """
    date_range = (f"{df_1h.index[0].date()} → {df_1h.index[-1].date()}"
                  if not df_1h.empty else "—")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC")

    nav = """
<nav>
  <span class="brand">BTCUSDT Strategy</span>
  <a href="#summary">Summary</a>
  <a href="#signals">Signals</a>
  <a href="#backtest">Backtest</a>
  <a href="#wfo">Walk-Forward</a>
  <a href="#mc-in-sample">MC In-Sample</a>
  <a href="#mc-oos">MC OOS</a>
  <a href="#scenarios">Scenarios</a>
  <a href="#leverage">Leverage</a>
</nav>"""

    body = "".join([
        _section_summary(kpis, oi_source, len(df_1h), date_range, scenario_name),
        _section_signals(),
        _section_backtest(bt_result, kpis),
        _section_wfo(wf_result) if wf_result else "",
        _section_mc(mc_insample, "In-Sample"),
        _section_mc(mc_oos,      "OOS Trades"),
        _section_scenarios(scenario_kpis) if scenario_kpis else "",
        _section_leverage(lev_result) if (lev_result is not None) else "",
    ])

    footer = f"""
<footer>
  BTCUSDT Multi-Timeframe Strategy Report &nbsp;·&nbsp; Generated {now}
  &nbsp;·&nbsp; Data: Binance Vision CDN &nbsp;·&nbsp; OI: {oi_source}
</footer>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BTCUSDT Strategy Report</title>
<style>{_CSS}</style>
</head>
<body>
{nav}
<div class="container">
{body}
</div>
{footer}
</body>
</html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path
