"""
create_orb_v2_report.py
════════════════════════════════════════════════════════════════════════
ICT Asian Range Sweep — Versione 2 (corretta dai test statistici)

Correzioni rispetto alla v1:
  ✗ v1: London + NY KZ  →  ✓ v2: solo London KZ (07:00-09:59 UTC)
  ✗ v1: tutti i sweep   →  ✓ v2: solo sweep superficiali (depth/ATR ≤ soglia)
  ✗ v1: NR filter       →  ✓ v2: nessun filtro NR (non supportato dai dati)
  ✗ v1: TP = Asian Range →  ✓ v2: TP/SL basati su ATR (ottimizzati IS)
  ✗ v1: score = CR×depth →  ✓ v2: score piatto (scoring contraddiceva i dati)

Grid IS ottimizzato:
  atr_sl       ∈ {0.5, 1.0, 1.5}
  atr_tp1      ∈ {0.75, 1.0, 1.5, 2.0}
  max_depth_atr ∈ {0.3, 0.5, 0.8}

Walk-Forward: 6m IS / 2m OOS / step 2m (invariato)

Output → reports/report_orb_v2_strategy.html
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
import scipy.stats as scipy_stats

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.strategy.data_fetcher import fetch_extended_data
from src.strategy.indicators   import add_indicators
from src.strategy.engine       import run_backtest, INIT_CAP
from src.strategy.orb_ict      import (
    build_asian_range,
    build_orb_ict_signals,
    LONDON_START, LONDON_END,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
START_YEAR  = 2020
WF_TRAIN_M  = 6
WF_OOS_M    = 2
WF_STEP_M   = 2

SL_GRID    = [0.5, 1.0, 1.5]
TP1_GRID   = [0.75, 1.0, 1.5, 2.0]
DEPTH_GRID = [0.3, 0.5, 0.8]   # max_depth_atr — shallow sweep filter

_BG   = "#0f1117"
_CARD = "#12151f"
_GRID = "#1e2130"
_TEXT = "#e0e0e0"
_ACC  = "#42a5f5"   # London blue
_GRN  = "#66bb6a"
_RED  = "#ef5350"
_YEL  = "#ffd54f"


# ─────────────────────────────────────────────────────────────────────────────
# WF helpers
# ─────────────────────────────────────────────────────────────────────────────
def _wf_windows(index: pd.DatetimeIndex) -> list[tuple]:
    start, end = index[0], index[-1]
    windows, cur = [], start
    while True:
        tr_end = cur + pd.DateOffset(months=WF_TRAIN_M)
        oo_s   = tr_end
        oo_e   = oo_s + pd.DateOffset(months=WF_OOS_M)
        if oo_e > end:
            break
        windows.append((cur, tr_end, oo_s, oo_e))
        cur = cur + pd.DateOffset(months=WF_STEP_M)
    return windows


def _best_is_params(df_is, asian_daily_is, min_trades: int = 5):
    best = {"sl": 0.5, "tp1": 1.0, "depth": 0.5, "metric": -np.inf}
    for sl in SL_GRID:
        for tp1 in TP1_GRID:
            for depth in DEPTH_GRID:
                try:
                    sigs = build_orb_ict_signals(df_is, asian_daily_is,
                                                 max_depth_atr=depth)
                    bt   = run_backtest(df_is, sigs,
                                        atr_sl_override=sl,
                                        atr_tp1_override=tp1)
                except Exception:
                    continue
                trd = bt.get("trades", pd.DataFrame())
                n   = len(trd) if isinstance(trd, pd.DataFrame) else 0
                if n < min_trades:
                    continue
                cal = float(bt["kpis"].get("calmar", 0.0))
                ret = float(bt["kpis"].get("total_return", 0.0))
                metric = cal if cal > 0 else ret
                if metric > best["metric"]:
                    best.update({"sl": sl, "tp1": tp1, "depth": depth,
                                 "metric": metric, "n": n})
    return best


def run_wf(df_15m, asian_daily, windows, min_is_trades: int = 5) -> list[dict]:
    results = []
    for i, (tr_s, tr_e, oo_s, oo_e) in enumerate(windows):
        df_is         = df_15m[(df_15m.index >= tr_s) & (df_15m.index < tr_e)]
        asian_is      = asian_daily[(asian_daily.index >= tr_s) &
                                    (asian_daily.index <  tr_e)]
        df_oos        = df_15m[(df_15m.index >= oo_s) & (df_15m.index < oo_e)]
        asian_oos     = asian_daily[(asian_daily.index >= oo_s) &
                                    (asian_daily.index <  oo_e)]

        if len(df_is) < 200 or len(df_oos) < 20:
            continue

        best = _best_is_params(df_is, asian_is, min_trades=min_is_trades)

        try:
            sig_oos = build_orb_ict_signals(df_oos, asian_oos,
                                            max_depth_atr=best["depth"])
            oos_bt  = run_backtest(df_oos, sig_oos,
                                   atr_sl_override=best["sl"],
                                   atr_tp1_override=best["tp1"])
        except Exception as exc:
            print(f"  W{i+1:02d} OOS error: {exc}")
            continue

        trd     = oos_bt.get("trades", pd.DataFrame())
        oos_n   = len(trd) if isinstance(trd, pd.DataFrame) else 0
        oos_ret = float(oos_bt["kpis"].get("total_return", 0.0)) * 100
        oos_dd  = abs(float(oos_bt["kpis"].get("max_drawdown", 0.0))) * 100
        oos_wr  = float(oos_bt["kpis"].get("win_rate", 0.0)) * 100
        oos_cal = float(oos_bt["kpis"].get("calmar", 0.0))
        n_sigs  = int((sig_oos["signal"] != 0).sum())

        results.append({
            "window": i + 1, "oos_s": oo_s, "oos_e": oo_e,
            "is_calmar": round(best["metric"], 3),
            "best_sl": best["sl"], "best_tp1": best["tp1"],
            "best_depth": best["depth"],
            "oos_sigs": n_sigs, "oos_n": oos_n,
            "oos_ret": round(oos_ret, 2), "oos_dd": round(oos_dd, 2),
            "oos_wr": round(oos_wr, 1), "oos_cal": round(oos_cal, 3),
            "oos_trades": trd,
            "equity": oos_bt.get("equity", pd.Series(dtype=float)),
        })
        print(f"  W{i+1:02d} [{oo_s.date()}→{oo_e.date()}] "
              f"sl={best['sl']} tp={best['tp1']} d={best['depth']} "
              f"IS_cal={best['metric']:+.2f} "
              f"| OOS ret={oos_ret:+.1f}% DD={oos_dd:.1f}% "
              f"N={oos_n} WR={oos_wr:.1f}%")
    return results


def _aggregate_oos(results: list[dict]) -> dict:
    """Stitch OOS equity curves and compute aggregate KPIs."""
    if not results:
        return {}
    equities = []
    cap = float(INIT_CAP)
    for r in results:
        eq = r["equity"]
        if isinstance(eq, pd.Series) and len(eq) > 0:
            scale = cap / float(eq.iloc[0])
            equities.append(eq * scale)
            cap = float(eq.iloc[-1]) * scale
    if not equities:
        return {}
    full_eq = pd.concat(equities)
    total_ret = (full_eq.iloc[-1] / full_eq.iloc[0] - 1) * 100
    roll_max  = full_eq.cummax()
    dd        = (full_eq - roll_max) / roll_max
    max_dd    = abs(dd.min()) * 100
    calmar    = total_ret / max_dd if max_dd > 0 else 0.0

    all_trades = pd.concat(
        [r["oos_trades"] for r in results
         if isinstance(r.get("oos_trades"), pd.DataFrame) and len(r["oos_trades"]) > 0],
        ignore_index=True)
    total_n = sum(r["oos_n"] for r in results)
    avg_wr  = np.mean([r["oos_wr"] for r in results if r["oos_n"] > 0])

    return {
        "equity":    full_eq,
        "total_ret": round(total_ret, 2),
        "max_dd":    round(max_dd, 2),
        "calmar":    round(calmar, 3),
        "total_n":   total_n,
        "avg_wr":    round(avg_wr, 1),
        "trades":    all_trades,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────
def _b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=_BG, edgecolor="none")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _style_ax(ax, title=""):
    ax.set_facecolor(_CARD)
    ax.tick_params(colors=_TEXT, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(_GRID)
    ax.grid(True, color=_GRID, linewidth=0.5, alpha=0.6)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    if title:
        ax.set_title(title, color=_TEXT, fontsize=9, pad=6)


def plot_equity(agg: dict, label: str) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5),
                             facecolor=_BG, gridspec_kw={"height_ratios": [3, 1]})
    eq = agg["equity"]
    ret_pct = (eq / eq.iloc[0] - 1) * 100

    axes[0].plot(eq.index, ret_pct.values, color=_ACC, linewidth=1.1)
    axes[0].axhline(0, color=_GRID, linewidth=0.8)
    axes[0].fill_between(eq.index, ret_pct.values, 0,
                         where=ret_pct.values >= 0,
                         alpha=0.18, color=_GRN)
    axes[0].fill_between(eq.index, ret_pct.values, 0,
                         where=ret_pct.values < 0,
                         alpha=0.18, color=_RED)
    _style_ax(axes[0], f"{label} — Equity curve (% OOS stitched)")
    axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    roll_max = eq.cummax()
    dd_pct   = (eq - roll_max) / roll_max * 100
    axes[1].fill_between(eq.index, dd_pct.values, 0,
                         alpha=0.65, color=_RED)
    axes[1].set_ylim(min(dd_pct.min() * 1.15, -0.5), 0.5)
    _style_ax(axes[1], "Drawdown %")
    axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    fig.tight_layout(pad=0.8)
    return _b64(fig)


def plot_oos_bar(results: list[dict], label: str) -> str:
    ws   = [f"W{r['window']:02d}" for r in results]
    rets = [r["oos_ret"] for r in results]
    cols = [_GRN if r >= 0 else _RED for r in rets]

    fig, ax = plt.subplots(figsize=(max(8, len(ws) * 0.4), 3.5), facecolor=_BG)
    ax.bar(ws, rets, color=cols, alpha=0.85, width=0.65)
    ax.axhline(0, color=_TEXT, linewidth=0.8)
    _style_ax(ax, f"{label} — OOS return per finestra WF (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    fig.tight_layout(pad=0.8)
    return _b64(fig)


def plot_win_rate_dist(trades: pd.DataFrame, label: str) -> str:
    if trades.empty or "pnl" not in trades.columns:
        fig, ax = plt.subplots(figsize=(7, 3), facecolor=_BG)
        ax.text(0.5, 0.5, "No trades", ha="center", va="center",
                color=_TEXT, transform=ax.transAxes)
        _style_ax(ax, label)
        return _b64(fig)

    wins = trades[trades["pnl"] > 0]["pnl"]
    loss = trades[trades["pnl"] < 0]["pnl"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), facecolor=_BG)
    rng = max(abs(trades["pnl"].max()), abs(trades["pnl"].min()))
    bins = np.linspace(-rng, rng, 60) if rng > 0.01 else 1

    axes[0].hist(wins.values, bins=bins, color=_GRN, alpha=0.7,
                 label=f"Win ({len(wins)})")
    axes[0].hist(loss.values, bins=bins, color=_RED, alpha=0.7,
                 label=f"Loss ({len(loss)})")
    axes[0].axvline(0, color=_TEXT, linewidth=0.8)
    axes[0].legend(fontsize=8, labelcolor=_TEXT,
                   facecolor=_CARD, edgecolor=_GRID)
    _style_ax(axes[0], f"{label} — Distribuzione P&L per trade")
    axes[0].set_xlabel("P&L ($)")

    cumulative = trades["pnl"].sort_values().cumsum().reset_index(drop=True)
    axes[1].plot(cumulative.index, cumulative.values, color=_ACC)
    axes[1].axhline(0, color=_GRID, linewidth=0.8)
    _style_ax(axes[1], "P&L cumulativo (ordinato)")
    axes[1].set_xlabel("Trade #")
    axes[1].set_ylabel("P&L ($)")

    fig.tight_layout(pad=0.8)
    return _b64(fig)


def plot_depth_filter_effect(df_15m, asian_daily) -> str:
    """Show win rate vs max_depth_atr threshold on full dataset."""
    thresholds = np.arange(0.1, 2.01, 0.1)
    wr_all = []
    counts = []

    HORIZONS_B = 16  # 4h in 15M bars

    IDX   = df_15m.index
    H     = IDX.hour
    HI    = df_15m["high"].values
    LO    = df_15m["low"].values
    CL    = df_15m["close"].values
    ATR   = df_15m["atr_14"].clip(lower=1.0).values if "atr_14" in df_15m.columns else np.ones(len(df_15m))

    date_idx = IDX.normalize()
    ah_map = asian_daily["asian_high"].to_dict()
    al_map = asian_daily["asian_low"].to_dict()
    ah_arr = np.array([ah_map.get(d, np.nan) for d in date_idx], dtype=float)
    al_arr = np.array([al_map.get(d, np.nan) for d in date_idx], dtype=float)

    N = len(df_15m)
    all_depths = []
    all_dir_ok = []

    for i in range(N - HORIZONS_B - 1):
        h_i = H[i]
        if not (LONDON_START <= h_i < LONDON_END):
            continue
        ah = ah_arr[i]; al = al_arr[i]
        if np.isnan(ah) or np.isnan(al):
            continue
        lo_i = LO[i]; hi_i = HI[i]; cl_i = CL[i]
        fwd = (CL[i + HORIZONS_B] - cl_i) / cl_i * 100

        if lo_i < al and cl_i >= al:
            pen = al - lo_i
            d_atr = pen / ATR[i]
            all_depths.append(d_atr)
            all_dir_ok.append(1 if fwd > 0 else 0)
        elif hi_i > ah and cl_i <= ah:
            pen = hi_i - ah
            d_atr = pen / ATR[i]
            all_depths.append(d_atr)
            all_dir_ok.append(1 if fwd < 0 else 0)

    all_depths = np.array(all_depths)
    all_dir_ok = np.array(all_dir_ok)

    for thr in thresholds:
        mask = all_depths <= thr
        n = mask.sum()
        counts.append(n)
        wr_all.append(all_dir_ok[mask].mean() * 100 if n > 0 else 50.0)

    fig, ax1 = plt.subplots(figsize=(9, 3.8), facecolor=_BG)
    ax2 = ax1.twinx()

    ax1.plot(thresholds, wr_all, color=_ACC, linewidth=1.8,
             marker="o", markersize=4, label="Dir accuracy %")
    ax1.axhline(50, color=_GRID, linewidth=0.8, linestyle="--", label="50% (casuale)")
    ax1.axhline(54, color=_YEL, linewidth=0.8, linestyle=":",
                label="54% (London KZ max)")
    ax2.bar(thresholds, counts, width=0.07, alpha=0.35, color=_GRN,
            label="N eventi")
    ax2.set_ylabel("N eventi", color=_TEXT, fontsize=8)
    ax2.tick_params(colors=_TEXT, labelsize=7)

    ax1.set_xlabel("max_depth_atr (soglia sweep superficiale)", color=_TEXT)
    ax1.set_ylabel("Accuratezza direzionale % (4h)", color=_TEXT)
    ax1.legend(fontsize=7, loc="upper right", facecolor=_CARD,
               edgecolor=_GRID, labelcolor=_TEXT)
    _style_ax(ax1, "Effetto soglia depth/ATR sull'accuratezza direzionale (4h)")
    ax1.set_facecolor(_CARD)
    fig.tight_layout(pad=0.8)
    return _b64(fig)


def plot_params_heatmap(results: list[dict]) -> str:
    """Heatmap: IS-best params chosen per window."""
    if not results:
        fig, ax = plt.subplots(facecolor=_BG)
        _style_ax(ax, "No data")
        return _b64(fig)

    depths = sorted(set(r["best_depth"] for r in results))
    tps    = sorted(set(r["best_tp1"]   for r in results))
    sls    = sorted(set(r["best_sl"]    for r in results))

    param_counts = {}
    for r in results:
        key = (r["best_sl"], r["best_tp1"], r["best_depth"])
        param_counts[key] = param_counts.get(key, 0) + 1

    labels = [f"SL={k[0]} TP={k[1]} d={k[2]}" for k in param_counts]
    vals   = list(param_counts.values())
    sort_idx = np.argsort(vals)[::-1]

    fig, ax = plt.subplots(figsize=(9, max(3, len(labels) * 0.35 + 1)),
                           facecolor=_BG)
    bars = ax.barh([labels[i] for i in sort_idx],
                   [vals[i]   for i in sort_idx],
                   color=_ACC, alpha=0.8)
    for bar, v in zip(bars, [vals[i] for i in sort_idx]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{v}", va="center", ha="left", color=_TEXT, fontsize=8)
    _style_ax(ax, "Frequenza parametri IS selezionati per finestra WF")
    ax.set_xlabel("# finestre WF", color=_TEXT)
    fig.tight_layout(pad=0.8)
    return _b64(fig)


# ─────────────────────────────────────────────────────────────────────────────
# HTML report
# ─────────────────────────────────────────────────────────────────────────────
def _card(title: str, body: str) -> str:
    return f"""
    <div class="card">
      <h3>{title}</h3>
      {body}
    </div>"""


def _metric(label: str, value: str, color: str = "#e0e0e0") -> str:
    return (f'<div class="metric"><span class="mlabel">{label}</span>'
            f'<span class="mval" style="color:{color}">{value}</span></div>')


def _table(headers: list, rows: list) -> str:
    th = "".join(f"<th>{h}</th>" for h in headers)
    tr = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
        for row in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>"


def build_html(results: dict, df_15m, asian_daily) -> str:
    # ── Section A: depth filter analysis ──────────────────────────────────
    img_depth = plot_depth_filter_effect(df_15m, asian_daily)

    # ── Section B: WF results ─────────────────────────────────────────────
    r = results
    wf_rows = []
    for w in r["wf"]:
        cal_c = _GRN if w["oos_cal"] > 0 else _RED
        ret_c = _GRN if w["oos_ret"] > 0 else _RED
        wf_rows.append([
            f"W{w['window']:02d}",
            f"{w['oos_s'].date()}",
            f"sl={w['best_sl']} tp={w['best_tp1']} d={w['best_depth']}",
            w["oos_sigs"],
            w["oos_n"],
            f'<span style="color:{ret_c}">{w["oos_ret"]:+.1f}%</span>',
            f"{w['oos_dd']:.1f}%",
            f'<span style="color:{cal_c}">{w["oos_cal"]:+.3f}</span>',
            f"{w['oos_wr']:.1f}%",
        ])

    agg = r["agg"]
    agg_cal_c = _GRN if agg.get("calmar", 0) > 0 else _RED
    agg_ret_c = _GRN if agg.get("total_ret", 0) > 0 else _RED

    img_eq  = plot_equity(agg, "London KZ Sweep v2") if "equity" in agg else ""
    img_bar = plot_oos_bar(r["wf"], "London KZ Sweep v2")
    img_pnl = plot_win_rate_dist(agg.get("trades", pd.DataFrame()), "v2")
    img_par = plot_params_heatmap(r["wf"])

    def imgt(b64):
        return (f'<img src="data:image/png;base64,{b64}"'
                ' style="width:100%;border-radius:6px;margin-top:8px">')

    # Pre-build all card HTML to avoid nested f-strings with backslashes
    card_depth = _card(
        "Accuratezza direzionale 4h vs soglia max_depth_atr",
        imgt(img_depth) + (
            '<p style="font-size:0.82em;color:#aaa;margin-top:6px">'
            "Il grafico mostra l'accuratezza direzionale a 4h (linea blu) "
            "in funzione della soglia di filtro. N crescente (barre verdi) "
            "con soglie piu' alte. La grid IS testa: 0.3, 0.5, 0.8.</p>"))

    card_wf_table = _card(
        "Risultati per finestra WF",
        _table(["W#", "OOS start", "Params IS",
                "N segnali", "N trade",
                "OOS ret%", "Max DD%", "Calmar", "WR%"], wf_rows))

    eq_inner = imgt(img_eq) if img_eq else "<p>No equity data</p>"
    card_eq  = _card("Equity curve OOS stitched", eq_inner)
    card_bar = _card("OOS return per finestra",   imgt(img_bar))
    card_pnl = _card("Distribuzione P&L per trade", imgt(img_pnl))
    card_par = _card("Frequenza parametri IS selezionati", imgt(img_par))

    tot_ret  = agg.get("total_ret", 0)
    max_dd   = agg.get("max_dd", 0)
    calmar   = agg.get("calmar", 0)
    total_n  = agg.get("total_n", 0)
    avg_wr   = agg.get("avg_wr", 0)
    kpi_inner = (
        '<div class="metric-row">'
        + _metric("Total Return",  f"{tot_ret:+.2f}%",  _GRN if tot_ret > 0 else _RED)
        + _metric("Max Drawdown",  f"{max_dd:.2f}%",    _RED)
        + _metric("Calmar",        f"{calmar:+.3f}",     _GRN if calmar > 0 else _RED)
        + _metric("N Trade OOS",   str(total_n))
        + _metric("Avg Win Rate",  f"{avg_wr:.1f}%")
        + "</div>")
    card_kpi = _card("KPI aggregati (equity OOS stitched)", kpi_inner)

    card_conclusion = _card("Edge e limitazioni", (
        "<ul>"
        "<li>La correlazione ICT (sweep + reversion nella London KZ) e' "
        "statisticamente confermata (~54% directional accuracy a 4h, p&lt;0.001).</li>"
        "<li>L'edge e' <strong>piccolo</strong>: richiede R:R &ge; 0.85:1 per la "
        "profittabilita' (breakeven WR = 46% a 54% accuracy).</li>"
        "<li>Sweep profondi (depth/ATR &gt; soglia) degradano il segnale: "
        "la soglia ottimale IS e' selezionata nella grid.</li>"
        "<li>Il segnale e' non-stazionario nel tempo: 2021-2022 piu' forte "
        "(~53%), 2023-2025 piu' debole (~49%).</li>"
        "</ul>"))

    body = f"""
    <section id="intro">
      <h2>ORB-ICT v2 — London KZ Shallow Sweep</h2>
      <p>Strategia ICT Asian Range Sweep ricalibrata sui risultati dei test statistici
         (<code>test_ict_correlation.py</code>). Dati: BTCUSDT Perpetual Futures
         Binance Vision, 15M, 2020-01 - 2026-05.</p>
      <div class="finding-box">
        <strong>Motivazione v2:</strong> i test statistici (7 test, 5.245 sweep events)
        hanno evidenziato che l&apos;edge ICT esiste esclusivamente nella
        <strong>London KZ (07:00-09:59 UTC)</strong> con 52.9% dir. accuracy a 15m
        e 54.0% a 4h (p&lt;0.001). La NY KZ restituisce 48-49% (contro la predizione).
        Il filtro NR e il compression_ratio non sono statisticamente supportati.
        La profondit&agrave; dello sweep e&apos; <em>negativamente</em> correlata
        col return forward (Spearman r = -0.039, p=0.004): solo sweep superficiali
        sono inclusi.
      </div>
    </section>

    <section id="depth">
      <h2>Analisi Soglia Depth/ATR</h2>
      {card_depth}
    </section>

    <section id="wf">
      <h2>Walk-Forward Backtest — 6m IS / 2m OOS</h2>
      {card_wf_table}
      {card_eq}
      {card_bar}
      {card_pnl}
      {card_par}
    </section>

    <section id="summary">
      <h2>Riepilogo Aggregato OOS</h2>
      {card_kpi}
    </section>

    <section id="conclusion">
      <h2>Interpretazione</h2>
      {card_conclusion}
    </section>
    """

    css = f"""
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: {_BG}; color: {_TEXT}; font-family: 'Segoe UI', sans-serif;
            font-size: 14px; padding: 24px; }}
    h1   {{ color: {_ACC}; font-size: 1.6em; margin-bottom: 4px; }}
    h2   {{ color: {_ACC}; font-size: 1.1em; margin: 24px 0 10px; border-bottom:
            1px solid {_GRID}; padding-bottom: 4px; }}
    h3   {{ color: {_TEXT}; font-size: 0.95em; margin-bottom: 8px; }}
    .card {{ background: {_CARD}; border: 1px solid {_GRID}; border-radius: 8px;
             padding: 14px; margin-bottom: 14px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.82em; }}
    th, td {{ border: 1px solid {_GRID}; padding: 5px 8px; text-align: center; }}
    th {{ background: {_GRID}; color: {_TEXT}; }}
    .metric-row {{ display: flex; flex-wrap: wrap; gap: 12px; margin-top: 6px; }}
    .metric {{ background: {_BG}; border: 1px solid {_GRID}; border-radius: 6px;
               padding: 10px 16px; min-width: 130px; }}
    .mlabel {{ display: block; font-size: 0.75em; color: #aaa; margin-bottom: 4px; }}
    .mval   {{ display: block; font-size: 1.3em; font-weight: bold; }}
    .finding-box {{ background: #1a2233; border-left: 3px solid {_ACC};
                    padding: 12px 16px; margin: 12px 0; border-radius: 4px;
                    font-size: 0.88em; line-height: 1.6; }}
    code {{ background: {_GRID}; padding: 1px 5px; border-radius: 3px; font-size: 0.85em; }}
    ul {{ padding-left: 20px; line-height: 1.8; font-size: 0.88em; }}
    p  {{ line-height: 1.6; font-size: 0.88em; margin: 6px 0; }}
    """

    return f"""<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="UTF-8">
  <title>ORB-ICT v2 Strategy Report — BTCUSDT</title>
  <style>{css}</style>
</head>
<body>
  <h1>ORB-ICT v2 — London KZ Shallow Sweep Strategy</h1>
  <p style="color:#aaa;font-size:0.82em">BTCUSDT Perpetual Futures · Binance Vision
     · 15M · 2020-01 → 2026-05 · WF: {WF_TRAIN_M}m IS / {WF_OOS_M}m OOS</p>
  {body}
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t0 = time.time()

    print("=" * 65)
    print("ORB-ICT v2 — London KZ Shallow Sweep Report")
    print("=" * 65)

    print("\n[1/4] Caricamento dati …")
    raw    = fetch_extended_data(start_year=START_YEAR, start_month=1,
                                  fetch_15m=True, fetch_1m=False,
                                  fetch_flow=False)
    df_1h  = add_indicators(raw["1H"])
    df_15m = add_indicators(raw["15M"])
    print(f"  1H : {len(df_1h):,} bars  ({df_1h.index[0].date()} → {df_1h.index[-1].date()})")
    print(f"  15M: {len(df_15m):,} bars")

    print("\n[2/4] Asian Range …")
    asian_daily = build_asian_range(df_1h)
    print(f"  Giorni completi: {len(asian_daily)}")

    print("\n[3/4] Walk-Forward (grid IS: "
          f"{len(SL_GRID)}×{len(TP1_GRID)}×{len(DEPTH_GRID)} = "
          f"{len(SL_GRID)*len(TP1_GRID)*len(DEPTH_GRID)} combinazioni) …")
    windows = _wf_windows(df_15m.index)
    print(f"  {len(windows)} finestre WF")

    wf_results = run_wf(df_15m, asian_daily, windows)
    agg = _aggregate_oos(wf_results)

    print(f"\n  ── Aggregato OOS ──────────────────────────────────")
    print(f"  Ret={agg.get('total_ret',0):+.2f}%  "
          f"DD={agg.get('max_dd',0):.1f}%  "
          f"Calmar={agg.get('calmar',0):+.3f}  "
          f"N={agg.get('total_n',0)}  "
          f"WR={agg.get('avg_wr',0):.1f}%")

    print("\n[4/4] Generazione HTML …")
    html = build_html({"wf": wf_results, "agg": agg}, df_15m, asian_daily)

    out = Path("reports/report_orb_v2_strategy.html")
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    sz = out.stat().st_size / 1024
    print(f"  ✓ {out}  ({sz:.0f} KB)  in {time.time()-t0:.0f}s")
