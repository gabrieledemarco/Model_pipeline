"""
create_orb_strategy_report.py
──────────────────────────────
ICT-style Asian Range Sweep ORB Strategy — Research + Walk-Forward Validation.

Pipeline
────────
1. Load 1H + 15M data (2020-2026)
2. Compute Asian Range (1H, 00-06 UTC) + NR compression filter
3. Detect sweeps on 15M in London KZ (07-09 UTC) + NY KZ (13-15 UTC)
4. Section A — Statistical Research:
   a. Sweep frequency per day-of-week and year
   b. Win rate by sweep type (London vs NY, long vs short)
   c. NR filter effect (NR days vs all days)
   d. Payoff profile: histogram of post-sweep returns
5. Section B — Walk-Forward Backtest (6m IS / 2m OOS):
   - IS grid: atr_sl ∈ {0.3, 0.5, 1.0} × atr_tp1 ∈ {1.0, 1.5, 2.0} × nr_filter {T/F}
   - Best by Calmar on IS, applied to OOS
6. HTML report → reports/report_orb_strategy.html

Usage
─────
  python create_orb_strategy_report.py
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
import matplotlib.patches as mpatches

from src.strategy.data_fetcher import fetch_extended_data
from src.strategy.indicators   import add_indicators
from src.strategy.engine       import run_backtest, INIT_CAP
from src.strategy.orb_ict      import (
    build_asian_range, build_orb_ict_signals,
    LONDON_START, LONDON_END, NY_START, NY_END,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
START_YEAR  = 2020
START_MONTH = 1
WF_TRAIN_M  = 6
WF_OOS_M    = 2
WF_STEP_M   = 2

SL_GRID    = [0.3, 0.5, 1.0]
TP1_GRID   = [1.0, 1.5, 2.0]
NR_GRID    = [True, False]   # nr_filter on/off

_BG   = "#0f1117"
_CARD = "#12151f"
_GRID = "#1e2130"
_TEXT = "#e0e0e0"
_COL  = {"London": "#42a5f5", "NY": "#ef5350", "NR": "#66bb6a", "All": "#90caf9"}


# ─────────────────────────────────────────────────────────────────────────────
# Research: sweep statistics (raw, before strategy entry)
# ─────────────────────────────────────────────────────────────────────────────

def compute_sweep_stats(df_15m: pd.DataFrame,
                        asian_daily: pd.DataFrame) -> dict:
    """
    For each day, find ALL sweep events in London KZ and NY KZ.
    Track whether the event led to a profitable follow-through:
      LONG  : price moves above Asian High within the same session → win
      SHORT : price moves below Asian Low within the same session → win
    """
    records = []
    dates = sorted(asian_daily.index)

    for date in dates:
        if date not in asian_daily.index:
            continue
        row = asian_daily.loc[date]
        ah  = row["asian_high"]
        al  = row["asian_low"]
        ar  = row["asian_range"]
        is_nr = bool(row["is_nr"])
        cr  = float(row["compression_ratio"])

        if pd.isna(ah) or pd.isna(al) or ar < 1.0:
            continue

        # Get full-day 15M bars for this date
        day_bars = df_15m[df_15m.index.date == date.date()].sort_index()
        if len(day_bars) < 4:
            continue

        for kz_name, kz_s, kz_e in [("London", LONDON_START, LONDON_END),
                                      ("NY",     NY_START,     NY_END)]:
            kz_bars = day_bars[(day_bars.index.hour >= kz_s) &
                               (day_bars.index.hour <  kz_e)]
            if kz_bars.empty:
                continue

            # Remaining bars after KZ (for outcome measurement)
            post_kz = day_bars[day_bars.index > kz_bars.index[-1]]

            for i, (ts, bar) in enumerate(kz_bars.iterrows()):
                # LONG sweep: low < AL, close >= AL
                if bar["low"] < al and bar["close"] >= al:
                    pen = (al - bar["low"])
                    # Outcome: does price reach AH (target = range) after event?
                    # Use remaining KZ bars + post-KZ bars
                    after = kz_bars.iloc[i+1:]
                    after = pd.concat([after, post_kz]) if not after.empty else post_kz
                    tp = al + ar            # target = bottom of range + range size
                    sl = al - pen * 1.5    # SL below penetration
                    hit_tp = (after["high"] >= tp).any()  if not after.empty else False
                    hit_sl = (after["low"]  <= sl).any()  if not after.empty else False
                    if hit_tp and hit_sl:
                        # whichever comes first
                        first_tp = after[after["high"] >= tp].index[0] if hit_tp else pd.NaT
                        first_sl = after[after["low"]  <= sl].index[0] if hit_sl else pd.NaT
                        outcome = 1 if first_tp < first_sl else -1
                    elif hit_tp:
                        outcome = 1
                    elif hit_sl:
                        outcome = -1
                    else:
                        # measure by session close
                        if not after.empty:
                            close_ret = (after["close"].iloc[-1] - al) / ar
                            outcome = 1 if close_ret > 0.3 else (-1 if close_ret < -0.3 else 0)
                        else:
                            outcome = 0

                    records.append({
                        "date": date, "kz": kz_name,
                        "direction": "LONG", "outcome": outcome,
                        "pen_pct": pen / al * 100,
                        "ar_pct": ar / al * 100,
                        "is_nr": is_nr, "cr": cr,
                    })

                # SHORT sweep: high > AH, close <= AH
                if bar["high"] > ah and bar["close"] <= ah:
                    pen = (bar["high"] - ah)
                    after = kz_bars.iloc[i+1:]
                    after = pd.concat([after, post_kz]) if not after.empty else post_kz
                    tp = ah - ar            # target = top of range - range size
                    sl = ah + pen * 1.5
                    hit_tp = (after["low"]  <= tp).any()  if not after.empty else False
                    hit_sl = (after["high"] >= sl).any()  if not after.empty else False
                    if hit_tp and hit_sl:
                        first_tp = after[after["low"]  <= tp].index[0] if hit_tp else pd.NaT
                        first_sl = after[after["high"] >= sl].index[0] if hit_sl else pd.NaT
                        outcome = 1 if first_tp < first_sl else -1
                    elif hit_tp:
                        outcome = 1
                    elif hit_sl:
                        outcome = -1
                    else:
                        if not after.empty:
                            close_ret = (ah - after["close"].iloc[-1]) / ar
                            outcome = 1 if close_ret > 0.3 else (-1 if close_ret < -0.3 else 0)
                        else:
                            outcome = 0

                    records.append({
                        "date": date, "kz": kz_name,
                        "direction": "SHORT", "outcome": outcome,
                        "pen_pct": pen / ah * 100,
                        "ar_pct": ar / ah * 100,
                        "is_nr": is_nr, "cr": cr,
                    })

    df = pd.DataFrame(records)
    if df.empty:
        return {"df": df, "by_kz": {}, "by_nr": {}}

    def win_rate(sub):
        n   = len(sub)
        wins = (sub["outcome"] == 1).sum()
        return {"n": n, "wins": wins, "wr": wins/n*100 if n > 0 else 0}

    by_kz = {}
    for kz in df["kz"].unique():
        sub = df[df["kz"] == kz]
        by_kz[kz] = {
            "all": win_rate(sub),
            "long":  win_rate(sub[sub["direction"] == "LONG"]),
            "short": win_rate(sub[sub["direction"] == "SHORT"]),
            "nr":    win_rate(sub[sub["is_nr"]]),
            "no_nr": win_rate(sub[~sub["is_nr"]]),
        }

    by_nr = {
        "NR days":   win_rate(df[df["is_nr"]]),
        "Non-NR days": win_rate(df[~df["is_nr"]]),
        "All days":    win_rate(df),
    }

    return {"df": df, "by_kz": by_kz, "by_nr": by_nr}


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward
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


def _best_is_params(df_is, sig_is, min_trades: int = 5):
    best_sl, best_tp1, best_metric = 0.5, 1.0, -np.inf
    any_valid = False
    for sl in SL_GRID:
        for tp1 in TP1_GRID:
            try:
                bt = run_backtest(df_is, sig_is,
                                  atr_sl_override=sl, atr_tp1_override=tp1)
            except Exception:
                continue
            trd = bt.get("trades", pd.DataFrame())
            n   = len(trd) if isinstance(trd, pd.DataFrame) else 0
            if n < min_trades:
                continue
            cal = float(bt["kpis"].get("calmar", 0.0))
            ret = float(bt["kpis"].get("total_return", 0.0))
            metric = cal if cal > 0 else ret
            any_valid = True
            if metric > best_metric:
                best_metric = metric
                best_sl, best_tp1 = sl, tp1
    if not any_valid:
        return 0.5, 1.0, 0.0, 0
    return best_sl, best_tp1, best_metric, 0


def run_wf(df_15m, signals, windows, vol_target=None, label="WF",
           min_is_trades: int = 5) -> list[dict]:
    results = []
    for i, (tr_s, tr_e, oo_s, oo_e) in enumerate(windows):
        df_is  = df_15m[(df_15m.index >= tr_s) & (df_15m.index < tr_e)]
        sig_is = signals[(signals.index >= tr_s) & (signals.index < tr_e)]
        df_oos  = df_15m[(df_15m.index >= oo_s) & (df_15m.index < oo_e)]
        sig_oos = signals[(signals.index >= oo_s) & (signals.index < oo_e)]

        if len(df_is) < 200 or len(df_oos) < 20:
            continue
        n_is_sigs = int((sig_is["signal"] != 0).sum())

        best_sl, best_tp1, is_calmar, _ = _best_is_params(
            df_is, sig_is, min_trades=min_is_trades)

        try:
            oos_bt = run_backtest(df_oos, sig_oos,
                                  atr_sl_override=best_sl,
                                  atr_tp1_override=best_tp1,
                                  vol_target=vol_target)
        except Exception as exc:
            continue

        oos_trades = oos_bt.get("trades", pd.DataFrame())
        oos_n   = len(oos_trades) if isinstance(oos_trades, pd.DataFrame) else 0
        oos_ret = float(oos_bt["kpis"].get("total_return", 0.0)) * 100
        oos_dd  = abs(float(oos_bt["kpis"].get("max_drawdown", 0.0))) * 100
        oos_wr  = float(oos_bt["kpis"].get("win_rate", 0.0)) * 100

        results.append({
            "window": i + 1, "oos_s": oo_s, "oos_e": oo_e,
            "is_sigs": n_is_sigs, "is_calmar": round(is_calmar, 3),
            "best_sl": best_sl, "best_tp1": best_tp1,
            "oos_ret": round(oos_ret, 2), "oos_dd": round(oos_dd, 2),
            "oos_n": oos_n, "oos_wr": round(oos_wr, 1),
            "oos_trades": oos_trades,
        })
        print(f"  {label} W{i+1:02d} [{oo_s.date()}→{oo_e.date()}] "
              f"IS={n_is_sigs} sl={best_sl} tp={best_tp1} CalIS={is_calmar:+.2f} "
              f"| OOS ret={oos_ret:+.1f}% DD={oos_dd:.1f}% N={oos_n}")
    return results


def _aggregate_oos(wf_results: list[dict]) -> dict:
    all_pnl, n_wins = [], 0
    for r in wf_results:
        trd = r.get("oos_trades", pd.DataFrame())
        if not isinstance(trd, pd.DataFrame) or trd.empty:
            continue
        pnls = trd["net_pnl"].tolist()
        all_pnl.extend(pnls)
        n_wins += sum(1 for p in pnls if p > 0)

    n = len(all_pnl)
    if n == 0:
        return {"ret": 0.0, "dd": 0.0, "calmar": 0.0,
                "n_trades": 0, "win_rate": 0.0, "sharpe": 0.0}

    eq = np.full(n + 1, float(INIT_CAP))
    for i, p in enumerate(all_pnl):
        eq[i + 1] = eq[i] + p
    ret = (eq[-1] / eq[0] - 1) * 100
    peak = np.maximum.accumulate(eq)
    dd   = abs(((eq - peak) / peak).min()) * 100
    cal  = ret / dd if dd > 0.001 else 0.0
    wr   = n_wins / n * 100
    tr   = np.array(all_pnl) / INIT_CAP * 100
    sr   = tr.mean() / (tr.std(ddof=1) + 1e-9) * np.sqrt(n)
    return {"ret": round(ret,2), "dd": round(dd,2), "calmar": round(cal,3),
            "n_trades": n, "win_rate": round(wr,1), "sharpe": round(sr,3)}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(_CARD)
    ax.tick_params(colors=_TEXT, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(_GRID)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    if title:  ax.set_title(title, color=_TEXT, fontsize=9, pad=5)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, color=_GRID, linewidth=0.5, alpha=0.7)


def _to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=_BG, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _img(b64, w=700):
    return f'<img src="data:image/png;base64,{b64}" width="{w}" style="border-radius:6px;margin:8px 0">'


def plot_asian_range_dist(asian_daily: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor(_BG)

    data = asian_daily["asian_range_pct"].dropna()
    q25  = data.quantile(0.25)
    q75  = data.quantile(0.75)

    ax = axes[0]
    ax.hist(data.clip(0, 5), bins=60, color=_COL["London"], alpha=0.8, edgecolor="none")
    ax.axvline(data.median(), color="white", lw=1.5, ls="--",
               label=f"Median: {data.median():.2f}%")
    ax.axvline(q25, color="#4caf50", lw=1.2, ls=":",
               label=f"Q25 (NR): {q25:.2f}%")
    ax.axvline(q75, color="#ff9800", lw=1.0, ls=":",
               label=f"Q75: {q75:.2f}%")
    _style_ax(ax, "Asian Range Distribution (% of open)", "Range %", "Freq")
    ax.legend(facecolor=_CARD, edgecolor=_GRID, labelcolor=_TEXT, fontsize=8)

    ax2 = axes[1]
    ax2.plot(asian_daily.index, asian_daily["asian_range_pct"].rolling(30).median(),
             color=_COL["London"], lw=1.2, label="30d rolling median")
    ax2.axhline(data.median(), color="white", lw=0.8, ls="--", alpha=0.5)
    ax2.fill_between(asian_daily.index,
                     asian_daily["asian_range_pct"].rolling(30).quantile(0.25),
                     asian_daily["asian_range_pct"].rolling(30).quantile(0.75),
                     alpha=0.15, color=_COL["London"])
    _style_ax(ax2, "Asian Range % — Rolling 30d median (shaded IQR)", "Date", "Range %")
    ax2.legend(facecolor=_CARD, edgecolor=_GRID, labelcolor=_TEXT, fontsize=8)

    fig.tight_layout(pad=1.2)
    return _to_b64(fig)


def plot_sweep_stats(stats: dict) -> str:
    df = stats["df"]
    if df.empty:
        fig, ax = plt.subplots(figsize=(6,3)); fig.patch.set_facecolor(_BG)
        ax.text(0.5, 0.5, "No sweep events", ha="center", color=_TEXT)
        return _to_b64(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor(_BG)
    fig.suptitle("Sweep Event Statistics (raw, before strategy filter)", color=_TEXT, fontsize=10)

    # 1: Win rate by KZ and direction
    ax = axes[0]
    kz_labels, wr_vals, colors, ns = [], [], [], []
    for kz_name, kz_col in [("London", _COL["London"]), ("NY", _COL["NY"])]:
        if kz_name not in stats["by_kz"]:
            continue
        bk = stats["by_kz"][kz_name]
        for label, key, col in [("Long", "long", "#4caf50"), ("Short", "short", "#ef5350")]:
            d = bk[key]
            kz_labels.append(f"{kz_name}\n{label}")
            wr_vals.append(d["wr"])
            colors.append(col)
            ns.append(d["n"])

    bars = ax.bar(kz_labels, wr_vals, color=colors, alpha=0.85, width=0.6)
    ax.axhline(50, color="white", lw=0.8, ls="--", alpha=0.6, label="50%")
    for bar, n, wr in zip(bars, ns, wr_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{wr:.1f}%\nn={n}", ha="center", va="bottom",
                color=_TEXT, fontsize=8)
    _style_ax(ax, "Win Rate by KZ + Direction", "", "Win Rate %")
    ax.set_ylim(30, 80)

    # 2: NR vs Non-NR win rate
    ax2 = axes[1]
    labels2 = list(stats["by_nr"].keys())
    wr2     = [stats["by_nr"][l]["wr"] for l in labels2]
    ns2     = [stats["by_nr"][l]["n"] for l in labels2]
    cols2   = [_COL["NR"], _COL["All"], _COL["London"]][:len(labels2)]
    bars2 = ax2.bar(labels2, wr2, color=cols2, alpha=0.85, width=0.6)
    ax2.axhline(50, color="white", lw=0.8, ls="--", alpha=0.6)
    for bar, n, wr in zip(bars2, ns2, wr2):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{wr:.1f}%\nn={n}", ha="center", va="bottom",
                 color=_TEXT, fontsize=8)
    _style_ax(ax2, "NR Filter Effect on Win Rate", "", "Win Rate %")
    ax2.set_ylim(30, 80)

    # 3: Penetration depth distribution
    ax3 = axes[2]
    for kz_name, kz_col in [("London", _COL["London"]), ("NY", _COL["NY"])]:
        sub = df[df["kz"] == kz_name]["pen_pct"].clip(0, 1.0)
        if len(sub) > 0:
            ax3.hist(sub, bins=40, color=kz_col, alpha=0.6,
                     label=f"{kz_name} (med={sub.median():.3f}%)")
    _style_ax(ax3, "Penetration Depth Distribution (% of level)", "Depth %", "Freq")
    ax3.legend(facecolor=_CARD, edgecolor=_GRID, labelcolor=_TEXT, fontsize=8)

    fig.tight_layout(pad=1.2)
    return _to_b64(fig)


def plot_sweep_calendar(stats: dict) -> str:
    df = stats["df"]
    if df.empty:
        return ""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.patch.set_facecolor(_BG)
    fig.suptitle("Sweep Events — Frequenza e Win Rate Temporale", color=_TEXT, fontsize=10)

    df["year"] = df["date"].dt.year
    df["dow"]  = df["date"].dt.dayofweek  # 0=Mon

    # By year
    ax = axes[0]
    years = sorted(df["year"].unique())
    yr_wr_all = [df[df["year"]==y]["outcome"].apply(lambda x: 1 if x==1 else 0).mean()*100 for y in years]
    yr_n      = [len(df[df["year"]==y]) for y in years]
    yr_nr_wr  = [df[(df["year"]==y)&df["is_nr"]]["outcome"].apply(lambda x: 1 if x==1 else 0).mean()*100
                 if (df["year"]==y).sum()>0 else 0 for y in years]

    x = np.arange(len(years))
    ax.bar(x - 0.2, yr_wr_all, 0.35, color=_COL["All"],    alpha=0.85, label="All sweeps")
    ax.bar(x + 0.2, yr_nr_wr,  0.35, color=_COL["NR"],     alpha=0.85, label="NR days only")
    ax.axhline(50, color="white", lw=0.7, ls="--", alpha=0.5)
    for i, (wr, n) in enumerate(zip(yr_wr_all, yr_n)):
        ax.text(i - 0.2, wr + 0.5, f"n={n}", ha="center", va="bottom",
                color=_TEXT, fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years], color=_TEXT, fontsize=8)
    _style_ax(ax, "Win Rate per Anno", "Anno", "Win Rate %")
    ax.legend(facecolor=_CARD, edgecolor=_GRID, labelcolor=_TEXT, fontsize=8)
    ax.set_ylim(30, 80)

    # By day-of-week
    ax2 = axes[1]
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dows = sorted(df["dow"].unique())
    dw_wr = [df[df["dow"]==d]["outcome"].apply(lambda x: 1 if x==1 else 0).mean()*100 for d in dows]
    dw_n  = [len(df[df["dow"]==d]) for d in dows]
    cols_dw = [_COL["NR"] if wr >= 55 else (_COL["NY"] if wr < 45 else _COL["All"])
               for wr in dw_wr]
    bars = ax2.bar([dow_names[d] for d in dows], dw_wr, color=cols_dw, alpha=0.85, width=0.6)
    ax2.axhline(50, color="white", lw=0.7, ls="--", alpha=0.5)
    for bar, n, wr in zip(bars, dw_n, dw_wr):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"n={n}", ha="center", va="bottom", color=_TEXT, fontsize=7)
    _style_ax(ax2, "Win Rate per Giorno della Settimana", "", "Win Rate %")
    ax2.set_ylim(30, 80)

    fig.tight_layout(pad=1.2)
    return _to_b64(fig)


def plot_oos_equity(wf_results_dict: dict) -> str:
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor(_BG)

    colors_map = {
        "London KZ (NR)":     _COL["NR"],
        "London KZ (All)":    _COL["London"],
        "London+NY KZ (NR)":  "#ab47bc",
    }

    for label, wf_res in wf_results_dict.items():
        all_pnl = []
        dates   = []
        for r in wf_res:
            trd = r.get("oos_trades", pd.DataFrame())
            if isinstance(trd, pd.DataFrame) and not trd.empty and "net_pnl" in trd.columns:
                all_pnl.extend(trd["net_pnl"].tolist())
                dates.extend(trd.index.tolist() if hasattr(trd.index, "tolist") else [r["oos_s"]]*len(trd))
        if not all_pnl:
            continue
        eq = np.full(len(all_pnl)+1, float(INIT_CAP))
        for i, p in enumerate(all_pnl):
            eq[i+1] = eq[i] + p
        col = colors_map.get(label, "#90caf9")
        x_range = range(len(eq))
        ax.plot(x_range, eq, color=col, lw=1.5, label=label)

    ax.axhline(INIT_CAP, color="white", lw=0.7, ls="--", alpha=0.4)
    _style_ax(ax, "OOS Stitched Equity Curve — Varianti ORB-ICT",
              "Trade #", "Equity (USDT)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    ax.legend(facecolor=_CARD, edgecolor=_GRID, labelcolor=_TEXT, fontsize=9)
    fig.tight_layout(pad=1.2)
    return _to_b64(fig)


def plot_nr_compression_vs_outcome(asian_daily: pd.DataFrame,
                                   stats: dict) -> str:
    df = stats["df"]
    if df.empty:
        return ""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor(_BG)
    fig.suptitle("Compression Ratio vs Outcome (Crabel NR Filter Validation)",
                 color=_TEXT, fontsize=10)

    # Scatter: compression_ratio vs win (colored)
    ax = axes[0]
    wins  = df[df["outcome"] ==  1]
    loses = df[df["outcome"] == -1]
    ax.scatter(loses["cr"], loses["pen_pct"],
               color="#ef5350", alpha=0.3, s=12, label="Loss")
    ax.scatter(wins["cr"],  wins["pen_pct"],
               color="#4caf50", alpha=0.3, s=12, label="Win")
    _style_ax(ax, "Compression Ratio vs Penetration Depth",
              "Compression Ratio", "Penetration Depth %")
    ax.legend(facecolor=_CARD, edgecolor=_GRID, labelcolor=_TEXT, fontsize=8)
    ax.set_xlim(0.4, 4.5)

    # Win rate by compression quartile
    ax2 = axes[1]
    df_c = df.copy()
    df_c["cr_q"] = pd.qcut(df_c["cr"], 4,
                            labels=["Q1 Low\nCompression","Q2","Q3","Q4 High\nCompression"])
    qs = ["Q1 Low\nCompression","Q2","Q3","Q4 High\nCompression"]
    qwr = [df_c[df_c["cr_q"]==q]["outcome"].apply(lambda x: 1 if x==1 else 0).mean()*100
           if (df_c["cr_q"]==q).sum()>0 else 0 for q in qs]
    qn  = [(df_c["cr_q"]==q).sum() for q in qs]
    cols = ["#ef5350","#ff9800","#ffeb3b","#4caf50"]
    bars = ax2.bar(qs, qwr, color=cols, alpha=0.85, width=0.6)
    ax2.axhline(50, color="white", lw=0.8, ls="--", alpha=0.5, label="50%")
    for bar, n, wr in zip(bars, qn, qwr):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f"{wr:.1f}%\nn={n}", ha="center", va="bottom",
                 color=_TEXT, fontsize=8)
    _style_ax(ax2, "Win Rate per Quartile Compression Ratio", "", "Win Rate %")
    ax2.set_ylim(30, 80)

    fig.tight_layout(pad=1.2)
    return _to_b64(fig)


# ─────────────────────────────────────────────────────────────────────────────
# HTML
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
body{font-family:system-ui,sans-serif;background:#0f1117;color:#e0e0e0;margin:0;
     padding:32px 24px;max-width:1200px;margin:0 auto}
h1{font-size:1.6rem;color:#fff;border-bottom:2px solid #42a5f5;padding-bottom:8px}
h2{font-size:1.1rem;color:#90caf9;margin-top:36px;border-left:3px solid #42a5f5;padding-left:10px}
h3{font-size:.92rem;color:#b0bec5;margin-top:18px}
p,li{font-size:.88rem;line-height:1.6;color:#cfd8dc}
p.meta{color:#546e7a;font-size:.78rem}
code{background:#1e2130;padding:2px 6px;border-radius:3px;font-size:.82rem;color:#80cbc4}
table{border-collapse:collapse;width:100%;margin-top:12px;font-size:.82rem}
th{background:#1e2130;color:#90caf9;padding:7px 12px;text-align:center;border-bottom:2px solid #42a5f5}
td{padding:6px 12px;border-bottom:1px solid #1e2130;text-align:center}
td:first-child{text-align:left}
tr:hover td{background:#1a1e2e}
.pos{color:#4caf50;font-weight:600} .neg{color:#f44336;font-weight:600}
.warn{color:#ff9800;font-weight:600}
.hl td{background:#1e2a1e!important;border-left:3px solid #4caf50}
.card{background:#12151f;border:1px solid #1e2130;border-radius:8px;padding:16px 20px;margin-top:16px}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px;margin-top:12px}
.kpi{background:#12151f;border:1px solid #1e2130;border-radius:8px;padding:12px;text-align:center}
.kpi .val{font-size:1.3rem;font-weight:700;color:#fff}
.kpi .lbl{font-size:.73rem;color:#78909c;margin-top:4px}
.scroll{overflow-x:auto}
img{max-width:100%;display:block}
.finding{background:#0d2137;border-left:3px solid #42a5f5;padding:10px 14px;margin:8px 0;border-radius:0 6px 6px 0}
.finding strong{color:#90caf9}
"""


def _kpi(val, lbl):
    return f'<div class="kpi"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>'


def _pnl(v):
    return "pos" if v > 0 else ("neg" if v < 0 else "")


def _finding(text):
    return f'<div class="finding">{text}</div>'


def build_html(asian_daily, stats, wf_variants, agg_variants,
               imgs, elapsed):

    # ── summary table
    sum_rows = ""
    best_cal = max(d["calmar"] for d in agg_variants.values()) if agg_variants else 0
    for label, agg in agg_variants.items():
        cal = agg["calmar"]
        cls = "pos" if cal > 0 else "neg"
        hl  = ' class="hl"' if abs(cal - best_cal) < 0.001 else ""
        sum_rows += (
            f"<tr{hl}><td>{label}</td>"
            f'<td class="{_pnl(agg["ret"])}">{agg["ret"]:+.2f}%</td>'
            f'<td class="neg">{agg["dd"]:.1f}%</td>'
            f'<td class="{cls}">{cal:.3f}</td>'
            f'<td>{agg["sharpe"]:.3f}</td>'
            f'<td>{agg["win_rate"]:.1f}%</td>'
            f'<td>{agg["n_trades"]}</td></tr>'
        )

    # ── sweep stats table
    bk = stats.get("by_kz", {})
    bn = stats.get("by_nr", {})
    sweep_rows = ""
    for kz in ["London", "NY"]:
        if kz not in bk:
            continue
        d = bk[kz]
        row = d["all"]
        sweep_rows += (
            f"<tr><td>{kz} KZ</td><td>All</td>"
            f"<td>{row['n']}</td>"
            f'<td class="{_pnl(row["wr"]-50)}">{row["wr"]:.1f}%</td></tr>'
        )
        for sub_lbl, sub_key in [("Long only", "long"), ("Short only", "short"),
                                  ("NR days", "nr"), ("Non-NR days", "no_nr")]:
            r2 = d[sub_key]
            sweep_rows += (
                f"<tr><td style='color:#546e7a'>&nbsp;&nbsp;{kz} — {sub_lbl}</td>"
                f"<td style='color:#546e7a'>{sub_lbl}</td>"
                f"<td>{r2['n']}</td>"
                f'<td class="{_pnl(r2["wr"]-50)}">{r2["wr"]:.1f}%</td></tr>'
            )

    # ── WF detail table (best variant)
    best_variant = max(agg_variants, key=lambda k: agg_variants[k]["calmar"]) if agg_variants else None
    wf_rows = ""
    if best_variant and best_variant in wf_variants:
        for r in wf_variants[best_variant]:
            rc = _pnl(r["oos_ret"])
            ic = _pnl(r["is_calmar"])
            wf_rows += (
                f'<tr><td>W{r["window"]:02d}</td>'
                f'<td>{r["oos_s"].date()}→{r["oos_e"].date()}</td>'
                f'<td>{r["is_sigs"]}</td>'
                f'<td>{r["best_sl"]}×</td>'
                f'<td>{r["best_tp1"]}×</td>'
                f'<td class="{ic}">{r["is_calmar"]:+.3f}</td>'
                f'<td class="{rc}">{r["oos_ret"]:+.2f}%</td>'
                f'<td>{r["oos_dd"]:.1f}%</td>'
                f'<td>{r["oos_n"]}</td>'
                f'<td>{r["oos_wr"]:.0f}%</td></tr>'
            )

    # ── findings
    f_sweep = ""
    if bk:
        lon_all = bk.get("London", {}).get("all", {})
        lon_nr  = bk.get("London", {}).get("nr",  {})
        f_sweep = (
            f"<strong>London KZ sweep:</strong> {lon_all.get('n',0)} eventi totali, "
            f"WR all={lon_all.get('wr',0):.1f}%, "
            f"WR NR days={lon_nr.get('wr',0):.1f}% "
            f"(+{lon_nr.get('wr',0)-lon_all.get('wr',0):.1f}pp con filtro NR)"
        )

    best_agg = agg_variants.get(best_variant, {}) if best_variant else {}
    f_wf = (
        f"<strong>Best variant ({best_variant}):</strong> "
        f"OOS Return={best_agg.get('ret',0):+.2f}%, "
        f"DD={best_agg.get('dd',0):.1f}%, "
        f"Calmar={best_agg.get('calmar',0):.3f}, "
        f"WR={best_agg.get('win_rate',0):.1f}%, "
        f"N={best_agg.get('n_trades',0)} trade"
    ) if best_agg else ""

    return f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<title>ORB-ICT Strategy — BTCUSDT 2020-2026</title>
<style>{_CSS}</style>
</head>
<body>
<h1>ICT Asian Range Sweep — ORB Strategy BTCUSDT</h1>
<p class="meta">
  Periodo: 2020-01 → 2026-05 &middot; Signal: 15M Asian Range Sweep &middot;
  Asian Range: 1H 00:00-06:59 UTC &middot;
  London KZ: 07:00-09:59 UTC &middot; NY KZ: 13:00-15:59 UTC &middot;
  WF: {WF_TRAIN_M}m IS / {WF_OOS_M}m OOS / step {WF_STEP_M}m &middot; {elapsed:.0f}s
</p>

<h2>Executive Summary</h2>
<div class="card">
  {_finding(f_sweep) if f_sweep else ""}
  {_finding(f_wf) if f_wf else ""}
  {_finding("<strong>Framework:</strong> Asian Range (1H, 00-06 UTC) come liquidity reference. London KZ (07-09 UTC) sweeppa il high/low asiatico su 15M. Reversion inside range = entry. NR filter (bottom Q25 Asian range) amplifica il segnale (Crabel compression logic).")}
</div>

<h2>A — Statistiche Asian Range (1H, 2020-2026)</h2>
{_img(imgs.get("asian_range",""), 1000)}
<div class="kpi-grid">
  {_kpi(f"{asian_daily['asian_range_pct'].median():.2f}%", "Median Range %")}
  {_kpi(f"{asian_daily['asian_range_pct'].quantile(0.25):.2f}%", "Q25 (NR soglia)")}
  {_kpi(f"{asian_daily['asian_range_pct'].quantile(0.75):.2f}%", "Q75")}
  {_kpi(f"{asian_daily['is_nr'].sum()}", "Giorni NR (Q25)")}
  {_kpi(f"{len(asian_daily)}", "Giorni totali")}
</div>

<h2>B — Statistiche Sweep Events (raw, no strategy filter)</h2>
<p>Win = prezzo raggiunge il target (≥ 1× Asian Range dalla entry) prima dello stop (1.5× penetrazione).</p>
{_img(imgs.get("sweep_stats",""), 1000)}
{_img(imgs.get("sweep_calendar",""), 1000)}
{_img(imgs.get("nr_compression",""), 1000)}

<div class="scroll">
<table>
<thead><tr><th>Killzone</th><th>Subset</th><th>N eventi</th><th>Win Rate</th></tr></thead>
<tbody>{sweep_rows}</tbody>
</table>
</div>

<h2>C — Walk-Forward Backtest (15M, OOS stitched)</h2>
<p>IS grid: SL ∈ {{{", ".join(str(x) for x in SL_GRID)}}} × TP1 ∈ {{{", ".join(str(x) for x in TP1_GRID)}}} × ATR.
Minimo 5 trade IS. Criterio selezione: Calmar (o return se Calmar &le; 0).</p>

<div class="scroll">
<table>
<thead><tr><th>Variante</th><th>Return OOS</th><th>Max DD</th>
  <th>Calmar</th><th>Sharpe</th><th>Win Rate</th><th>Trades</th></tr></thead>
<tbody>{sum_rows}</tbody>
</table>
</div>

{_img(imgs.get("oos_equity",""), 1000)}

<h2>D — Walk-Forward Detail ({best_variant or "n/a"})</h2>
<div class="scroll">
<table>
<thead><tr>
  <th>#</th><th>OOS Window</th><th>IS Sigs</th>
  <th>Best SL</th><th>Best TP1</th><th>IS Calmar</th>
  <th>OOS Ret</th><th>OOS DD</th><th>OOS N</th><th>OOS WR</th>
</tr></thead>
<tbody>{wf_rows}</tbody>
</table>
</div>

<h2>E — Logica Strategia</h2>
<div class="card">
  <h3>Definizione Asian Range (1H)</h3>
  <p>Asian High = max(high) delle barre ore 00-06 UTC. Asian Low = min(low) stesse barre.
  Il range viene calcolato sulla barra <em>precedente</em> al giorno di trading:
  il range del giorno D viene usato per il trading del giorno D (le 7 barre 00-06
  chiudono prima che inizi la London KZ alle 07).</p>

  <h3>NR Filter (Crabel)</h3>
  <p>Il filtro NR (Narrow Range) emette segnale solo quando il range asiatico del giorno
  &egrave; nel quartile inferiore (Q25) degli ultimi 20 giorni. Questo implementa la logica
  di Crabel: <em>volatility contraction precedes expansion</em> — i giorni con range
  stretto precedono i breakout pi&ugrave; ampi.</p>

  <h3>Sweep Detection (15M, London KZ 07-09 UTC)</h3>
  <p><strong>LONG setup</strong>: una barra 15M nel London KZ ha il low sotto Asian Low
  E chiude sopra Asian Low (falsa rottura del supporto). Entry sul open della barra successiva.</p>
  <p><strong>SHORT setup</strong>: una barra 15M nel London KZ ha il high sopra Asian High
  E chiude sotto Asian High (falsa rottura della resistenza). Entry sul open della barra successiva.</p>

  <h3>SL / TP (ottimizzati via IS)</h3>
  <p>SL = entry &plusmn; atr_sl &times; ATR(14 su 15M). TP1 = entry &plusmn; atr_tp1 &times; ATR(14).
  TP2 = 2&times;TP1, TP3 = 3&times;TP1. Dopo TP1 hit: stop a break-even sulla quota residua.</p>

  <h3>Composite Score</h3>
  <p>Score = compression_ratio &times; (penetration_depth / ATR) &times; 2.
  Giorni pi&ugrave; compressi + penetrazione pi&ugrave; profonda = score maggiore.</p>
</div>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("══ ORB-ICT Strategy Report ══\n")
    t0 = time.time()

    # ── [1] Dati
    print("[1/6] Loading data …")
    raw  = fetch_extended_data(start_year=START_YEAR, start_month=START_MONTH,
                               fetch_flow=False, fetch_15m=True, fetch_1m=False)
    df_1h  = add_indicators(raw["1H"])
    df_15m = add_indicators(raw["15M"])
    print(f"  1H : {len(df_1h):,} bars")
    print(f"  15M: {len(df_15m):,} bars")

    # ── [2] Asian Range
    print("[2/6] Computing Asian Range (1H) …")
    asian_daily = build_asian_range(df_1h)
    nr_days = asian_daily["is_nr"].sum()
    print(f"  {len(asian_daily)} days | NR days (Q25): {nr_days} | "
          f"median range: {asian_daily['asian_range_pct'].median():.2f}%")

    # ── [3] Sweep statistics (research)
    print("[3/6] Computing sweep statistics …")
    stats = compute_sweep_stats(df_15m, asian_daily)
    df_sw = stats["df"]
    print(f"  Total sweep events: {len(df_sw)}")
    for kz, d in stats.get("by_kz", {}).items():
        print(f"  {kz}: n={d['all']['n']}  WR_all={d['all']['wr']:.1f}%  "
              f"WR_NR={d['nr']['wr']:.1f}%  WR_Long={d['long']['wr']:.1f}%  "
              f"WR_Short={d['short']['wr']:.1f}%")
    for k, v in stats.get("by_nr", {}).items():
        print(f"  {k}: n={v['n']}  WR={v['wr']:.1f}%")

    # ── [4] Build 15M signals (3 variants)
    print("[4/6] Building 15M signals …")
    sigs_lon_nr  = build_orb_ict_signals(df_15m, asian_daily,
                                          use_london_kz=True,  use_ny_kz=False, nr_filter=True)
    sigs_lon_all = build_orb_ict_signals(df_15m, asian_daily,
                                          use_london_kz=True,  use_ny_kz=False, nr_filter=False)
    sigs_both_nr = build_orb_ict_signals(df_15m, asian_daily,
                                          use_london_kz=True,  use_ny_kz=True,  nr_filter=True)

    for label, sig in [("London KZ (NR)", sigs_lon_nr),
                       ("London KZ (All)", sigs_lon_all),
                       ("London+NY KZ (NR)", sigs_both_nr)]:
        n = int((sig["signal"] != 0).sum())
        nl = int((sig["signal"] == 1).sum())
        ns = int((sig["signal"] == -1).sum())
        print(f"  {label}: {n} signals ({nl} long, {ns} short)")

    # ── [5] Walk-forward
    print("[5/6] Walk-forward validation …")
    windows = _wf_windows(df_15m.index)
    print(f"  WF windows: {len(windows)}")

    wf_variants: dict[str, list[dict]] = {}
    agg_variants: dict[str, dict]      = {}

    for label, sigs in [
        ("London KZ (NR)",    sigs_lon_nr),
        ("London KZ (All)",   sigs_lon_all),
        ("London+NY KZ (NR)", sigs_both_nr),
    ]:
        print(f"\n  ── {label} ──")
        wf = run_wf(df_15m, sigs, windows, label=label, min_is_trades=5)
        agg = _aggregate_oos(wf)
        wf_variants[label]  = wf
        agg_variants[label] = agg
        print(f"  OOS: ret={agg['ret']:+.2f}%  DD={agg['dd']:.1f}%  "
              f"Calmar={agg['calmar']:.3f}  N={agg['n_trades']}  WR={agg['win_rate']:.1f}%")

    # ── [6] Charts
    print("\n[6/6] Generating charts …")
    imgs = {
        "asian_range":   plot_asian_range_dist(asian_daily),
        "sweep_stats":   plot_sweep_stats(stats),
        "sweep_calendar":plot_sweep_calendar(stats),
        "nr_compression":plot_nr_compression_vs_outcome(asian_daily, stats),
        "oos_equity":    plot_oos_equity(wf_variants),
    }

    elapsed = time.time() - t0
    html = build_html(asian_daily, stats, wf_variants, agg_variants, imgs, elapsed)

    out = Path("reports/report_orb_strategy.html")
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"\n  → {out}  ({out.stat().st_size//1024}KB, {elapsed:.0f}s)")


if __name__ == "__main__":
    main()
