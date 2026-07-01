"""
create_wyckoff_multitf_report.py
──────────────────────────────────
Wyckoff Spring/Upthrust strategy tested across four timeframes:
  15M · 1H · 4H · 1D

For each timeframe:
  • n_range and range_width_max are scaled to represent ~2-4 weeks of price
    action in that bar size
  • Walk-forward: 6m train / 2m OOS / step 2m  (~35 windows)
  • IS grid search: atr_sl ∈ {0.5,1.0,1.5,2.0,2.5}
                   × atr_tp1 ∈ {1.0,1.5,2.0,3.0}  → 20 combos
  • OOS stitched equity → Calmar, Return, Max DD

Output → reports/report_wyckoff_multitf.html

Usage
─────
  python create_wyckoff_multitf_report.py
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from src.strategy.data_fetcher  import fetch_extended_data
from src.strategy.indicators    import add_indicators
from src.strategy.engine        import run_backtest, INIT_CAP
from src.strategy.wyckoff       import build_wyckoff_signals

START_YEAR  = 2020
START_MONTH = 1
WF_TRAIN_M  = 6
WF_OOS_M    = 2
WF_STEP_M   = 2

# ── Per-TF Wyckoff parameters ────────────────────────────────────────────────
# n_range represents roughly how many bars span the consolidation zone:
#   15M → 192 bars = 2 days
#   1H  →  48 bars = 2 days
#   4H  →  42 bars = 7 days
#   1D  →  30 bars = 1 month
#
# range_width_max widens for longer TFs because BTC ranges are naturally bigger.
# session_hours = (8,21) for sub-daily; None for 4H/1D (24h crypto market).

TF_PARAMS: dict[str, dict] = {
    "15M": {
        "df_key":          "15M",
        "label":           "15-minute",
        "n_range":         192,
        "adx_max":         25.0,
        "range_width_max": 0.07,
        "vol_threshold":   1.3,
        "session_hours":   (8, 21),
        "min_is_trades":   8,
    },
    "1H": {
        "df_key":          "1H",
        "label":           "1-hour",
        "n_range":         48,
        "adx_max":         25.0,
        "range_width_max": 0.10,
        "vol_threshold":   1.3,
        "session_hours":   (8, 21),
        "min_is_trades":   5,
    },
    "4H": {
        "df_key":          "4H",
        "label":           "4-hour",
        "n_range":         42,
        "adx_max":         25.0,
        "range_width_max": 0.12,
        "vol_threshold":   1.3,
        "session_hours":   None,
        "min_is_trades":   5,
    },
    "1D": {
        "df_key":          "1D",
        "label":           "Daily",
        "n_range":         30,
        "adx_max":         25.0,
        "range_width_max": 0.25,
        "vol_threshold":   1.3,
        "session_hours":   None,
        "min_is_trades":   3,
    },
}

SL_GRID  = [0.5, 1.0, 1.5, 2.0, 2.5]
TP1_GRID = [1.0, 1.5, 2.0, 3.0]


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward windows
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


# ─────────────────────────────────────────────────────────────────────────────
# IS grid search
# ─────────────────────────────────────────────────────────────────────────────

def _best_params_is(
    df_tf: pd.DataFrame,
    sig_tf: pd.DataFrame,
    min_trades: int = 5,
) -> tuple[float, float, float, int]:
    """Return (best_sl, best_tp1, best_metric, n_trades_of_best)."""
    best_sl, best_tp1, best_metric = 2.0, 2.0, -np.inf
    best_n = 0
    any_valid = False

    for sl in SL_GRID:
        for tp1 in TP1_GRID:
            try:
                bt = run_backtest(df_tf, sig_tf,
                                  atr_sl_override=sl, atr_tp1_override=tp1)
            except Exception:
                continue
            trd = bt.get("trades", pd.DataFrame())
            n = len(trd) if isinstance(trd, pd.DataFrame) else 0
            if n < min_trades:
                continue
            cal    = float(bt["kpis"].get("calmar", 0.0))
            ret    = float(bt["kpis"].get("total_return", 0.0))
            metric = cal if cal > 0 else ret
            any_valid = True
            if metric > best_metric:
                best_metric = metric
                best_sl, best_tp1, best_n = sl, tp1, n

    if not any_valid:
        return 2.0, 2.0, 0.0, 0
    return best_sl, best_tp1, best_metric, best_n


# ─────────────────────────────────────────────────────────────────────────────
# Per-TF walk-forward run
# ─────────────────────────────────────────────────────────────────────────────

def run_wf_tf(
    df_tf: pd.DataFrame,
    signals: pd.DataFrame,
    windows: list[tuple],
    min_is_trades: int,
    tf_label: str,
) -> list[dict]:
    results = []
    for i, (tr_s, tr_e, oo_s, oo_e) in enumerate(windows):
        df_is  = df_tf.loc[(df_tf.index >= tr_s) & (df_tf.index < tr_e)]
        sig_is = signals.loc[(signals.index >= tr_s) & (signals.index < tr_e)]
        df_oos  = df_tf.loc[(df_tf.index >= oo_s) & (df_tf.index < oo_e)]
        sig_oos = signals.loc[(signals.index >= oo_s) & (signals.index < oo_e)]

        if len(df_is) < 30 or len(df_oos) < 5:
            continue

        n_is_sigs = int((sig_is["signal"] != 0).sum())
        best_sl, best_tp1, is_cal, is_n = _best_params_is(
            df_is, sig_is, min_trades=min_is_trades,
        )

        try:
            oos_bt = run_backtest(df_oos, sig_oos,
                                  atr_sl_override=best_sl,
                                  atr_tp1_override=best_tp1)
        except Exception as exc:
            print(f"    [{tf_label}] W{i+1:02d} OOS error: {exc}")
            continue

        oos_trd = oos_bt.get("trades", pd.DataFrame())
        oos_n   = len(oos_trd) if isinstance(oos_trd, pd.DataFrame) else 0
        oos_ret = float(oos_bt["kpis"].get("total_return", 0.0)) * 100
        oos_dd  = abs(float(oos_bt["kpis"].get("max_drawdown", 0.0))) * 100
        oos_wr  = float(oos_bt["kpis"].get("win_rate", 0.0)) * 100

        results.append({
            "window":    i + 1,
            "oos_s":     oo_s,
            "oos_e":     oo_e,
            "is_sigs":   n_is_sigs,
            "is_calmar": round(is_cal, 3),
            "is_n":      is_n,
            "best_sl":   best_sl,
            "best_tp1":  best_tp1,
            "oos_ret":   round(oos_ret, 2),
            "oos_dd":    round(oos_dd,  2),
            "oos_n":     oos_n,
            "oos_wr":    round(oos_wr,  1),
            "oos_trades": oos_trd,
        })

        print(f"    W{i+1:02d} [{oo_s.date()}→{oo_e.date()}] "
              f"IS sigs={n_is_sigs} sl={best_sl} tp={best_tp1} "
              f"Cal={is_cal:+.2f} N={is_n} | "
              f"OOS ret={oos_ret:+.1f}% DD={oos_dd:.1f}% N={oos_n}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate OOS from stitched trades
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_oos(wf_results: list[dict]) -> dict:
    all_pnl: list[float] = []
    n_wins = 0
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
                "n_trades": 0, "win_rate": 0.0, "sharpe": 0.0,
                "oos_windows": 0, "pos_windows": 0}

    eq = np.full(n + 1, float(INIT_CAP))
    for i, p in enumerate(all_pnl):
        eq[i + 1] = eq[i] + p
    total_ret = (eq[-1] / eq[0] - 1.0) * 100
    peak = np.maximum.accumulate(eq)
    max_dd = abs(((eq - peak) / peak).min()) * 100
    calmar = total_ret / max_dd if max_dd > 0.001 else 0.0
    win_rate = n_wins / n * 100

    rets_arr = np.array(all_pnl) / INIT_CAP * 100
    sharpe = float(rets_arr.mean() / (rets_arr.std(ddof=1) + 1e-9) * np.sqrt(n))

    pos_win = sum(1 for r in wf_results if r["oos_ret"] > 0)

    return {
        "ret":         round(float(total_ret), 2),
        "dd":          round(float(max_dd),    2),
        "calmar":      round(float(calmar),    3),
        "n_trades":    int(n),
        "win_rate":    round(float(win_rate),  1),
        "sharpe":      round(float(sharpe),    3),
        "oos_windows": len(wf_results),
        "pos_windows": int(pos_win),
    }


def _ttest(wf_results: list[dict]) -> tuple[float, float]:
    rets = np.array([r["oos_ret"] for r in wf_results])
    if len(rets) < 3 or rets.std() == 0:
        return 0.0, 1.0
    t, _ = st.ttest_1samp(rets, 0)
    p    = float(st.t.sf(float(t), df=len(rets) - 1))
    return round(float(t), 2), round(p, 4)


# ─────────────────────────────────────────────────────────────────────────────
# HTML helpers
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
body{font-family:system-ui,sans-serif;background:#0f1117;color:#e0e0e0;margin:0;padding:24px}
h1{font-size:1.6rem;color:#fff;border-bottom:2px solid #2196f3;padding-bottom:8px}
h2{font-size:1.1rem;color:#90caf9;margin-top:32px}
h3{font-size:.95rem;color:#b0bec5;margin-top:20px}
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
.hl td{background:#1e2a1e!important;border-left:3px solid #4caf50}
.card{background:#12151f;border:1px solid #1e2130;border-radius:8px;
      padding:16px;margin-top:16px}
.tf-section{border-top:1px solid #2196f3;margin-top:40px;padding-top:8px}
.scroll{overflow-x:auto}
.badge{display:inline-block;padding:2px 10px;border-radius:10px;
       font-size:.8rem;font-weight:600;margin-right:8px}
.b15{background:#1a237e;color:#90caf9}
.b1h{background:#1b5e20;color:#a5d6a7}
.b4h{background:#4a148c;color:#ce93d8}
.b1d{background:#b71c1c;color:#ffcdd2}
"""


def _pnl_cls(v: float) -> str:
    return "pos" if v > 0 else ("neg" if v < 0 else "")


def _summary_table(tf_results: dict[str, dict]) -> str:
    rows = ""
    best_calmar = max(r["agg"]["calmar"] for r in tf_results.values())
    for tf_key, res in tf_results.items():
        agg  = res["agg"]
        t, p = res["ttest"]
        cal  = agg["calmar"]
        cal_c = "pos" if cal > 0.5 else ("warn" if cal > 0 else "neg")
        sig_flag = " ★" if p < 0.05 else ""
        hl = ' class="hl"' if cal >= best_calmar - 0.001 else ""
        rows += (
            f'<tr{hl}>'
            f'<td><span class="badge b{tf_key.lower()}">{tf_key}</span>'
            f'{TF_PARAMS[tf_key]["label"]}</td>'
            f'<td class="{_pnl_cls(agg["ret"])}">{agg["ret"]:+.2f}%</td>'
            f'<td class="neg">{agg["dd"]:.1f}%</td>'
            f'<td class="{cal_c}">{cal:.3f}</td>'
            f'<td>{agg["sharpe"]:.3f}</td>'
            f'<td>{agg["win_rate"]:.1f}%</td>'
            f'<td>{agg["n_trades"]}</td>'
            f'<td>{agg["pos_windows"]}/{agg["oos_windows"]}</td>'
            f'<td>{t:.2f} / {p:.4f}{sig_flag}</td>'
            f'</tr>'
        )
    return (
        '<div class="scroll"><table>'
        '<thead><tr>'
        '<th>Timeframe</th><th>OOS Return</th><th>Max DD</th>'
        '<th>Calmar</th><th>Sharpe</th><th>Win Rate</th><th>Trades</th>'
        '<th>Pos Windows</th><th>t / p-val</th>'
        '</tr></thead>'
        f'<tbody>{rows}</tbody></table></div>'
    )


def _calmar_bar(tf_results: dict[str, dict]) -> str:
    vals = [(k, v["agg"]["calmar"], v["agg"]["dd"], v["agg"]["ret"],
             v["agg"]["n_trades"]) for k, v in tf_results.items()]
    max_abs = max(abs(x[1]) for x in vals) + 0.1
    rows = ""
    badge_cls = {"15M": "b15", "1H": "b1h", "4H": "b4h", "1D": "b1d"}
    for tf_key, cal, dd, ret, nt in vals:
        w   = min(abs(cal) / max_abs * 260, 260)
        col = "#4caf50" if cal > 0 else "#f44336"
        rows += (
            f'<tr>'
            f'<td style="width:80px">'
            f'<span class="badge {badge_cls[tf_key]}">{tf_key}</span></td>'
            f'<td><div style="background:{col};height:14px;width:{w:.0f}px;'
            f'border-radius:3px;display:inline-block"></div> {cal:.3f}</td>'
            f'<td style="color:#f44336">{dd:.1f}%</td>'
            f'<td class="{_pnl_cls(ret)}">{ret:+.1f}%</td>'
            f'<td>{nt} trades</td>'
            f'</tr>'
        )
    return (
        '<table><thead><tr>'
        '<th>TF</th><th>Calmar</th><th>Max DD</th>'
        '<th>Return</th><th>Trades</th>'
        '</tr></thead><tbody>' + rows + '</tbody></table>'
    )


def _param_freq(wf_results: list[dict], top_n: int = 5) -> str:
    from collections import Counter
    cnt = Counter((r["best_sl"], r["best_tp1"]) for r in wf_results)
    total = sum(cnt.values())
    rows = ""
    for (sl, tp), n in sorted(cnt.items(), key=lambda x: -x[1])[:top_n]:
        w = n / total * 100
        rows += (
            f'<tr><td>SL={sl:.1f}×  TP={tp:.1f}×</td>'
            f'<td>{n}</td><td>{w:.0f}%</td>'
            f'<td><div style="background:#2196f3;height:8px;'
            f'width:{w*2:.0f}px;border-radius:3px;display:inline-block">'
            f'</div></td></tr>'
        )
    return (
        '<table><thead><tr><th>Params</th><th>Count</th>'
        '<th>Freq</th><th></th></tr></thead>'
        f'<tbody>{rows}</tbody></table>'
    )


def _wf_detail_table(wf_results: list[dict]) -> str:
    rows = ""
    for r in wf_results:
        rc  = _pnl_cls(r["oos_ret"])
        isc = _pnl_cls(r["is_calmar"])
        rows += (
            f'<tr>'
            f'<td>W{r["window"]:02d}</td>'
            f'<td>{r["oos_s"].date()}→{r["oos_e"].date()}</td>'
            f'<td>{r["is_sigs"]}</td>'
            f'<td>{r["best_sl"]:.1f}×</td>'
            f'<td>{r["best_tp1"]:.1f}×</td>'
            f'<td class="{isc}">{r["is_calmar"]:+.3f}</td>'
            f'<td>{r["is_n"]}</td>'
            f'<td class="{rc}">{r["oos_ret"]:+.2f}%</td>'
            f'<td>{r["oos_dd"]:.1f}%</td>'
            f'<td>{r["oos_n"]}</td>'
            f'<td>{r["oos_wr"]:.0f}%</td>'
            f'</tr>'
        )
    return (
        '<div class="scroll"><table>'
        '<thead><tr>'
        '<th>#</th><th>OOS Window</th><th>IS Sigs</th>'
        '<th>Best SL</th><th>Best TP</th>'
        '<th>IS Cal</th><th>IS N</th>'
        '<th>OOS Ret</th><th>OOS DD</th><th>OOS N</th><th>OOS WR</th>'
        '</tr></thead>'
        f'<tbody>{rows}</tbody></table></div>'
    )


def _signal_counts_row(tf_key: str, n_long: int, n_short: int, cfg: dict) -> str:
    badge_cls = {"15M": "b15", "1H": "b1h", "4H": "b4h", "1D": "b1d"}
    sess = f'{cfg["session_hours"]}' if cfg["session_hours"] else "24h"
    return (
        f'<tr>'
        f'<td><span class="badge {badge_cls[tf_key]}">{tf_key}</span>'
        f'  {cfg["label"]}</td>'
        f'<td>{cfg["n_range"]} bars</td>'
        f'<td>{cfg["range_width_max"]*100:.0f}%</td>'
        f'<td>{cfg["adx_max"]}</td>'
        f'<td>{sess}</td>'
        f'<td class="pos">{n_long}</td>'
        f'<td class="neg">{n_short}</td>'
        f'<td>{n_long+n_short}</td>'
        f'</tr>'
    )


def build_html(
    tf_results: dict[str, dict],
    elapsed: float,
) -> str:

    summary   = _summary_table(tf_results)
    cal_bars  = _calmar_bar(tf_results)

    # Signal count table
    sig_rows = "".join(
        _signal_counts_row(k, v["n_long"], v["n_short"], TF_PARAMS[k])
        for k, v in tf_results.items()
    )
    sig_table = (
        '<table><thead><tr>'
        '<th>Timeframe</th><th>n_range</th><th>Max Width</th>'
        '<th>ADX max</th><th>Session</th>'
        '<th class="pos">Springs (L)</th><th class="neg">Upthrusts (S)</th>'
        '<th>Total</th>'
        '</tr></thead>'
        f'<tbody>{sig_rows}</tbody></table>'
    )

    # Per-TF sections
    tf_sections = ""
    badge_cls = {"15M": "b15", "1H": "b1h", "4H": "b4h", "1D": "b1d"}
    for tf_key, res in tf_results.items():
        agg    = res["agg"]
        t, p   = res["ttest"]
        detail = _wf_detail_table(res["wf"])
        pfreq  = _param_freq(res["wf"])
        pct_pos = res["agg"]["pos_windows"] / res["agg"]["oos_windows"] * 100 \
                  if res["agg"]["oos_windows"] > 0 else 0.0
        tf_sections += f"""
<div class="tf-section">
<h2><span class="badge {badge_cls[tf_key]}">{tf_key}</span>
    {TF_PARAMS[tf_key]["label"]} — Wyckoff WF-optimised</h2>
<p class="meta">
  Springs: {res['n_long']} · Upthrusts: {res['n_short']} · Total signals: {res['n_long']+res['n_short']} ·
  OOS stitched: ret={agg['ret']:+.1f}%  DD={agg['dd']:.1f}%  Calmar={agg['calmar']:.3f}
  Trades={agg['n_trades']} · Win rate={agg['win_rate']:.1f}% ·
  Positive windows: {agg['pos_windows']}/{agg['oos_windows']} ({pct_pos:.0f}%) ·
  t={t:.2f}  p={p:.4f}{"  ★" if p < 0.05 else ""}
</p>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:12px">
  <div class="card">
  <h3>Walk-Forward Detail</h3>
  {detail}
  </div>
  <div class="card">
  <h3>IS-Selected Param Frequency</h3>
  {pfreq}
  </div>
</div>
</div>
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Wyckoff Multi-TF — BTCUSDT 2020-2026</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Wyckoff Spring/Upthrust — Multi-Timeframe Comparison</h1>
<p class="meta">
BTCUSDT Perpetual Futures · Period: 2020-01 → 2026-05 ·
WF: {WF_TRAIN_M}m train / {WF_OOS_M}m OOS / step {WF_STEP_M}m ·
IS grid: SL∈{{0.5,1.0,1.5,2.0,2.5}} × TP∈{{1.0,1.5,2.0,3.0}} ·
Fees: 0.04% per side · Generated in {elapsed:.0f}s
</p>
<p class="meta">
All OOS metrics are computed on stitched OOS trades (each 2m window contributes trades
starting from a shared equity base). Positive-window rate = fraction of 2m windows
where OOS P&amp;L &gt; 0. ★ = t-test p &lt; 0.05 (OOS returns &gt; 0).
</p>

<h2>Summary — All Timeframes (OOS stitched)</h2>
{summary}

<h2>Calmar Ratio by Timeframe</h2>
<div class="card">{cal_bars}</div>

<h2>Signal Counts by Timeframe &amp; Parameters</h2>
<div class="card">{sig_table}</div>

{tf_sections}

<h2>Methodology Notes</h2>
<div class="card">
<p><strong>Range detection</strong>: rolling max(high) and min(low) over the last <em>n_range</em>
bars (shifted 1 bar to prevent look-ahead), classified as ranging when ADX &lt; adx_max
AND bandwidth &lt; range_width_max.</p>
<p><strong>Accumulation / Distribution bias</strong>: OBV above/below its 21-bar EMA
within a ranging period — smart money absorbing vs unloading supply.</p>
<p><strong>Spring (long)</strong>: bar pierces below range support AND closes back inside,
on vol_ratio ≥ 1.3, in accumulation context. Entry on next bar open.</p>
<p><strong>Upthrust (short)</strong>: bar pierces above range resistance AND closes back inside,
on elevated volume, in distribution context. Entry on next bar open.</p>
<p><strong>IS optimisation</strong>: for each 6-month window, 20 (SL, TP1) combos are evaluated
on IS trades. Best combo (highest Calmar or return if Calmar &le; 0) applied to 2m OOS window.</p>
</div>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("══ Wyckoff Multi-TF Report ══\n")
    t0 = time.time()

    # ── Load data (all TFs at once) ───────────────────────────────────────────
    print("[1/3] Loading data …")
    raw = fetch_extended_data(
        start_year=START_YEAR, start_month=START_MONTH,
        fetch_flow=True, fetch_15m=True, fetch_1m=False,
    )
    tf_data: dict[str, pd.DataFrame] = {}
    for tf_key in ["15M", "1H", "4H", "1D"]:
        df = raw.get(tf_key, pd.DataFrame())
        tf_data[tf_key] = add_indicators(df) if not df.empty and len(df) > 50 else df
        if not tf_data[tf_key].empty:
            df_ = tf_data[tf_key]
            print(f"  [{tf_key}] {len(df_):,} bars  "
                  f"[{df_.index[0].date()} → {df_.index[-1].date()}]")
        else:
            print(f"  [{tf_key}] empty — skipping")

    # ── Build Wyckoff signals + WF for each TF ────────────────────────────────
    print("\n[2/3] Signals + Walk-forward …")
    tf_results: dict[str, dict] = {}

    for tf_key, params in TF_PARAMS.items():
        df_tf = tf_data.get(tf_key, pd.DataFrame())
        if df_tf.empty:
            print(f"  [{tf_key}] skipped (no data)")
            continue

        print(f"\n  ── {tf_key} ({params['label']}) ──")

        # Build signals with TF-appropriate parameters
        signals = build_wyckoff_signals(
            df_tf,
            n_range         = params["n_range"],
            adx_max         = params["adx_max"],
            range_width_max = params["range_width_max"],
            vol_threshold   = params["vol_threshold"],
            session_hours   = params["session_hours"],
        )
        n_long  = int((signals["signal"] ==  1).sum())
        n_short = int((signals["signal"] == -1).sum())
        print(f"    Springs: {n_long}  Upthrusts: {n_short}  Total: {n_long+n_short}")

        windows = _wf_windows(df_tf.index)
        print(f"    WF windows: {len(windows)}")

        wf = run_wf_tf(df_tf, signals, windows,
                       min_is_trades=params["min_is_trades"],
                       tf_label=tf_key)

        agg   = _aggregate_oos(wf)
        t, p  = _ttest(wf)

        tf_results[tf_key] = {
            "n_long":  n_long,
            "n_short": n_short,
            "wf":      wf,
            "agg":     agg,
            "ttest":   (t, p),
        }

        print(f"    → OOS aggregate: ret={agg['ret']:+.1f}%  DD={agg['dd']:.1f}%  "
              f"Calmar={agg['calmar']:.3f}  N={agg['n_trades']}  "
              f"t={t:.2f}  p={p:.4f}")

    # ── HTML ──────────────────────────────────────────────────────────────────
    print("\n[3/3] Generating HTML …")
    elapsed = time.time() - t0
    html    = build_html(tf_results, elapsed)

    out = Path("reports/report_wyckoff_multitf.html")
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"  → {out}  ({elapsed:.0f}s)")

    # Final summary
    print("\n── Final Summary ──")
    print(f"  {'TF':<5} {'Return':>8} {'Max DD':>8} {'Calmar':>8} {'Trades':>7}")
    print(f"  {'─'*5} {'─'*8} {'─'*8} {'─'*8} {'─'*7}")
    for tf_key, res in tf_results.items():
        a = res["agg"]
        print(f"  {tf_key:<5} {a['ret']:>+8.1f}% {a['dd']:>7.1f}% "
              f"{a['calmar']:>8.3f} {a['n_trades']:>7}")


if __name__ == "__main__":
    main()
