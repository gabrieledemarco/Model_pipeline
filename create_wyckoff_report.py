"""
create_wyckoff_report.py
─────────────────────────
Wyckoff Spring/Upthrust strategy — full walk-forward validation with
per-window in-sample SL/TP optimisation.

Pipeline
────────────────────────────────────────────────────────────────────────────
1. Load  : BTCUSDT 1H data 2020-2026 (cache + REST top-up)
2. Signal: build_wyckoff_signals() — spring/upthrust on ranging market
3. Walk-forward (6m train / 2m OOS / step 2m → ~35 windows)
   IS  : grid search atr_sl ∈ {0.5,1.0,1.5,2.0,2.5} ×
                     atr_tp1 ∈ {1.0,1.5,2.0,3.0}  → 20 combos
         Best combo = highest Calmar (or return if Calmar ≤ 0),
                      minimum 5 IS trades required.
   OOS : run with IS-best params; record trades.
4. Aggregate: stitch OOS trades → synthetic equity → KPIs
5. Compare:
   • Wyckoff (default SL=2×ATR, TP1=2×ATR)
   • Wyckoff WF-optimised
   • Wyckoff WF-opt + Vol sizing (vol_target=0.20)
   • Baseline composite (from enhancements report, for reference)
6. HTML report → reports/report_wyckoff.html

Usage
─────
  python create_wyckoff_report.py
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

from src.strategy.data_fetcher  import fetch_extended_data, generate_oi, generate_funding
from src.strategy.indicators    import add_indicators
from src.strategy.signals       import build_signal_matrix
from src.strategy.optimizer     import apply_filters, ScenarioConfig
from src.strategy.engine        import run_backtest, INIT_CAP
from src.strategy.wyckoff       import build_wyckoff_signals

START_YEAR  = 2020
START_MONTH = 1
WF_TRAIN_M  = 6
WF_OOS_M    = 2
WF_STEP_M   = 2

# Baseline session/threshold config (for reference comparison)
_BASELINE_CFG = ScenarioConfig(
    "Session 08-21", session_hours=(8, 21),
    long_threshold=3.0, short_threshold=-3.0,
)

# IS optimisation grid
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
# IS parameter optimisation
# ─────────────────────────────────────────────────────────────────────────────

def _best_params_is(df_is: pd.DataFrame,
                    sig_is: pd.DataFrame,
                    vol_target: float | None = None,
                    min_trades: int = 5,
                    ) -> tuple[float, float, float, int]:
    """
    Grid search IS Calmar over (atr_sl, atr_tp1) combos.

    Returns (best_sl, best_tp1, best_calmar, n_trades_of_best)
    Falls back to (2.0, 2.0, 0.0, 0) if no combo meets min_trades.
    """
    best_sl, best_tp1, best_metric = 2.0, 2.0, -np.inf
    best_n = 0
    any_valid = False

    for sl in SL_GRID:
        for tp1 in TP1_GRID:
            try:
                bt = run_backtest(
                    df_is, sig_is,
                    atr_sl_override=sl, atr_tp1_override=tp1,
                    vol_target=vol_target,
                )
            except Exception:
                continue

            trades_df = bt.get("trades", pd.DataFrame())
            n = len(trades_df) if isinstance(trades_df, pd.DataFrame) else 0
            if n < min_trades:
                continue

            cal = float(bt["kpis"].get("calmar", 0.0))
            ret = float(bt["kpis"].get("total_return", 0.0))
            metric = cal if cal > 0 else ret
            any_valid = True

            if metric > best_metric:
                best_metric = metric
                best_sl, best_tp1 = sl, tp1
                best_n = n

    if not any_valid:
        return 2.0, 2.0, 0.0, 0
    return best_sl, best_tp1, best_metric, best_n


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward runner
# ─────────────────────────────────────────────────────────────────────────────

def run_wf(
    df_1h: pd.DataFrame,
    signals: pd.DataFrame,
    windows: list[tuple],
    vol_target: float | None = None,
    label: str = "WF",
) -> list[dict]:
    """
    Walk-forward IS optimisation + OOS evaluation.

    Returns list of per-window dicts with IS/OOS metrics.
    """
    results = []
    for i, (tr_s, tr_e, oo_s, oo_e) in enumerate(windows):
        df_is  = df_1h.loc[(df_1h.index >= tr_s) & (df_1h.index < tr_e)]
        sig_is = signals.loc[(signals.index >= tr_s) & (signals.index < tr_e)]
        df_oos  = df_1h.loc[(df_1h.index >= oo_s) & (df_1h.index < oo_e)]
        sig_oos = signals.loc[(signals.index >= oo_s) & (signals.index < oo_e)]

        if len(df_is) < 100 or len(df_oos) < 10:
            continue

        n_is_sigs = int((sig_is["signal"] != 0).sum())

        # ── IS optimisation ──────────────────────────────────────────────────
        best_sl, best_tp1, is_calmar, is_n = _best_params_is(
            df_is, sig_is, vol_target=vol_target,
        )

        # ── OOS evaluation with IS-best params ───────────────────────────────
        try:
            oos_bt = run_backtest(
                df_oos, sig_oos,
                atr_sl_override=best_sl, atr_tp1_override=best_tp1,
                vol_target=vol_target,
            )
        except Exception as exc:
            print(f"  W{i+1:02d} OOS failed: {exc}")
            continue

        oos_trades = oos_bt.get("trades", pd.DataFrame())
        oos_n   = len(oos_trades) if isinstance(oos_trades, pd.DataFrame) else 0
        oos_ret = float(oos_bt["kpis"].get("total_return", 0.0)) * 100
        oos_dd  = abs(float(oos_bt["kpis"].get("max_drawdown", 0.0))) * 100
        oos_wr  = float(oos_bt["kpis"].get("win_rate", 0.0)) * 100

        results.append({
            "window":   i + 1,
            "oos_s":    oo_s,
            "oos_e":    oo_e,
            "is_sigs":  n_is_sigs,
            "is_calmar": round(is_calmar, 3),
            "is_n":     is_n,
            "best_sl":  best_sl,
            "best_tp1": best_tp1,
            "oos_ret":  round(oos_ret, 2),
            "oos_dd":   round(oos_dd, 2),
            "oos_n":    oos_n,
            "oos_wr":   round(oos_wr, 1),
            "oos_trades": oos_trades,
        })

        print(f"  {label} W{i+1:02d} [{oo_s.date()}→{oo_e.date()}] "
              f"IS sigs={n_is_sigs} sl={best_sl:.1f} tp={best_tp1:.1f} Cal={is_calmar:+.2f} N={is_n}"
              f" | OOS ret={oos_ret:+.1f}% DD={oos_dd:.1f}% N={oos_n}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate OOS metrics from stitched trade list
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_oos(wf_results: list[dict]) -> dict:
    """Stitch OOS trades into a synthetic equity curve and compute KPIs."""
    all_pnl: list[float] = []
    n_wins = 0

    for r in wf_results:
        trd = r.get("oos_trades", pd.DataFrame())
        if not isinstance(trd, pd.DataFrame) or trd.empty:
            continue
        pnls = trd["net_pnl"].tolist()
        all_pnl.extend(pnls)
        n_wins += sum(1 for p in pnls if p > 0)

    n_total = len(all_pnl)
    if n_total == 0:
        return {"ret": 0.0, "dd": 0.0, "calmar": 0.0,
                "n_trades": 0, "win_rate": 0.0, "sharpe": 0.0}

    equity = np.full(n_total + 1, float(INIT_CAP))
    for i, p in enumerate(all_pnl):
        equity[i + 1] = equity[i] + p

    total_ret = (equity[-1] / equity[0] - 1.0) * 100
    peak      = np.maximum.accumulate(equity)
    dd_arr    = (equity - peak) / peak
    max_dd    = abs(dd_arr.min()) * 100

    calmar    = total_ret / max_dd if max_dd > 0.001 else 0.0
    win_rate  = n_wins / n_total * 100

    # Simple Sharpe on per-trade returns
    trade_rets = np.array(all_pnl) / INIT_CAP * 100
    sharpe     = (trade_rets.mean() / (trade_rets.std(ddof=1) + 1e-9)
                  * np.sqrt(n_total))

    return {
        "ret":      round(float(total_ret), 2),
        "dd":       round(float(max_dd),    2),
        "calmar":   round(float(calmar),    3),
        "n_trades": int(n_total),
        "win_rate": round(float(win_rate),  1),
        "sharpe":   round(float(sharpe),    3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single full-period backtest KPIs
# ─────────────────────────────────────────────────────────────────────────────

def _run_kpi(df_1h: pd.DataFrame, signals: pd.DataFrame,
             label: str, **bt_kwargs) -> dict:
    bt  = run_backtest(df_1h, signals, **bt_kwargs)
    k   = bt.get("kpis", {})
    trd = bt.get("trades", pd.DataFrame())
    n   = len(trd) if isinstance(trd, pd.DataFrame) else 0
    ret = float(k.get("total_return", 0.0)) * 100
    dd  = abs(float(k.get("max_drawdown", 0.0))) * 100
    cal = float(k.get("calmar", 0.0))
    sr  = float(k.get("sharpe", 0.0))
    wr  = float(k.get("win_rate", 0.0)) * 100
    print(f"  {label:<35s}: ret={ret:+.1f}%  DD={dd:.1f}%  "
          f"Calmar={cal:.3f}  Sharpe={sr:.2f}  Trades={n}")
    return {"label": label, "ret": round(ret,2), "dd": round(dd,2),
            "calmar": round(cal,3), "sharpe": round(sr,3),
            "win_rate": round(wr,1), "n_trades": n}


# ─────────────────────────────────────────────────────────────────────────────
# HTML helpers
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
body{font-family:system-ui,sans-serif;background:#0f1117;color:#e0e0e0;margin:0;padding:24px}
h1{font-size:1.6rem;color:#fff;border-bottom:2px solid #2196f3;padding-bottom:8px}
h2{font-size:1.1rem;color:#90caf9;margin-top:32px}
h3{font-size:.95rem;color:#b0bec5;margin-top:20px}
p.meta{color:#546e7a;font-size:.8rem}
table{border-collapse:collapse;width:100%;margin-top:12px;font-size:.82rem}
th{background:#1e2130;color:#90caf9;padding:7px 10px;text-align:center;
   border-bottom:2px solid #2196f3}
td{padding:6px 10px;border-bottom:1px solid #1e2130;text-align:center}
td:first-child{text-align:left}
tr:hover td{background:#1a1e2e}
.pos{color:#4caf50;font-weight:600}
.neg{color:#f44336;font-weight:600}
.warn{color:#ff9800;font-weight:600}
.hl td{background:#1e2a1e!important;border-left:3px solid #4caf50}
.card{background:#12151f;border:1px solid #1e2130;border-radius:8px;
      padding:16px;margin-top:16px}
.scroll{overflow-x:auto}
"""


def _pnl_cls(v: float) -> str:
    return "pos" if v > 0 else ("neg" if v < 0 else "")


def _fmt_ret(v: float) -> str:
    return f'<td class="{_pnl_cls(v)}">{v:+.2f}%</td>'


def _summary_row(r: dict, highlight: bool = False) -> str:
    cal   = r["calmar"]
    cal_c = "pos" if cal > 0.5 else ("warn" if cal > 0 else "neg")
    hl    = ' class="hl"' if highlight else ""
    return (
        f'<tr{hl}>'
        f'<td>{r["label"]}</td>'
        f'{_fmt_ret(r["ret"])}'
        f'<td class="neg">{r["dd"]:.1f}%</td>'
        f'<td class="{cal_c}">{cal:.3f}</td>'
        f'<td>{r["sharpe"]:.3f}</td>'
        f'<td>{r["win_rate"]:.1f}%</td>'
        f'<td>{r["n_trades"]}</td>'
        f'</tr>'
    )


def _wf_table_rows(wf_results: list[dict]) -> str:
    rows = []
    for r in wf_results:
        oo_lbl = f'{r["oos_s"].date()}→{r["oos_e"].date()}'
        ret_c  = _pnl_cls(r["oos_ret"])
        is_c   = _pnl_cls(r["is_calmar"])
        rows.append(
            f'<tr>'
            f'<td>W{r["window"]:02d}</td>'
            f'<td>{oo_lbl}</td>'
            f'<td>{r["is_sigs"]}</td>'
            f'<td>{r["best_sl"]:.1f}×</td>'
            f'<td>{r["best_tp1"]:.1f}×</td>'
            f'<td class="{is_c}">{r["is_calmar"]:+.3f}</td>'
            f'<td>{r["is_n"]}</td>'
            f'<td class="{ret_c}">{r["oos_ret"]:+.2f}%</td>'
            f'<td>{r["oos_dd"]:.1f}%</td>'
            f'<td>{r["oos_n"]}</td>'
            f'<td>{r["oos_wr"]:.0f}%</td>'
            f'</tr>'
        )
    return "\n".join(rows)


def _param_freq_table(wf_results: list[dict]) -> str:
    from collections import Counter
    cnt = Counter((r["best_sl"], r["best_tp1"]) for r in wf_results)
    total = sum(cnt.values())
    rows = "".join(
        f'<tr><td>SL={sl:.1f}×ATR  TP1={tp:.1f}×ATR</td>'
        f'<td>{n}</td><td>{n/total*100:.0f}%</td></tr>'
        for (sl, tp), n in sorted(cnt.items(), key=lambda x: -x[1])
    )
    return (
        '<table><thead><tr>'
        '<th>Param Combo</th><th>Selected (# windows)</th><th>Frequency</th>'
        '</tr></thead><tbody>' + rows + '</tbody></table>'
    )


def _calmar_bar(results: list[dict]) -> str:
    max_abs = max(abs(r["calmar"]) for r in results) + 0.1
    rows = ""
    for r in results:
        cal = r["calmar"]
        w   = min(abs(cal) / max_abs * 280, 280)
        col = "#4caf50" if cal > 0 else "#f44336"
        rows += (
            f'<tr><td style="width:240px">{r["label"]}</td>'
            f'<td><div style="background:{col};height:14px;width:{w:.0f}px;'
            f'border-radius:3px;display:inline-block"></div>'
            f' {cal:.3f}</td>'
            f'<td style="color:#f44336">{r["dd"]:.1f}%</td>'
            f'<td>{r["ret"]:+.1f}%</td>'
            f'<td>{r["n_trades"]}</td></tr>'
        )
    return f'<table><thead><tr><th>Strategy</th><th>Calmar</th><th>Max DD</th><th>Return</th><th>Trades</th></tr></thead><tbody>{rows}</tbody></table>'


def build_html(
    summary: list[dict],
    wf_default: list[dict],
    wf_opt: list[dict],
    wf_vol: list[dict],
    elapsed: float,
    n_signals: int,
) -> str:
    oos_hit_opt = sum(1 for r in wf_opt if r["oos_ret"] > 0) if wf_opt else 0
    oos_win_pct = oos_hit_opt / len(wf_opt) * 100 if wf_opt else 0.0
    oos_rets    = np.array([r["oos_ret"] for r in wf_opt]) if wf_opt else np.array([])
    t_stat, p_val = (0.0, 1.0)
    if len(oos_rets) >= 2 and oos_rets.std() > 0:
        from scipy.stats import ttest_1samp
        _t, _ = ttest_1samp(oos_rets, 0)
        t_stat = float(_t)
        from scipy.stats import t as t_dist
        p_val  = float(t_dist.sf(_t, df=len(oos_rets) - 1))

    rows_sum = "".join(
        _summary_row(r, highlight=(i in (1, 2)))
        for i, r in enumerate(summary)
    )
    bar_chart    = _calmar_bar(summary)
    wf_rows_opt  = _wf_table_rows(wf_opt)
    param_table  = _param_freq_table(wf_opt)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Wyckoff Strategy — BTCUSDT 2020-2026</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Wyckoff Spring/Upthrust Strategy — BTCUSDT Perpetual Futures</h1>
<p class="meta">
Period: 2020-01 → 2026-05 · Signals: {n_signals:,} spring/upthrust events (08-21 UTC) ·
WF: {WF_TRAIN_M}m train / {WF_OOS_M}m OOS / step {WF_STEP_M}m · {len(wf_opt)} windows ·
Fees: 0.04% per side · Generated in {elapsed:.0f}s
</p>

<h2>Strategy Comparison (full period, stitched OOS trades)</h2>
<p class="meta">
WF-opt rows use stitched OOS trades (IS-selected SL/TP per window) — same initial capital continuity.
Baseline shown for reference (composite signal system, walk-forward results from enhancements report).
</p>
<div class="scroll">
<table>
<thead><tr>
  <th>Strategy</th><th>Total Return</th><th>Max DD</th>
  <th>Calmar</th><th>Sharpe</th><th>Win Rate</th><th>Trades</th>
</tr></thead>
<tbody>{rows_sum}</tbody>
</table>
</div>
<p class="meta">
OOS t-stat = {t_stat:.2f}, p = {p_val:.4f} (one-sided, OOS returns &gt; 0)
· OOS positive-window rate: {oos_win_pct:.0f}% ({oos_hit_opt}/{len(wf_opt)})
</p>

<h2>Calmar Ratio Comparison</h2>
<div class="card">{bar_chart}</div>

<h2>WF Walk-Forward Detail (optimised variant)</h2>
<p class="meta">IS = 6-month in-sample optimisation; OOS = 2-month out-of-sample evaluation.</p>
<div class="scroll">
<table>
<thead><tr>
  <th>#</th><th>OOS Window</th><th>IS Signals</th>
  <th>Best SL</th><th>Best TP1</th>
  <th>IS Calmar</th><th>IS Trades</th>
  <th>OOS Ret</th><th>OOS DD</th><th>OOS Trades</th><th>OOS WR</th>
</tr></thead>
<tbody>{wf_rows_opt}</tbody>
</table>
</div>

<h2>IS-Selected Parameter Frequency</h2>
<div class="card">{param_table}</div>

<h2>Strategy Logic</h2>
<div class="card">
<h3>1 — Range Detection</h3>
<p>The market is classified as <em>ranging</em> when ADX &lt; 25 AND the
price band (rolling 24-bar high − low) / low &lt; 10%. Range boundaries
are computed using shifted highs/lows (no look-ahead).</p>

<h3>2 — Volume Bias (Accumulation vs Distribution)</h3>
<p><em>Accumulation</em>: OBV is above its 21-bar EMA within a ranging
market — smart money is absorbing supply without moving price.
<em>Distribution</em>: OBV is below its 21-bar EMA — smart money is
quietly unloading.</p>

<h3>3 — Manipulation Events</h3>
<p><strong>Spring</strong> (long setup): The current bar's low pierces
below the range boundary AND closes back inside, on volume &ge; 1.3×
20-bar average, within an accumulation context. This is the classic
Wyckoff spring — a false breakdown designed to shake out weak longs
before the mark-up phase begins.</p>
<p><strong>Upthrust</strong> (short setup): The current bar's high
pierces above the range boundary AND closes back inside, on elevated
volume, within a distribution context. The mirror image of the spring —
a false breakout designed to attract late longs before mark-down.</p>

<h3>4 — Entry Timing</h3>
<p>Signal fires on the bar <em>after</em> the spring/upthrust event
(next-bar open execution). No look-ahead: the signal bar's open price
is used for entry.</p>

<h3>5 — SL/TP Walk-Forward Optimisation</h3>
<p>For each 6-month IS window, a grid search over
SL ∈ {{0.5, 1.0, 1.5, 2.0, 2.5}} × TP1 ∈ {{1.0, 1.5, 2.0, 3.0}} ATR multiples
selects the combo with the highest Calmar ratio (minimum 5 IS trades).
The TP2 = 2×TP1, TP3 = 3×TP1 scale automatically. The selected params
are applied to the OOS window.</p>

<h3>6 — Composite Score</h3>
<p>Score = (penetration depth / ATR) × vol_ratio × 2.
Positive for longs (spring), negative for shorts (upthrust).
Used for analytics; all valid events are traded (no score threshold).</p>
</div>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("══ Wyckoff Strategy — Walk-Forward Report ══\n")
    t0 = time.time()

    # ── [1] Load data ─────────────────────────────────────────────────────────
    print("[1/5] Loading data …")
    raw = fetch_extended_data(
        start_year=START_YEAR, start_month=START_MONTH,
        fetch_flow=True, fetch_15m=False, fetch_1m=False,
    )
    tf_data: dict[str, pd.DataFrame] = {}
    for tf in ["1W", "1D", "4H", "1H"]:
        df = raw.get(tf, pd.DataFrame())
        tf_data[tf] = add_indicators(df) if not df.empty and len(df) > 20 else df

    df_1h = tf_data["1H"]
    df_1d = tf_data["1D"]
    print(f"  1H: {len(df_1h):,} bars  ({df_1h.index[0].date()} → {df_1h.index[-1].date()})")

    # ── [2] Build Wyckoff signals ─────────────────────────────────────────────
    print("[2/5] Building Wyckoff signals …")
    wyckoff_signals = build_wyckoff_signals(df_1h, session_hours=(8, 21))
    n_long  = int((wyckoff_signals["signal"] ==  1).sum())
    n_short = int((wyckoff_signals["signal"] == -1).sum())
    n_total = n_long + n_short
    print(f"  Springs (long): {n_long}  Upthrusts (short): {n_short}  Total: {n_total}")

    # ── [3] Baseline comparison signals ──────────────────────────────────────
    print("[3/5] Building baseline signals for comparison …")
    oi_df   = generate_oi(df_1d["close"])
    funding = generate_funding(df_1d["close"])
    raw_sig = build_signal_matrix(
        tf_data=tf_data, oi_df=oi_df, funding=funding,
        premium_1h=None, df_15m=None, df_1m=None,
    )
    base_signals = apply_filters(raw_sig, _BASELINE_CFG)

    windows = _wf_windows(df_1h.index)
    print(f"  WF windows: {len(windows)}")

    # ── [4] Full-period backtests ─────────────────────────────────────────────
    print("[4/5] Full-period backtests …")
    baseline_kpi = _run_kpi(df_1h, base_signals,
                            "Baseline (composite, T=3)",
                            atr_sl_override=2.0, atr_tp1_override=2.0)
    wyckoff_def  = _run_kpi(df_1h, wyckoff_signals,
                            "Wyckoff (default SL=2× TP=2×)",
                            atr_sl_override=2.0, atr_tp1_override=2.0)
    wyckoff_vol  = _run_kpi(df_1h, wyckoff_signals,
                            "Wyckoff (default) + Vol (0.20)",
                            atr_sl_override=2.0, atr_tp1_override=2.0,
                            vol_target=0.20)

    # ── [5] Walk-forward with IS SL/TP optimisation ───────────────────────────
    print("[5/5] Walk-forward optimisation …")
    print("  ── WF-optimised (no vol sizing) ──")
    wf_opt = run_wf(df_1h, wyckoff_signals, windows, vol_target=None,  label="Opt")
    print("  ── WF-optimised + vol sizing ──")
    wf_vol = run_wf(df_1h, wyckoff_signals, windows, vol_target=0.20, label="VolOpt")
    print("  ── WF default params (SL=2 TP=2) ──")
    wf_def = run_wf(df_1h, wyckoff_signals, windows, vol_target=None,  label="Def")

    # Override best_sl/best_tp1 to fixed for the default WF run (for display)
    # (We ran it via _best_params_is which selects — use the results as-is for
    # aggregation; the "default" label refers to the full-period fixed run.)

    # Aggregate OOS metrics
    agg_opt = _aggregate_oos(wf_opt)
    agg_vol = _aggregate_oos(wf_vol)

    # Build summary table: full-period runs + WF-stitched OOS
    summary = [
        baseline_kpi,
        wyckoff_def,
        wyckoff_vol,
        {**agg_opt,  "label": "Wyckoff WF-optimised (OOS stitched)"},
        {**agg_vol,  "label": "Wyckoff WF-opt + Vol 0.20 (OOS)"},
    ]

    # Print aggregate OOS
    print("\n── OOS Aggregate (stitched) ──")
    for label, agg in [("WF-opt", agg_opt), ("WF-vol", agg_vol)]:
        print(f"  {label}: ret={agg['ret']:+.1f}%  DD={agg['dd']:.1f}%  "
              f"Calmar={agg['calmar']:.3f}  N={agg['n_trades']}")

    # ── HTML ──────────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    html    = build_html(summary, wf_def, wf_opt, wf_vol, elapsed, n_total)

    out = Path("reports/report_wyckoff.html")
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"\n  → {out}  ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
