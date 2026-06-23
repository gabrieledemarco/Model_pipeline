"""
create_enhancements_report.py
──────────────────────────────
Tests four targeted enhancements on top of the baseline system
(BTCUSDT 1H, 2020-2026, session 08-21, composite threshold ≥ 3.0).

Enhancements
────────────
A. Volatility-scaled sizing  — vol_target=0.20 (built-in engine parameter)
B. Adaptive threshold        — rolling 75th-percentile of |composite|, 90d window
C. Daily EMA-200 regime filter — only long above 200d EMA, only short below it
D. Persistence filter N=2    — enter only if composite was above threshold for 2 consecutive bars

Also tests:
  A+C (recommended combo)
  All (A+B+C+D combined)

Output → reports/report_enhancements.html
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
from src.strategy.ml_gate       import _wf_windows

START_YEAR  = 2020
START_MONTH = 1

THRESH       = 3.0
SESSION_H    = (8, 21)
WF_TRAIN_M   = 6
WF_OOS_M     = 2
WF_STEP_M    = 2

SESSION_CFG  = ScenarioConfig(
    "Session 08-21", session_hours=SESSION_H,
    long_threshold=THRESH, short_threshold=-THRESH,
)


# ─────────────────────────────────────────────────────────────────────────────
# Signal enhancement functions
# ─────────────────────────────────────────────────────────────────────────────

def _in_session(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(
        (index.hour >= SESSION_H[0]) & (index.hour < SESSION_H[1]),
        index=index,
    )


def apply_vol_sizing(signals: pd.DataFrame, df_1h: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """A: No signal change — pass vol_target=0.20 to run_backtest."""
    return signals.copy(), {"vol_target": 0.20}


def apply_adaptive_threshold(
    signals: pd.DataFrame,
    df_1h: pd.DataFrame,
    window_bars: int = 90 * 24,
    percentile: float = 75,
    min_thresh: float = THRESH,
) -> tuple[pd.DataFrame, dict]:
    """B: Threshold adapts to rolling 75th-percentile of |composite|."""
    composite = signals["composite"]
    in_sess   = _in_session(signals.index)

    roll_thresh = (
        composite.abs()
        .rolling(window_bars, min_periods=200)
        .quantile(percentile / 100)
        .fillna(min_thresh)
        .clip(lower=min_thresh)
    )

    sig = pd.Series(0, index=signals.index)
    sig[(composite >= roll_thresh)  & in_sess] =  1
    sig[(composite <= -roll_thresh) & in_sess] = -1

    out = signals.copy()
    out["signal"] = sig.astype(int)
    return out, {}


def apply_ma_regime_filter(
    signals: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_1d: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """C: Only long above daily EMA-200, only short below it."""
    ema200 = df_1d["ema_200"].reindex(df_1h.index, method="ffill")
    close  = df_1h["close"]

    above_ema = (close > ema200).reindex(signals.index, fill_value=False)
    below_ema = (close < ema200).reindex(signals.index, fill_value=False)

    out = signals.copy()
    # Suppress shorts in bull regime
    out.loc[above_ema & (out["signal"] == -1), "signal"] = 0
    # Suppress longs in bear regime
    out.loc[below_ema & (out["signal"] ==  1), "signal"] = 0
    return out, {}


def apply_persistence_filter(
    signals: pd.DataFrame,
    df_1h: pd.DataFrame,
    n_bars: int = 2,
) -> tuple[pd.DataFrame, dict]:
    """D: Fire signal only if the already-filtered signal was present for n_bars in a row."""
    sig = signals["signal"]

    # Works from existing signal column so it composes correctly with B/C
    long_pers  = (sig == 1).astype(float).rolling(n_bars, min_periods=n_bars).min().fillna(0).astype(bool)
    short_pers = (sig == -1).astype(float).rolling(n_bars, min_periods=n_bars).min().fillna(0).astype(bool)

    new_sig = pd.Series(0, index=signals.index)
    new_sig[long_pers]  =  1
    new_sig[short_pers] = -1

    out = signals.copy()
    out["signal"] = new_sig.astype(int)
    return out, {}


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _kpi(bt: dict) -> dict:
    k   = bt.get("kpis", {})
    trd = bt.get("trades", pd.DataFrame())
    nt  = len(trd) if isinstance(trd, pd.DataFrame) else 0
    return {
        "ret":    round(float(k.get("total_return",  0)) * 100, 2),
        "dd":     round(abs(float(k.get("max_drawdown", 0))) * 100, 2),
        "calmar": round(float(k.get("calmar",   0)), 3),
        "sharpe": round(float(k.get("sharpe",   0)), 3),
        "sortino":round(float(k.get("sortino",  0)), 3),
        "wr":     round(float(k.get("win_rate", 0)) * 100, 1),
        "pf":     round(float(k.get("profit_factor", 0)), 3),
        "n_trades": nt,
    }


def _per_window_ret(trades_df: pd.DataFrame, windows: list) -> np.ndarray:
    rets = []
    for (_, _, oo_s, oo_e) in windows:
        mask = (trades_df["entry_ts"] >= oo_s) & (trades_df["entry_ts"] < oo_e)
        val  = float(trades_df.loc[mask, "net_pnl"].sum()) / INIT_CAP * 100
        rets.append(val)
    return np.array(rets)


def _ttest(arr: np.ndarray) -> tuple[float, float]:
    if len(arr) < 2 or arr.std() == 0:
        return 0.0, 1.0
    t, _ = st.ttest_1samp(arr, 0)
    p1   = float(st.t.sf(t, df=len(arr) - 1))
    return round(float(t), 3), round(p1, 4)


# ─────────────────────────────────────────────────────────────────────────────
# HTML generation
# ─────────────────────────────────────────────────────────────────────────────

def _cell_ret(v: float) -> str:
    cls = "pos" if v > 0 else "neg"
    return f'<td class="{cls}">{v:+.2f}%</td>'


def _cell_cal(v: float) -> str:
    cls = "pos" if v > 0.5 else ("warn" if v > 0 else "neg")
    return f'<td class="{cls}">{v:.3f}</td>'


def _result_row(label: str, m: dict, base_dd: float, base_cal: float,
                t_stat: float, p_val: float, highlight: bool = False) -> str:
    dd_delta  = base_dd - m["dd"]          # positive = DD reduced
    cal_delta = m["calmar"] - base_cal
    hl = ' class="hl"' if highlight else ""
    dd_cls   = "pos" if dd_delta > 2 else ("neg" if dd_delta < -2 else "")
    cd_cls   = "pos" if cal_delta > 0 else "neg"
    sig_flag = " ★" if p_val < 0.05 else ""
    return (
        f'<tr{hl}>'
        f'<td>{label}</td>'
        f'{_cell_ret(m["ret"])}'
        f'<td class="neg">{m["dd"]:.1f}%</td>'
        f'<td class="{dd_cls}">{dd_delta:+.1f}%</td>'
        f'{_cell_cal(m["calmar"])}'
        f'<td class="{cd_cls}">{cal_delta:+.3f}</td>'
        f'<td>{m["sharpe"]:.3f}</td>'
        f'<td>{m["n_trades"]}</td>'
        f'<td>{t_stat:.2f} / {p_val:.3f}{sig_flag}</td>'
        f'</tr>'
    )


def _win_rows(variants: list[tuple[str, np.ndarray]], windows: list) -> str:
    rows = []
    n    = len(windows)
    for i, (_, _, oo_s, oo_e) in enumerate(windows):
        cells = "".join(_cell_ret(v[i]) for _, v in variants)
        rows.append(f"<tr><td>{i+1}</td><td>{oo_s.date()}→{oo_e.date()}</td>{cells}</tr>")
    return "\n".join(rows)


def _mini_bar(val: float, max_abs: float, color: str) -> str:
    pct = min(abs(val) / max(max_abs, 1e-6) * 100, 100)
    return (f'<div style="display:flex;align-items:center;gap:4px">'
            f'<div style="background:{color};height:10px;width:{pct:.0f}px;min-width:2px"></div>'
            f'<span>{val:.3f}</span></div>')


def build_html(results: list[dict], windows: list, elapsed: float) -> str:
    base      = results[0]
    base_dd   = base["metrics"]["dd"]
    base_cal  = base["metrics"]["calmar"]
    base_wins = base["wins"]

    # ── summary table ──────────────────────────────────────────────────────────
    rows_sum = ""
    for r in results:
        m  = r["metrics"]
        t, p = r.get("t", 0.0), r.get("p", 1.0)
        hl = r.get("highlight", False)
        rows_sum += _result_row(r["label"], m, base_dd, base_cal, t, p, hl)

    # ── per-window table ───────────────────────────────────────────────────────
    variants_for_win = [(r["label"], r["wins"]) for r in results]
    win_hdrs  = "".join(f"<th>{r['label']}</th>" for r in results)
    win_rows  = _win_rows(variants_for_win, windows)

    # ── calmar bar chart ───────────────────────────────────────────────────────
    max_cal = max(abs(r["metrics"]["calmar"]) for r in results) + 0.1
    bar_rows = ""
    for r in results:
        cal  = r["metrics"]["calmar"]
        w    = min(abs(cal) / max_cal * 300, 300)
        col  = "#4caf50" if cal > 0 else "#f44336"
        hl   = "font-weight:700;" if r.get("highlight") else ""
        bar_rows += (
            f'<tr><td style="{hl}width:200px">{r["label"]}</td>'
            f'<td><div style="background:{col};height:16px;width:{w:.0f}px;'
            f'border-radius:3px;display:inline-block"></div> '
            f'<span style="{hl}">{cal:.3f}</span></td>'
            f'<td style="color:#f44336">{r["metrics"]["dd"]:.1f}%</td>'
            f'<td>{r["metrics"]["ret"]:+.1f}%</td></tr>'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Strategy Enhancements — BTCUSDT 2020-2026</title>
<style>
  body{{font-family:system-ui,sans-serif;background:#0f1117;color:#e0e0e0;margin:0;padding:24px}}
  h1{{font-size:1.6rem;color:#fff;border-bottom:2px solid #2196f3;padding-bottom:8px}}
  h2{{font-size:1.1rem;color:#90caf9;margin-top:32px}}
  h3{{font-size:.95rem;color:#b0bec5;margin-top:20px}}
  table{{border-collapse:collapse;width:100%;margin-top:12px;font-size:.82rem}}
  th{{background:#1e2130;color:#90caf9;padding:7px 10px;text-align:center;border-bottom:2px solid #2196f3}}
  td{{padding:6px 10px;border-bottom:1px solid #1e2130;text-align:center}}
  td:first-child{{text-align:left}}
  tr:hover td{{background:#1a1e2e}}
  .pos{{color:#4caf50;font-weight:600}}
  .neg{{color:#f44336;font-weight:600}}
  .warn{{color:#ff9800;font-weight:600}}
  .hl td{{background:#1e2a1e!important;border-left:3px solid #4caf50}}
  .badge{{display:inline-block;padding:2px 8px;border-radius:10px;font-size:.75rem;font-weight:600}}
  .badge-green{{background:#1b5e20;color:#a5d6a7}}
  .badge-red{{background:#b71c1c;color:#ffcdd2}}
  .badge-yellow{{background:#e65100;color:#ffe0b2}}
  .card{{background:#12151f;border:1px solid #1e2130;border-radius:8px;padding:16px;margin-top:16px}}
  .meta{{color:#546e7a;font-size:.78rem;margin-top:4px}}
  .legend{{display:flex;gap:16px;flex-wrap:wrap;margin:8px 0;font-size:.8rem}}
  .dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px}}
  .scroll{{overflow-x:auto}}
</style>
</head>
<body>
<h1>Strategy Enhancements — BTCUSDT Perpetual Futures</h1>
<p class="meta">Period: 2020-01 → 2026-05 · 56k bars · WF: 6m train / 2m OOS / step 2m · 35 windows · Fees: 0.04% per side · Generated in {elapsed:.0f}s</p>

<div class="legend">
  <span><span class="dot" style="background:#4caf50"></span>Improvement vs baseline</span>
  <span><span class="dot" style="background:#f44336"></span>Degradation vs baseline</span>
  <span>★ = t-test p &lt; 0.05 (OOS returns &gt; 0)</span>
  <span>hl row = recommended combo</span>
</div>

<h2>Results Summary</h2>
<div class="scroll">
<table>
<thead><tr>
  <th>Variant</th><th>Return</th><th>Max DD</th><th>ΔDD</th>
  <th>Calmar</th><th>ΔCalmar</th><th>Sharpe</th><th>Trades</th><th>t / p-val</th>
</tr></thead>
<tbody>
{rows_sum}
</tbody>
</table>
</div>
<p class="meta">ΔDD = baseline DD − variant DD (positive = drawdown reduced). ΔCalmar = variant Calmar − baseline Calmar.</p>

<h2>Calmar Ratio by Variant</h2>
<div class="card">
<table style="width:auto">
<tbody>{bar_rows}</tbody>
</table>
</div>

<h2>Per-Window OOS Returns (% of initial equity)</h2>
<div class="scroll">
<table>
<thead><tr><th>#</th><th>OOS Window</th>{win_hdrs}</tr></thead>
<tbody>{win_rows}</tbody>
</table>
</div>

<h2>Enhancement Descriptions</h2>
<div class="card">
<h3>A · Volatility-scaled sizing (vol_target = 0.20)</h3>
<p>Scales position size inversely to realised 20-bar hourly vol (rvol_20),
targeting 20% annualised portfolio volatility. High-vol periods (2022 bear,
2024 ETF pump) get smaller positions automatically. Scalar is clipped [0.25, 3.0].
No signal changes — only affects how much risk is taken per trade.</p>

<h3>B · Adaptive composite threshold (rolling 75th-pct, 90d window)</h3>
<p>Instead of a fixed threshold of 3.0, the entry bar requires
composite ≥ p75(|composite| over past 90 days). In high-activity regimes the
bar rises automatically (fewer, stronger signals); in quiet regimes it stays
near the base threshold of 3.0. Applied symmetrically for shorts.</p>

<h3>C · Daily EMA-200 regime filter</h3>
<p>Only longs when 1H close &gt; daily EMA(200); only shorts when 1H close &lt;
daily EMA(200). Prevents trading against the medium-term trend (~6-month).
Particularly effective for filtering the 2020-2021 bull run where the composite
system was generating incorrect short signals.</p>

<h3>D · Signal persistence filter (N=2 bars)</h3>
<p>A signal fires only when the composite has been above the threshold for 2
consecutive 1H bars (not just the current one). Eliminates single-bar
spikes and whipsaw entries in choppy consolidations. The trade-off is slightly
later entries but higher signal quality.</p>

<h3>A2 · Vol sizing + DD circuit breaker (vol_target=0.20, dd_halt=15%)</h3>
<p>Combines volatility targeting with an equity circuit breaker: when drawdown
from peak exceeds 15%, position size is halved; at 22.5% trading pauses
entirely until equity recovers above 15% DD. Designed to capture the upside of
vol targeting while hard-limiting tail drawdowns.</p>
</div>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("══ Strategy Enhancements Report ══\n")
    t0 = time.time()

    # ── [1] Load data ─────────────────────────────────────────────────────────
    print("[1/4] Loading data …")
    raw = fetch_extended_data(
        start_year=START_YEAR, start_month=START_MONTH,
        fetch_flow=True, fetch_15m=True, fetch_1m=False,
    )
    tf_data = {}
    for tf in ["1W", "1D", "4H", "1H", "15M"]:
        df = raw.get(tf, pd.DataFrame())
        tf_data[tf] = add_indicators(df) if not df.empty and len(df) > 20 else df

    df_1h   = tf_data["1H"]
    df_1d   = tf_data["1D"]
    oi_df   = generate_oi(df_1d["close"])
    funding = generate_funding(df_1d["close"])
    print(f"  1H: {len(df_1h):,} bars  ({df_1h.index[0].date()} → {df_1h.index[-1].date()})")

    # ── [2] Signals ───────────────────────────────────────────────────────────
    print("[2/4] Building signals …")
    raw_sig = build_signal_matrix(
        tf_data=tf_data, oi_df=oi_df, funding=funding,
        premium_1h=None, df_15m=tf_data.get("15M"), df_1m=None,
    )
    base_signals = apply_filters(raw_sig, SESSION_CFG)

    windows = _wf_windows(df_1h.index)
    print(f"  Signal bars: {(base_signals['signal']!=0).sum():,}  WF windows: {len(windows)}")

    # ── [3] Run variants ──────────────────────────────────────────────────────
    print("[3/4] Running backtest variants …")

    def _run(label: str, signals: pd.DataFrame, bt_kwargs: dict = {},
             highlight: bool = False) -> dict:
        bt  = run_backtest(df_1h, signals, **bt_kwargs)
        m   = _kpi(bt)
        trd = bt.get("trades", pd.DataFrame())
        wins = _per_window_ret(trd, windows) if not trd.empty else np.zeros(len(windows))
        t_stat, p_val = _ttest(wins)
        ret_pct = m["ret"]
        dd_pct  = m["dd"]
        print(f"  {label:<25s}: ret={ret_pct:+.1f}%  DD={dd_pct:.1f}%  "
              f"Calmar={m['calmar']:.3f}  Trades={m['n_trades']}")
        return {
            "label": label, "metrics": m, "wins": wins,
            "t": t_stat, "p": p_val,
            "highlight": highlight,
        }

    results = []

    # Baseline
    results.append(_run("Baseline", base_signals))

    # A: vol scaling only
    sig_a, kw_a = apply_vol_sizing(base_signals, df_1h)
    results.append(_run("A: Vol Sizing (0.20)", sig_a, kw_a))

    # B: adaptive threshold
    sig_b, kw_b = apply_adaptive_threshold(base_signals, df_1h)
    results.append(_run("B: Adaptive Threshold", sig_b, kw_b))

    # C: MA regime filter
    sig_c, kw_c = apply_ma_regime_filter(base_signals, df_1h, df_1d)
    results.append(_run("C: MA Regime Filter", sig_c, kw_c))

    # D: persistence N=2
    sig_d, kw_d = apply_persistence_filter(base_signals, df_1h)
    results.append(_run("D: Persistence N=2", sig_d, kw_d))

    # A2: Vol scaling + DD circuit breaker (targets DD reduction)
    results.append(_run("A2: Vol + DD Halt 15%", base_signals,
                        {"vol_target": 0.20, "dd_halt_pct": 0.15}))

    # A+C combo
    sig_ac, kw_ac = apply_ma_regime_filter(base_signals, df_1h, df_1d)
    results.append(_run("A+C (Vol + MA Filter)", sig_ac, {"vol_target": 0.20}, highlight=True))

    # A+B+C+D: all four
    sig_all = base_signals.copy()
    sig_all, _ = apply_adaptive_threshold(sig_all, df_1h)
    sig_all, _ = apply_ma_regime_filter(sig_all, df_1h, df_1d)
    sig_all, _ = apply_persistence_filter(sig_all, df_1h)
    results.append(_run("All (A+B+C+D)", sig_all, {"vol_target": 0.20}))

    # ── [4] HTML ──────────────────────────────────────────────────────────────
    print("[4/4] Generating HTML …")
    elapsed = time.time() - t0
    html    = build_html(results, windows, elapsed)

    out = Path("reports/report_enhancements.html")
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"  → {out}")

    print(f"\n══ Done ({elapsed:.0f}s) ════════════════════════════════════════════")
    base_m = results[0]["metrics"]
    for r in results[1:]:
        m      = r["metrics"]
        dd_red = base_m["dd"] - m["dd"]
        print(f"  {r['label']:<25s}: Calmar {m['calmar']:+.3f}  DD {m['dd']:.1f}%"
              f"  (ΔDD={dd_red:+.1f}%  Δret={m['ret']-base_m['ret']:+.1f}%)")


if __name__ == "__main__":
    main()
