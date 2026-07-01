"""
Four-direction experiment: systematic exploration after lookahead-bias fix.

Direction 1 – Momentum signals     : ROC-based replacements for EMA/MACD crossovers
Direction 2 – Threshold sweep      : composite threshold ±2 … ±10
Direction 3 – 1H entry variants    : Donchian breakout vs RSI mean-reversion vs current
Direction 4 – Weight configurations: 6 presets ranging from macro-only to micro-heavy

Baseline = bias-fixed signal matrix with current weights & threshold ±5.
Session filter (08-21 UTC) applied to ALL experiments for consistency with live setup.

Output → reports/experiment_report.html
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.strategy.data_fetcher   import fetch_extended_data, generate_oi, generate_funding
from src.strategy.indicators     import add_indicators
from src.strategy.signals        import (
    build_signal_matrix, WEIGHTS, LONG_THRESH, SHORT_THRESH,
    weekly_trend, daily_trend, fourfour_setup, one_hour_entry,
    oi_signal, basis_oi_signal, fifteen_min_entry, one_min_entry,
    funding_signal, volume_signal, cyclicality_signal,
    _align,
)
from src.strategy.engine         import run_backtest, INIT_CAP
from src.strategy.optimizer      import apply_filters, ScenarioConfig

# ── Session-filter config applied to every experiment ─────────────────────────
SESSION_CFG = ScenarioConfig("Session 08-21", session_hours=(8, 21))


def _run(df_1h, signals, thr=5.0):
    """Apply session filter + threshold, then run backtest."""
    cfg = ScenarioConfig(
        f"thr{thr}",
        long_threshold=thr, short_threshold=-thr,
        session_hours=(8, 21),
    )
    return run_backtest(df_1h, apply_filters(signals, cfg))


# ─────────────────────────────────────────────────────────────────────────────
# Momentum signal helpers (Direction 1)
# ─────────────────────────────────────────────────────────────────────────────

def weekly_momentum(df: pd.DataFrame) -> pd.Series:
    """
    ROC-based weekly momentum.  Score: -3 to +3.
      ±2  ROC-10 (weekly bars) sign × magnitude threshold
      ±1  ROC-10 vs 52-week EMA(ROC)
    """
    roc = df["close"].pct_change(10).fillna(0)
    ema_roc = roc.ewm(span=52, adjust=False).mean()
    s = pd.Series(0.0, index=df.index)
    s += np.where(roc >  0.05, 2.0, np.where(roc >  0.0,  1.0,
         np.where(roc < -0.05, -2.0, np.where(roc < 0.0, -1.0, 0.0))))
    s += np.where(roc > ema_roc, 1.0, -1.0)
    return s.clip(-3, 3).rename("s_weekly")


def daily_momentum(df: pd.DataFrame) -> pd.Series:
    """
    ROC-based daily momentum.  Score: -4 to +4.
      ±2  ROC-10 threshold (strong/weak)
      ±1  ROC-3 direction (short-term impulse)
      ±1  ROC-10 vs EMA-21(ROC)
    """
    roc10 = df["close"].pct_change(10).fillna(0)
    roc3  = df["close"].pct_change(3).fillna(0)
    ema21_roc = roc10.ewm(span=21, adjust=False).mean()
    s = pd.Series(0.0, index=df.index)
    s += np.where(roc10 >  0.07, 2.0, np.where(roc10 >  0.0,  1.0,
         np.where(roc10 < -0.07, -2.0, np.where(roc10 < 0.0, -1.0, 0.0))))
    s += np.where(roc3 > 0, 1.0, -1.0)
    s += np.where(roc10 > ema21_roc, 1.0, -1.0)
    return s.clip(-4, 4).rename("s_daily")


def fourfour_momentum(df: pd.DataFrame) -> pd.Series:
    """
    ROC-based 4H momentum.  Score: -3 to +3.
      ±2  ROC-18 (3-day lookback at 4H resolution)
      ±1  ROC-18 vs EMA-21(ROC)
    """
    roc = df["close"].pct_change(18).fillna(0)
    ema_roc = roc.ewm(span=21, adjust=False).mean()
    s = pd.Series(0.0, index=df.index)
    s += np.where(roc >  0.03, 2.0, np.where(roc >  0.0,  1.0,
         np.where(roc < -0.03, -2.0, np.where(roc < 0.0, -1.0, 0.0))))
    s += np.where(roc > ema_roc, 1.0, -1.0)
    return s.clip(-3, 3).rename("s_4h")


def hourly_momentum(df: pd.DataFrame) -> pd.Series:
    """
    ROC-based 1H momentum.  Score: -3 to +3.
      ±1  ROC-4 (4-bar, ~4h lookback)
      ±1  ROC-4 vs EMA-14(ROC)
      ±1  Volume-weighted ROC sign
    """
    roc4  = df["close"].pct_change(4).fillna(0)
    ema_roc = roc4.ewm(span=14, adjust=False).mean()
    vw_roc = (roc4 * df["vol_ratio"]).clip(-3, 3)
    s = pd.Series(0.0, index=df.index)
    s += np.where(roc4 > 0, 1.0, -1.0)
    s += np.where(roc4 > ema_roc, 1.0, -1.0)
    s += np.sign(vw_roc)
    return s.clip(-3, 3).rename("s_1h")


# ─────────────────────────────────────────────────────────────────────────────
# 1H entry variants (Direction 3)
# ─────────────────────────────────────────────────────────────────────────────

def donchian_breakout(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Donchian channel breakout.  Score: -3 to +3.
      ±2  close above/below N-bar channel
      ±1  volume confirmation (vol_ratio × direction)
    """
    high_n = df["high"].shift(1).rolling(period, min_periods=1).max()
    low_n  = df["low"].shift(1).rolling(period, min_periods=1).min()
    s = pd.Series(0.0, index=df.index)
    s[df["close"] >= high_n] =  2.0
    s[df["close"] <= low_n]  = -2.0
    vol_conf = np.where(df["vol_ratio"] > 1.3, np.sign(df["log_ret"]), 0.0)
    s += vol_conf
    return s.clip(-3, 3).rename("s_1h")


def rsi_mean_reversion(df: pd.DataFrame) -> pd.Series:
    """
    RSI mean-reversion (contrarian).  Score: -3 to +3.
      Buy oversold (RSI < 30) → +2/+3
      Sell overbought (RSI > 70) → -2/-3
      Neutral zone → ±1 based on direction from 50
    """
    rsi = df["rsi_14"]
    s = pd.Series(0.0, index=df.index)
    s[rsi < 20]  =  3.0   # extreme oversold – strong buy
    s[(rsi >= 20) & (rsi < 30)] =  2.0
    s[(rsi >= 30) & (rsi < 45)] =  1.0
    s[(rsi > 55)  & (rsi <= 70)] = -1.0
    s[(rsi > 70)  & (rsi <= 80)] = -2.0
    s[rsi > 80]  = -3.0   # extreme overbought – strong sell
    return s.rename("s_1h")


def mixed_entry(df: pd.DataFrame) -> pd.Series:
    """
    Blend of breakout (50%) + current trend entry (50%).
    Score: -3 to +3.
    """
    br  = donchian_breakout(df)
    cur = one_hour_entry(df)
    s   = (0.5 * br + 0.5 * cur).clip(-3, 3)
    return s.rename("s_1h")


# ─────────────────────────────────────────────────────────────────────────────
# Signal matrix builder with custom overrides
# ─────────────────────────────────────────────────────────────────────────────

def build_custom_matrix(
    tf_data, oi_df, funding, premium_1h=None, df_15m=None, df_1m=None,
    *,
    weekly_fn=None, daily_fn=None, fourfour_fn=None, hourly_fn=None,
    weights=None,
) -> pd.DataFrame:
    """
    Variant of build_signal_matrix that accepts pluggable signal functions
    and custom weights/thresholds, but preserves the bias-free alignment.
    """
    df_1w = tf_data["1W"]
    df_1d = tf_data["1D"]
    df_4h = tf_data["4H"]
    df_1h = tf_data["1H"]

    base        = df_1h.index
    close_times = base + pd.Timedelta(hours=1)

    # Choose signal functions
    wk_fn  = weekly_fn   or weekly_trend
    dy_fn  = daily_fn    or daily_trend
    fh_fn  = fourfour_fn or fourfour_setup
    hr_fn  = hourly_fn   or one_hour_entry

    use_real_basis = (premium_1h is not None and
                      not premium_1h.empty and len(premium_1h) > 100)
    if use_real_basis:
        s_oi_raw = basis_oi_signal(premium_1h, df_1h)
    else:
        s_oi_raw = oi_signal(oi_df, df_1d)

    has_15m = (df_15m is not None and not df_15m.empty and
               "rsi_7" in df_15m.columns and len(df_15m) > 100)
    s_15m_raw = fifteen_min_entry(df_15m) if has_15m else pd.Series(0.0, index=base)

    has_1m = (df_1m is not None and not df_1m.empty and
              "vol_ratio" in df_1m.columns and len(df_1m) > 100)
    s_1m_raw = one_min_entry(df_1m) if has_1m else pd.Series(0.0, index=base)

    def _shift(series, delta):
        s = series.copy()
        s.index = s.index + delta
        return s

    out = pd.DataFrame(index=base)

    out["s_weekly"] = _align(_shift(wk_fn(df_1w),  pd.Timedelta(weeks=1)),  base)
    out["s_daily"]  = _align(_shift(dy_fn(df_1d),  pd.Timedelta(days=1)),   base)
    out["s_4h"]     = _align(_shift(fh_fn(df_4h),  pd.Timedelta(hours=4)),  base)

    if use_real_basis:
        out["s_oi"] = _align(s_oi_raw, base)
    else:
        out["s_oi"] = _align(_shift(s_oi_raw, pd.Timedelta(days=1)), base)

    out["s_funding"] = _align(_shift(funding_signal(funding), pd.Timedelta(days=1)), base)
    out["s_15m"] = _align(_shift(s_15m_raw, pd.Timedelta(minutes=15)), close_times)
    out["s_1m"]  = _align(_shift(s_1m_raw,  pd.Timedelta(minutes=1)),  close_times)

    out["s_1h"]    = hr_fn(df_1h).reindex(base, fill_value=0).values
    out["s_vol"]   = volume_signal(df_1h).reindex(base, fill_value=0).values
    out["s_cycle"] = cyclicality_signal(df_1h).reindex(base, fill_value=0).values

    out["oi_source"] = "real_basis" if use_real_basis else "synthetic"
    out["has_15m"]   = int(has_15m)
    out["has_1m"]    = int(has_1m)

    W = weights or WEIGHTS
    out["composite"] = sum(out[k] * W[k] for k in W)

    # Signal placeholder — will be overridden by apply_filters with actual threshold
    out["signal"] = np.where(
        out["composite"] >= 5.0,  1,
        np.where(out["composite"] <= -5.0, -1, 0),
    ).astype(int)
    out["strong"] = (out["composite"].abs() >= 8.0).astype(int)

    bull_regime = (df_1d["close"] > df_1d["ema_200"]).astype(int)
    out["regime"] = _align(
        _shift(bull_regime.map({1: "bull", 0: "bear"}).rename("regime"),
               pd.Timedelta(days=1)),
        base,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# KPI summariser
# ─────────────────────────────────────────────────────────────────────────────

def kpi_row(name: str, bt: dict) -> dict:
    k = bt["kpis"]
    return {
        "name":         name,
        "total_return": round(k.get("total_return", 0) * 100, 2),
        "max_dd":       round(k.get("max_drawdown", 0) * 100, 2),
        "sharpe":       round(k.get("sharpe", 0), 3),
        "sortino":      round(k.get("sortino", 0), 3),
        "calmar":       round(k.get("calmar", 0), 3),
        "n_trades":     k.get("n_trades", 0),
        "win_rate":     round(k.get("win_rate", 0) * 100, 1),
        "profit_factor":round(k.get("profit_factor", 0), 2),
        "expectancy":   round(k.get("expectancy", 0), 0),
        "final_equity": round(k.get("final_equity", 0), 0),
        "equity":       bt["equity"].tolist(),
        "drawdown":     bt["drawdown"].tolist(),
        "index":        [str(t) for t in bt["equity"].index],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n══ Experiment Report ══════════════════════════════════════════════")

    # ── Load & prepare data ───────────────────────────────────────────────────
    print("\n[1/6] Loading data from cache …")
    raw = fetch_extended_data(start_year=2022, start_month=1, fetch_1m=False)

    tf_ind = {}
    for tf in ["1W", "1D", "4H", "1H", "15M"]:
        df = raw.get(tf, pd.DataFrame())
        tf_ind[tf] = add_indicators(df) if not df.empty and len(df) > 20 else df

    df_1h  = tf_ind["1H"]
    df_15m = tf_ind["15M"]
    oi_df  = generate_oi(tf_ind["1D"]["close"]) if not tf_ind["1D"].empty else pd.DataFrame()
    funding = generate_funding(tf_ind["1D"]["close"]) if not tf_ind["1D"].empty else pd.Series(dtype=float)

    shared_args = dict(
        tf_data    = tf_ind,
        oi_df      = oi_df,
        funding    = funding,
        premium_1h = None,      # no live premium in backtest context
        df_15m     = df_15m,
        df_1m      = None,
    )

    def run(signals, thr=5.0):
        return _run(df_1h, signals, thr=thr)

    results = {}   # group → list of kpi_row dicts

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("\n[2/6] Baseline (current bias-fixed signals) …")
    base_sig = build_signal_matrix(**shared_args)
    base_bt  = run(base_sig)
    baseline = kpi_row("Baseline (current)", base_bt)
    print(f"  Return={baseline['total_return']:.1f}%  DD={baseline['max_dd']:.1f}%  "
          f"Sharpe={baseline['sharpe']:.3f}  Trades={baseline['n_trades']}")

    # ══════════════════════════════════════════════════════════════════════════
    # Direction 1 – Momentum signals
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[3/6] Direction 1: Momentum signals …")
    dir1 = []

    dir1.append(baseline)   # reference

    # Full momentum replacement
    sig_mom = build_custom_matrix(
        **shared_args,
        weekly_fn=weekly_momentum, daily_fn=daily_momentum,
        fourfour_fn=fourfour_momentum, hourly_fn=hourly_momentum,
    )
    bt = run(sig_mom)
    r  = kpi_row("Full Momentum (ROC)", bt)
    dir1.append(r)
    print(f"  Full Momentum:   Return={r['total_return']:.1f}%  DD={r['max_dd']:.1f}%  Sharpe={r['sharpe']:.3f}")

    # Weekly + Daily momentum, keep 4H + 1H trend
    sig_htf_mom = build_custom_matrix(
        **shared_args,
        weekly_fn=weekly_momentum, daily_fn=daily_momentum,
    )
    bt = run(sig_htf_mom)
    r  = kpi_row("HTF Momentum + Trend", bt)
    dir1.append(r)
    print(f"  HTF Momentum:    Return={r['total_return']:.1f}%  DD={r['max_dd']:.1f}%  Sharpe={r['sharpe']:.3f}")

    # LTF momentum only (keep trend for HTF)
    sig_ltf_mom = build_custom_matrix(
        **shared_args,
        fourfour_fn=fourfour_momentum, hourly_fn=hourly_momentum,
    )
    bt = run(sig_ltf_mom)
    r  = kpi_row("LTF Momentum + HTF Trend", bt)
    dir1.append(r)
    print(f"  LTF Momentum:    Return={r['total_return']:.1f}%  DD={r['max_dd']:.1f}%  Sharpe={r['sharpe']:.3f}")

    results["dir1"] = dir1

    # ══════════════════════════════════════════════════════════════════════════
    # Direction 2 – Threshold sweep
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[4/6] Direction 2: Threshold sweep …")
    dir2 = []
    thresholds = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15]

    for thr in thresholds:
        bt = run(base_sig, thr=thr)      # same composite, different threshold
        r  = kpi_row(f"Threshold ±{thr}", bt)
        dir2.append(r)
        print(f"  ±{thr:2d}:  Return={r['total_return']:+7.1f}%  DD={r['max_dd']:.1f}%  "
              f"Sharpe={r['sharpe']:+.3f}  Trades={r['n_trades']}")

    results["dir2"] = dir2

    # ══════════════════════════════════════════════════════════════════════════
    # Direction 3 – 1H entry variants
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[5/6] Direction 3: 1H entry variants …")
    dir3 = [baseline]

    for label, fn in [
        ("Donchian Breakout",       donchian_breakout),
        ("RSI Mean-reversion",      rsi_mean_reversion),
        ("Mixed (Breakout+Trend)",   mixed_entry),
    ]:
        sig_v = build_custom_matrix(**shared_args, hourly_fn=fn)
        bt    = run(sig_v)
        r     = kpi_row(label, bt)
        dir3.append(r)
        print(f"  {label:<28s}  Return={r['total_return']:+7.1f}%  DD={r['max_dd']:.1f}%  Sharpe={r['sharpe']:+.3f}")

    results["dir3"] = dir3

    # ══════════════════════════════════════════════════════════════════════════
    # Direction 4 – Weight configurations
    # ══════════════════════════════════════════════════════════════════════════
    print("\n[6/6] Direction 4: Weight configurations …")
    dir4 = [baseline]

    weight_presets = {
        "Trend-heavy":   {"s_weekly":5,"s_daily":6,"s_4h":5,"s_1h":2,
                          "s_oi":1,"s_funding":1,"s_vol":1,"s_cycle":1,"s_15m":1,"s_1m":1},
        "Micro-heavy":   {"s_weekly":1,"s_daily":2,"s_4h":2,"s_1h":4,
                          "s_oi":2,"s_funding":1,"s_vol":3,"s_cycle":1,"s_15m":4,"s_1m":3},
        "Equal weights": {"s_weekly":1,"s_daily":1,"s_4h":1,"s_1h":1,
                          "s_oi":1,"s_funding":1,"s_vol":1,"s_cycle":1,"s_15m":1,"s_1m":1},
        "Macro-only":    {"s_weekly":5,"s_daily":5,"s_4h":5,"s_1h":0,
                          "s_oi":0,"s_funding":0,"s_vol":0,"s_cycle":0,"s_15m":0,"s_1m":0},
        "No-macro":      {"s_weekly":0,"s_daily":0,"s_4h":0,"s_1h":5,
                          "s_oi":3,"s_funding":2,"s_vol":3,"s_cycle":2,"s_15m":4,"s_1m":2},
        "OI+Fund boost": {"s_weekly":3,"s_daily":4,"s_4h":3,"s_1h":3,
                          "s_oi":4,"s_funding":3,"s_vol":2,"s_cycle":1,"s_15m":2,"s_1m":1},
    }

    for label, wts in weight_presets.items():
        sig_w = build_custom_matrix(**shared_args, weights=wts)
        bt    = run(sig_w)
        r     = kpi_row(label, bt)
        dir4.append(r)
        print(f"  {label:<22s}  Return={r['total_return']:+7.1f}%  DD={r['max_dd']:.1f}%  Sharpe={r['sharpe']:+.3f}")

    results["dir4"] = dir4

    # ── HTML report ────────────────────────────────────────────────────────────
    print("\nGenerating HTML report …")
    html = _build_html(results, baseline)
    out  = Path("reports/experiment_report.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"  Saved → {out}")
    print("\n══ Done ═══════════════════════════════════════════════════════════")


# ─────────────────────────────────────────────────────────────────────────────
# HTML builder
# ─────────────────────────────────────────────────────────────────────────────

def _metric_color(val, metric):
    """Return a CSS class based on whether the metric value is good/bad."""
    if metric in ("total_return", "sharpe", "sortino", "calmar", "win_rate", "profit_factor"):
        return "pos" if val > 0 else "neg"
    if metric in ("max_dd",):
        return "pos" if val > -15 else "neg"
    return ""


def _build_html(results: dict, baseline: dict) -> str:
    base_ret = baseline["total_return"]
    base_dd  = baseline["max_dd"]

    dir1 = results["dir1"]
    dir2 = results["dir2"]
    dir3 = results["dir3"]
    dir4 = results["dir4"]

    # best threshold by return
    best_thr = max(dir2, key=lambda r: r["total_return"])

    # overall winner by total return (primary) then min DD (secondary)
    all_variants = dir1[1:] + dir2 + dir3[1:] + dir4[1:]
    best_overall = max(all_variants, key=lambda r: r["total_return"])
    # count profitable variants
    n_profitable = sum(1 for r in all_variants if r["total_return"] > 0)

    def rows_html(items, highlight_key="total_return"):
        best_val = max(r[highlight_key] for r in items)
        html = ""
        for r in items:
            is_base = r["name"].startswith("Baseline")
            is_best = (r[highlight_key] == best_val and not is_base)
            row_cls = " class='best-row'" if is_best else (" class='base-row'" if is_base else "")
            html += f"<tr{row_cls}>"
            html += f"<td>{r['name']}</td>"
            for col in ["total_return","max_dd","sharpe","win_rate","profit_factor","n_trades","expectancy"]:
                v   = r[col]
                cls = _metric_color(v, col)
                sfx = "%" if col in ("total_return","max_dd","win_rate") else ""
                html += f"<td class='{cls}'>{v}{sfx}</td>"
            html += "</tr>\n"
        return html

    def equity_datasets(items, id_prefix):
        """Return JS datasets array string."""
        colours = [
            "#4CAF50","#2196F3","#FF9800","#9C27B0",
            "#F44336","#00BCD4","#FFEB3B","#E91E63","#795548","#607D8B",
        ]
        ds = []
        for i, r in enumerate(items):
            col   = colours[i % len(colours)]
            label = r["name"].replace('"', "'")
            vals  = [round(v, 2) for v in r["equity"]]
            ds.append(f"""{{
  label: "{label}",
  data: {json.dumps(vals)},
  borderColor: "{col}",
  backgroundColor: "{col}22",
  borderWidth: {"2" if i == 0 else "1.5"},
  pointRadius: 0,
  tension: 0.1,
  fill: false,
}}""")
        return "[" + ",\n".join(ds) + "]"

    # Use first item's index for x-axis (all share same 1H base)
    idx = dir1[0]["index"]
    # Downsample to daily for readability in chart
    idx_series = pd.DatetimeIndex(idx)
    # Take every 24th point
    step = max(1, len(idx) // 800)
    chart_idx = [str(idx_series[i].date()) for i in range(0, len(idx), step)]

    def ds_down(items):
        """Datasets with downsampled equity."""
        colours = [
            "#4CAF50","#2196F3","#FF9800","#9C27B0",
            "#F44336","#00BCD4","#FFEB3B","#E91E63","#795548","#607D8B",
        ]
        ds = []
        for i, r in enumerate(items):
            col  = colours[i % len(colours)]
            vals = [round(r["equity"][j], 2) for j in range(0, len(r["equity"]), step)]
            label = r["name"].replace('"', "'")
            bw   = "2" if r["name"].startswith("Baseline") else "1.5"
            ds.append(f"""{{
  label: "{label}",
  data: {json.dumps(vals)},
  borderColor: "{col}",
  backgroundColor: "{col}22",
  borderWidth: {bw},
  pointRadius: 0,
  tension: 0.1,
  fill: false,
}}""")
        return "[" + ",\n".join(ds) + "]"

    idx_js = json.dumps(chart_idx)

    # Threshold chart: bar chart of sharpe vs threshold
    thr_labels = json.dumps([r["name"] for r in dir2])
    thr_sharpe = json.dumps([r["sharpe"] for r in dir2])
    thr_ret    = json.dumps([r["total_return"] for r in dir2])
    thr_dd     = json.dumps([r["max_dd"] for r in dir2])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Experiment Report — BTCUSDT Strategy</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{box-sizing:border-box; margin:0; padding:0;}}
  body {{font-family:'Segoe UI',Arial,sans-serif; background:#0d1117; color:#c9d1d9; font-size:14px;}}
  h1 {{text-align:center; padding:24px; font-size:22px; color:#58a6ff; border-bottom:1px solid #30363d;}}
  h2 {{color:#79c0ff; font-size:16px; margin:20px 0 10px; padding-left:4px; border-left:3px solid #388bfd;}}
  h3 {{color:#8b949e; font-size:13px; margin:10px 0 6px;}}
  .section {{background:#161b22; border:1px solid #30363d; border-radius:8px; padding:20px; margin:16px;}}
  .highlight-box {{background:#1c2333; border:1px solid #388bfd; border-radius:6px; padding:12px 16px; margin-bottom:14px;}}
  .highlight-box p {{line-height:1.7; color:#c9d1d9;}}
  .highlight-box strong {{color:#58a6ff;}}
  table {{width:100%; border-collapse:collapse; font-size:13px; margin-top:8px;}}
  th {{background:#1c2333; color:#8b949e; padding:8px 10px; text-align:right; font-weight:600; position:sticky; top:0;}}
  th:first-child {{text-align:left;}}
  td {{padding:7px 10px; border-bottom:1px solid #21262d; text-align:right;}}
  td:first-child {{text-align:left; color:#c9d1d9;}}
  tr:hover td {{background:#1c2333;}}
  .pos {{color:#3fb950;}}
  .neg {{color:#f85149;}}
  .base-row td {{background:#161b22; font-style:italic; color:#8b949e;}}
  .best-row td {{background:#1a2d1a; font-weight:600;}}
  .chart-container {{position:relative; height:320px; margin:10px 0;}}
  .chart-sm {{position:relative; height:220px; margin:6px 0;}}
  .grid2 {{display:grid; grid-template-columns:1fr 1fr; gap:16px;}}
  .summary-grid {{display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:12px 0;}}
  .stat-card {{background:#1c2333; border:1px solid #30363d; border-radius:6px; padding:12px; text-align:center;}}
  .stat-card .val {{font-size:20px; font-weight:700; margin:4px 0;}}
  .stat-card .lbl {{font-size:11px; color:#8b949e;}}
  .dir-badge {{display:inline-block; background:#1c2333; border:1px solid #388bfd; color:#58a6ff;
               border-radius:4px; padding:2px 8px; font-size:11px; margin-right:6px;}}
  @media(max-width:700px){{.grid2{{grid-template-columns:1fr;}} .summary-grid{{grid-template-columns:1fr 1fr;}}}}
</style>
</head>
<body>
<h1>BTCUSDT Strategy — Experimental Directions Report</h1>

<!-- ── Summary ─────────────────────────────────────────────── -->
<div class="section">
<h2>Executive Summary</h2>
<div class="highlight-box">
<p>
All results use the <strong>bias-corrected signal matrix</strong> (no lookahead).
Session filter <strong>08:00–21:00 UTC</strong> applied to every experiment.
Baseline: <strong class="{'pos' if base_ret>0 else 'neg'}">{base_ret:+.1f}% return</strong>,
Max DD <strong class="neg">{base_dd:.1f}%</strong>.
<strong class="pos">{n_profitable} profitable variant(s)</strong> found across {len(all_variants)} tested.
<br>
Best overall: <strong>{best_overall['name']}</strong> →
Return <strong class="{'pos' if best_overall['total_return']>0 else 'neg'}">{best_overall['total_return']:+.1f}%</strong>,
Max DD <strong class="neg">{best_overall['max_dd']:.1f}%</strong>.
</p>
<p style="margin-top:8px;color:#e3b341;">
⚠ Note: Sharpe ratios are computed on bar-by-bar equity changes (1H granularity). Strategies that are in a trade
most of the time may show inflated Sharpe values. Use <strong>Return %</strong> and <strong>Max DD</strong>
as primary decision metrics.
</p>
</div>

<div class="summary-grid">
  <div class="stat-card">
    <div class="lbl">Best Return</div>
    <div class="val {'pos' if best_overall['total_return']>0 else 'neg'}">{best_overall['total_return']:+.1f}%</div>
    <div class="lbl">{best_overall['name'][:24]}</div>
  </div>
  <div class="stat-card">
    <div class="lbl">Best Threshold (Return)</div>
    <div class="val {'pos' if best_thr['total_return']>0 else 'neg'}">{best_thr['name']}</div>
    <div class="lbl">Return {best_thr['total_return']:+.1f}%</div>
  </div>
  <div class="stat-card">
    <div class="lbl">Lowest Max DD</div>
    <div class="val pos">{min(all_variants,key=lambda r:abs(r['max_dd']))['max_dd']:.1f}%</div>
    <div class="lbl">{min(all_variants,key=lambda r:abs(r['max_dd']))['name'][:24]}</div>
  </div>
  <div class="stat-card">
    <div class="lbl">Profitable Variants</div>
    <div class="val {'pos' if n_profitable>0 else 'neg'}">{n_profitable} / {len(all_variants)}</div>
    <div class="lbl">across all 4 directions</div>
  </div>
</div>
</div>

<!-- ── Direction 1 ──────────────────────────────────────────── -->
<div class="section">
<h2><span class="dir-badge">Dir 1</span>Momentum Signals (ROC-based)</h2>
<p style="color:#8b949e;font-size:12px;margin-bottom:12px;">
Replaces EMA/MACD crossover logic with Rate-of-Change (ROC) based signals across weekly, daily, 4H, and 1H timeframes.
ROC captures price velocity directly; EMA crossovers add lag. Variants: Full replacement, HTF-only, LTF-only.
</p>
<div class="chart-container"><canvas id="c_dir1"></canvas></div>
<div style="overflow-x:auto"><table>
<tr><th>Variant</th><th>Return%</th><th>Max DD%</th><th>Sharpe</th><th>Win%</th><th>PF</th><th>Trades</th><th>Expect($)</th></tr>
{rows_html(dir1)}
</table></div>
</div>

<!-- ── Direction 2 ──────────────────────────────────────────── -->
<div class="section">
<h2><span class="dir-badge">Dir 2</span>Composite Threshold Sweep</h2>
<p style="color:#8b949e;font-size:12px;margin-bottom:12px;">
Testing composite score thresholds from ±2 (very permissive, many trades) to ±15 (very selective, few trades).
Lower threshold → more signals but lower average quality. Higher → fewer but stronger signals.
</p>
<div class="grid2">
  <div>
    <h3>Return % by Threshold</h3>
    <div class="chart-sm"><canvas id="c_thr_sharpe"></canvas></div>
  </div>
  <div>
    <h3>Return % vs Max DD</h3>
    <div class="chart-sm"><canvas id="c_thr_ret"></canvas></div>
  </div>
</div>
<div class="chart-container" style="height:260px"><canvas id="c_dir2"></canvas></div>
<div style="overflow-x:auto"><table>
<tr><th>Threshold</th><th>Return%</th><th>Max DD%</th><th>Sharpe</th><th>Win%</th><th>PF</th><th>Trades</th><th>Expect($)</th></tr>
{rows_html(dir2)}
</table></div>
</div>

<!-- ── Direction 3 ──────────────────────────────────────────── -->
<div class="section">
<h2><span class="dir-badge">Dir 3</span>1H Entry Signal Variants</h2>
<p style="color:#8b949e;font-size:12px;margin-bottom:12px;">
Replaces the current 1H entry (RSI zone + MACD sign + vol spike) with:
Donchian breakout (trend-following, enters on N-bar high/low breakout),
RSI mean-reversion (contrarian, buys oversold/sells overbought),
or a 50/50 blend of breakout + current trend.
</p>
<div class="chart-container"><canvas id="c_dir3"></canvas></div>
<div style="overflow-x:auto"><table>
<tr><th>Variant</th><th>Return%</th><th>Max DD%</th><th>Sharpe</th><th>Win%</th><th>PF</th><th>Trades</th><th>Expect($)</th></tr>
{rows_html(dir3)}
</table></div>
</div>

<!-- ── Direction 4 ──────────────────────────────────────────── -->
<div class="section">
<h2><span class="dir-badge">Dir 4</span>Weight Configurations</h2>
<p style="color:#8b949e;font-size:12px;margin-bottom:12px;">
Six weight presets: Current (baseline), Trend-heavy (amplifies HTF signals),
Micro-heavy (amplifies 1H/15M/1M), Equal weights (all components equal),
Macro-only (weekly+daily+4H only), No-macro (intraday + sentiment only), OI+Fund boost.
</p>
<div class="chart-container"><canvas id="c_dir4"></canvas></div>
<div style="overflow-x:auto"><table>
<tr><th>Weights</th><th>Return%</th><th>Max DD%</th><th>Sharpe</th><th>Win%</th><th>PF</th><th>Trades</th><th>Expect($)</th></tr>
{rows_html(dir4)}
</table></div>
</div>

<!-- ── Scripts ──────────────────────────────────────────────── -->
<script>
const IDX = {idx_js};
const CFG = {{
  type:'line',
  options:{{
    responsive:true, maintainAspectRatio:false,
    animation:{{duration:0}},
    plugins:{{legend:{{labels:{{color:'#8b949e',font:{{size:11}}}}}}}},
    scales:{{
      x:{{ticks:{{color:'#8b949e',maxTicksLimit:12,font:{{size:10}}}},grid:{{color:'#21262d'}}}},
      y:{{ticks:{{color:'#8b949e',font:{{size:10}},callback:v=>v.toLocaleString()}},grid:{{color:'#21262d'}}}}
    }}
  }}
}};
function mkChart(id, datasets){{
  new Chart(document.getElementById(id),{{...CFG, data:{{labels:IDX, datasets}}}});
}}

mkChart('c_dir1', {ds_down(dir1)});
mkChart('c_dir2', {ds_down(dir2)});
mkChart('c_dir3', {ds_down(dir3)});
mkChart('c_dir4', {ds_down(dir4)});

// Threshold bar charts
new Chart(document.getElementById('c_thr_sharpe'),{{
  type:'bar',
  data:{{labels:{thr_labels},datasets:[
    {{label:'Return%',data:{thr_ret},
     backgroundColor:{thr_ret}.map(v=>v>0?'#3fb95088':'#f8514988'),
     borderColor:{thr_ret}.map(v=>v>0?'#3fb950':'#f85149'),
     borderWidth:1}}
  ]}},
  options:{{
    responsive:true, maintainAspectRatio:false, animation:{{duration:0}},
    plugins:{{legend:{{display:false}}}},
    scales:{{
      x:{{ticks:{{color:'#8b949e',font:{{size:10}}}},grid:{{color:'#21262d'}}}},
      y:{{ticks:{{color:'#8b949e',font:{{size:10}}}},grid:{{color:'#21262d'}}}}
    }}
  }}
}});

new Chart(document.getElementById('c_thr_ret'),{{
  type:'bar',
  data:{{labels:{thr_labels},datasets:[
    {{label:'Return%',data:{thr_ret},
     backgroundColor:{thr_ret}.map(v=>v>0?'#2196F388':'#FF980088'),
     borderColor:{thr_ret}.map(v=>v>0?'#2196F3':'#FF9800'),
     borderWidth:1,yAxisID:'y'}},
    {{label:'Max DD%',data:{thr_dd},type:'line',
     borderColor:'#f85149',backgroundColor:'transparent',
     borderWidth:1.5,pointRadius:3,yAxisID:'y'}}
  ]}},
  options:{{
    responsive:true, maintainAspectRatio:false, animation:{{duration:0}},
    plugins:{{legend:{{labels:{{color:'#8b949e',font:{{size:10}}}}}}}},
    scales:{{
      x:{{ticks:{{color:'#8b949e',font:{{size:10}}}},grid:{{color:'#21262d'}}}},
      y:{{ticks:{{color:'#8b949e',font:{{size:10}}}},grid:{{color:'#21262d'}}}}
    }}
  }}
}});
</script>
</body>
</html>"""
    return html


if __name__ == "__main__":
    main()
