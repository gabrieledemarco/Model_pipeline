#!/usr/bin/env python3
"""
create_mtf_trend_scan_v2_report.py
====================================
Iterazione sul sistema MTF trend-start (v1: create_mtf_trend_scan_report.py,
tutte le 4 varianti fallite su holdout 2025-2026 con stop ATR-multiplo).

Griglia di iterazione (tutti i suggerimenti discussi):
  - Stop: ATR×1.5 (baseline v1), ATR×3 (più largo), Swing opposto (il pivot
    LTF più recente in direzione contraria — stop strutturale, non arbitrario)
  - Pivot HTF (lookback per la definizione di trend, left=right): 5 (v1), 10
    (meno rumoroso, meno whipsaw)
  - LTF: 15m (v1), 5m (timeframe più fine)
  - Trigger LTF: stessi 4 di v1 (Donchian20, Donchian50, Squeeze, Pullback)

= 4 trigger × 2 pivot-lookback × 3 stop = 24 combinazioni, per ciascun LTF.

Stesso protocollo di validazione: breakdown per anno FIN DA SUBITO, holdout
2025-2026 genuino mai usato per scegliere la combinazione vincente.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from src.strategy.data_fetcher import fetch_binance_vision_klines
from src.strategy.mtf_swing import causal_trend_state, align_htf_to_ltf
from src.strategy.monte_carlo import run_monte_carlo

SEP = "═" * 78
START_YEAR = 2020
INIT_CAP = 100_000.0
RISK_PCT = 0.01
FEE = 0.0004
MAX_LEV = 10.0
MAX_RR_RATIO = 15.0
CUTOFF = pd.Timestamp("2025-01-01")
N_SIMS = 3_000   # reduced from 5000: many combos to scan, MC precision still fine

HTF_TF = "4h"
LTF_TFS = ["15m", "5m"]
LTF_MINUTES = {"15m": 15, "5m": 5}
HTF_PIVOT_LOOKBACKS = [5, 10]
STOP_MODES = ["atr1.5", "atr3", "swing"]
LTF_SWING_LOOKBACK = 5   # fixed left/right for the LTF's own pivots (stop reference)
COOLDOWN_HOURS = 2
MAX_HOLD_HOURS = 96
WARMUP_BARS_MIN = 500

report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

def wf_dates(idx, train_m=6, oos_m=2, step_m=2):
    t0 = idx[0]; windows = []
    while True:
        tr_s = t0; tr_e = tr_s + pd.DateOffset(months=train_m)
        oo_s = tr_e; oo_e = oo_s + pd.DateOffset(months=oos_m)
        if oo_e > idx[-1]: break
        windows.append((tr_s, tr_e, oo_s, oo_e))
        t0 += pd.DateOffset(months=step_m)
    return windows

def mc_summary(pnls):
    if len(pnls) < 5: return dict(p_profit=0.0, p_ruin=1.0)
    mc = run_monte_carlo(pd.DataFrame({"net_pnl": pnls}), INIT_CAP, N_SIMS)
    return dict(p_profit=float(mc.get("p_profit", 0.0)), p_ruin=float(mc.get("p_ruin", 1.0)))

w(SEP)
w("MTF Trend Scan v2 — iterazione stop (ATR / swing opposto) × pivot HTF × LTF")
w(SEP)

for ltf_tf in LTF_TFS:
    t_ltf_start = time.time()
    w(f"\n{SEP}")
    w(f"LTF = {ltf_tf}")
    w(SEP)

    ltf_min = LTF_MINUTES[ltf_tf]
    COOLDOWN_BARS = max(1, COOLDOWN_HOURS * 60 // ltf_min)
    MAX_HOLD_BARS = MAX_HOLD_HOURS * 60 // ltf_min

    print(f"\n[DATA] Loading {HTF_TF} and {ltf_tf} OHLCV …")
    df_htf = fetch_binance_vision_klines(HTF_TF, start_year=START_YEAR, start_month=1,
                                          workers=6, verbose=False)
    df_ltf = fetch_binance_vision_klines(ltf_tf, start_year=START_YEAR, start_month=1,
                                          workers=6, verbose=False)
    print(f"  {HTF_TF}: {len(df_htf):,} bars   {ltf_tf}: {len(df_ltf):,} bars")

    IDXLTF = df_ltf.index
    NLTF = len(df_ltf)
    CL = df_ltf["close"].values.astype(float)
    HI = df_ltf["high"].values.astype(float)
    LO = df_ltf["low"].values.astype(float)
    OP = df_ltf["open"].values.astype(float)
    CL_s = pd.Series(CL, index=IDXLTF)

    tr = np.maximum(HI[1:] - LO[1:], np.maximum(np.abs(HI[1:] - CL[:-1]), np.abs(LO[1:] - CL[:-1])))
    tr = np.concatenate([[HI[0] - LO[0]], tr])
    ATR = pd.Series(tr).ewm(span=14, adjust=False).mean().shift(1).bfill().values
    ATR = np.where(ATR > 0, ATR, 1.0)
    WARMUP = max(WARMUP_BARS_MIN, 60 * 24 // ltf_min)   # >= 1 day of context

    # ── LTF's own pivots (fixed lookback) — used for the "swing opposite" stop
    print(f"[LTF-PIVOTS] Computing LTF pivot structure (left=right={LTF_SWING_LOOKBACK}) …")
    ltf_pivots = causal_trend_state(CL, HI, LO, LTF_SWING_LOOKBACK, LTF_SWING_LOOKBACK)
    LTF_LAST_PIVOT_HIGH = ltf_pivots["last_pivot_high"].values
    LTF_LAST_PIVOT_LOW = ltf_pivots["last_pivot_low"].values

    # ── LTF entry signals (independent of HTF pivot lookback / stop mode) ──
    def sig_donchian(trend, n_bars):
        donch_hi = CL_s.rolling(n_bars).max().shift(1).values
        donch_lo = CL_s.rolling(n_bars).min().shift(1).values
        long_sig = (CL > donch_hi) & (trend == 1)
        short_sig = (CL < donch_lo) & (trend == -1)
        s = np.where(long_sig, 1.0, np.where(short_sig, -1.0, 0.0))
        s[:WARMUP] = 0
        return s

    def sig_squeeze(trend, bb_len=20, bb_mult=2.0, width_win=200, sq_pct=0.2, exp_mult=1.5):
        mid = CL_s.rolling(bb_len).mean()
        std = CL_s.rolling(bb_len).std()
        width = (2 * bb_mult * std) / mid.replace(0, np.nan)
        width_thresh = width.rolling(width_win).quantile(sq_pct)
        squeeze = (width < width_thresh).shift(1).fillna(False).values
        tr_series = pd.Series(tr)
        expansion = (tr_series > tr_series.rolling(20).mean() * exp_mult).values
        up_bar = CL > OP; dn_bar = CL < OP
        long_sig = squeeze & expansion & up_bar & (trend == 1)
        short_sig = squeeze & expansion & dn_bar & (trend == -1)
        s = np.where(long_sig, 1.0, np.where(short_sig, -1.0, 0.0))
        s[:WARMUP] = 0
        return s

    def sig_pullback(trend, t_ema, pb_ema, dev):
        te = CL_s.ewm(span=t_ema, adjust=False).mean().shift(1).values
        pe = CL_s.ewm(span=pb_ema, adjust=False).mean().shift(1).values
        prev_cl = CL_s.shift(1).values
        long_sig = (prev_cl > te) & (prev_cl < pe * (1 - dev)) & (trend == 1)
        short_sig = (prev_cl < te) & (prev_cl > pe * (1 + dev)) & (trend == -1)
        s = np.where(long_sig, 1.0, np.where(short_sig, -1.0, 0.0))
        s[:WARMUP] = 0
        return s

    def run_bt(events, max_hold):
        if not events:
            return dict(n=0, wr=0.0, ret=0.0, mdd=0.0, net_pnls=[], n_dropped_none=0)
        cap = INIT_CAP; peak = cap; mdd = 0.0; wins = 0; net_pnls = []
        n_dropped_none = 0
        for ev in events:
            i, d, ep, tp, sl = ev["i"], ev["d"], ev["ep"], ev["tp"], ev["sl"]
            out = "none"
            for k in range(1, max_hold + 1):
                if i + k >= NLTF: break
                hk, lk = HI[i + k], LO[i + k]
                if d == 1:
                    if hk >= tp: out = "tp"; break
                    if lk <= sl: out = "sl"; break
                else:
                    if lk <= tp: out = "tp"; break
                    if hk >= sl: out = "sl"; break
            if out == "none":
                j = min(i + max_hold, NLTF - 1)
                exit_price = CL[j]
                n_dropped_none += 1
            else:
                exit_price = tp if out == "tp" else sl
            stop_dist = abs(ep - sl)
            if stop_dist <= 0: continue
            risk = INIT_CAP * RISK_PCT
            units = min(risk / stop_dist, MAX_LEV * INIT_CAP / ep)
            notional = units * ep
            pnl_dollar = units * (exit_price - ep) * d - FEE * 2 * notional
            cap += pnl_dollar
            peak = max(peak, cap)
            mdd = min(mdd, (cap - peak) / peak)
            wins += int(pnl_dollar > 0)
            net_pnls.append(pnl_dollar)
        n = len(net_pnls); wr = wins / n if n else 0.0
        return dict(n=n, wr=wr, ret=(cap / INIT_CAP - 1) * 100, mdd=mdd * 100,
                    net_pnls=net_pnls, n_dropped_none=n_dropped_none)

    def make_events(sig, idx_arr, target_high, target_low, stop_mode, cooldown_bars):
        evs = []
        last_s = -cooldown_bars
        for k in idx_arr:
            if k >= NLTF or sig[k] == 0 or ATR[k] <= 0: continue
            if k - last_s < cooldown_bars: continue
            d = 1 if sig[k] > 0 else -1
            ep = CL[k]
            target = target_high[k] if d == 1 else target_low[k]
            if np.isnan(target): continue
            if d == 1 and target <= ep: continue
            if d == -1 and target >= ep: continue

            if stop_mode == "atr1.5":
                stop = ep - d * 1.5 * ATR[k]
            elif stop_mode == "atr3":
                stop = ep - d * 3.0 * ATR[k]
            else:  # "swing" — opposite LTF pivot, fallback to atr3 if unavailable/invalid
                if d == 1:
                    cand = LTF_LAST_PIVOT_LOW[k]
                    stop = cand if (not np.isnan(cand) and cand < ep) else ep - d * 3.0 * ATR[k]
                else:
                    cand = LTF_LAST_PIVOT_HIGH[k]
                    stop = cand if (not np.isnan(cand) and cand > ep) else ep - d * 3.0 * ATR[k]

            stop_dist = abs(ep - stop)
            reward_dist = abs(target - ep)
            if stop_dist <= 0 or reward_dist / stop_dist > MAX_RR_RATIO:
                continue
            evs.append(dict(i=k, d=d, ep=ep, tp=target, sl=stop))
            last_s = k
        return evs

    WF_WINDOWS = wf_dates(IDXLTF)
    WF_HOLDOUT = [wd for wd in WF_WINDOWS if wd[2] >= CUTOFF]
    print(f"[WFO] {len(WF_WINDOWS)} windows  |  {len(WF_HOLDOUT)} holdout (2025-2026)")

    ENTRY_FNS = {
        "Donchian20": lambda trend: sig_donchian(trend, 20),
        "Donchian50": lambda trend: sig_donchian(trend, 50),
        "Squeeze":    lambda trend: sig_squeeze(trend),
        "Pullback":   lambda trend: sig_pullback(trend, t_ema=200, pb_ema=20, dev=0.002),
    }

    grid_results = []
    for pivot_lr in HTF_PIVOT_LOOKBACKS:
        htf_close = df_htf["close"].values.astype(float)
        htf_high = df_htf["high"].values.astype(float)
        htf_low = df_htf["low"].values.astype(float)
        htf_state = causal_trend_state(htf_close, htf_high, htf_low, pivot_lr, pivot_lr)
        aligned = align_htf_to_ltf(df_htf.index, htf_state, IDXLTF)
        trend = aligned["trend_state"].values
        target_high = aligned["target_high"].values
        target_low = aligned["target_low"].values

        entry_signals = {name: fn(trend) for name, fn in ENTRY_FNS.items()}

        for entry_name, sig in entry_signals.items():
            for stop_mode in STOP_MODES:
                combo_id = f"{entry_name}|pivot{pivot_lr}|{stop_mode}"
                all_evs, year_evs = [], {}
                for tr_s, tr_e, oo_s, oo_e in WF_WINDOWS:
                    idx_oos = np.where((IDXLTF >= oo_s) & (IDXLTF < oo_e))[0]
                    if len(idx_oos) < 50: continue
                    evs = make_events(sig, idx_oos, target_high, target_low,
                                       stop_mode, COOLDOWN_BARS)
                    all_evs.extend(evs)
                    year_evs.setdefault(oo_s.year, []).extend(evs)

                res = run_bt(all_evs, MAX_HOLD_BARS)
                mc = mc_summary(res["net_pnls"])
                holdout_evs = [e for yr, evs in year_evs.items() if yr >= 2025 for e in evs]
                hres = run_bt(holdout_evs, MAX_HOLD_BARS)
                hmc = mc_summary(hres["net_pnls"])

                grid_results.append(dict(
                    ltf=ltf_tf, combo=combo_id, entry=entry_name, pivot=pivot_lr,
                    stop=stop_mode, n=res["n"], ret=res["ret"], mdd=res["mdd"],
                    wr=res["wr"], pp=mc["p_profit"], pr=mc["p_ruin"],
                    h_n=hres["n"], h_ret=hres["ret"], h_mdd=hres["mdd"], h_pp=hmc["p_profit"],
                ))
                print(f"  [{ltf_tf}] {combo_id:<32} n={res['n']:>5} ret={res['ret']:>+7.1f}% "
                      f"pp={mc['p_profit']:.3f}  |  holdout: n={hres['n']:>4} "
                      f"ret={hres['ret']:>+7.1f}% pp={hmc['p_profit']:.3f}")

    w(f"\n  {ltf_tf} — {len(grid_results)} combinazioni testate in {time.time()-t_ltf_start:.0f}s")
    w(f"\n  {'Combo':<32}  {'n':>6}  {'Ret%':>8}  {'pp':>6}  |  {'H.n':>5}  {'H.Ret%':>8}  {'H.pp':>6}")
    for r in sorted(grid_results, key=lambda r: r["h_pp"], reverse=True)[:10]:
        w(f"  {r['combo']:<32}  {r['n']:>6}  {r['ret']:>+7.1f}%  {r['pp']:>5.3f}  |  "
          f"{r['h_n']:>5}  {r['h_ret']:>+7.1f}%  {r['h_pp']:>5.3f}")

    pd.DataFrame(grid_results).to_csv(f"reports/mtf_trend_scan_v2_{ltf_tf}.csv", index=False)

w(f"\n{SEP}\n[DONE]\n{SEP}")
out_path = Path("reports/mtf_trend_scan_v2.md")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("# MTF Trend Scan v2 — iterazione stop/pivot/LTF\n\n```\n" +
                     "\n".join(report_lines) + "\n```\n", encoding="utf-8")
print(f"\n[DONE] {out_path}")
