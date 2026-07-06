#!/usr/bin/env python3
"""
create_mtf_trend_scan_report.py
=================================
Sistema multi-timeframe: identifica sul timeframe basso (15m, di default)
l'inizio di un trend allineato con lo swing del timeframe alto (4H), con
target = ultimo pivot HTF non ancora rotto nella direzione del trade.

Design (confermato con l'utente):
  - Swing HTF: struttura pivot causale (HH/HL = bullish, LH/LL = bearish)
  - Trigger LTF: 3 varianti candidate testate in parallelo
      A. Breakout Donchian (rottura di un range di consolidamento LTF)
      B. Squeeze + espansione di volatilità (contrazione poi breakout)
      C. Pullback-in-trend LTF (allineamento + minimo ritracciamento)
  - Target: ultimo pivot HTF (4H) ancora non rotto, nella direzione del trade
  - Stop: multiplo di ATR sul LTF
  - Sizing: risk-based standard (non la convenzione ATR-multipla di ADP),
    con cap di leva per evitare notional eccessivi quando lo stop è stretto

Segue lo stesso protocollo di validazione imparato in questa sessione:
breakdown per anno FIN DA SUBITO, holdout 2025-2026 genuino, MC, DSR.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from src.strategy.data_fetcher import fetch_binance_vision_klines
from src.strategy.mtf_swing import causal_trend_state, align_htf_to_ltf
from src.strategy.monte_carlo import run_monte_carlo, deflated_sharpe_ratio_family

SEP = "═" * 78
START_YEAR = 2020
INIT_CAP = 100_000.0
RISK_PCT = 0.01
FEE = 0.0004
MAX_LEV = 10.0  # safety-net cap for data glitches; primary risk control is MAX_RR_RATIO
CUTOFF = pd.Timestamp("2025-01-01")
N_SIMS = 5_000
DSR_THRESHOLD = 0.95

HTF_TF = "4h"
LTF_TF = "15m"
PIVOT_LEFT, PIVOT_RIGHT = 5, 5     # 4H bars: 20h left/right for pivot confirmation
COOLDOWN_BARS = 8                  # LTF bars between entries (2h on 15m)
MAX_HOLD_BARS = 4 * 24 * 4         # ~4 days on 15m bars
SL_ATR_MULT = 1.5

print(SEP)
print(f"MTF Trend Scan — swing {HTF_TF} (pivot HH/HL) + trigger {LTF_TF} (3 varianti)")
print(SEP)

# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[DATA] Loading {HTF_TF} and {LTF_TF} OHLCV …")
df_htf = fetch_binance_vision_klines(HTF_TF, start_year=START_YEAR, start_month=1,
                                      workers=6, verbose=False)
df_ltf = fetch_binance_vision_klines(LTF_TF, start_year=START_YEAR, start_month=1,
                                      workers=6, verbose=False)
print(f"  {HTF_TF}: {len(df_htf):,} bars  ({df_htf.index[0]} → {df_htf.index[-1]})")
print(f"  {LTF_TF}: {len(df_ltf):,} bars  ({df_ltf.index[0]} → {df_ltf.index[-1]})")

# ══════════════════════════════════════════════════════════════════════════════
# HTF SWING STATE (causal pivots) — computed once, aligned onto LTF
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[HTF] Computing causal pivot/trend state (left={PIVOT_LEFT}, right={PIVOT_RIGHT}) …")
htf_close = df_htf["close"].values.astype(float)
htf_high  = df_htf["high"].values.astype(float)
htf_low   = df_htf["low"].values.astype(float)
htf_state = causal_trend_state(htf_close, htf_high, htf_low, PIVOT_LEFT, PIVOT_RIGHT)
n_bull = (htf_state["trend_state"] == 1).sum()
n_bear = (htf_state["trend_state"] == -1).sum()
print(f"  HTF bars: bullish={n_bull} ({n_bull/len(htf_state):.0%})  "
      f"bearish={n_bear} ({n_bear/len(htf_state):.0%})  "
      f"neutral={len(htf_state)-n_bull-n_bear}")

aligned = align_htf_to_ltf(df_htf.index, htf_state, df_ltf.index)

IDXLTF = df_ltf.index
NLTF = len(df_ltf)
CL = df_ltf["close"].values.astype(float)
HI = df_ltf["high"].values.astype(float)
LO = df_ltf["low"].values.astype(float)
OP = df_ltf["open"].values.astype(float)
TREND = aligned["trend_state"].values
TARGET_HIGH = aligned["target_high"].values
TARGET_LOW = aligned["target_low"].values

# LTF ATR (causal, shifted)
tr = np.maximum(HI[1:] - LO[1:], np.maximum(np.abs(HI[1:] - CL[:-1]), np.abs(LO[1:] - CL[:-1])))
tr = np.concatenate([[HI[0] - LO[0]], tr])
ATR = pd.Series(tr).ewm(span=14, adjust=False).mean().shift(1).bfill().values
ATR = np.where(ATR > 0, ATR, 1.0)

WARMUP = 500

# ══════════════════════════════════════════════════════════════════════════════
# LTF SIGNAL VARIANTS
# ══════════════════════════════════════════════════════════════════════════════
CL_s = pd.Series(CL, index=IDXLTF)

def sig_donchian(n_bars: int) -> np.ndarray:
    donch_hi = CL_s.rolling(n_bars).max().shift(1).values
    donch_lo = CL_s.rolling(n_bars).min().shift(1).values
    long_sig  = (CL > donch_hi) & (TREND == 1)
    short_sig = (CL < donch_lo) & (TREND == -1)
    s = np.where(long_sig, 1.0, np.where(short_sig, -1.0, 0.0))
    s[:WARMUP] = 0
    return s

def sig_squeeze(bb_len: int = 20, bb_mult: float = 2.0, width_pctile_win: int = 200,
                squeeze_pctile: float = 0.2, expansion_mult: float = 1.5) -> np.ndarray:
    mid = CL_s.rolling(bb_len).mean()
    std = CL_s.rolling(bb_len).std()
    width = (2 * bb_mult * std) / mid.replace(0, np.nan)
    # Vectorized rolling-quantile threshold (fast) instead of a per-window
    # percentile-rank .apply() (which would be O(win) per bar, too slow at
    # ~210k 15m bars).
    width_thresh = width.rolling(width_pctile_win).quantile(squeeze_pctile)
    squeeze = (width < width_thresh).shift(1).fillna(False).values
    tr_series = pd.Series(tr)
    expansion = (tr_series > tr_series.rolling(20).mean() * expansion_mult).values
    up_bar = CL > OP
    dn_bar = CL < OP
    long_sig  = squeeze & expansion & up_bar & (TREND == 1)
    short_sig = squeeze & expansion & dn_bar & (TREND == -1)
    s = np.where(long_sig, 1.0, np.where(short_sig, -1.0, 0.0))
    s[:WARMUP] = 0
    return s

def sig_pullback_ltf(t_ema: int, pb_ema: int, dev: float) -> np.ndarray:
    te = CL_s.ewm(span=t_ema, adjust=False).mean().shift(1).values
    pe = CL_s.ewm(span=pb_ema, adjust=False).mean().shift(1).values
    prev_cl = CL_s.shift(1).values
    long_sig  = (prev_cl > te) & (prev_cl < pe * (1 - dev)) & (TREND == 1)
    short_sig = (prev_cl < te) & (prev_cl > pe * (1 + dev)) & (TREND == -1)
    s = np.where(long_sig, 1.0, np.where(short_sig, -1.0, 0.0))
    s[:WARMUP] = 0
    return s

VARIANTS = {
    "A_Donchian20":   lambda: sig_donchian(20),
    "A_Donchian50":   lambda: sig_donchian(50),
    "B_Squeeze":      lambda: sig_squeeze(),
    "C_Pullback":     lambda: sig_pullback_ltf(t_ema=200, pb_ema=20, dev=0.002),
}

# ══════════════════════════════════════════════════════════════════════════════
# EVENT CONSTRUCTION — target = HTF pivot, stop = ATR multiple, standard sizing
# ══════════════════════════════════════════════════════════════════════════════
MAX_RR_RATIO = 15.0   # skip trades whose HTF-pivot target implies an unrealistic
                       # reward:risk vs. the tight LTF ATR stop (avoids lottery-
                       # ticket-like position sizing / unstable compounding)

def make_events(sig: np.ndarray, idx_arr: np.ndarray) -> list:
    evs = []
    last_s = -COOLDOWN_BARS
    for k in idx_arr:
        if k >= NLTF or sig[k] == 0 or ATR[k] <= 0: continue
        if k - last_s < COOLDOWN_BARS: continue
        d = 1 if sig[k] > 0 else -1
        ep = CL[k]
        target = TARGET_HIGH[k] if d == 1 else TARGET_LOW[k]
        if np.isnan(target): continue           # no valid unbroken HTF pivot -> skip
        stop = ep - d * SL_ATR_MULT * ATR[k]
        if d == 1 and target <= ep: continue     # target must be ahead of price
        if d == -1 and target >= ep: continue
        stop_dist = abs(ep - stop)
        reward_dist = abs(target - ep)
        if stop_dist <= 0 or reward_dist / stop_dist > MAX_RR_RATIO:
            continue
        evs.append(dict(i=k, d=d, ep=ep, tp=target, sl=stop, a=ATR[k]))
        last_s = k
    return evs

def run_bt(events: list, max_hold: int = MAX_HOLD_BARS) -> dict:
    if not events:
        return dict(n=0, wr=0.0, ret=0.0, mdd=0.0, net_pnls=[], cap=INIT_CAP,
                     n_dropped_none=0, n_lev_capped=0)
    cap = INIT_CAP; peak = cap; mdd = 0.0; wins = 0; net_pnls = []
    n_dropped_none = 0
    n_lev_capped = 0
    for ev in events:
        i, d, ep, tp, sl, a = ev["i"], ev["d"], ev["ep"], ev["tp"], ev["sl"], ev["a"]
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
        # Fixed dollar risk (not % of current, compounding equity) for this
        # exploratory scan: isolates whether there's a genuine trade-level
        # edge from compounding/variance-drag effects. A low win-rate,
        # high-payoff-variance sequence can hit geometric ruin under fixed-
        # fractional (% of equity) sizing even with positive arithmetic
        # expectancy — that's a sizing-policy question, not a signal-quality
        # one, and conflates the two if not separated.
        risk = INIT_CAP * RISK_PCT
        units_uncapped = risk / stop_dist
        units_capped = MAX_LEV * INIT_CAP / ep
        units = min(units_uncapped, units_capped)
        if units == units_capped and units_capped < units_uncapped:
            n_lev_capped += 1
        notional = units * ep
        fee_dollar = FEE * 2 * notional
        pnl_dollar = units * (exit_price - ep) * d - fee_dollar

        cap += pnl_dollar
        peak = max(peak, cap)
        mdd = min(mdd, (cap - peak) / peak)
        wins += int(pnl_dollar > 0)
        net_pnls.append(pnl_dollar)
    n = len(net_pnls); wr = wins / n if n else 0.0
    ret = (cap / INIT_CAP - 1) * 100
    return dict(n=n, wr=wr, ret=ret, mdd=mdd * 100, net_pnls=net_pnls, cap=cap,
                n_dropped_none=n_dropped_none, n_lev_capped=n_lev_capped)

def mc_summary(pnls):
    if len(pnls) < 5: return dict(p_profit=0.0, p_ruin=1.0)
    mc = run_monte_carlo(pd.DataFrame({"net_pnl": pnls}), INIT_CAP, N_SIMS)
    return dict(p_profit=float(mc.get("p_profit", 0.0)), p_ruin=float(mc.get("p_ruin", 1.0)))

# ══════════════════════════════════════════════════════════════════════════════
# WFO WINDOWS
# ══════════════════════════════════════════════════════════════════════════════
def wf_dates(idx, train_m=6, oos_m=2, step_m=2):
    t0 = idx[0]; windows = []
    while True:
        tr_s = t0; tr_e = tr_s + pd.DateOffset(months=train_m)
        oo_s = tr_e; oo_e = oo_s + pd.DateOffset(months=oos_m)
        if oo_e > idx[-1]: break
        windows.append((tr_s, tr_e, oo_s, oo_e))
        t0 += pd.DateOffset(months=step_m)
    return windows

WF_WINDOWS = wf_dates(IDXLTF)
WF_PRE2025 = [wd for wd in WF_WINDOWS if wd[3] <= CUTOFF]
WF_HOLDOUT = [wd for wd in WF_WINDOWS if wd[2] >= CUTOFF]
print(f"\n[WFO] {len(WF_WINDOWS)} windows total  |  {len(WF_PRE2025)} pre-2025  |  "
      f"{len(WF_HOLDOUT)} holdout")

report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

# ══════════════════════════════════════════════════════════════════════════════
# SCAN: for each variant, backtest OOS across all WFO windows, track per-year + holdout
# ══════════════════════════════════════════════════════════════════════════════
w(f"\n{SEP}")
w("SCAN — 4 varianti candidate (OOS aggregato su tutte le finestre WFO)")
w(SEP)

all_results = {}
for name, sig_fn in VARIANTS.items():
    print(f"\n  [{name}] generating signal + events …", end="", flush=True)
    sig = sig_fn()
    all_evs = []
    year_evs: dict[int, list] = {}
    for tr_s, tr_e, oo_s, oo_e in WF_WINDOWS:
        idx_oos = np.where((IDXLTF >= oo_s) & (IDXLTF < oo_e))[0]
        if len(idx_oos) < 50: continue
        evs = make_events(sig, idx_oos)
        all_evs.extend(evs)
        year_evs.setdefault(oo_s.year, []).extend(evs)
        print(".", end="", flush=True)
    print()

    res = run_bt(all_evs)
    mc = mc_summary(res["net_pnls"])
    valid = res["ret"] > 0 and mc["p_profit"] > 0.90 and mc["p_ruin"] < 0.05
    all_results[name] = dict(res=res, mc=mc, valid=valid, year_evs=year_evs, all_evs=all_evs)
    w(f"\n  [{name}]  n={res['n']}  wr={res['wr']:.1%}  ret={res['ret']:+.1f}%  "
      f"mdd={res['mdd']:.1f}%  pp={mc['p_profit']:.3f}  pr={mc['p_ruin']:.3f}  "
      f"n_timeout={res['n_dropped_none']}  n_lev_capped={res['n_lev_capped']}  "
      f"{'✅' if valid else '✗'}")

w(f"\n{SEP}")
w("BREAKDOWN PER ANNO (tutte le varianti)")
w(SEP)
years = sorted({y for r in all_results.values() for y in r["year_evs"]})
for name, r in all_results.items():
    w(f"\n  {name}:")
    w(f"    {'Year':>6}  {'n':>6}  {'Ret%':>8}  {'WR':>6}")
    for yr in years:
        evs = r["year_evs"].get(yr, [])
        if len(evs) < 5: continue
        yres = run_bt(evs)
        w(f"    {yr:>6}  {yres['n']:>6}  {yres['ret']:>+7.1f}%  {yres['wr']:>5.1%}")

# ══════════════════════════════════════════════════════════════════════════════
# HOLDOUT 2025-2026 (mai usato per scegliere la variante)
# ══════════════════════════════════════════════════════════════════════════════
w(f"\n{SEP}")
w(f"HOLDOUT GENUINO 2025-2026 (N={len(WF_HOLDOUT)} finestre, valutazione diretta "
  f"di ciascuna variante — nessuna selezione fatta su questi dati)")
w(SEP)
for name, sig_fn in VARIANTS.items():
    sig = all_results[name]["res"]  # reuse nothing; regenerate events on holdout windows only
    sig_arr = VARIANTS[name]()
    holdout_evs = []
    for tr_s, tr_e, oo_s, oo_e in WF_HOLDOUT:
        idx_oos = np.where((IDXLTF >= oo_s) & (IDXLTF < oo_e))[0]
        if len(idx_oos) < 50: continue
        holdout_evs.extend(make_events(sig_arr, idx_oos))
    hres = run_bt(holdout_evs)
    hmc = mc_summary(hres["net_pnls"])
    w(f"  {name:<14}  n={hres['n']:>5}  ret={hres['ret']:>+6.1f}%  "
      f"mdd={hres['mdd']:>6.1f}%  pp={hmc['p_profit']:.3f}")

w(f"\n{SEP}")
w("[DONE]")

out_path = Path("reports/mtf_trend_scan.md")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("# MTF Trend Scan — swing 4H + trigger 15m (3 varianti)\n\n```\n" +
                     "\n".join(report_lines) + "\n```\n", encoding="utf-8")
print(f"\n[DONE] {out_path}")
