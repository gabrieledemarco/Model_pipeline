#!/usr/bin/env python3
"""
create_forward_montecarlo_14d_report.py
==========================================
Probabilità di rendimento positivo a 14 giorni di calendario (2 settimane)
da oggi, per ciascuna delle 3 strategie validate in questa sessione
(ICT Fade Standalone, Carver Breakout pool, TSMOM-sign pool) + un
portafoglio combinato equal-weight (1/3 ciascuna, per la diversificazione
di stile già osservata: mean-reversion contrarian / trend-following /
momentum direzionale).

Metodo A (raccomandato, il più prudente — stessa metodologia già usata in
create_ict_fade_forward_montecarlo.py): finestre storiche di calendario
REALIZZATE, ogni possibile punto di partenza nella storia 2020-2026 ->
rendimento realmente realizzato nei 14 giorni successivi. Cattura
fedelmente come si sono distribuiti nel tempo sia la frequenza di
trade/rendimento sia la loro redditività, senza assumere indipendenza.

Ricostruzione delle serie di P&L giornaliero (identica logica agli script
di validazione originali, stessi parametri, nessuna ri-ottimizzazione):
  - ICT Fade Standalone: trade-based (SWING_LOOKBACK=10, STOP_BUFFER_ATR=
    0.1, RR=3.0, MAX_HOLD_BARS=96) -> P&L raggruppato per data di uscita
  - Breakout pool (Carver): mark-to-market giornaliero continuo
  - TSMOM-sign pool: mark-to-market giornaliero continuo
Portafoglio combinato: somma dei 3 P&L giornalieri con allocazione 1/3
del capitale a ciascuna strategia (capitale iniziale unico, diviso in 3).
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

from src.strategy.data_fetcher import fetch_extended_data
from src.strategy.indicators import add_indicators
from src.strategy.mtf_swing import find_pivots

SEP = "═" * 78
START_YEAR = 2020
INIT_CAP = 100_000.0
HORIZON_DAYS = 14

report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

w(SEP)
w("Probabilità di rendimento positivo a 14 giorni — 3 strategie validate + portafoglio")
w(SEP)

t0 = time.time()
print("\n[DATA] Loading 4H, 1D, 15M …")
raw = fetch_extended_data(start_year=START_YEAR, start_month=1,
                           fetch_15m=True, fetch_1m=False, fetch_flow=False)
df4h = add_indicators(raw["4H"])
df15 = add_indicators(raw["15M"])
df1d_raw = raw["1D"]
print(f"  4H: {len(df4h):,}  15M: {len(df15):,}  1D: {len(df1d_raw):,}  (loaded in {time.time()-t0:.0f}s)")

# ═══════════════════════════════════════════════════════════════════════
# 1) ICT FADE STANDALONE — ricostruzione trade -> P&L giornaliero
# ═══════════════════════════════════════════════════════════════════════
print("\n[1/3] Rebuilding ICT Fade Standalone trades …")
RISK_PCT_ICT = 0.01
FEE_TAKER = 0.00055
SLIPPAGE_BASE = 0.00015
MAX_LEV = 10.0
FVG_MIN_ATR_FRAC = 0.05
OB_LOCAL_WIN = 3
OB_MAX_CONFIRM_BARS = 20
SWEEP_MAX_REJECT_BARS = 3
BREAKER_MAX_WAIT_BARS = 60
ACC_BARS = 8
ACC_RANGE_ATR_MULT = 1.5
MANIP_MAX_BARS = 10
ZONE_MAX_AGE_BARS_4H = 60
SWING_LOOKBACK = 10
STOP_BUFFER_ATR = 0.1
MAX_HOLD_BARS = 96
RR_ICT = 3.0

IDX4H = df4h.index; N4H = len(df4h)
IDX15 = df15.index; N15 = len(df15)
CL4, HI4, LO4, OP4 = (df4h[c].values.astype(float) for c in ("close", "high", "low", "open"))
ATR4 = np.where(df4h["atr_14"].values > 0, df4h["atr_14"].values, np.nan)
CL15, HI15, LO15, OP15 = (df15[c].values.astype(float) for c in ("close", "high", "low", "open"))
ATR15 = np.where(df15["atr_14"].values > 0, df15["atr_14"].values, np.nan)
IDX15_vals = IDX15.values

def bar15_at_or_after(ts):
    pos = np.searchsorted(IDX15_vals, np.datetime64(ts), side="left")
    return pos if pos < N15 else None

def detect_fvgs(HIx, LOx, ATRx):
    n = len(HIx); events = []; zones = []
    for i in range(1, n - 1):
        atr = ATRx[i]
        if atr <= 0 or np.isnan(atr): continue
        if HIx[i - 1] < LOx[i + 1]:
            fbot, ftop = HIx[i - 1], LOx[i + 1]
            if (ftop - fbot) < FVG_MIN_ATR_FRAC * atr: continue
            events.append(dict(idx=i + 1, dir=1)); zones.append(dict(idx=i + 1, dir=1, lo=fbot, hi=ftop))
        elif LOx[i - 1] > HIx[i + 1]:
            ftop, fbot = LOx[i - 1], HIx[i + 1]
            if (ftop - fbot) < FVG_MIN_ATR_FRAC * atr: continue
            events.append(dict(idx=i + 1, dir=-1)); zones.append(dict(idx=i + 1, dir=-1, lo=fbot, hi=ftop))
    return events, zones

def detect_order_blocks(OPx, CLx):
    n = len(CLx); events = []; zones = []
    for i in range(OB_LOCAL_WIN, n - OB_MAX_CONFIRM_BARS - 1):
        if CLx[i] < OPx[i]:
            if CLx[i] == CLx[i - OB_LOCAL_WIN + 1:i + 1].min():
                for j in range(i + 1, i + 1 + OB_MAX_CONFIRM_BARS):
                    if CLx[j] > OPx[i]:
                        events.append(dict(idx=j, dir=1))
                        zones.append(dict(idx=j, dir=1, lo=min(OPx[i], CLx[i]), hi=max(OPx[i], CLx[i])))
                        break
        elif CLx[i] > OPx[i]:
            if CLx[i] == CLx[i - OB_LOCAL_WIN + 1:i + 1].max():
                for j in range(i + 1, i + 1 + OB_MAX_CONFIRM_BARS):
                    if CLx[j] < OPx[i]:
                        events.append(dict(idx=j, dir=-1))
                        zones.append(dict(idx=j, dir=-1, lo=min(OPx[i], CLx[i]), hi=max(OPx[i], CLx[i])))
                        break
    return events, zones

def detect_sweeps(HIx, LOx, CLx, pivots):
    n = len(CLx); events = []
    confirmed_highs, confirmed_lows = [], []; p_idx = 0
    for i in range(n):
        while p_idx < len(pivots) and pivots[p_idx]["confirm_idx"] == i:
            piv = pivots[p_idx]
            (confirmed_highs if piv["kind"] == 1 else confirmed_lows).append(piv["price"])
            p_idx += 1
        if confirmed_highs and HIx[i] > confirmed_highs[-1]:
            level = confirmed_highs[-1]
            for k in range(0, min(SWEEP_MAX_REJECT_BARS, n - i)):
                if CLx[i + k] < level: events.append(dict(idx=i + k, dir=-1)); break
        if confirmed_lows and LOx[i] < confirmed_lows[-1]:
            level = confirmed_lows[-1]
            for k in range(0, min(SWEEP_MAX_REJECT_BARS, n - i)):
                if CLx[i + k] > level: events.append(dict(idx=i + k, dir=1)); break
    return events

def detect_breakers(zones_ob, HIx, LOx, CLx):
    n = len(CLx); events = []
    for z in zones_ob:
        j0 = z["idx"]; lo, hi = z["lo"], z["hi"]
        invalid_i = None
        for k in range(j0, min(j0 + BREAKER_MAX_WAIT_BARS, n)):
            if z["dir"] == 1 and CLx[k] < lo: invalid_i = k; break
            if z["dir"] == -1 and CLx[k] > hi: invalid_i = k; break
        if invalid_i is None: continue
        for k in range(invalid_i + 1, min(invalid_i + 1 + BREAKER_MAX_WAIT_BARS, n)):
            if z["dir"] == 1 and LOx[k] <= hi and HIx[k] >= lo and CLx[k] < lo:
                events.append(dict(idx=k, dir=-1)); break
            if z["dir"] == -1 and HIx[k] >= lo and LOx[k] <= hi and CLx[k] > hi:
                events.append(dict(idx=k, dir=1)); break
    return events

def detect_ifvgs(zones_fvg, CLx):
    n = len(CLx); events = []
    for z in zones_fvg:
        j0 = z["idx"]; lo, hi, dd = z["lo"], z["hi"], z["dir"]
        for k in range(j0, min(j0 + BREAKER_MAX_WAIT_BARS, n)):
            if dd == 1 and CLx[k] < lo: events.append(dict(idx=k, dir=-1)); break
            if dd == -1 and CLx[k] > hi: events.append(dict(idx=k, dir=1)); break
    return events

def detect_amd(HIx, LOx, CLx, ATRx):
    n = len(CLx); events = []
    for i in range(ACC_BARS, n - MANIP_MAX_BARS - 1):
        atr = ATRx[i]
        if atr <= 0 or np.isnan(atr): continue
        seg_hi = HIx[i - ACC_BARS:i].max(); seg_lo = LOx[i - ACC_BARS:i].min()
        if (seg_hi - seg_lo) > ACC_RANGE_ATR_MULT * atr: continue
        for k in range(i, min(i + MANIP_MAX_BARS, n)):
            if HIx[k] > seg_hi:
                for m in range(k, min(k + SWEEP_MAX_REJECT_BARS, n)):
                    if CLx[m] < seg_hi: events.append(dict(idx=m, dir=-1)); break
                break
            if LOx[k] < seg_lo:
                for m in range(k, min(k + SWEEP_MAX_REJECT_BARS, n)):
                    if CLx[m] > seg_lo: events.append(dict(idx=m, dir=1)); break
                break
    return events

_, ob_zones_4h = detect_order_blocks(OP4, CL4)
_, fvg_zones_4h = detect_fvgs(HI4, LO4, ATR4)
raw_zones_4h = ob_zones_4h + fvg_zones_4h
htf_zones_15 = []
for z in raw_zones_4h:
    j0 = z["idx"]
    if j0 >= N4H: continue
    invalid_i = None
    for k in range(j0, min(j0 + ZONE_MAX_AGE_BARS_4H, N4H)):
        if z["dir"] == 1 and CL4[k] < z["lo"]: invalid_i = k; break
        if z["dir"] == -1 and CL4[k] > z["hi"]: invalid_i = k; break
    end_4h = invalid_i if invalid_i is not None else min(N4H - 1, j0 + ZONE_MAX_AGE_BARS_4H)
    start15 = bar15_at_or_after(IDX4H[j0])
    end15 = bar15_at_or_after(IDX4H[end_4h])
    if start15 is None: continue
    if end15 is None: end15 = N15
    if end15 <= start15: continue
    htf_zones_15.append(dict(dir=z["dir"], start=start15, end=end15))

htf_by_dir = {1: sorted([(z["start"], z["end"]) for z in htf_zones_15 if z["dir"] == 1]),
              -1: sorted([(z["start"], z["end"]) for z in htf_zones_15 if z["dir"] == -1])}
starts_1 = np.array([s for s, e in htf_by_dir[1]]); ends_1 = np.array([e for s, e in htf_by_dir[1]])
starts_m1 = np.array([s for s, e in htf_by_dir[-1]]); ends_m1 = np.array([e for s, e in htf_by_dir[-1]])

def in_active_zone(idx, dd):
    starts, ends = (starts_1, ends_1) if dd == 1 else (starts_m1, ends_m1)
    if len(starts) == 0: return False
    pos = np.searchsorted(starts, idx, side="right") - 1
    while pos >= 0:
        if starts[pos] <= idx <= ends[pos]: return True
        if idx - starts[pos] > 4 * ZONE_MAX_AGE_BARS_4H: break
        pos -= 1
    return False

pivots15 = find_pivots(HI15, LO15, 1, 1)
sweeps = detect_sweeps(HI15, LO15, CL15, pivots15)
fvgs, _ = detect_fvgs(HI15, LO15, ATR15)
obs, ob_zones15 = detect_order_blocks(OP15, CL15)
_, fvg_zones15 = detect_fvgs(HI15, LO15, ATR15)
breakers = detect_breakers(ob_zones15, HI15, LO15, CL15)
ifvgs = detect_ifvgs(fvg_zones15, CL15)
amd = detect_amd(HI15, LO15, CL15, ATR15)
ALL_TRIGGERS = sweeps + fvgs + obs + breakers + ifvgs + amd
standalone = [e for e in ALL_TRIGGERS if not in_active_zone(e["idx"], e["dir"])]
standalone.sort(key=lambda e: e["idx"])

def build_fade_events():
    evs = []; last_exit = -1
    for e in standalone:
        i = e["idx"]
        if i <= last_exit: continue
        if i < SWING_LOOKBACK + 2 or i >= N15 - 1: continue
        if np.isnan(ATR15[i]) or ATR15[i] <= 0: continue
        bias = -e["dir"]
        if bias == 1:
            swing = LO15[i - SWING_LOOKBACK:i + 1].min(); sl = swing - STOP_BUFFER_ATR * ATR15[i]
        else:
            swing = HI15[i - SWING_LOOKBACK:i + 1].max(); sl = swing + STOP_BUFFER_ATR * ATR15[i]
        entry_i = i + 1
        ep = OP15[entry_i]
        risk = abs(ep - sl)
        if risk <= 0: continue
        hold = min(MAX_HOLD_BARS, N15 - 1 - entry_i)
        if hold < 1: continue
        evs.append(dict(entry_i=entry_i, d=bias, ep=ep, sl=sl, risk=risk, hold=hold))
        last_exit = entry_i + hold
    return evs

ict_events = build_fade_events()
ict_trade_pnls = []
ict_trade_exit_ts = []
cap = INIT_CAP
for ev in ict_events:
    dd, ep, sl, risk, entry_i, hold = ev["d"], ev["ep"], ev["sl"], ev["risk"], ev["entry_i"], ev["hold"]
    target = ep + dd * RR_ICT * risk
    out = "time"; exit_price = None; exit_j = None
    for k in range(hold):
        j = entry_i + k
        if j >= N15: break
        hk, lk = HI15[j], LO15[j]
        hit_sl = (lk <= sl) if dd == 1 else (hk >= sl)
        hit_tp = (hk >= target) if dd == 1 else (lk <= target)
        if hit_sl: exit_price = sl; exit_j = j; break
        if hit_tp: exit_price = target; exit_j = j; break
    if exit_price is None:
        exit_j = min(entry_i + hold, N15 - 1); exit_price = CL15[exit_j]
    r = INIT_CAP * RISK_PCT_ICT
    units = min(r / risk, MAX_LEV * INIT_CAP / ep)
    slip_pct = SLIPPAGE_BASE
    fill_ep = ep * (1 + dd * slip_pct); fill_xp = exit_price * (1 - dd * slip_pct)
    pnl = units * (fill_xp - fill_ep) * dd - FEE_TAKER * units * fill_ep - FEE_TAKER * units * fill_xp
    ict_trade_pnls.append(pnl)
    ict_trade_exit_ts.append(IDX15[exit_j])

ict_daily = pd.Series(ict_trade_pnls, index=pd.DatetimeIndex(ict_trade_exit_ts).normalize()).groupby(level=0).sum()
print(f"  {len(ict_events):,} trade ICT Fade ricostruiti  ({len(ict_daily)} giorni con attività)")

# ═══════════════════════════════════════════════════════════════════════
# 2) CARVER BREAKOUT POOL — P&L giornaliero (mark-to-market)
# ═══════════════════════════════════════════════════════════════════════
print("[2/3] Rebuilding Breakout pool daily P&L …")
TARGET_ANNUAL_VOL = 0.20
FRICTION = FEE_TAKER + SLIPPAGE_BASE
VOL_SPAN = 25
FORECAST_CAP = 20.0
FORECAST_TARGET_ABS = 10.0
BREAKOUT_NS = [10, 20, 40, 80, 160, 320]

IDX1D = df1d_raw.index; N1D = len(df1d_raw)
CL1D = df1d_raw["close"].values.astype(float)
price1d = pd.Series(CL1D, index=IDX1D)
daily_ret1d = price1d.diff()
price_vol1d = daily_ret1d.ewm(span=VOL_SPAN, min_periods=VOL_SPAN).std()
annualized_price_vol1d = (price_vol1d * np.sqrt(365)).values

def calibrate_forecast(raw_arr):
    raw_s = pd.Series(raw_arr)
    abs_expanding_mean = raw_s.abs().expanding(min_periods=60).mean()
    scalar = FORECAST_TARGET_ABS / abs_expanding_mean.replace(0, np.nan)
    return (raw_s * scalar).clip(-FORECAST_CAP, FORECAST_CAP).values

breakout_forecasts = {}
for Nb in BREAKOUT_NS:
    roll_max = price1d.rolling(Nb, min_periods=Nb).max()
    roll_min = price1d.rolling(Nb, min_periods=Nb).min()
    mid = (roll_max + roll_min) / 2.0
    half_range = (roll_max - roll_min) / 2.0
    raw_b = np.where(half_range.values > 0, 40.0 * (price1d.values - mid.values) / half_range.values, np.nan)
    raw_b_s = pd.Series(raw_b).ewm(span=max(Nb // 4, 2), min_periods=max(Nb // 4, 2)).mean()
    breakout_forecasts[Nb] = raw_b_s.clip(-FORECAST_CAP, FORECAST_CAP).values
breakout_pool_fcst = np.nanmean(np.column_stack(list(breakout_forecasts.values())), axis=1)

def run_bt_daily(forecast, CLx, ann_vol, idx):
    n = len(forecast)
    valid = np.isfinite(forecast) & (ann_vol > 0)
    raw_units = np.where(valid, (forecast / FORECAST_TARGET_ABS) * INIT_CAP * TARGET_ANNUAL_VOL /
                          np.where(ann_vol > 0, ann_vol, np.nan), 0.0)
    max_units = MAX_LEV * INIT_CAP / CLx
    units = np.clip(raw_units, -max_units, max_units)
    units = np.where(valid, units, 0.0)
    net_daily = np.zeros(n)
    prev_units = 0.0
    for t in range(1, n):
        cost = FRICTION * abs(units[t] - prev_units) * CLx[t]
        pnl = prev_units * (CLx[t] - CLx[t - 1])
        net_daily[t] = pnl - cost
        prev_units = units[t]
    return pd.Series(net_daily, index=idx)

breakout_daily = run_bt_daily(breakout_pool_fcst, CL1D, annualized_price_vol1d, IDX1D)
print(f"  Breakout pool ricostruito  ({len(breakout_daily)} giorni)")

# ═══════════════════════════════════════════════════════════════════════
# 3) TSMOM-SIGN POOL — P&L giornaliero (mark-to-market)
# ═══════════════════════════════════════════════════════════════════════
print("[3/3] Rebuilding TSMOM-sign pool daily P&L …")
TSMOM_LOOKBACKS = [30, 60, 90, 120, 252]
sign_forecasts = {}
for L in TSMOM_LOOKBACKS:
    past_price = price1d.shift(L)
    raw_sign = np.sign(price1d.values - past_price.values)
    raw_sign = np.where(np.isfinite(past_price.values), raw_sign, np.nan)
    sign_forecasts[L] = calibrate_forecast(raw_sign)
tsmom_sign_pool_fcst = np.nanmean(np.column_stack(list(sign_forecasts.values())), axis=1)
tsmom_daily = run_bt_daily(tsmom_sign_pool_fcst, CL1D, annualized_price_vol1d, IDX1D)
print(f"  TSMOM-sign pool ricostruito  ({len(tsmom_daily)} giorni)")

# ═══════════════════════════════════════════════════════════════════════
# Serie giornaliere allineate + portafoglio combinato (1/3 ciascuna)
# ═══════════════════════════════════════════════════════════════════════
full_idx = pd.date_range(IDX1D[0].normalize(), IDX1D[-1].normalize(), freq="D")
ict_s = ict_daily.reindex(full_idx, fill_value=0.0)
brk_s = breakout_daily.reindex(full_idx, fill_value=0.0)
tsm_s = tsmom_daily.reindex(full_idx, fill_value=0.0)
combo_s = ict_s / 3.0 + brk_s / 3.0 + tsm_s / 3.0   # 1/3 del capitale ciascuna, stesso INIT_CAP totale

STRATS = {
    "ICT Fade Standalone": ict_s,
    "Breakout pool": brk_s,
    "TSMOM-sign pool": tsm_s,
    "Portafoglio combinato (1/3 ciascuna)": combo_s,
}


def forward_stats(daily_pnl: pd.Series, horizon: int, cap_frac: float = 1.0):
    cap_base = INIT_CAP * cap_frac
    cum = daily_pnl.cumsum()
    n = len(cum)
    rets = []
    for i in range(n - horizon):
        window_pnl = cum.iloc[i + horizon] - (cum.iloc[i - 1] if i > 0 else 0.0)
        rets.append(window_pnl / cap_base * 100)
    rets = np.array(rets)
    return dict(n=len(rets), mean=rets.mean(), median=np.median(rets), p_pos=(rets > 0).mean() * 100,
                p05=np.percentile(rets, 5), p10=np.percentile(rets, 10), p25=np.percentile(rets, 25),
                p75=np.percentile(rets, 75), p90=np.percentile(rets, 90), p95=np.percentile(rets, 95))


w(f"\nMetodo A — finestre storiche di calendario REALIZZATE, orizzonte {HORIZON_DAYS} giorni")
w(f"(ogni possibile punto di partenza 2020-2026 -> rendimento nei {HORIZON_DAYS} giorni successivi)")
w(SEP)
for name, series in STRATS.items():
    s = forward_stats(series, HORIZON_DAYS)
    w(f"\n  {name}")
    w(f"    n finestre storiche = {s['n']:,}")
    w(f"    Media: {s['mean']:+.2f}%   Mediana: {s['median']:+.2f}%   "
      f"P(rendimento>0): {s['p_pos']:.1f}%")
    w(f"    Percentili:  5%={s['p05']:+.2f}%  10%={s['p10']:+.2f}%  25%={s['p25']:+.2f}%  "
      f"75%={s['p75']:+.2f}%  90%={s['p90']:+.2f}%  95%={s['p95']:+.2f}%")

w(f"\n{SEP}")
w("SINTESI — P(rendimento positivo dopo 14 giorni)")
w(SEP)
w(f"\n  {'Strategia':<40}{'P(>0)':>10}{'Mediana':>12}{'Media':>10}")
for name, series in STRATS.items():
    s = forward_stats(series, HORIZON_DAYS)
    w(f"  {name:<40}{s['p_pos']:>9.1f}%{s['median']:>+11.2f}%{s['mean']:>+9.2f}%")

w(f"\n{SEP}\n[DONE]\n{SEP}")
out_path = Path("reports/forward_montecarlo_14d.md")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("# Probabilità di rendimento positivo a 14 giorni — 3 strategie + portafoglio\n\n```\n" +
                     "\n".join(report_lines) + "\n```\n", encoding="utf-8")
print(f"\n[DONE] {out_path}   (total runtime {time.time()-t0:.0f}s)")
