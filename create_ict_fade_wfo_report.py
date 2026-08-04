#!/usr/bin/env python3
"""
create_ict_fade_wfo_report.py
================================
Walk-forward optimization di STOP (costruzione strutturale, parametro
SWING_LOOKBACK) e TARGET (RR) per la strategia validata "Fade dei trigger
ICT standalone" (`ict_fade_standalone.md`). A differenza del report base
(RR fisso scelto su tutta la storia con correzione DSR), qui la selezione
è CAUSALE per finestra: 6 mesi IS / 2 mesi OOS / step 2 mesi (stessa
cadenza usata per VWAP MR e il regime-filter HMM in questa sessione),
scelta per Sharpe IS su una griglia SWING_LOOKBACK × RR, applicata
SOLO su barre OOS mai viste durante quella selezione — zero look-ahead
per costruzione, nessuna correzione DSR necessaria (la selezione stessa
è già causale/out-of-sample per ogni finestra).

Griglia: SWING_LOOKBACK ∈ {5, 10, 15} barre (costruzione dello stop
strutturale) × RR ∈ {1.5, 2.0, 2.5, 3.0} (target) = 12 combinazioni/finestra.

Il segnale di ingresso (6 oggetti ICT combinati, fade dei trigger
standalone) resta IDENTICO al report validato — qui si ottimizza solo
l'uscita (stop/target), non il segnale.
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
from src.strategy.monte_carlo import run_monte_carlo, run_monte_carlo_block, trade_level_sharpe

INIT_CAP = 100_000.0
RISK_PCT = 0.01
FEE_TAKER = 0.00055
SLIPPAGE_BASE = 0.00015
MAX_LEV = 10.0
START_YEAR = 2020
N_SIMS = 5_000

FVG_MIN_ATR_FRAC = 0.05
OB_LOCAL_WIN = 3
OB_MAX_CONFIRM_BARS = 20
SWEEP_MAX_REJECT_BARS = 3
BREAKER_MAX_WAIT_BARS = 60
ACC_BARS = 8
ACC_RANGE_ATR_MULT = 1.5
MANIP_MAX_BARS = 10
ZONE_MAX_AGE_BARS_4H = 60
STOP_BUFFER_ATR = 0.1
MAX_HOLD_BARS = 96

SWING_LOOKBACK_GRID = [5, 10, 15]
RR_GRID = [1.5, 2.0, 2.5, 3.0]
WF_TRAIN_M, WF_OOS_M, WF_STEP_M = 6, 2, 2
CUTOFF = pd.Timestamp("2025-01-01")

report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

SEP = "═" * 78
w(SEP)
w("Fade ICT standalone — Walk-Forward Optimization stop (SWING_LOOKBACK) × target (RR)")
w(SEP)

t0 = time.time()
print("\n[DATA] Loading 4H + 15M …")
raw = fetch_extended_data(start_year=START_YEAR, start_month=1,
                           fetch_15m=True, fetch_1m=False, fetch_flow=False)
df4h = add_indicators(raw["4H"])
df15 = add_indicators(raw["15M"])
IDX4H = df4h.index; N4H = len(df4h)
IDX = df15.index; N = len(df15)
CL4, HI4, LO4, OP4 = (df4h[c].values.astype(float) for c in ("close", "high", "low", "open"))
ATR4 = np.where(df4h["atr_14"].values > 0, df4h["atr_14"].values, np.nan)
CL, HI, LO, OP = (df15[c].values.astype(float) for c in ("close", "high", "low", "open"))
ATR = np.where(df15["atr_14"].values > 0, df15["atr_14"].values, np.nan)
print(f"  4H: {N4H:,}  15M: {N:,}  (loaded in {time.time()-t0:.0f}s)")

IDX15_vals = IDX.values
def bar15_at_or_after(ts):
    pos = np.searchsorted(IDX15_vals, np.datetime64(ts), side="left")
    return pos if pos < N else None


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
        j0 = z["idx"]; lo, hi, d = z["lo"], z["hi"], z["dir"]
        for k in range(j0, min(j0 + BREAKER_MAX_WAIT_BARS, n)):
            if d == 1 and CLx[k] < lo: events.append(dict(idx=k, dir=-1)); break
            if d == -1 and CLx[k] > hi: events.append(dict(idx=k, dir=1)); break
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


print("[HTF] Detecting 4H demand/supply zones …")
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
    if end15 is None: end15 = N
    if end15 <= start15: continue
    htf_zones_15.append(dict(dir=z["dir"], start=start15, end=end15))
print(f"  {len(htf_zones_15)} zone 4H mappate su 15M")

htf_by_dir = {1: sorted([(z["start"], z["end"]) for z in htf_zones_15 if z["dir"] == 1]),
              -1: sorted([(z["start"], z["end"]) for z in htf_zones_15 if z["dir"] == -1])}
starts_1 = np.array([s for s, e in htf_by_dir[1]]); ends_1 = np.array([e for s, e in htf_by_dir[1]])
starts_m1 = np.array([s for s, e in htf_by_dir[-1]]); ends_m1 = np.array([e for s, e in htf_by_dir[-1]])


def in_active_zone(idx, d):
    starts, ends = (starts_1, ends_1) if d == 1 else (starts_m1, ends_m1)
    if len(starts) == 0: return False
    pos = np.searchsorted(starts, idx, side="right") - 1
    while pos >= 0:
        if starts[pos] <= idx <= ends[pos]: return True
        if idx - starts[pos] > 4 * ZONE_MAX_AGE_BARS_4H: break
        pos -= 1
    return False


print("[LTF] Detecting 15M triggers …")
pivots15 = find_pivots(HI, LO, 1, 1)
sweeps = detect_sweeps(HI, LO, CL, pivots15)
fvgs, _ = detect_fvgs(HI, LO, ATR)
obs, ob_zones15 = detect_order_blocks(OP, CL)
_, fvg_zones15 = detect_fvgs(HI, LO, ATR)
breakers = detect_breakers(ob_zones15, HI, LO, CL)
ifvgs = detect_ifvgs(fvg_zones15, CL)
amd = detect_amd(HI, LO, CL, ATR)

ALL_TRIGGERS = sweeps + fvgs + obs + breakers + ifvgs + amd
standalone = [e for e in ALL_TRIGGERS if not in_active_zone(e["idx"], e["dir"])]
standalone.sort(key=lambda e: e["idx"])
print(f"  {len(ALL_TRIGGERS)} trigger totali -> {len(standalone)} standalone")


def build_fade_events(swing_lookback):
    evs = []
    last_exit = -1
    for e in standalone:
        i = e["idx"]
        if i <= last_exit: continue
        if i < swing_lookback + 2 or i >= N - 1: continue
        if np.isnan(ATR[i]) or ATR[i] <= 0: continue
        bias = -e["dir"]
        if bias == 1:
            swing = LO[i - swing_lookback:i + 1].min()
            sl = swing - STOP_BUFFER_ATR * ATR[i]
        else:
            swing = HI[i - swing_lookback:i + 1].max()
            sl = swing + STOP_BUFFER_ATR * ATR[i]
        entry_i = i + 1
        ep = OP[entry_i]
        risk = abs(ep - sl)
        if risk <= 0: continue
        hold = min(MAX_HOLD_BARS, N - 1 - entry_i)
        if hold < 1: continue
        evs.append(dict(entry_i=entry_i, d=bias, ep=ep, sl=sl, risk=risk, hold=hold))
        last_exit = entry_i + hold
    return evs


print("[EVENTS] Building event sets per SWING_LOOKBACK …")
EVENTS_BY_LOOKBACK = {sl: build_fade_events(sl) for sl in SWING_LOOKBACK_GRID}
for sl, evs in EVENTS_BY_LOOKBACK.items():
    print(f"  SWING_LOOKBACK={sl}: {len(evs)} trade")


def run_bt(evs, rr, slippage_extra=0.0):
    slip_pct = SLIPPAGE_BASE + slippage_extra
    if not evs:
        return dict(n=0, wr=0.0, ret=0.0, mdd=0.0, net_pnls=[], exits=[])
    cap = INIT_CAP; peak = cap; mdd = 0.0; wins = 0; net_pnls = []; exits = []
    for ev in evs:
        d, ep, sl, risk, entry_i, hold = ev["d"], ev["ep"], ev["sl"], ev["risk"], ev["entry_i"], ev["hold"]
        target = ep + d * rr * risk
        out = "time"; exit_price = None; exit_j = None
        for k in range(hold):
            j = entry_i + k
            if j >= N: break
            hk, lk = HI[j], LO[j]
            if d == 1:
                hit_sl = lk <= sl; hit_tp = hk >= target
            else:
                hit_sl = hk >= sl; hit_tp = lk <= target
            if hit_sl: out = "sl"; exit_price = sl; exit_j = j; break
            if hit_tp: out = "tp"; exit_price = target; exit_j = j; break
        if exit_price is None:
            exit_j = min(entry_i + hold, N - 1)
            exit_price = CL[exit_j]
        if risk <= 0: continue
        r = INIT_CAP * RISK_PCT
        units = min(r / risk, MAX_LEV * INIT_CAP / ep)
        fill_ep = ep * (1 + d * slip_pct)
        fill_xp = exit_price * (1 - d * slip_pct)
        pnl = units * (fill_xp - fill_ep) * d - FEE_TAKER * units * fill_ep - FEE_TAKER * units * fill_xp
        cap += pnl
        peak = max(peak, cap)
        mdd = min(mdd, (cap - peak) / peak)
        wins += int(pnl > 0)
        net_pnls.append(pnl)
        exits.append(dict(ts=IDX[exit_j], out=out))
    n = len(net_pnls); wr = wins / n if n else 0.0
    return dict(n=n, wr=wr, ret=(cap / INIT_CAP - 1) * 100, mdd=mdd * 100, net_pnls=net_pnls, exits=exits)


def wf_dates(idx):
    t0_ = idx[0]; windows = []
    while True:
        tr_s = t0_; tr_e = tr_s + pd.DateOffset(months=WF_TRAIN_M)
        oo_s = tr_e; oo_e = oo_s + pd.DateOffset(months=WF_OOS_M)
        if oo_e > idx[-1]: break
        windows.append((tr_s, tr_e, oo_s, oo_e))
        t0_ += pd.DateOffset(months=WF_STEP_M)
    return windows

WF_WINDOWS = wf_dates(IDX)
w(f"\n[WFO] {len(WF_WINDOWS)} finestre ({WF_TRAIN_M}m IS / {WF_OOS_M}m OOS / step {WF_STEP_M}m)")
w(f"[WFO] Griglia: SWING_LOOKBACK {SWING_LOOKBACK_GRID} × RR {RR_GRID} = "
  f"{len(SWING_LOOKBACK_GRID)*len(RR_GRID)} combinazioni/finestra")

selection_log = []
all_oos_trades = []   # (exit_ts, pnl, out, entry_ts_year)

for tr_s, tr_e, oo_s, oo_e in WF_WINDOWS:
    best = None
    for sl_look in SWING_LOOKBACK_GRID:
        evs_all = EVENTS_BY_LOOKBACK[sl_look]
        is_evs = [e for e in evs_all if tr_s <= IDX[e["entry_i"]] < tr_e]
        if len(is_evs) < 15:
            continue
        for rr in RR_GRID:
            is_res = run_bt(is_evs, rr)
            if is_res["n"] < 15:
                continue
            sharpe = trade_level_sharpe(is_res["net_pnls"])
            if best is None or sharpe > best["sharpe"]:
                best = dict(sl_look=sl_look, rr=rr, sharpe=sharpe, n_is=is_res["n"])
    if best is None:
        continue
    selection_log.append(dict(window_start=tr_s, **best))

    oos_evs = [e for e in EVENTS_BY_LOOKBACK[best["sl_look"]] if oo_s <= IDX[e["entry_i"]] < oo_e]
    oos_res = run_bt(oos_evs, best["rr"])
    for pnl, ex in zip(oos_res["net_pnls"], oos_res["exits"]):
        all_oos_trades.append(dict(exit_ts=ex["ts"], pnl=pnl, out=ex["out"],
                                    sl_look=best["sl_look"], rr=best["rr"]))
    print(".", end="", flush=True)

print()
sel_df = pd.DataFrame(selection_log)
w(f"\n[WFO] Selezione per finestra ({len(sel_df)} finestre valide):")
w(f"  SWING_LOOKBACK più scelto: {sel_df['sl_look'].value_counts().to_dict() if not sel_df.empty else {}}")
w(f"  RR più scelto             : {sel_df['rr'].value_counts().to_dict() if not sel_df.empty else {}}")

# ── Aggregate OOS results ──────────────────────────────────────────────
all_oos_trades.sort(key=lambda t: t["exit_ts"])
net_pnls = [t["pnl"] for t in all_oos_trades]
n = len(net_pnls)
cap_curve = INIT_CAP + np.cumsum(net_pnls)
wins = sum(1 for p in net_pnls if p > 0)
wr = wins / n if n else 0.0
ret_pct = (cap_curve[-1] / INIT_CAP - 1) * 100 if n else 0.0
peak = np.maximum.accumulate(np.concatenate([[INIT_CAP], cap_curve]))[1:]
mdd_pct = ((cap_curve - peak) / peak).min() * 100 if n else 0.0

w(f"\n{SEP}")
w("RISULTATI AGGREGATI — walk-forward OOS (stop+target ottimizzati per finestra)")
w(SEP)
w(f"\n  FULL WFO-OOS: n={n}  wr={wr:.1%}  ret={ret_pct:+.1f}%  mdd={mdd_pct:.1f}%")

def mc_summary(pnls):
    if len(pnls) < 5: return dict(p_profit=0.0, p_ruin=1.0)
    mc = run_monte_carlo(pd.DataFrame({"net_pnl": pnls}), INIT_CAP, N_SIMS)
    return dict(p_profit=float(mc.get("p_profit", 0.0)), p_ruin=float(mc.get("p_ruin", 1.0)))

def mc_block_summary(pnls, block_size=10):
    if len(pnls) < 5: return dict(p_profit=0.0, p_ruin=1.0)
    mc = run_monte_carlo_block(pd.DataFrame({"net_pnl": pnls}), INIT_CAP, N_SIMS, block_size=block_size)
    return dict(p_profit=float(mc.get("p_profit", 0.0)), p_ruin=float(mc.get("p_ruin", 1.0)))

mc = mc_summary(net_pnls)
mc_blk = mc_block_summary(net_pnls)
w(f"    MC i.i.d.  : pp={mc['p_profit']:.3f}  pr={mc['p_ruin']:.3f}")
w(f"    MC block   : pp={mc_blk['p_profit']:.3f}  pr={mc_blk['p_ruin']:.3f}")

w(f"\n  Breakdown per anno:")
w(f"    {'Year':>6}  {'n':>6}  {'Ret%':>8}  {'WR':>6}")
year_pnls: dict[int, list] = {}
for t in all_oos_trades:
    year_pnls.setdefault(t["exit_ts"].year, []).append(t["pnl"])
for yr in sorted(year_pnls):
    yp = year_pnls[yr]
    if len(yp) < 5: continue
    ycap = INIT_CAP + np.cumsum(yp)
    ywins = sum(1 for p in yp if p > 0)
    w(f"    {yr:>6}  {len(yp):>6}  {(ycap[-1]/INIT_CAP-1)*100:>+7.1f}%  {ywins/len(yp):>5.1%}")

holdout_trades = [t for t in all_oos_trades if t["exit_ts"] >= CUTOFF]
hpnls = [t["pnl"] for t in holdout_trades]
if hpnls:
    hcap = INIT_CAP + np.cumsum(hpnls)
    hwins = sum(1 for p in hpnls if p > 0)
    hmc = mc_summary(hpnls); hmc_blk = mc_block_summary(hpnls)
    w(f"\n  HOLDOUT GENUINO 2025-2026 (sub-slice del WFO-OOS): n={len(hpnls)}  wr={hwins/len(hpnls):.1%}  "
      f"ret={(hcap[-1]/INIT_CAP-1)*100:+.1f}%")
    w(f"    MC i.i.d.  : pp={hmc['p_profit']:.3f}  pr={hmc['p_ruin']:.3f}")
    w(f"    MC block   : pp={hmc_blk['p_profit']:.3f}  pr={hmc_blk['p_ruin']:.3f}")

# ── Confronto con la versione a RR fisso (ict_fade_standalone.md) ────────
w(f"\n{SEP}")
w("CONFRONTO — WFO (stop/target dinamici) vs RR=3.0 fisso (ict_fade_standalone.md)")
w(SEP)
w(f"\n  {'':>20}  {'n':>6}  {'Ret%':>8}  {'WR':>6}  {'MC pp':>7}")
w(f"  {'WFO dinamico':>20}  {n:>6}  {ret_pct:>+7.1f}%  {wr:>5.1%}  {mc['p_profit']:>6.3f}")
w(f"  {'RR=3.0 fisso (rif.)':>20}  {'1399':>6}  {'+512.7':>7}%  {'47.0%':>6}  {'1.000':>6}")

w(f"\n{SEP}\n[DONE]\n{SEP}")
out_path = Path("reports/ict_fade_wfo.md")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("# Fade ICT standalone — Walk-Forward Optimization stop × target\n\n```\n" +
                     "\n".join(report_lines) + "\n```\n", encoding="utf-8")

# ── Salva dati per il report HTML ──────────────────────────────────────
import pickle
wfo_data = dict(
    all_oos_trades=all_oos_trades, selection_log=selection_log, sel_df=sel_df,
    n=n, wr=wr, ret_pct=ret_pct, mdd_pct=mdd_pct, mc=mc, mc_blk=mc_blk,
    cap_curve=cap_curve.tolist(), net_pnls=net_pnls,
    year_pnls={k: v for k, v in year_pnls.items()},
    holdout_trades=holdout_trades, hpnls=hpnls,
    wf_windows=WF_WINDOWS, idx_start=IDX[0], idx_end=IDX[-1],
)
with open("reports/ict_fade_wfo_data.pkl", "wb") as f:
    pickle.dump(wfo_data, f)

print(f"\n[DONE] {out_path}   (total runtime {time.time()-t0:.0f}s)")
