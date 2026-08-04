#!/usr/bin/env python3
"""
create_ict_fade_forward_montecarlo.py
========================================
Rendimento atteso del capitale investito nella strategia validata (Fade
ICT standalone, RR=3.0) a 30/50/60 giorni da oggi — proiezione forward,
non le statistiche a posteriori già viste in `ict_fade_standalone.md`.

Due metodi incrociati (per non fidarsi di uno solo):

  A) FINESTRE STORICHE REALIZZATE: per ogni punto di partenza possibile
     nella storia 2020-2026, calcola il rendimento REALMENTE realizzato
     nei successivi N giorni di calendario (usando la curva di equity
     giornaliera della strategia già validata) — la distribuzione
     empirica di "cosa sarebbe successo investendo in un giorno a caso e
     tenendo N giorni". Preserva la vera struttura temporale (cluster di
     trade, periodi senza trade, autocorrelazione reale).

  B) MONTE CARLO BLOCK-BOOTSTRAP FORWARD: stima il tasso di trade/giorno
     dalla storia (full-sample), lo usa per determinare quanti trade
     aspettarsi in N giorni, poi ricampiona a blocchi (block bootstrap,
     block_size=10, stessa disciplina usata altrove in sessione) dalla
     sequenza storica di P&L per trade fino a raggiungere quel numero di
     trade, ripetuto N_SIMS volte — distribuzione sintetica del
     rendimento a N giorni.

Stesso sizing non-compounding (capitale fisso INIT_CAP per il rischio per
trade) usato in tutta la sessione — i risultati sono espressi in % di
INIT_CAP, coerenti con `ict_fade_standalone.md`.
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

INIT_CAP = 100_000.0
RISK_PCT = 0.01
FEE_TAKER = 0.00055
SLIPPAGE_BASE = 0.00015
MAX_LEV = 10.0
START_YEAR = 2020

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
RR = 3.0

HORIZONS_DAYS = [30, 50, 60]
BLOCK_SIZE = 10
N_SIMS = 20_000

report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

SEP = "═" * 78
w(SEP)
w("Fade ICT standalone (RR=3.0) — Rendimento atteso a 30/50/60 giorni (forward)")
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


def build_fade_events():
    evs = []
    last_exit = -1
    for e in standalone:
        i = e["idx"]
        if i <= last_exit: continue
        if i < SWING_LOOKBACK + 2 or i >= N - 1: continue
        if np.isnan(ATR[i]) or ATR[i] <= 0: continue
        bias = -e["dir"]
        if bias == 1:
            swing = LO[i - SWING_LOOKBACK:i + 1].min()
            sl = swing - STOP_BUFFER_ATR * ATR[i]
        else:
            swing = HI[i - SWING_LOOKBACK:i + 1].max()
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


ALL_EVENTS = build_fade_events()
print(f"[EVENTS] {len(ALL_EVENTS)} trade fade")


def run_bt_trades(evs, rr):
    """Ritorna lista di dict(exit_ts, pnl) in ordine cronologico."""
    trades = []
    for ev in evs:
        d, ep, sl, risk, entry_i, hold = ev["d"], ev["ep"], ev["sl"], ev["risk"], ev["entry_i"], ev["hold"]
        target = ep + d * rr * risk
        exit_price = None; exit_j = None
        for k in range(hold):
            j = entry_i + k
            if j >= N: break
            hk, lk = HI[j], LO[j]
            if d == 1:
                hit_sl = lk <= sl; hit_tp = hk >= target
            else:
                hit_sl = hk >= sl; hit_tp = lk <= target
            if hit_sl: exit_price = sl; exit_j = j; break
            if hit_tp: exit_price = target; exit_j = j; break
        if exit_price is None:
            exit_j = min(entry_i + hold, N - 1)
            exit_price = CL[exit_j]
        if risk <= 0: continue
        r = INIT_CAP * RISK_PCT
        units = min(r / risk, MAX_LEV * INIT_CAP / ep)
        fill_ep = ep * (1 + d * SLIPPAGE_BASE)
        fill_xp = exit_price * (1 - d * SLIPPAGE_BASE)
        pnl = units * (fill_xp - fill_ep) * d - FEE_TAKER * units * fill_ep - FEE_TAKER * units * fill_xp
        trades.append(dict(exit_ts=IDX[exit_j], pnl=pnl))
    return trades


trades = run_bt_trades(ALL_EVENTS, RR)
print(f"[TRADES] {len(trades)} trade eseguiti, ultimo exit: {trades[-1]['exit_ts']}")

trade_pnls = np.array([t["pnl"] for t in trades])
trade_ts = pd.DatetimeIndex([t["exit_ts"] for t in trades])
n_trades = len(trade_pnls)

total_days = (IDX[-1] - IDX[0]).days
rate_per_day = n_trades / total_days
w(f"\nTasso storico: {n_trades} trade in {total_days} giorni -> {rate_per_day:.3f} trade/giorno "
  f"(~1 ogni {1/rate_per_day:.1f} giorni)")

# ── Metodo A: finestre storiche realizzate ────────────────────────────────
daily_cum = pd.Series(trade_pnls, index=trade_ts.normalize()).groupby(level=0).sum()
full_daily_index = pd.date_range(IDX[0].normalize(), IDX[-1].normalize(), freq="D")
daily_cum = daily_cum.reindex(full_daily_index, fill_value=0.0)
equity_daily = INIT_CAP + daily_cum.cumsum()

w(f"\n{SEP}")
w("METODO A — Finestre storiche realizzate (rendimento reale a N giorni, ogni punto di partenza)")
w(SEP)
method_a_results = {}
for H in HORIZONS_DAYS:
    fwd_ret = (equity_daily.shift(-H) - equity_daily) / equity_daily
    fwd_ret = fwd_ret.dropna().values * 100
    if len(fwd_ret) < 10:
        continue
    pcts = np.percentile(fwd_ret, [5, 10, 25, 50, 75, 90, 95])
    prob_pos = (fwd_ret > 0).mean()
    method_a_results[H] = dict(mean=fwd_ret.mean(), pcts=pcts, prob_pos=prob_pos, n=len(fwd_ret))
    w(f"\n  Orizzonte {H} giorni  (n finestre storiche = {len(fwd_ret)}):")
    w(f"    Media: {fwd_ret.mean():+.2f}%   Mediana: {pcts[3]:+.2f}%   P(rendimento>0): {prob_pos:.1%}")
    w(f"    Percentili:  5%={pcts[0]:+.2f}%  10%={pcts[1]:+.2f}%  25%={pcts[2]:+.2f}%  "
      f"75%={pcts[4]:+.2f}%  90%={pcts[5]:+.2f}%  95%={pcts[6]:+.2f}%")

# ── Metodo B: Monte Carlo block-bootstrap forward ─────────────────────────
w(f"\n{SEP}")
w(f"METODO B — Monte Carlo block-bootstrap forward (block_size={BLOCK_SIZE}, {N_SIMS:,} simulazioni)")
w(SEP)

rng = np.random.default_rng(42)
method_b_results = {}
for H in HORIZONS_DAYS:
    n_target = max(1, round(rate_per_day * H))
    sim_totals = np.empty(N_SIMS)
    n_blocks_needed = int(np.ceil(n_target / BLOCK_SIZE))
    max_start = n_trades - BLOCK_SIZE
    for s in range(N_SIMS):
        pieces = []
        collected = 0
        while collected < n_target:
            start = rng.integers(0, max(1, max_start + 1))
            block = trade_pnls[start:start + BLOCK_SIZE]
            pieces.append(block)
            collected += len(block)
        seq = np.concatenate(pieces)[:n_target]
        sim_totals[s] = seq.sum()
    sim_ret_pct = sim_totals / INIT_CAP * 100
    pcts = np.percentile(sim_ret_pct, [5, 10, 25, 50, 75, 90, 95])
    prob_pos = (sim_ret_pct > 0).mean()
    prob_ruin = (sim_totals < -0.5 * INIT_CAP).mean()
    method_b_results[H] = dict(mean=sim_ret_pct.mean(), pcts=pcts, prob_pos=prob_pos,
                                prob_ruin=prob_ruin, n_trades_expected=n_target)
    w(f"\n  Orizzonte {H} giorni  (~{n_target} trade attesi):")
    w(f"    Media: {sim_ret_pct.mean():+.2f}%   Mediana: {pcts[3]:+.2f}%   "
      f"P(rendimento>0): {prob_pos:.1%}   P(perdita>50% capitale): {prob_ruin:.3f}")
    w(f"    Percentili:  5%={pcts[0]:+.2f}%  10%={pcts[1]:+.2f}%  25%={pcts[2]:+.2f}%  "
      f"75%={pcts[4]:+.2f}%  90%={pcts[5]:+.2f}%  95%={pcts[6]:+.2f}%")

# ── Confronto ───────────────────────────────────────────────────────────
w(f"\n{SEP}")
w("CONFRONTO — Metodo A (storico realizzato) vs Metodo B (Monte Carlo)")
w(SEP)
w(f"\n  {'Giorni':>7}  {'A: mediana':>11}  {'A: P(>0)':>9}  {'B: mediana':>11}  {'B: P(>0)':>9}  "
  f"{'B: 5-95%':>20}")
for H in HORIZONS_DAYS:
    a = method_a_results.get(H); b = method_b_results.get(H)
    if not a or not b: continue
    w(f"  {H:>6}g  {a['pcts'][3]:>+10.2f}%  {a['prob_pos']:>8.1%}  {b['pcts'][3]:>+10.2f}%  "
      f"{b['prob_pos']:>8.1%}  [{b['pcts'][0]:>+.1f}%, {b['pcts'][6]:>+.1f}%]")

w(f"\n{SEP}\n[DONE]\n{SEP}")
out_path = Path("reports/ict_fade_forward_montecarlo.md")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("# Fade ICT standalone (RR=3.0) — Rendimento atteso a 30/50/60 giorni\n\n```\n" +
                     "\n".join(report_lines) + "\n```\n", encoding="utf-8")
print(f"\n[DONE] {out_path}   (total runtime {time.time()-t0:.0f}s)")
