#!/usr/bin/env python3
"""
create_vp_vwap_confluence_liquidity_trailing_report.py
==========================================================
Valuta le DUE direzioni strutturali proposte dopo il fallimento del
regime-filter HMM (`vp_vwap_confluence_regime.md`), sulla stessa entry
logic validata (POC rolling + VWAP confluence + trend-context + rejection
+ stop strutturale, RR=3.0 base):

  A) LIQUIDITY: filtro di "volume distribution" — invece di un regime di
     prezzo (che ha peggiorato il risultato), richiede che il POC del
     giorno sia realmente "significativo": PROMINENCE = volume del bin
     POC / volume totale della finestra rolling. Un POC alto-prominence
     indica un vero livello di consenso (molto volume concentrato in una
     stretta fascia di prezzo); un POC basso-prominence indica un profilo
     piatto/disperso, dove "il livello con più volume" è solo rumore
     statistico. Soglia: percentile mediano rolling (non un numero magico
     scelto a posteriori) calcolato sui 60 giorni precedenti.

  B) TRAILING: dynamic risk management — sostituisce il target fisso
     RR=3.0 con uno stop ATR trailing (chandelier-style): una volta che il
     prezzo si muove a favore di ACTIVATION_ATR_MULT×ATR, lo stop segue a
     distanza TRAIL_ATR_MULT×ATR dal miglior prezzo raggiunto (mai
     retrocede). Lascia correre i vincitori oltre RR=3 invece di
     tagliarli lì, mantenendo lo stesso stop strutturale iniziale per i
     perdenti. Asimmetria entry/exit strutturale, non un nuovo parametro
     sul segnale di ingresso.

  C) COMBINED: A + B insieme.

Confronto diretto A/B/C vs BASELINE (RR=3.0 fisso, nessun filtro
liquidità) sullo STESSO pool di eventi causali full-history (non WFO-OOS
come nel report del regime filter, perché né A né B richiedono un modello
fittato — sono trasformazioni deterministiche di OHLCV/volume, quindi si
confrontano correttamente contro il pool più ampio già validato in
`vp_vwap_confluence.md`, n=160/45).

Frizioni Bybit obbligatorie (identiche al report del regime filter):
taker 0.055% + slippage 0.015%/lato = 0.14% round-trip, ordini a mercato,
nessun look-ahead (segnale su barra i chiusa, esecuzione all'apertura
di i+1).
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
from src.strategy.monte_carlo import run_monte_carlo, run_monte_carlo_block

SEP = "═" * 78
START_YEAR = 2020
INIT_CAP = 100_000.0
RISK_PCT = 0.01

FEE_TAKER = 0.00055
SLIPPAGE_BASE = 0.00015
MAX_LEV = 10.0
CUTOFF = pd.Timestamp("2025-01-01")
N_SIMS = 5_000

LOOKBACK = 5
MIN_SEP_ATR = 0.5
LOOKBACK_DAYS = 20
N_BINS = 50
CONF_ATR_MULT = 0.5
SWING_LOOKBACK = 10
STOP_BUFFER_ATR = 0.1
MAX_HOLD_BARS = 48
RR_FIXED = 3.0

PROM_ROLL_DAYS = 60          # finestra rolling per il percentile di prominence
PROM_PCTL_MIN = 0.50         # soglia: mediana rolling (non un numero scelto a posteriori)

ACTIVATION_ATR_MULT = 1.5    # profitto (in ATR) prima che il trailing si attivi
TRAIL_ATR_MULT = 2.0         # distanza di trailing (in ATR) dal miglior prezzo raggiunto

SLIPPAGE_STRESS_BPS = [0, 2, 5, 10]

report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

w(SEP)
w("VP + VWAP Confluence — Liquidity filter & ATR Trailing Stop — Validation Pipeline")
w(SEP)
w("\nDue direzioni strutturali dopo il fallimento del regime-filter HMM:")
w("A) LIQUIDITY: filtro di prominence del POC (volume distribution, non prezzo)")
w("B) TRAILING: stop ATR dinamico al posto del target RR fisso")
w("C) COMBINED. Confronto diretto vs BASELINE, stesso pool di eventi causali.")

# ── DATA ─────────────────────────────────────────────────────────────────
t0 = time.time()
print("\n[DATA] Loading 1H …")
raw = fetch_extended_data(start_year=START_YEAR, start_month=1,
                           fetch_15m=False, fetch_1m=False, fetch_flow=False)
df1h = add_indicators(raw["1H"])
IDX1H = df1h.index
N1H = len(df1h)
print(f"  1H: {N1H:,} bars  ({IDX1H[0].date()} → {IDX1H[-1].date()})  "
      f"(loaded in {time.time()-t0:.0f}s)")

CL = df1h["close"].values.astype(float)
HI = df1h["high"].values.astype(float)
LO = df1h["low"].values.astype(float)
OP = df1h["open"].values.astype(float)
VOL = df1h["volume"].values.astype(float)
ATR = np.where(df1h["atr_14"].values > 0, df1h["atr_14"].values, 1.0)

dates = IDX1H.normalize().values
day_change = np.r_[True, dates[1:] != dates[:-1]]
day_id = np.cumsum(day_change) - 1
day_start_idx = np.where(day_change)[0]
n_days = len(day_start_idx)

tp = (HI + LO + CL) / 3.0
tmp = pd.DataFrame({"day_id": day_id, "pv": tp * VOL, "vol": VOL})
g = tmp.groupby("day_id")
cum_pv = g["pv"].cumsum().values
cum_v = g["vol"].cumsum().values
bar_in_day = g.cumcount().values
VWAP = np.where(cum_v > 0, cum_pv / cum_v, np.nan)

# ── Volume Profile / POC + PROMINENCE giornaliero rolling, causale ────────
print(f"[VP] Computing rolling daily POC + prominence (lookback={LOOKBACK_DAYS}d, bins={N_BINS}) …")
daily_poc = np.full(n_days, np.nan)
daily_prom = np.full(n_days, np.nan)
MIN_WIN_BARS = 100
for d in range(n_days):
    win_start_day = d - LOOKBACK_DAYS
    if win_start_day < 0:
        continue
    lo_bar = day_start_idx[win_start_day]
    hi_bar = day_start_idx[d]
    if hi_bar - lo_bar < MIN_WIN_BARS:
        continue
    seg_tp = tp[lo_bar:hi_bar]
    seg_vol = VOL[lo_bar:hi_bar]
    seg_lo = LO[lo_bar:hi_bar].min()
    seg_hi = HI[lo_bar:hi_bar].max()
    if seg_hi <= seg_lo:
        continue
    hist, edges = np.histogram(seg_tp, bins=N_BINS, range=(seg_lo, seg_hi), weights=seg_vol)
    poc_idx = int(np.argmax(hist))
    daily_poc[d] = (edges[poc_idx] + edges[poc_idx + 1]) / 2.0
    tot = hist.sum()
    daily_prom[d] = hist[poc_idx] / tot if tot > 0 else np.nan
POC = daily_poc[day_id]

prom_pctl_daily = pd.Series(daily_prom).rolling(PROM_ROLL_DAYS, min_periods=PROM_ROLL_DAYS).rank(pct=True).values
PROM_PCTL = prom_pctl_daily[day_id]
print(f"  valid POC bars: {np.isfinite(POC).sum():,} / {N1H:,}   "
      f"valid prominence-pctl bars: {np.isfinite(PROM_PCTL).sum():,} / {N1H:,}")


def build_events(require_liquidity):
    evs = []
    last_exit = -1
    start_i = max(LOOKBACK, SWING_LOOKBACK) + 2
    for i in range(start_i, N1H - 1):
        if i <= last_exit:
            continue
        if not np.isfinite(VWAP[i]) or not np.isfinite(POC[i]) or bar_in_day[i] < 2 or ATR[i] <= 0:
            continue

        ref = i - LOOKBACK
        if not np.isfinite(VWAP[ref]) or ATR[ref] <= 0:
            continue
        sep = CL[ref] - VWAP[ref]
        if abs(sep) < MIN_SEP_ATR * ATR[ref]:
            continue
        bias = 1 if sep > 0 else -1

        if abs(POC[i] - VWAP[i]) > CONF_ATR_MULT * ATR[i]:
            continue
        zone_lo = min(POC[i], VWAP[i]); zone_hi = max(POC[i], VWAP[i])
        touch = LO[i] <= zone_hi and HI[i] >= zone_lo
        if not touch:
            continue
        zone_mid = (POC[i] + VWAP[i]) / 2.0
        rejection = (bias == 1 and CL[i] > zone_mid) or (bias == -1 and CL[i] < zone_mid)
        if not rejection:
            continue

        if require_liquidity:
            if not np.isfinite(PROM_PCTL[i]) or PROM_PCTL[i] < PROM_PCTL_MIN:
                continue

        if bias == 1:
            swing = LO[i - SWING_LOOKBACK:i + 1].min()
            sl = swing - STOP_BUFFER_ATR * ATR[i]
        else:
            swing = HI[i - SWING_LOOKBACK:i + 1].max()
            sl = swing + STOP_BUFFER_ATR * ATR[i]

        entry_i = i + 1
        ep = OP[entry_i]
        risk = abs(ep - sl)
        if risk <= 0:
            continue
        hold = min(MAX_HOLD_BARS, N1H - 1 - entry_i)
        if hold < 1:
            continue
        evs.append(dict(i=i, entry_i=entry_i, d=bias, ep=ep, sl=sl, risk=risk, hold=hold, atr=ATR[i]))
        last_exit = entry_i + hold
    return evs


EVENTS_BASELINE = build_events(require_liquidity=False)
EVENTS_LIQUIDITY = build_events(require_liquidity=True)
print(f"[EVENTS] baseline={len(EVENTS_BASELINE)}  liquidity-filtered={len(EVENTS_LIQUIDITY)}")


def _settle(cap, peak, mdd, wins, net_pnls, d, ep, exit_price, risk, slip_pct):
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
    return cap, peak, mdd, wins


def run_bt_fixed(evs, extra_slippage_pct=0.0):
    """Baseline / LIQUIDITY: stop strutturale + target RR fisso."""
    slip_pct = SLIPPAGE_BASE + extra_slippage_pct
    if not evs:
        return dict(n=0, wr=0.0, ret=0.0, mdd=0.0, net_pnls=[], n_tp=0, n_sl=0, n_time=0)
    cap = INIT_CAP; peak = cap; mdd = 0.0; wins = 0; net_pnls = []
    n_tp = n_sl = n_time = 0
    for ev in evs:
        d, ep, sl, risk, entry_i, hold = ev["d"], ev["ep"], ev["sl"], ev["risk"], ev["entry_i"], ev["hold"]
        target = ep + d * RR_FIXED * risk
        out = "time"; exit_price = None
        for k in range(hold):
            j = entry_i + k
            if j >= N1H: break
            hk, lk = HI[j], LO[j]
            if d == 1:
                hit_sl = lk <= sl; hit_tp = hk >= target
            else:
                hit_sl = hk >= sl; hit_tp = lk <= target
            if hit_sl: out = "sl"; exit_price = sl; break
            if hit_tp: out = "tp"; exit_price = target; break
        if exit_price is None:
            exit_price = CL[min(entry_i + hold, N1H - 1)]
        if out == "tp": n_tp += 1
        elif out == "sl": n_sl += 1
        else: n_time += 1
        if risk <= 0: continue
        cap, peak, mdd, wins = _settle(cap, peak, mdd, wins, net_pnls, d, ep, exit_price, risk, slip_pct)
    n = len(net_pnls); wr = wins / n if n else 0.0
    return dict(n=n, wr=wr, ret=(cap / INIT_CAP - 1) * 100, mdd=mdd * 100, net_pnls=net_pnls,
                n_tp=n_tp, n_sl=n_sl, n_time=n_time)


def run_bt_trailing(evs, extra_slippage_pct=0.0):
    """TRAILING / COMBINED: stop strutturale iniziale + ATR chandelier trailing, nessun target fisso."""
    slip_pct = SLIPPAGE_BASE + extra_slippage_pct
    if not evs:
        return dict(n=0, wr=0.0, ret=0.0, mdd=0.0, net_pnls=[], n_trail=0, n_time=0)
    cap = INIT_CAP; peak = cap; mdd = 0.0; wins = 0; net_pnls = []
    n_trail = n_time = 0
    for ev in evs:
        d, ep, sl0, risk, entry_i, hold, atr0 = ev["d"], ev["ep"], ev["sl"], ev["risk"], ev["entry_i"], ev["hold"], ev["atr"]
        cur_stop = sl0
        best_px = ep
        activated = False
        out = "time"; exit_price = None
        for k in range(hold):
            j = entry_i + k
            if j >= N1H: break
            hk, lk = HI[j], LO[j]
            if d == 1:
                best_px = max(best_px, hk)
                if not activated and (best_px - ep) >= ACTIVATION_ATR_MULT * atr0:
                    activated = True
                if activated:
                    cur_stop = max(cur_stop, best_px - TRAIL_ATR_MULT * atr0)
                if lk <= cur_stop:
                    out = "trail" if activated else "sl"; exit_price = cur_stop; break
            else:
                best_px = min(best_px, lk)
                if not activated and (ep - best_px) >= ACTIVATION_ATR_MULT * atr0:
                    activated = True
                if activated:
                    cur_stop = min(cur_stop, best_px + TRAIL_ATR_MULT * atr0)
                if hk >= cur_stop:
                    out = "trail" if activated else "sl"; exit_price = cur_stop; break
        if exit_price is None:
            exit_price = CL[min(entry_i + hold, N1H - 1)]
        if out in ("trail", "sl"): n_trail += 1
        else: n_time += 1
        if risk <= 0: continue
        cap, peak, mdd, wins = _settle(cap, peak, mdd, wins, net_pnls, d, ep, exit_price, risk, slip_pct)
    n = len(net_pnls); wr = wins / n if n else 0.0
    return dict(n=n, wr=wr, ret=(cap / INIT_CAP - 1) * 100, mdd=mdd * 100, net_pnls=net_pnls,
                n_trail=n_trail, n_time=n_time)


def mc_summary(pnls):
    if len(pnls) < 5: return dict(p_profit=0.0, p_ruin=1.0)
    mc = run_monte_carlo(pd.DataFrame({"net_pnl": pnls}), INIT_CAP, N_SIMS)
    return dict(p_profit=float(mc.get("p_profit", 0.0)), p_ruin=float(mc.get("p_ruin", 1.0)))

def mc_block_summary(pnls, block_size=10):
    if len(pnls) < 5: return dict(p_profit=0.0, p_ruin=1.0)
    mc = run_monte_carlo_block(pd.DataFrame({"net_pnl": pnls}), INIT_CAP, N_SIMS, block_size=block_size)
    return dict(p_profit=float(mc.get("p_profit", 0.0)), p_ruin=float(mc.get("p_ruin", 1.0)))


VARIANTS = [
    ("BASELINE (RR=3.0 fisso, no filtro)", EVENTS_BASELINE, run_bt_fixed),
    ("LIQUIDITY (prominence POC >= mediana)", EVENTS_LIQUIDITY, run_bt_fixed),
    ("TRAILING (ATR chandelier, no filtro)", EVENTS_BASELINE, run_bt_trailing),
    ("COMBINED (liquidity + trailing)", EVENTS_LIQUIDITY, run_bt_trailing),
]

summary_rows = []

for label, evs, runner in VARIANTS:
    w(f"\n{SEP}")
    w(f"{label}")
    w(SEP)

    res = runner(evs)
    n = res["n"]
    mc = mc_summary(res["net_pnls"])
    mc_blk = mc_block_summary(res["net_pnls"])
    w(f"\n  FULL-SAMPLE: n={n}  wr={res['wr']:.1%}  ret={res['ret']:+.1f}%  mdd={res['mdd']:.1f}%")
    w(f"    MC i.i.d.  : pp={mc['p_profit']:.3f}  pr={mc['p_ruin']:.3f}")
    w(f"    MC block   : pp={mc_blk['p_profit']:.3f}  pr={mc_blk['p_ruin']:.3f}")

    w(f"\n  Breakdown per anno:")
    w(f"    {'Year':>6}  {'n':>6}  {'Ret%':>8}  {'WR':>6}")
    year_evs: dict[int, list] = {}
    for e in evs:
        year_evs.setdefault(IDX1H[e["entry_i"]].year, []).append(e)
    for yr in sorted(year_evs):
        yevs = year_evs[yr]
        if len(yevs) < 5: continue
        yres = runner(yevs)
        w(f"    {yr:>6}  {yres['n']:>6}  {yres['ret']:>+7.1f}%  {yres['wr']:>5.1%}")

    holdout_evs = [e for e in evs if IDX1H[e["entry_i"]] >= CUTOFF]
    hres = runner(holdout_evs)
    hmc = mc_summary(hres["net_pnls"])
    hmc_blk = mc_block_summary(hres["net_pnls"])
    w(f"\n  HOLDOUT GENUINO 2025-2026: n={hres['n']}  wr={hres['wr']:.1%}  ret={hres['ret']:+.1f}%  "
      f"mdd={hres['mdd']:.1f}%")
    w(f"    MC i.i.d.  : pp={hmc['p_profit']:.3f}  pr={hmc['p_ruin']:.3f}")
    w(f"    MC block   : pp={hmc_blk['p_profit']:.3f}  pr={hmc_blk['p_ruin']:.3f}")

    w(f"\n  Slippage-stress (sopra frizione base 0.14% RT già inclusa):")
    w(f"    {'Extra slip':>10}  {'Scope':>10}  {'n':>6}  {'Ret%':>8}  {'WR':>6}  {'MC pp':>7}  {'MC pr':>7}")
    for bps in SLIPPAGE_STRESS_BPS:
        extra = bps / 10_000.0
        for scope_name, scope_evs in [("full-sample", evs), ("holdout", holdout_evs)]:
            r = runner(scope_evs, extra_slippage_pct=extra)
            m = mc_summary(r["net_pnls"])
            w(f"    {bps:>7}bps  {scope_name:>10}  {r['n']:>6}  {r['ret']:>+7.1f}%  "
              f"{r['wr']:>5.1%}  {m['p_profit']:>6.3f}  {m['p_ruin']:>6.3f}")

    summary_rows.append(dict(label=label, n=n, ret=res["ret"], wr=res["wr"], mc_pp=mc["p_profit"],
                              n_hold=hres["n"], ret_hold=hres["ret"], mc_pp_hold=hmc["p_profit"]))

w(f"\n{SEP}")
w("RIEPILOGO")
w(SEP)
w(f"\n  {'Variante':<42}  {'n':>5}  {'Ret%':>8}  {'WR':>6}  {'MCpp':>6}  "
  f"{'n(h)':>5}  {'Ret%(h)':>8}  {'MCpp(h)':>7}")
for r in summary_rows:
    w(f"  {r['label']:<42}  {r['n']:>5}  {r['ret']:>+7.1f}%  {r['wr']:>5.1%}  {r['mc_pp']:>6.3f}  "
      f"{r['n_hold']:>5}  {r['ret_hold']:>+7.1f}%  {r['mc_pp_hold']:>7.3f}")

w(f"\n{SEP}\n[DONE]\n{SEP}")
out_path = Path("reports/vp_vwap_confluence_liquidity_trailing.md")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("# VP + VWAP Confluence — Liquidity filter & ATR Trailing Stop — Validation Pipeline\n\n```\n" +
                     "\n".join(report_lines) + "\n```\n", encoding="utf-8")
print(f"\n[DONE] {out_path}   (total runtime {time.time()-t0:.0f}s)")
