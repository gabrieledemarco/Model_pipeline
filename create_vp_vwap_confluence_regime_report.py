#!/usr/bin/env python3
"""
create_vp_vwap_confluence_regime_report.py
=============================================
v3 — Structural regime filter su Volume Profile + VWAP Confluence (l'unica
strategia di questa sessione con DSR full-sample>0.8): RR=3.0, l'edge è di
tipo trend-continuation (win rate strutturalmente <50% compensato da un
RR asimmetrico) — il sospetto è che il filtro di trend-context "a buon
mercato" (separazione close/VWAP di LOOKBACK barre fa) generi falsi
positivi in mercati genuinamente laterali/choppy, dove non c'è vero
regime direzionale a sostenere la continuazione dopo l'assorbimento sul
POC.

STRUTTURALE, non un nuovo parametro sul grid: si aggiunge un HMM 3-stati
(bear/sideways/bull, hmmlearn, causale) come CONFERMA indipendente del
regime, rifittato ad ogni finestra walk-forward (6m IS / 2m OOS / step 2m,
stessa cadenza usata per VWAP MR in questa sessione) — il modello è
sempre allenato SOLO su barre passate (IS) e applicato SOLO a barre future
(OOS), zero look-ahead. Un trade viene accettato solo se il regime HMM
all'apertura del segnale è coerente con la direzione del trend-context
(bias long -> richiede stato "bull"; bias short -> richiede stato "bear").
Nessun IS-scan/selezione: le due varianti (NO_FILTER baseline vs
REGIME_MATCH) sono confrontate direttamente sullo STESSO pool di eventi
OOS, senza scegliere a posteriori quale sia "migliore" — questo evita di
reintrodurre selection bias nel confronto stesso.

Entry logic invariata rispetto a `create_vp_vwap_confluence_report.py`:
  POC rolling (Volume Profile, 20gg) + VWAP di sessione + trend-context
  (LOOKBACK=5h, MIN_SEP_ATR=0.5) + tocco POC (CONF_ATR_MULT=0.5) +
  rejection + stop strutturale (minimo/massimo su 10h) + target RR=3.0
  fisso (il migliore per DSR nel report base).

FRIZIONI BYBIT (obbligatorie, applicate di default su ogni trade, non solo
come sensitivity check):
  - Esecuzione: ordini a MERCATO (taker) su ENTRAMBI i lati.
  - Fee taker: 0.055% per transazione (0.00055).
  - Slippage: 0.015% per transazione (0.00015), a simulare execution lag
    e profondità del book.
  - Frizione round-trip totale: 0.14% (0.0014), dedotta automaticamente
    da ogni trade (fee*2 + slippage*2).
  - Nessun look-ahead: il segnale sulla barra i usa solo dati fino a i
    incluso (barra i già chiusa); l'esecuzione avviene all'apertura della
    barra i+1.

Validazione: per-anno, holdout 2025-2026 genuino, Monte Carlo i.i.d.+block,
confronto diretto NO_FILTER vs REGIME_MATCH sullo stesso pool OOS,
slippage-stress aggiuntivo sopra la frizione obbligatoria di base.
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
from src.strategy.hmm_regime import fit_hmm, predict_hmm_features
from src.strategy.monte_carlo import run_monte_carlo, run_monte_carlo_block

SEP = "═" * 78
START_YEAR = 2020
INIT_CAP = 100_000.0
RISK_PCT = 0.01

# ── Frizioni Bybit obbligatorie (non-VIP futures) ─────────────────────────
FEE_TAKER = 0.00055          # per transazione
SLIPPAGE_BASE = 0.00015      # per transazione (execution lag / book depth)
# round-trip totale = (FEE_TAKER + SLIPPAGE_BASE) * 2 = 0.0014 (0.14%)

MAX_LEV = 10.0
CUTOFF = pd.Timestamp("2025-01-01")
N_SIMS = 5_000
WF_TRAIN_M, WF_OOS_M, WF_STEP_M = 6, 2, 2

# ── Entry logic (identica al report base, RR=3.0 già validato) ───────────
LOOKBACK = 5
MIN_SEP_ATR = 0.5
LOOKBACK_DAYS = 20
N_BINS = 50
CONF_ATR_MULT = 0.5
SWING_LOOKBACK = 10
STOP_BUFFER_ATR = 0.1
MAX_HOLD_BARS = 48
RR = 3.0
SLIPPAGE_STRESS_BPS = [0, 2, 5, 10]   # extra sopra la frizione base, per stress-test

report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

w(SEP)
w("VP + VWAP Confluence v3 — Regime Filter strutturale (HMM walk-forward) — Validation Pipeline")
w(SEP)
w("\nStessa entry logic del report base (RR=3.0, DSR full=0.884/holdout=0.498).")
w("Aggiunta: HMM 3-stati rifittato per finestra WFO (6m IS/2m OOS), causale,")
w("gate 'regime coerente col bias' vs baseline NO_FILTER sullo stesso pool OOS.")
w("Frizioni Bybit: taker 0.055% + slippage 0.015%/lato = 0.14% round-trip (default).")

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

# ── VWAP di sessione (reset giornaliero UTC), cumulativo, causale ─────────
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

# ── Volume Profile / POC giornaliero rolling, causale (identico al base) ──
print(f"[VP] Computing rolling daily POC (lookback={LOOKBACK_DAYS}d, bins={N_BINS}) …")
daily_poc = np.full(n_days, np.nan)
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
POC = daily_poc[day_id]
print(f"  valid POC bars: {np.isfinite(POC).sum():,} / {N1H:,}")


def build_events(idx_range, hmm_state, filter_variant):
    evs = []
    last_exit = -1
    for i in idx_range:
        if i <= last_exit:
            continue
        if i < max(LOOKBACK, SWING_LOOKBACK) + 2 or i >= N1H - 1:
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

        if filter_variant == "REGIME_MATCH":
            st = hmm_state[i]
            if np.isnan(st):
                continue
            # 0=bear 1=sideways 2=bull (semantica hmm_regime.py)
            if bias == 1 and st != 2:
                continue
            if bias == -1 and st != 0:
                continue
        elif filter_variant == "REGIME_SOFT":
            # esclude solo il regime direttamente OPPOSTO al bias (bear per un
            # long, bull per uno short); sideways è ammesso per entrambe le
            # direzioni -- un filtro più permissivo del match esatto
            st = hmm_state[i]
            if np.isnan(st):
                continue
            if bias == 1 and st == 0:
                continue
            if bias == -1 and st == 2:
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
        evs.append(dict(i=i, entry_i=entry_i, d=bias, ep=ep, sl=sl, risk=risk, hold=hold))
        last_exit = entry_i + hold
    return evs


def run_bt(evs, extra_slippage_pct=0.0):
    fee_pct = FEE_TAKER
    slip_pct = SLIPPAGE_BASE + extra_slippage_pct
    if not evs:
        return dict(n=0, wr=0.0, ret=0.0, mdd=0.0, net_pnls=[], n_tp=0, n_sl=0, n_time=0)
    cap = INIT_CAP; peak = cap; mdd = 0.0; wins = 0; net_pnls = []
    n_tp = n_sl = n_time = 0
    for ev in evs:
        d, ep, sl, risk, entry_i, hold = ev["d"], ev["ep"], ev["sl"], ev["risk"], ev["entry_i"], ev["hold"]
        target = ep + d * RR * risk
        out = "time"; exit_price = None
        for k in range(hold):
            j = entry_i + k
            if j >= N1H: break
            hk, lk = HI[j], LO[j]
            if d == 1:
                hit_sl = lk <= sl; hit_tp = hk >= target
            else:
                hit_sl = hk >= sl; hit_tp = lk <= target
            if hit_sl:
                out = "sl"; exit_price = sl; break
            if hit_tp:
                out = "tp"; exit_price = target; break
        if exit_price is None:
            j = min(entry_i + hold, N1H - 1)
            exit_price = CL[j]
        if out == "tp": n_tp += 1
        elif out == "sl": n_sl += 1
        else: n_time += 1

        if risk <= 0: continue
        r = INIT_CAP * RISK_PCT
        units = min(r / risk, MAX_LEV * INIT_CAP / ep)
        notional = units * ep
        fill_ep = ep * (1 + d * slip_pct)
        fill_xp = exit_price * (1 - d * slip_pct)
        pnl = units * (fill_xp - fill_ep) * d - fee_pct * units * fill_ep - fee_pct * units * fill_xp
        cap += pnl
        peak = max(peak, cap)
        mdd = min(mdd, (cap - peak) / peak)
        wins += int(pnl > 0)
        net_pnls.append(pnl)
    n = len(net_pnls); wr = wins / n if n else 0.0
    return dict(n=n, wr=wr, ret=(cap / INIT_CAP - 1) * 100, mdd=mdd * 100, net_pnls=net_pnls,
                n_tp=n_tp, n_sl=n_sl, n_time=n_time)


def mc_summary(pnls):
    if len(pnls) < 5: return dict(p_profit=0.0, p_ruin=1.0)
    mc = run_monte_carlo(pd.DataFrame({"net_pnl": pnls}), INIT_CAP, N_SIMS)
    return dict(p_profit=float(mc.get("p_profit", 0.0)), p_ruin=float(mc.get("p_ruin", 1.0)))

def mc_block_summary(pnls, block_size=10):
    if len(pnls) < 5: return dict(p_profit=0.0, p_ruin=1.0)
    mc = run_monte_carlo_block(pd.DataFrame({"net_pnl": pnls}), INIT_CAP, N_SIMS, block_size=block_size)
    return dict(p_profit=float(mc.get("p_profit", 0.0)), p_ruin=float(mc.get("p_ruin", 1.0)))


def wf_dates(idx):
    t0_ = idx[0]; windows = []
    while True:
        tr_s = t0_; tr_e = tr_s + pd.DateOffset(months=WF_TRAIN_M)
        oo_s = tr_e; oo_e = oo_s + pd.DateOffset(months=WF_OOS_M)
        if oo_e > idx[-1]: break
        windows.append((tr_s, tr_e, oo_s, oo_e))
        t0_ += pd.DateOffset(months=WF_STEP_M)
    return windows

WF_WINDOWS = wf_dates(IDX1H)
print(f"[WFO] {len(WF_WINDOWS)} finestre (6m IS / 2m OOS / step 2m)")

print("\n[WFO] Fitting HMM causale per finestra …")
t1 = time.time()
oos_evs_nofilter = []
oos_evs_regime = []
oos_evs_soft = []
state_counts = {0: 0, 1: 0, 2: 0}

for tr_s, tr_e, oo_s, oo_e in WF_WINDOWS:
    idx_is = np.where((IDX1H >= tr_s) & (IDX1H < tr_e))[0]
    idx_oos = np.where((IDX1H >= oo_s) & (IDX1H < oo_e))[0]
    if len(idx_is) < 500 or len(idx_oos) < 50:
        continue

    model, sorted_idx = fit_hmm(df1h.iloc[idx_is], n_states=3, random_state=42)
    hmm_oos = predict_hmm_features(model, sorted_idx, df1h.iloc[idx_oos])

    state_oos_full = np.full(N1H, np.nan)
    state_oos_full[idx_oos] = hmm_oos["hmm_state"].values
    for s in hmm_oos["hmm_state"].values:
        state_counts[int(s)] += 1

    oos_evs_nofilter.extend(build_events(idx_oos, state_oos_full, "NO_FILTER"))
    oos_evs_regime.extend(build_events(idx_oos, state_oos_full, "REGIME_MATCH"))
    oos_evs_soft.extend(build_events(idx_oos, state_oos_full, "REGIME_SOFT"))
    print(".", end="", flush=True)

print(f"\n  done in {time.time()-t1:.0f}s")
tot_states = sum(state_counts.values())
w(f"\n[HMM] Distribuzione stati OOS (tutte le finestre): "
  f"bear={state_counts[0]/tot_states:.1%}  sideways={state_counts[1]/tot_states:.1%}  "
  f"bull={state_counts[2]/tot_states:.1%}")

# ── Confronto diretto NO_FILTER vs REGIME_MATCH sullo stesso pool OOS ────
for label, evs in [("NO_FILTER (baseline)", oos_evs_nofilter),
                    ("REGIME_MATCH (HMM gate esatto)", oos_evs_regime),
                    ("REGIME_SOFT (esclude solo regime opposto)", oos_evs_soft)]:
    w(f"\n{SEP}")
    w(f"{label}")
    w(SEP)

    res = run_bt(evs)
    n = res["n"]
    mc = mc_summary(res["net_pnls"])
    mc_blk = mc_block_summary(res["net_pnls"])
    w(f"\n  FULL OOS: n={n}  wr={res['wr']:.1%}  ret={res['ret']:+.1f}%  mdd={res['mdd']:.1f}%")
    w(f"    Exit: TP={res['n_tp']}  SL={res['n_sl']}  time={res['n_time']}")
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
        yres = run_bt(yevs)
        w(f"    {yr:>6}  {yres['n']:>6}  {yres['ret']:>+7.1f}%  {yres['wr']:>5.1%}")

    holdout_evs = [e for e in evs if IDX1H[e["entry_i"]] >= CUTOFF]
    hres = run_bt(holdout_evs)
    hmc = mc_summary(hres["net_pnls"])
    hmc_blk = mc_block_summary(hres["net_pnls"])
    w(f"\n  HOLDOUT GENUINO 2025-2026: n={hres['n']}  wr={hres['wr']:.1%}  ret={hres['ret']:+.1f}%  "
      f"mdd={hres['mdd']:.1f}%")
    w(f"    MC i.i.d.  : pp={hmc['p_profit']:.3f}  pr={hmc['p_ruin']:.3f}")
    w(f"    MC block   : pp={hmc_blk['p_profit']:.3f}  pr={hmc_blk['p_ruin']:.3f}")

    w(f"\n  Slippage-stress (sopra la frizione base 0.14% round-trip già inclusa):")
    w(f"    {'Extra slip':>10}  {'Scope':>10}  {'n':>6}  {'Ret%':>8}  {'WR':>6}  {'MC pp':>7}  {'MC pr':>7}")
    for bps in SLIPPAGE_STRESS_BPS:
        extra = bps / 10_000.0
        for scope_name, scope_evs in [("full-OOS", evs), ("holdout", holdout_evs)]:
            r = run_bt(scope_evs, extra_slippage_pct=extra)
            m = mc_summary(r["net_pnls"])
            w(f"    {bps:>7}bps  {scope_name:>10}  {r['n']:>6}  {r['ret']:>+7.1f}%  "
              f"{r['wr']:>5.1%}  {m['p_profit']:>6.3f}  {m['p_ruin']:>6.3f}")

w(f"\n{SEP}\n[DONE]\n{SEP}")
out_path = Path("reports/vp_vwap_confluence_regime.md")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("# VP + VWAP Confluence v3 — Regime Filter strutturale (HMM walk-forward) — Validation Pipeline\n\n```\n" +
                     "\n".join(report_lines) + "\n```\n", encoding="utf-8")
print(f"\n[DONE] {out_path}   (total runtime {time.time()-t0:.0f}s)")
