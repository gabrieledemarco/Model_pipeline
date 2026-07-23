#!/usr/bin/env python3
"""
create_fvg_sniper_fade_report.py
====================================
Versione FADE di IFVG Flip — tre cambi STRUTTURALI (non filtri/soglie da
ottimizzare, nessun nuovo grid), motivati dalle diagnosi precedenti:

  1. DIREZIONE INVERTITA: invece di tradare la continuazione dopo
     l'inversione del gap (IFVG Flip originale), si fada — si scommette
     che la rottura decisiva sia un fakeout e il prezzo torni indietro.
     Replica il pattern strutturale dell'unica strategia validata in
     questa sessione (ICT Fade Standalone: fade dei trigger standalone).
  2. STOP STRUTTURALE: minimo/massimo (nel verso del fade) sulle ultime
     SWING_LOOKBACK barre + buffer ATR — IDENTICO, parametri non
     ri-derivati, allo stop di ICT Fade Standalone (SWING_LOOKBACK=10,
     STOP_BUFFER_ATR=0.1) — sostituisce lo stop "Gap-Protected" con
     floor ATR stretto che l'autopsy ha indicato come causa tecnica
     degli stop-out rapidi.
  3. ENTRY SU RETEST: invece di entrare a mercato alla barra successiva
     al segnale, si aspetta (fino a RETEST_MAX_BARS barre) che il
     prezzo RECLAMI il livello del gap rotto (chiusura di nuovo dentro
     il gap originale) prima di entrare — se non arriva entro la
     finestra, il segnale è scartato. Evita di entrare esattamente
     sullo spike che si esaurisce subito (pattern trovato nell'autopsy:
     tutti i 25 trade peggiori erano stop-out in 1-8 barre).

Target: RR=3.0 fisso, RIUSATO da ICT Fade Standalone (non ri-derivato).
Segnale di base (FVG grading, HTF bias, MIN_GRADE=4.0) invariato.
Pipeline standard: full-sample, per-anno, holdout 2025-2026 genuino, MC
i.i.d.+block, slippage-stress.
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

from src.strategy.data_fetcher import fetch_extended_data
from src.strategy.indicators import add_indicators
from src.strategy.monte_carlo import run_monte_carlo, run_monte_carlo_block, trade_level_sharpe

SEP = "═" * 78
START_YEAR = 2020
INIT_CAP = 100_000.0
RISK_PCT = 0.01
MAX_TOTAL_RISK_PCT = 0.05
FEE_TAKER = 0.00055
SLIPPAGE_BASE = 0.00015
MAX_LEV = 10.0
CUTOFF = pd.Timestamp("2025-01-01")
N_SIMS = 5_000

ATR_LEN = 14
MIN_GAP_ATR = 0.25
MIN_BODY_RATIO = 0.45
MIN_GRADE = 4.0
HTF_EMA_LEN = 50
MAX_TRADE_BARS = 120
GAP_MAX_AGE_BARS = 220

# Parametri RIUSATI (non ri-derivati) da ICT Fade Standalone
SWING_LOOKBACK = 10
STOP_BUFFER_ATR = 0.1
RR = 3.0
RETEST_MAX_BARS = 10

SLIPPAGE_STRESS_BPS = [0, 2, 5, 10]

report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

w(SEP)
w("FVG Sniper FADE — direzione invertita + stop strutturale + entry su retest")
w(SEP)
w("\nTre cambi strutturali (non filtri da ottimizzare): (1) fade invece di")
w("continuazione, (2) stop strutturale 10-barre (param. riusati da ICT Fade),")
w("(3) entry su reclaim del livello invece che a mercato immediato.")

t0 = time.time()
print("\n[DATA] Loading 4H, 1H …")
raw = fetch_extended_data(start_year=START_YEAR, start_month=1,
                           fetch_15m=False, fetch_1m=False, fetch_flow=False)
df4h = add_indicators(raw["4H"])
d = add_indicators(raw["1H"])
IDX = d.index; N = len(d)
print(f"  4H: {len(df4h):,} bars   1H: {N:,} bars  (loaded in {time.time()-t0:.0f}s)")

ema50_4h = df4h["close"].ewm(span=HTF_EMA_LEN, adjust=False).mean().values
htf_bias_4h = np.where(df4h["close"].values > ema50_4h, 1, -1)
IDX4H_vals = df4h.index.values


def map_htf_bias(idx_ltf):
    pos = np.searchsorted(IDX4H_vals, idx_ltf.values, side="right") - 1
    pos = np.clip(pos, 0, len(htf_bias_4h) - 1)
    bias = htf_bias_4h[pos]
    bias[pos < 0] = 0
    return bias


HI = d["high"].values.astype(float); LO = d["low"].values.astype(float)
OP = d["open"].values.astype(float); CL = d["close"].values.astype(float)
VOL = d["volume"].values.astype(float)
ATR = np.where(d["atr_14"].values > 0, d["atr_14"].values, np.nan)
htf_bias = map_htf_bias(d.index)


def detect_graded_fvgs(HI, LO, OP, CL, VOL, ATR):
    n = len(CL)
    vol_avg = pd.Series(VOL).rolling(20, min_periods=1).mean().values
    events = []
    for i in range(2, n):
        atr_i = ATR[i]
        safe_atr = atr_i if (not np.isnan(atr_i) and atr_i > 0) else 1e-9
        body_ratio = abs(CL[i - 1] - OP[i - 1]) / max(HI[i - 1] - LO[i - 1], 1e-9)
        disp3 = (HI[i - 1] - LO[i - 1]) / safe_atr
        vol_score = min(VOL[i - 1] / vol_avg[i - 1], 2.0) / 2.0 if (vol_avg[i - 1] > 0 and not np.isnan(vol_avg[i - 1])) else 0.5
        if LO[i] > HI[i - 2]:
            g_top, g_bot = LO[i], HI[i - 2]
            gap_atr = (g_top - g_bot) / safe_atr
            if gap_atr >= MIN_GAP_ATR and body_ratio >= MIN_BODY_RATIO:
                size_score = min(gap_atr / 1.0, 1.0); disp_score = min(disp3 / 2.0, 1.0)
                grade = min(10.0, (size_score * 0.40 + disp_score * 0.35 + vol_score * 0.25) * 10.0)
                events.append(dict(idx=i, dir=1, top=g_top, bot=g_bot, grade=grade))
        elif HI[i] < LO[i - 2]:
            g_top, g_bot = LO[i - 2], HI[i]
            gap_atr = (g_top - g_bot) / safe_atr
            if gap_atr >= MIN_GAP_ATR and body_ratio >= MIN_BODY_RATIO:
                size_score = min(gap_atr / 1.0, 1.0); disp_score = min(disp3 / 2.0, 1.0)
                grade = min(10.0, (size_score * 0.40 + disp_score * 0.35 + vol_score * 0.25) * 10.0)
                events.append(dict(idx=i, dir=-1, top=g_top, bot=g_bot, grade=grade))
    return events


def run_engine_ifvg_only(fvg_births, HI, LO, OP, CL, N):
    """Identico al motore v2, MA senza il filtro HTF qui (si applica dopo,
    sulla direzione di FADE, non su quella originale — vedi build_fade_candidates)."""
    births_by_idx = {}
    for f in fvg_births:
        births_by_idx.setdefault(f["idx"], []).append(f)
    active = []
    signals = []
    for i in range(N):
        still_active = []
        for f in active:
            age = i - f["born"]
            if not f["inverted"]:
                if f["dir"] == 1:
                    if LO[i] <= f["bot"]:
                        f["mitigated"] = True
                    if CL[i] < f["bot"]:
                        f["inverted"] = True
                        if f["grade"] >= MIN_GRADE:
                            signals.append(dict(idx=i, dir=-1, top=f["top"], bot=f["bot"]))
                else:
                    if HI[i] >= f["top"]:
                        f["mitigated"] = True
                    if CL[i] > f["top"]:
                        f["inverted"] = True
                        if f["grade"] >= MIN_GRADE:
                            signals.append(dict(idx=i, dir=1, top=f["top"], bot=f["bot"]))
            if age <= GAP_MAX_AGE_BARS:
                still_active.append(f)
        active = still_active
        for f in births_by_idx.get(i, []):
            active.append(dict(born=i, dir=f["dir"], top=f["top"], bot=f["bot"], grade=f["grade"],
                                mitigated=False, inverted=False))
    return signals


def build_fade_candidates(signals, HI, LO, OP, CL, ATR, N, htf_bias):
    """(1) direzione = OPPOSTA al segnale originale (fade). (2) entry SOLO
    dopo reclaim del livello rotto entro RETEST_MAX_BARS (altrimenti
    scartato). (3) stop strutturale swing 10-barre + buffer ATR."""
    evs = []
    for s in sorted(signals, key=lambda s: s["idx"]):
        i = s["idx"]
        trade_dir = -s["dir"]                      # (1) FADE
        level = s["bot"] if s["dir"] == 1 else s["top"]

        retest_j = None
        for j in range(i + 1, min(i + 1 + RETEST_MAX_BARS, N)):
            if trade_dir == 1 and CL[j] > level:
                retest_j = j; break
            if trade_dir == -1 and CL[j] < level:
                retest_j = j; break
        if retest_j is None:                        # (3) nessun reclaim -> scarta
            continue

        entry_i = retest_j + 1
        if entry_i >= N - 1:
            continue
        if htf_bias[entry_i] * trade_dir < 0:        # bias 4H allineato alla direzione di FADE
            continue
        if entry_i < SWING_LOOKBACK + 2:
            continue
        atr_i = ATR[entry_i]
        if np.isnan(atr_i) or atr_i <= 0:
            continue

        ep = OP[entry_i]
        if trade_dir == 1:
            swing = LO[entry_i - SWING_LOOKBACK:entry_i + 1].min()
            sl = swing - STOP_BUFFER_ATR * atr_i
        else:
            swing = HI[entry_i - SWING_LOOKBACK:entry_i + 1].max()
            sl = swing + STOP_BUFFER_ATR * atr_i
        risk = abs(ep - sl)
        if risk <= 0:
            continue
        if risk < RISK_PCT * ep / MAX_LEV:
            continue
        target = ep + trade_dir * RR * risk
        hold = min(MAX_TRADE_BARS, N - 1 - entry_i)
        if hold < 1:
            continue
        evs.append(dict(entry_i=entry_i, d=trade_dir, ep=ep, sl=sl, target=target, risk=risk, hold=hold))
    return evs


def run_bt_portfolio(evs, extra_slippage_pct=0.0, max_total_risk_pct=MAX_TOTAL_RISK_PCT):
    slip_pct = SLIPPAGE_BASE + extra_slippage_pct
    risk_budget_dollar = INIT_CAP * RISK_PCT
    max_total_risk_dollar = INIT_CAP * max_total_risk_pct
    if not evs:
        return dict(n=0, wr=0.0, ret=0.0, mdd=0.0, net_pnls=[], n_tp=0, n_sl=0, n_time=0)
    by_entry: dict[int, list] = {}
    for ev in evs:
        by_entry.setdefault(ev["entry_i"], []).append(ev)
    min_i = min(by_entry); max_i = max(ev["entry_i"] + ev["hold"] for ev in evs)
    open_positions = []; total_open_risk = 0.0
    cap = INIT_CAP; peak = cap; mdd = 0.0; wins = 0; net_pnls = []
    n_tp = n_sl = n_time = 0

    def close_position(pos, exit_price, reason):
        nonlocal cap, peak, mdd, wins, total_open_risk, n_tp, n_sl, n_time
        dd, ep, risk = pos["d"], pos["ep"], pos["risk"]
        units = risk_budget_dollar / risk
        fill_ep = ep * (1 + dd * slip_pct); fill_xp = exit_price * (1 - dd * slip_pct)
        pnl = units * (fill_xp - fill_ep) * dd - FEE_TAKER * units * fill_ep - FEE_TAKER * units * fill_xp
        cap += pnl; peak = max(peak, cap); mdd = min(mdd, (cap - peak) / peak)
        wins += int(pnl > 0); net_pnls.append(pnl)
        if reason == "tp": n_tp += 1
        elif reason == "sl": n_sl += 1
        else: n_time += 1
        total_open_risk -= risk_budget_dollar

    for i in range(min_i, max_i + 1):
        still_open = []
        for pos in open_positions:
            dd, sl, target, deadline_i = pos["d"], pos["sl"], pos["target"], pos["deadline_i"]
            hk, lk = HI[i], LO[i]
            hit_sl = (lk <= sl) if dd == 1 else (hk >= sl)
            hit_tp = (hk >= target) if dd == 1 else (lk <= target)
            if hit_sl: close_position(pos, sl, "sl"); continue
            if hit_tp: close_position(pos, target, "tp"); continue
            if i >= deadline_i: close_position(pos, CL[min(i, N - 1)], "time"); continue
            still_open.append(pos)
        open_positions = still_open
        for ev in by_entry.get(i, []):
            if total_open_risk + risk_budget_dollar > max_total_risk_dollar + 1e-9:
                continue
            open_positions.append(dict(entry_i=ev["entry_i"], d=ev["d"], ep=ev["ep"], sl=ev["sl"],
                                        target=ev["target"], risk=ev["risk"], deadline_i=ev["entry_i"] + ev["hold"]))
            total_open_risk += risk_budget_dollar
    for pos in open_positions:
        close_position(pos, CL[min(max_i, N - 1)], "time")

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


print("[ENGINE] Detecting FVGs + IFVG Flip signals (base, pre-fade) …")
fvgs = detect_graded_fvgs(HI, LO, OP, CL, VOL, ATR)
signals = run_engine_ifvg_only(fvgs, HI, LO, OP, CL, N)
evs = build_fade_candidates(signals, HI, LO, OP, CL, ATR, N, htf_bias)
holdout_evs = [e for e in evs if IDX[e["entry_i"]] >= CUTOFF]
print(f"  {len(signals):,} segnali base -> {len(evs):,} trade fade validi "
      f"(dopo retest+bias+stop; holdout: {len(holdout_evs):,})")

res = run_bt_portfolio(evs)
n = res["n"]
pval = st.binomtest(int(round(res["wr"] * n)), n, 0.5, alternative="greater").pvalue if n else 1.0
mc = mc_summary(res["net_pnls"]); mc_blk = mc_block_summary(res["net_pnls"])
sharpe_full = trade_level_sharpe(res["net_pnls"])

w(f"\n{SEP}")
w("FULL-SAMPLE 2020-2026")
w(SEP)
w(f"\n  n={n}  wr={res['wr']:.1%} (p={pval:.4f} vs 50%)  ret={res['ret']:+.1f}%  mdd={res['mdd']:.1f}%  "
  f"Sharpe_trade={sharpe_full:.3f}")
w(f"    Exit: TP={res['n_tp']}  SL={res['n_sl']}  time={res['n_time']}")
w(f"    MC i.i.d.  : pp={mc['p_profit']:.3f}  pr={mc['p_ruin']:.3f}")
w(f"    MC block   : pp={mc_blk['p_profit']:.3f}  pr={mc_blk['p_ruin']:.3f}")

w(f"\n  Breakdown per anno:")
w(f"    {'Year':>6}  {'n':>6}  {'Ret%':>8}  {'WR':>6}")
year_evs: dict[int, list] = {}
for e in evs:
    year_evs.setdefault(IDX[e["entry_i"]].year, []).append(e)
for yr in sorted(year_evs):
    yevs = year_evs[yr]
    if len(yevs) < 5: continue
    yres = run_bt_portfolio(yevs)
    w(f"    {yr:>6}  {yres['n']:>6}  {yres['ret']:>+7.1f}%  {yres['wr']:>5.1%}")

hres = run_bt_portfolio(holdout_evs)
hmc = mc_summary(hres["net_pnls"]); hmc_blk = mc_block_summary(hres["net_pnls"])
sharpe_h = trade_level_sharpe(hres["net_pnls"])
w(f"\n{SEP}")
w("HOLDOUT GENUINO 2025-2026")
w(SEP)
w(f"\n  n={hres['n']}  wr={hres['wr']:.1%}  ret={hres['ret']:+.1f}%  mdd={hres['mdd']:.1f}%  "
  f"Sharpe_trade={sharpe_h:.3f}")
w(f"    Exit: TP={hres['n_tp']}  SL={hres['n_sl']}  time={hres['n_time']}")
w(f"    MC i.i.d.  : pp={hmc['p_profit']:.3f}  pr={hmc['p_ruin']:.3f}")
w(f"    MC block   : pp={hmc_blk['p_profit']:.3f}  pr={hmc_blk['p_ruin']:.3f}")

w(f"\n{SEP}")
w("Slippage-stress")
w(SEP)
w(f"\n  {'Extra slip':>10}  {'Scope':>10}  {'n':>6}  {'Ret%':>8}  {'WR':>6}  {'MC pp':>7}  {'MC pr':>7}")
for bps in SLIPPAGE_STRESS_BPS:
    extra = bps / 10_000.0
    for scope_name, evs_s in [("full-sample", evs), ("holdout", holdout_evs)]:
        r = run_bt_portfolio(evs_s, extra_slippage_pct=extra)
        m = mc_summary(r["net_pnls"])
        w(f"  {bps:>7}bps  {scope_name:>10}  {r['n']:>6}  {r['ret']:>+7.1f}%  "
          f"{r['wr']:>5.1%}  {m['p_profit']:>6.3f}  {m['p_ruin']:>6.3f}")

w(f"\n{SEP}")
w("CONFRONTO — FADE (qui) vs continuazione (baseline/filtro grade)")
w(SEP)
w(f"\n                              n      Ret%      WR    MC pp(full)   ret_hold%   MC pp(hold)")
w(f"  FADE (qui)               {n:>6}   {res['ret']:>+6.1f}%   {res['wr']:>5.1%}      {mc['p_profit']:>6.3f}      "
  f"{hres['ret']:>+7.1f}%       {hmc['p_profit']:>6.3f}")
w(f"  Continuazione, grade=5.0   1725   +225.3%   33.9%      1.000        +15.2%        0.644")
w(f"  Continuazione, baseline    1995   +200.9%   33.2%      0.999         -9.8%        0.414")

w(f"\n{SEP}\n[DONE]\n{SEP}")
out_path = Path("reports/fvg_sniper_fade.md")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("# FVG Sniper FADE — direzione invertita + stop strutturale + retest\n\n```\n" +
                     "\n".join(report_lines) + "\n```\n", encoding="utf-8")
print(f"\n[DONE] {out_path}   (total runtime {time.time()-t0:.0f}s)")
