#!/usr/bin/env python3
"""
create_fvg_sniper_grade5_final_report.py
============================================
Retest completo su tutto 2020-2026 della configurazione finale — NESSUN
grid aggiuntivo (l'utente ha correttamente segnalato il rischio di
overfitting nel continuare a impilare filtri): parametri FISSI, scelti
una sola volta dalle analisi precedenti e non ri-ottimizzati qui.

  - Segnale: IFVG Flip Only, 1H, filtro bias 4H EMA50 (identico a v1/v2)
  - MIN_GRADE = 5.0 (soglia selezionata più di frequente dalla WFO
    causale in create_fvg_sniper_grade_wfo_report.py — 12/35 finestre,
    la più scelta della griglia)
  - Motore: posizioni concorrenti + filtro anti-leva (v2), stop
    gap-protected, target singolo 3R

Pipeline di validazione completa e standard di sessione: full-sample,
breakdown per anno, holdout genuino 2025-2026, Monte Carlo i.i.d.+block
su entrambi gli scope, slippage-stress. Nessuna correzione DSR
necessaria: un'unica configurazione fissa, non un grid scelto a
posteriori sull'intera storia.
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
MIN_GRADE = 5.0                  # FISSO — nessuna ri-ottimizzazione qui
HTF_EMA_LEN = 50
SL_BUF_ATR = 0.20
MIN_RISK_ATR = 0.40
MAX_RISK_ATR = 4.00
TP3_R = 3.0
MAX_TRADE_BARS = 120
GAP_MAX_AGE_BARS = 220
SLIPPAGE_STRESS_BPS = [0, 2, 5, 10]

report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

w(SEP)
w("FVG Sniper — IFVG Flip Only, MIN_GRADE=5.0 — Retest finale 2020-2026")
w(SEP)
w("\nConfigurazione fissa (nessun grid, nessuna ri-ottimizzazione qui — per")
w("evitare overfitting da filtri impilati): MIN_GRADE=5.0 selezionato dalla")
w("WFO causale precedente, applicato staticamente su tutto lo storico.")

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


def run_engine_ifvg_only(fvg_births, HI, LO, OP, CL, N, htf_bias):
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
                        if htf_bias[i] <= 0 and f["grade"] >= MIN_GRADE:
                            signals.append(dict(idx=i, dir=-1, top=f["top"], bot=f["bot"]))
                else:
                    if HI[i] >= f["top"]:
                        f["mitigated"] = True
                    if CL[i] > f["top"]:
                        f["inverted"] = True
                        if htf_bias[i] >= 0 and f["grade"] >= MIN_GRADE:
                            signals.append(dict(idx=i, dir=1, top=f["top"], bot=f["bot"]))
            if age <= GAP_MAX_AGE_BARS:
                still_active.append(f)
        active = still_active
        for f in births_by_idx.get(i, []):
            active.append(dict(born=i, dir=f["dir"], top=f["top"], bot=f["bot"], grade=f["grade"],
                                mitigated=False, inverted=False))
    return signals


def build_candidates(signals, HI, LO, OP, CL, ATR, N):
    evs = []
    for s in sorted(signals, key=lambda s: s["idx"]):
        i = s["idx"]; entry_i = i + 1
        if entry_i >= N - 1: continue
        atr_i = ATR[i]
        if np.isnan(atr_i) or atr_i <= 0: continue
        dd = s["dir"]; ep = OP[entry_i]
        raw_stop = (s["bot"] - atr_i * SL_BUF_ATR) if dd == 1 else (s["top"] + atr_i * SL_BUF_ATR)
        risk0 = abs(ep - raw_stop)
        risk = min(max(risk0, atr_i * MIN_RISK_ATR), atr_i * MAX_RISK_ATR)
        if risk <= 0: continue
        if risk < RISK_PCT * ep / MAX_LEV: continue
        sl = ep - dd * risk; target = ep + dd * TP3_R * risk
        hold = min(MAX_TRADE_BARS, N - 1 - entry_i)
        if hold < 1: continue
        evs.append(dict(entry_i=entry_i, d=dd, ep=ep, sl=sl, target=target, risk=risk, hold=hold))
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


print("[ENGINE] Detecting FVGs + IFVG Flip signals (MIN_GRADE=5.0) …")
fvgs = detect_graded_fvgs(HI, LO, OP, CL, VOL, ATR)
signals = run_engine_ifvg_only(fvgs, HI, LO, OP, CL, N, htf_bias)
evs = build_candidates(signals, HI, LO, OP, CL, ATR, N)
holdout_evs = [e for e in evs if IDX[e["entry_i"]] >= CUTOFF]
print(f"  {len(signals):,} segnali -> {len(evs):,} trade (holdout: {len(holdout_evs):,})")

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
w("CONFRONTO — MIN_GRADE=5.0 vs baseline MIN_GRADE=4.0 (create_fvg_sniper_v2_report.py)")
w(SEP)
w(f"\n                          n      Ret%      WR    MC pp(full)   ret_hold%   MC pp(hold)")
w(f"  MIN_GRADE=5.0 (qui)  {n:>6}   {res['ret']:>+6.1f}%   {res['wr']:>5.1%}      {mc['p_profit']:>6.3f}      "
  f"{hres['ret']:>+7.1f}%       {hmc['p_profit']:>6.3f}")
w(f"  MIN_GRADE=4.0 (rif.)   1995   +200.9%   33.2%      0.999        -9.8%        0.414")

w(f"\n{SEP}\n[DONE]\n{SEP}")
out_path = Path("reports/fvg_sniper_grade5_final.md")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("# FVG Sniper — IFVG Flip Only, MIN_GRADE=5.0 — Retest finale 2020-2026\n\n```\n" +
                     "\n".join(report_lines) + "\n```\n", encoding="utf-8")
print(f"\n[DONE] {out_path}   (total runtime {time.time()-t0:.0f}s)")
