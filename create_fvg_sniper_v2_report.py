#!/usr/bin/env python3
"""
create_fvg_sniper_v2_report.py
=================================
v2 — corregge la distorsione identificata in v1 (create_fvg_sniper_report.py):
lo stop "Gap-Protected" con floor ATR molto stretto (0.4x) produce spesso
un rischio in prezzo così piccolo che il sizing a rischio fisso (1% capitale
per trade) richiederebbe una leva > MAX_LEV=10x; il codice v1 troncava
silenziosamente la size al cap di leva, MA continuava a pagare le fee sul
notional pieno (leveraged) — pagando commissioni sproporzionate rispetto
al rischio realmente assunto. Questo, non l'assenza di edge, era la causa
principale del crollo su 15M (-250/-436% con win rate sopra il breakeven
teorico).

Due cambi, entrambi richiesti esplicitamente dall'utente:

1. FILTRO ANTI-CAP DI LEVA: un trade viene scartato (non troncato) se il
   rischio in prezzo è troppo stretto per raggiungere il rischio fisso
   all'1% del capitale entro MAX_LEV=10x — cioè si scarta se
   risk_price < RISK_PCT * entry_price / MAX_LEV. Ogni trade che sopravvive
   usa SEMPRE size = capitale*RISK_PCT/risk (mai troncata), quindi la fee è
   sempre proporzionale al rischio realmente assunto.

2. TRADE CONCORRENTI CON CAP SUL CAPITALE A RISCHIO: invece di un solo
   trade alla volta (gating sequenziale di v1), si simula un vero motore a
   posizioni multiple: ogni nuovo segnale apre una posizione SOLO SE il
   rischio totale delle posizioni già aperte + il rischio dell'1% di questo
   trade non supera MAX_TOTAL_RISK_PCT del capitale iniziale (default 5%,
   equivalente a un tetto di ~5 posizioni a rischio pieno aperte insieme).
   Simulazione bar-per-bar: a ogni barra si chiudono prima le posizioni che
   toccano SL/TP/scadenza, poi si aprono i nuovi segnali che rientrano nel
   budget di rischio libero in quel momento.

Tutto il resto (detection FVG, grading, state machine, segnali Rejection +
IFVG, filtro bias 4H EMA50) è identico a v1. Stessa pipeline di
validazione: fee reali Bybit, DSR family sulle 3 modalità di segnale,
breakdown annuale, holdout 2025-2026 genuino, MC i.i.d.+block,
slippage-stress.
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
from src.strategy.monte_carlo import (
    run_monte_carlo, run_monte_carlo_block, deflated_sharpe_ratio_family,
)

SEP = "═" * 78
START_YEAR = 2020
INIT_CAP = 100_000.0
RISK_PCT = 0.01
MAX_TOTAL_RISK_PCT = 0.05          # NUOVO: tetto al rischio aggregato aperto
FEE_TAKER = 0.00055
SLIPPAGE_BASE = 0.00015
MAX_LEV = 10.0
CUTOFF = pd.Timestamp("2025-01-01")
N_SIMS = 5_000
TIMEFRAMES = ["15M", "1H"]

ATR_LEN = 14
MIN_GAP_ATR = 0.25
MIN_BODY_RATIO = 0.45
MIN_GRADE = 4.0
HTF_EMA_LEN = 50
REQ_DIR_CLOSE = True
SL_BUF_ATR = 0.20
MIN_RISK_ATR = 0.40
MAX_RISK_ATR = 4.00
TP3_R = 3.0
MAX_TRADE_BARS = 120
GAP_MAX_AGE_BARS = 220
SIGNAL_MODES = ["Rejection Only", "IFVG Flip Only", "Rejection + IFVG"]
SLIPPAGE_STRESS_BPS = [0, 2, 5, 10]

report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

w(SEP)
w('FVG Sniper [JOAT] v2 — anti-leverage-cap filter + concurrent positions')
w(SEP)
w("\nv2: (1) scarta i trade il cui rischio in prezzo è troppo stretto per")
w("il sizing 1% entro MAX_LEV=10x (invece di troncare la size pagando fee")
w("sproporzionate); (2) trade CONCORRENTI con tetto sul rischio aggregato")
w(f"aperto ({MAX_TOTAL_RISK_PCT:.0%} del capitale) invece di un trade alla volta.")

t0 = time.time()
print("\n[DATA] Loading 4H, 1H, 15M …")
raw = fetch_extended_data(start_year=START_YEAR, start_month=1,
                           fetch_15m=True, fetch_1m=False, fetch_flow=False)
df4h = add_indicators(raw["4H"])
dfs = {tf: add_indicators(raw[tf]) for tf in TIMEFRAMES}
print(f"  4H: {len(df4h):,} bars   " + "  ".join(f"{tf}: {len(dfs[tf]):,}" for tf in TIMEFRAMES))
print(f"  (loaded in {time.time()-t0:.0f}s)")

ema50_4h = df4h["close"].ewm(span=HTF_EMA_LEN, adjust=False).mean().values
htf_bias_4h = np.where(df4h["close"].values > ema50_4h, 1, -1)
IDX4H_vals = df4h.index.values


def map_htf_bias(idx_ltf):
    pos = np.searchsorted(IDX4H_vals, idx_ltf.values, side="right") - 1
    pos = np.clip(pos, 0, len(htf_bias_4h) - 1)
    bias = htf_bias_4h[pos]
    bias[pos < 0] = 0
    return bias


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
                size_score = min(gap_atr / 1.0, 1.0)
                disp_score = min(disp3 / 2.0, 1.0)
                grade = min(10.0, (size_score * 0.40 + disp_score * 0.35 + vol_score * 0.25) * 10.0)
                events.append(dict(idx=i, dir=1, top=g_top, bot=g_bot, grade=grade))
        elif HI[i] < LO[i - 2]:
            g_top, g_bot = LO[i - 2], HI[i]
            gap_atr = (g_top - g_bot) / safe_atr
            if gap_atr >= MIN_GAP_ATR and body_ratio >= MIN_BODY_RATIO:
                size_score = min(gap_atr / 1.0, 1.0)
                disp_score = min(disp3 / 2.0, 1.0)
                grade = min(10.0, (size_score * 0.40 + disp_score * 0.35 + vol_score * 0.25) * 10.0)
                events.append(dict(idx=i, dir=-1, top=g_top, bot=g_bot, grade=grade))
    return events


def run_engine(fvg_births, HI, LO, OP, CL, N, htf_bias, mode):
    use_rej = mode in ("Rejection Only", "Rejection + IFVG")
    use_inv = mode in ("IFVG Flip Only", "Rejection + IFVG")

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
                    f["ext"] = min(f["ext"], LO[i])
                    if LO[i] <= f["bot"]:
                        f["mitigated"] = True
                    if CL[i] < f["bot"]:
                        f["inverted"] = True
                        if use_inv and htf_bias[i] <= 0 and f["grade"] >= MIN_GRADE:
                            signals.append(dict(idx=i, dir=-1, tag="IFVG", top=f["top"], bot=f["bot"]))
                else:
                    f["ext"] = max(f["ext"], HI[i])
                    if HI[i] >= f["top"]:
                        f["mitigated"] = True
                    if CL[i] > f["top"]:
                        f["inverted"] = True
                        if use_inv and htf_bias[i] >= 0 and f["grade"] >= MIN_GRADE:
                            signals.append(dict(idx=i, dir=1, tag="IFVG", top=f["top"], bot=f["bot"]))

                if use_rej and not f["mitigated"] and not f["inverted"] and not f["reacted"] and age >= 1:
                    if f["dir"] == 1:
                        tapped = f["bot"] <= LO[i] <= f["top"] and CL[i] > f["bot"]
                        ok_close = (not REQ_DIR_CLOSE) or (CL[i] > OP[i])
                        if tapped and ok_close and f["grade"] >= MIN_GRADE and htf_bias[i] >= 0:
                            f["reacted"] = True
                            signals.append(dict(idx=i, dir=1, tag="REJ", top=f["top"], bot=f["bot"]))
                    else:
                        tapped = f["bot"] <= HI[i] <= f["top"] and CL[i] < f["top"]
                        ok_close = (not REQ_DIR_CLOSE) or (CL[i] < OP[i])
                        if tapped and ok_close and f["grade"] >= MIN_GRADE and htf_bias[i] <= 0:
                            f["reacted"] = True
                            signals.append(dict(idx=i, dir=-1, tag="REJ", top=f["top"], bot=f["bot"]))

            if age <= GAP_MAX_AGE_BARS:
                still_active.append(f)
        active = still_active

        for f in births_by_idx.get(i, []):
            active.append(dict(born=i, dir=f["dir"], top=f["top"], bot=f["bot"], grade=f["grade"],
                                ext=(f["top"] if f["dir"] == 1 else f["bot"]),
                                mitigated=False, inverted=False, reacted=False))
    return signals


def build_candidates(signals, HI, LO, OP, CL, ATR, N):
    """v2: NESSUN gating sequenziale (i trade possono sovrapporsi nel tempo
    — la gestione della concorrenza avviene nel motore a posizioni multiple
    più sotto). Filtro anti-leva: scarta il trade se il rischio in prezzo
    non basta a raggiungere il sizing 1% entro MAX_LEV."""
    evs = []
    for s in sorted(signals, key=lambda s: s["idx"]):
        i = s["idx"]
        entry_i = i + 1
        if entry_i >= N - 1:
            continue
        atr_i = ATR[i]
        if np.isnan(atr_i) or atr_i <= 0:
            continue
        d = s["dir"]
        ep = OP[entry_i]
        raw_stop = (s["bot"] - atr_i * SL_BUF_ATR) if d == 1 else (s["top"] + atr_i * SL_BUF_ATR)
        risk0 = abs(ep - raw_stop)
        risk = min(max(risk0, atr_i * MIN_RISK_ATR), atr_i * MAX_RISK_ATR)
        if risk <= 0:
            continue
        min_risk_for_lev = RISK_PCT * ep / MAX_LEV
        if risk < min_risk_for_lev:
            continue                                     # NUOVO: scarta, non tronca
        sl = ep - d * risk
        target = ep + d * TP3_R * risk
        hold = min(MAX_TRADE_BARS, N - 1 - entry_i)
        if hold < 1:
            continue
        evs.append(dict(entry_i=entry_i, d=d, ep=ep, sl=sl, target=target, risk=risk, hold=hold))
    return evs


def run_bt_portfolio(evs, HI, LO, CL, N, extra_slippage_pct=0.0,
                      max_total_risk_pct=MAX_TOTAL_RISK_PCT):
    """Motore a posizioni multiple concorrenti, simulazione bar-per-bar.
    Tetto: somma dei rischio$ delle posizioni aperte <= max_total_risk_pct
    * INIT_CAP. Un segnale che non rientra nel budget libero al momento
    della sua barra di entry viene scartato (nessuna coda/retry)."""
    slip_pct = SLIPPAGE_BASE + extra_slippage_pct
    if not evs:
        return dict(n=0, wr=0.0, ret=0.0, mdd=0.0, net_pnls=[], n_tp=0, n_sl=0, n_time=0,
                     n_skipped_budget=0)

    risk_budget_dollar = INIT_CAP * RISK_PCT
    max_total_risk_dollar = INIT_CAP * max_total_risk_pct

    by_entry: dict[int, list] = {}
    for ev in evs:
        by_entry.setdefault(ev["entry_i"], []).append(ev)
    min_i = min(by_entry) if by_entry else 0
    max_i = max(ev["entry_i"] + ev["hold"] for ev in evs)

    open_positions = []          # dict(d, ep, sl, target, deadline_i)
    total_open_risk = 0.0
    cap = INIT_CAP
    peak = cap
    mdd = 0.0
    wins = 0
    net_pnls = []
    n_tp = n_sl = n_time = 0
    n_skipped_budget = 0

    def close_position(pos, exit_price):
        nonlocal cap, peak, mdd, wins, total_open_risk
        d, ep = pos["d"], pos["ep"]
        risk = pos["risk"]
        units = risk_budget_dollar / risk               # mai troncata: filtro anti-leva già garantisce fattibilità
        fill_ep = ep * (1 + d * slip_pct)
        fill_xp = exit_price * (1 - d * slip_pct)
        pnl = units * (fill_xp - fill_ep) * d - FEE_TAKER * units * fill_ep - FEE_TAKER * units * fill_xp
        cap += pnl
        peak = max(peak, cap)
        mdd = min(mdd, (cap - peak) / peak)
        wins += int(pnl > 0)
        net_pnls.append(pnl)
        total_open_risk -= risk_budget_dollar

    for i in range(min_i, max_i + 1):
        still_open = []
        for pos in open_positions:
            d, sl, target, deadline_i = pos["d"], pos["sl"], pos["target"], pos["deadline_i"]
            hk, lk = HI[i], LO[i]
            hit_sl = (lk <= sl) if d == 1 else (hk >= sl)
            hit_tp = (hk >= target) if d == 1 else (lk <= target)
            if hit_sl:
                n_sl += 1; close_position(pos, sl); continue
            if hit_tp:
                n_tp += 1; close_position(pos, target); continue
            if i >= deadline_i:
                n_time += 1; close_position(pos, CL[min(i, N - 1)]); continue
            still_open.append(pos)
        open_positions = still_open

        for ev in by_entry.get(i, []):
            if total_open_risk + risk_budget_dollar > max_total_risk_dollar + 1e-9:
                n_skipped_budget += 1
                continue
            open_positions.append(dict(d=ev["d"], ep=ev["ep"], sl=ev["sl"], target=ev["target"],
                                        risk=ev["risk"], deadline_i=ev["entry_i"] + ev["hold"]))
            total_open_risk += risk_budget_dollar

    # chiudi eventuali posizioni residue a mercato all'ultimo bar disponibile
    for pos in open_positions:
        n_time += 1
        close_position(pos, CL[min(max_i, N - 1)])

    n = len(net_pnls); wr = wins / n if n else 0.0
    return dict(n=n, wr=wr, ret=(cap / INIT_CAP - 1) * 100, mdd=mdd * 100, net_pnls=net_pnls,
                n_tp=n_tp, n_sl=n_sl, n_time=n_time, n_skipped_budget=n_skipped_budget)


def mc_summary(pnls):
    if len(pnls) < 5:
        return dict(p_profit=0.0, p_ruin=1.0)
    mc = run_monte_carlo(pd.DataFrame({"net_pnl": pnls}), INIT_CAP, N_SIMS)
    return dict(p_profit=float(mc.get("p_profit", 0.0)), p_ruin=float(mc.get("p_ruin", 1.0)))

def mc_block_summary(pnls, block_size=10):
    if len(pnls) < 5:
        return dict(p_profit=0.0, p_ruin=1.0)
    mc = run_monte_carlo_block(pd.DataFrame({"net_pnl": pnls}), INIT_CAP, N_SIMS, block_size=block_size)
    return dict(p_profit=float(mc.get("p_profit", 0.0)), p_ruin=float(mc.get("p_ruin", 1.0)))


for tf in TIMEFRAMES:
    d = dfs[tf]
    HI = d["high"].values.astype(float); LO = d["low"].values.astype(float)
    OP = d["open"].values.astype(float); CL = d["close"].values.astype(float)
    VOL = d["volume"].values.astype(float)
    ATR = np.where(d["atr_14"].values > 0, d["atr_14"].values, np.nan)
    N = len(d)
    htf_bias = map_htf_bias(d.index)

    w(f"\n{SEP}")
    w(f"[{tf}] Detection")
    w(SEP)
    fvgs = detect_graded_fvgs(HI, LO, OP, CL, VOL, ATR)
    w(f"\n  {len(fvgs):,} FVG (grade>={MIN_GRADE}, size>={MIN_GAP_ATR}xATR, body>={MIN_BODY_RATIO})")

    dsr_full, dsr_holdout = [], []
    for mode in SIGNAL_MODES:
        signals = run_engine(fvgs, HI, LO, OP, CL, N, htf_bias, mode)
        cands = build_candidates(signals, HI, LO, OP, CL, ATR, N)
        cands_holdout = [e for e in cands if d.index[e["entry_i"]] >= CUTOFF]

        w(f"\n{SEP}")
        w(f"[{tf}] Mode = {mode}")
        w(SEP)
        w(f"\n  {len(signals):,} segnali -> {len(cands):,} candidati (post filtro anti-leva; "
          f"holdout 2025-2026: {len(cands_holdout):,})")

        res = run_bt_portfolio(cands, HI, LO, CL, N)
        n = res["n"]
        pval = st.binomtest(int(round(res["wr"] * n)), n, 0.5, alternative="greater").pvalue if n else 1.0
        mc = mc_summary(res["net_pnls"])
        mc_blk = mc_block_summary(res["net_pnls"])
        w(f"\n  FULL-SAMPLE: n={n} (scartati per budget rischio: {res['n_skipped_budget']})  "
          f"wr={res['wr']:.1%} (p={pval:.4f} vs 50%)  ret={res['ret']:+.1f}%  mdd={res['mdd']:.1f}%")
        w(f"    Exit: TP={res['n_tp']}  SL={res['n_sl']}  time={res['n_time']}")
        w(f"    MC i.i.d.  : pp={mc['p_profit']:.3f}  pr={mc['p_ruin']:.3f}")
        w(f"    MC block   : pp={mc_blk['p_profit']:.3f}  pr={mc_blk['p_ruin']:.3f}")

        w(f"\n  Breakdown per anno:")
        w(f"    {'Year':>6}  {'n':>6}  {'Ret%':>8}  {'WR':>6}")
        year_evs: dict[int, list] = {}
        for e in cands:
            year_evs.setdefault(d.index[e["entry_i"]].year, []).append(e)
        for yr in sorted(year_evs):
            yevs = year_evs[yr]
            if len(yevs) < 5:
                continue
            yres = run_bt_portfolio(yevs, HI, LO, CL, N)
            w(f"    {yr:>6}  {yres['n']:>6}  {yres['ret']:>+7.1f}%  {yres['wr']:>5.1%}")

        hres = run_bt_portfolio(cands_holdout, HI, LO, CL, N)
        hmc = mc_summary(hres["net_pnls"])
        hmc_blk = mc_block_summary(hres["net_pnls"])
        w(f"\n  HOLDOUT GENUINO 2025-2026: n={hres['n']}  wr={hres['wr']:.1%}  ret={hres['ret']:+.1f}%  "
          f"mdd={hres['mdd']:.1f}%")
        w(f"    MC i.i.d.  : pp={hmc['p_profit']:.3f}  pr={hmc['p_ruin']:.3f}")
        w(f"    MC block   : pp={hmc_blk['p_profit']:.3f}  pr={hmc_blk['p_ruin']:.3f}")

        dsr_full.append(dict(mode=mode, net_pnls=res["net_pnls"], ret=res["ret"]))
        dsr_holdout.append(dict(mode=mode, net_pnls=hres["net_pnls"], ret=hres["ret"]))

    w(f"\n{SEP}")
    w(f"[{tf}] DSR — famiglia N={len(SIGNAL_MODES)} (griglia: modalità segnale)")
    w(SEP)
    deflated_sharpe_ratio_family(dsr_full, sharpe_key="sharpe_hat", dsr_key="dsr", pnls_key="net_pnls")
    deflated_sharpe_ratio_family(dsr_holdout, sharpe_key="sharpe_hat", dsr_key="dsr", pnls_key="net_pnls")
    w(f"\n  Full-sample:")
    w(f"  {'Mode':<20}{'n':>6}  {'Ret%':>8}  {'Sharpe_hat':>11}  {'DSR':>7}")
    for r in dsr_full:
        w(f"  {r['mode']:<20}{len(r['net_pnls']):>6}  {r['ret']:>+7.1f}%  {r['sharpe_hat']:>11.3f}  {r['dsr']:>7.3f}")
    w(f"\n  Holdout 2025-2026:")
    w(f"  {'Mode':<20}{'n':>6}  {'Ret%':>8}  {'Sharpe_hat':>11}  {'DSR':>7}")
    for r in dsr_holdout:
        w(f"  {r['mode']:<20}{len(r['net_pnls']):>6}  {r['ret']:>+7.1f}%  {r['sharpe_hat']:>11.3f}  {r['dsr']:>7.3f}")

    holdout_dsr_by_mode = {r["mode"]: r["dsr"] for r in dsr_holdout}
    holdout_sharpe_by_mode = {r["mode"]: r["sharpe_hat"] for r in dsr_holdout}
    best_mode = max(dsr_full, key=lambda r: (r["dsr"], holdout_dsr_by_mode[r["mode"]],
                                              holdout_sharpe_by_mode[r["mode"]]))["mode"]
    w(f"\n{SEP}")
    w(f"[{tf}] Slippage-stress — modalità migliore per DSR full+holdout = {best_mode}")
    w(SEP)
    w(f"\n  {'Extra slip':>10}  {'Scope':>10}  {'n':>6}  {'Ret%':>8}  {'WR':>6}  {'MC pp':>7}  {'MC pr':>7}")
    best_signals = run_engine(fvgs, HI, LO, OP, CL, N, htf_bias, best_mode)
    best_cands = build_candidates(best_signals, HI, LO, OP, CL, ATR, N)
    best_cands_holdout = [e for e in best_cands if d.index[e["entry_i"]] >= CUTOFF]
    for bps in SLIPPAGE_STRESS_BPS:
        extra = bps / 10_000.0
        for scope_name, evs_s in [("full-sample", best_cands), ("holdout", best_cands_holdout)]:
            r = run_bt_portfolio(evs_s, HI, LO, CL, N, extra_slippage_pct=extra)
            m = mc_summary(r["net_pnls"])
            w(f"  {bps:>7}bps  {scope_name:>10}  {r['n']:>6}  {r['ret']:>+7.1f}%  "
              f"{r['wr']:>5.1%}  {m['p_profit']:>6.3f}  {m['p_ruin']:>6.3f}")

w(f"\n{SEP}\n[DONE]\n{SEP}")
out_path = Path("reports/fvg_sniper_v2_validation.md")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("# FVG Sniper [JOAT] v2 — anti-leverage-cap + concurrent positions\n\n```\n" +
                     "\n".join(report_lines) + "\n```\n", encoding="utf-8")
print(f"\n[DONE] {out_path}   (total runtime {time.time()-t0:.0f}s)")
