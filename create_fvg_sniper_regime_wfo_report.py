#!/usr/bin/env python3
"""
create_fvg_sniper_regime_wfo_report.py
==========================================
Segue la diagnosi del 2025 (analyze_fvg_sniper_2025.py): il segnale IFVG
Flip Only (1H, v2 anti-leva + posizioni concorrenti) funziona in trend
puliti (su o giù) e fallisce nei regimi di chop a bassa efficiency ratio
(2025: ER=0.0025, un ordine di grandezza sotto ogni altro anno). Qui si
aggiunge un FILTRO DI REGIME basato sull'efficiency ratio (ER = |mossa
netta| / somma dei movimenti assoluti, finestra rolling causale) e lo si
seleziona con WALK-FORWARD OPTIMIZATION — non un grid+pick-best sull'intera
storia, che sarebbe hindsight bias plateale dato che il 2025 "cattivo" è
già noto: ogni finestra IS (6 mesi) seleziona (lookback ER, soglia ER
minima) con lo Sharpe IS, applicata poi a barre OOS (2 mesi) mai viste
durante la selezione — stesso principio già usato per lo stop/target di
ICT Fade (create_ict_fade_wfo_report.py). NESSUNA correzione DSR
necessaria (selezione causale per finestra).

Griglia: lookback ER in [96, 168, 336] barre 1H (4/7/14 giorni) × soglia
ER minima in [0.0 (nessun filtro), 0.03, 0.05, 0.08, 0.12] = 15
combinazioni/finestra. Segnale/entry/stop/target/motore a posizioni
concorrenti identici a v2 (create_fvg_sniper_v2_report.py).
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
SL_BUF_ATR = 0.20
MIN_RISK_ATR = 0.40
MAX_RISK_ATR = 4.00
TP3_R = 3.0
MAX_TRADE_BARS = 120
GAP_MAX_AGE_BARS = 220

ER_WINDOW_GRID = [96, 168, 336]
ER_MIN_GRID = [0.0, 0.03, 0.05, 0.08, 0.12]
WF_TRAIN_M, WF_OOS_M, WF_STEP_M = 6, 2, 2
MIN_IS_TRADES = 15

report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

w(SEP)
w("FVG Sniper — Filtro di regime (efficiency ratio) via Walk-Forward Optimization")
w(SEP)
w("\nSegue la diagnosi del 2025 (chop estremo, ER=0.0025 vs 0.007-0.053 negli")
w("altri anni): filtro ER selezionato causalmente per finestra (WFO 6m IS/2m")
w("OOS), non scelto a posteriori sull'intera storia (hindsight bias).")

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
        evs.append(dict(idx=i, entry_i=entry_i, d=dd, ep=ep, sl=sl, target=target, risk=risk, hold=hold))
    return evs


print("[ENGINE] Detecting FVGs + IFVG Flip signals …")
fvgs = detect_graded_fvgs(HI, LO, OP, CL, VOL, ATR)
signals = run_engine_ifvg_only(fvgs, HI, LO, OP, CL, N, htf_bias)
all_cands = build_candidates(signals, HI, LO, OP, CL, ATR, N)
print(f"  {len(signals):,} segnali -> {len(all_cands):,} candidati (post filtro anti-leva)")

# ── Efficiency ratio rolling causale, per ciascuna finestra della griglia ──
print("[REGIME] Computing rolling efficiency ratio …")
abs_diff = np.abs(np.diff(CL, prepend=CL[0]))
er_by_w = {}
for W in ER_WINDOW_GRID:
    path_sum = pd.Series(abs_diff).rolling(W, min_periods=W).sum().values
    net_move = np.abs(CL - np.roll(CL, W))
    net_move[:W] = np.nan
    er = np.where(path_sum > 0, net_move / path_sum, np.nan)
    er_by_w[W] = er

for c in all_cands:
    c["er"] = {W: er_by_w[W][c["idx"]] for W in ER_WINDOW_GRID}


def run_bt_portfolio(evs, max_total_risk_pct=MAX_TOTAL_RISK_PCT):
    slip_pct = SLIPPAGE_BASE
    risk_budget_dollar = INIT_CAP * RISK_PCT
    max_total_risk_dollar = INIT_CAP * max_total_risk_pct
    if not evs:
        return dict(n=0, net_pnls=[], exits=[])
    by_entry: dict[int, list] = {}
    for ev in evs:
        by_entry.setdefault(ev["entry_i"], []).append(ev)
    min_i = min(by_entry); max_i = max(ev["entry_i"] + ev["hold"] for ev in evs)
    open_positions = []; total_open_risk = 0.0
    net_pnls = []; exits = []

    def close_position(pos, exit_price, exit_i):
        nonlocal total_open_risk
        dd, ep, risk = pos["d"], pos["ep"], pos["risk"]
        units = risk_budget_dollar / risk
        fill_ep = ep * (1 + dd * slip_pct); fill_xp = exit_price * (1 - dd * slip_pct)
        pnl = units * (fill_xp - fill_ep) * dd - FEE_TAKER * units * fill_ep - FEE_TAKER * units * fill_xp
        net_pnls.append(pnl)
        exits.append(dict(entry_ts=IDX[pos["entry_i"]], exit_ts=IDX[min(exit_i, N - 1)], pnl=pnl))
        total_open_risk -= risk_budget_dollar

    for i in range(min_i, max_i + 1):
        still_open = []
        for pos in open_positions:
            dd, sl, target, deadline_i = pos["d"], pos["sl"], pos["target"], pos["deadline_i"]
            hk, lk = HI[i], LO[i]
            hit_sl = (lk <= sl) if dd == 1 else (hk >= sl)
            hit_tp = (hk >= target) if dd == 1 else (lk <= target)
            if hit_sl: close_position(pos, sl, i); continue
            if hit_tp: close_position(pos, target, i); continue
            if i >= deadline_i: close_position(pos, CL[min(i, N - 1)], i); continue
            still_open.append(pos)
        open_positions = still_open
        for ev in by_entry.get(i, []):
            if total_open_risk + risk_budget_dollar > max_total_risk_dollar + 1e-9:
                continue
            open_positions.append(dict(entry_i=ev["entry_i"], d=ev["d"], ep=ev["ep"], sl=ev["sl"],
                                        target=ev["target"], risk=ev["risk"], deadline_i=ev["entry_i"] + ev["hold"]))
            total_open_risk += risk_budget_dollar
    for pos in open_positions:
        close_position(pos, CL[min(max_i, N - 1)], max_i)
    return dict(n=len(net_pnls), net_pnls=net_pnls, exits=exits)


def filter_cands(cands, W, er_min, t_start, t_end):
    out = []
    for c in cands:
        if not (t_start <= IDX[c["entry_i"]] < t_end):
            continue
        erv = c["er"][W]
        if er_min > 0.0 and (np.isnan(erv) or erv < er_min):
            continue
        out.append(c)
    return out


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
w(f"[WFO] Griglia: ER lookback {ER_WINDOW_GRID} x ER minima {ER_MIN_GRID} = "
  f"{len(ER_WINDOW_GRID)*len(ER_MIN_GRID)} combinazioni/finestra")

selection_log = []
all_oos_trades = []

for tr_s, tr_e, oo_s, oo_e in WF_WINDOWS:
    best = None
    for W in ER_WINDOW_GRID:
        for er_min in ER_MIN_GRID:
            is_cands = filter_cands(all_cands, W, er_min, tr_s, tr_e)
            if len(is_cands) < MIN_IS_TRADES:
                continue
            is_res = run_bt_portfolio(is_cands)
            if is_res["n"] < MIN_IS_TRADES:
                continue
            sharpe = trade_level_sharpe(is_res["net_pnls"])
            if best is None or sharpe > best["sharpe"]:
                best = dict(W=W, er_min=er_min, sharpe=sharpe, n_is=is_res["n"])
    if best is None:
        continue
    selection_log.append(dict(window_start=tr_s, **best))

    oos_cands = filter_cands(all_cands, best["W"], best["er_min"], oo_s, oo_e)
    oos_res = run_bt_portfolio(oos_cands)
    for pnl, ex in zip(oos_res["net_pnls"], oos_res["exits"]):
        all_oos_trades.append(dict(exit_ts=ex["exit_ts"], entry_ts=ex["entry_ts"], pnl=pnl,
                                    W=best["W"], er_min=best["er_min"]))
    print(".", end="", flush=True)

print()
sel_df = pd.DataFrame(selection_log)
w(f"\n[WFO] Selezione per finestra ({len(sel_df)} finestre valide):")
w(f"  ER lookback più scelto : {sel_df['W'].value_counts().to_dict() if not sel_df.empty else {}}")
w(f"  ER minima più scelta   : {sel_df['er_min'].value_counts().to_dict() if not sel_df.empty else {}}")

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
w("RISULTATI AGGREGATI — walk-forward OOS (filtro ER selezionato per finestra)")
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
w(f"      {'Year':>6}  {'n':>6}  {'Ret%':>8}  {'WR':>6}")
year_trades: dict[int, list] = {}
for t in all_oos_trades:
    year_trades.setdefault(t["entry_ts"].year, []).append(t["pnl"])
for yr in sorted(year_trades):
    ypnls = year_trades[yr]
    yret = sum(ypnls) / INIT_CAP * 100
    ywr = sum(1 for p in ypnls if p > 0) / len(ypnls) if ypnls else 0.0
    w(f"      {yr:>6}  {len(ypnls):>6}  {yret:>+7.1f}%  {ywr:>5.1%}")

holdout_trades = [t for t in all_oos_trades if t["entry_ts"] >= CUTOFF]
h_pnls = [t["pnl"] for t in holdout_trades]
h_n = len(h_pnls)
h_ret = sum(h_pnls) / INIT_CAP * 100 if h_n else 0.0
h_wr = sum(1 for p in h_pnls if p > 0) / h_n if h_n else 0.0
hmc = mc_summary(h_pnls)
hmc_blk = mc_block_summary(h_pnls)
w(f"\n  HOLDOUT GENUINO 2025-2026 (sub-slice del WFO-OOS): n={h_n}  wr={h_wr:.1%}  ret={h_ret:+.1f}%")
w(f"    MC i.i.d.  : pp={hmc['p_profit']:.3f}  pr={hmc['p_ruin']:.3f}")
w(f"    MC block   : pp={hmc_blk['p_profit']:.3f}  pr={hmc_blk['p_ruin']:.3f}")

# ── Confronto con v2 (nessun filtro di regime) ─────────────────────────
w(f"\n{SEP}")
w("CONFRONTO — WFO con filtro ER vs v2 senza filtro (create_fvg_sniper_v2_report.py)")
w(SEP)
w(f"\n                          n      Ret%      WR")
w(f"  WFO + filtro ER      {n:>6}   {ret_pct:>+6.1f}%   {wr:>5.1%}")
w(f"  v2 nessun filtro (rif.) 1995   +200.9%   33.2%   (full-sample, non-WFO, per riferimento)")

w(f"\n{SEP}\n[DONE]\n{SEP}")
out_path = Path("reports/fvg_sniper_regime_wfo.md")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("# FVG Sniper — Filtro di regime (efficiency ratio) via WFO\n\n```\n" +
                     "\n".join(report_lines) + "\n```\n", encoding="utf-8")
print(f"\n[DONE] {out_path}   (total runtime {time.time()-t0:.0f}s)")
