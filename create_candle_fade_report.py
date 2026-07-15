#!/usr/bin/env python3
"""
create_candle_fade_report.py
===============================
Variante FADE (mean-reversion) di create_candle_momentum_report.py.

La versione momentum (long dopo verde, short dopo rosso) ha fallito in modo
netto e uniforme su tutte le 8 varianti testate — ma con un pattern
interessante: win rate ben SOTTO il 50% (9.7% a 1 minuto), che implica
un segnale invertito. Questo script testa esattamente la strategia
speculare per vedere se quell'inversione è sfruttabile al netto delle fee:

  SHORT dopo una candela VERDE (close > open)  — fade del rialzo
  LONG  dopo una candela ROSSA (close < open)  — fade del ribasso
  Filtro volume opzionale: richiede che il volume della candela di segnale
  sia superiore alla media mobile delle 20 barre precedenti (soglia
  "volume sopra la media", causale — usa solo barre passate)
  Holding period fisso, testato per 1, 5, 10, 15 minuti

Regole:
  - Segnale confermato alla chiusura della barra i (colore + volume)
  - Entry causale: open della barra i+1 (prima barra tradeabile dopo il
    segnale, mai il close della stessa barra del segnale)
  - Exit: open della barra i+1+hold_min (uscita a tempo fisso, nessun
    TP/SL — è un puro test di persistenza/momentum direzionale)
  - Trade SEQUENZIALI, non sovrapposti: dopo un'entry, nessun nuovo
    segnale è considerato finché la posizione corrente non è chiusa
    (cooldown = hold_min barre) — permette un vero equity curve
    single-position, coerente con la pipeline usata in tutta la sessione,
    invece di migliaia di trade sovrapposti non tradeabili singolarmente

8 varianti totali: hold ∈ {1,5,10,15} × filtro_volume ∈ {No, Sì}.
Sizing: notional fisso = capitale iniziale (nessuna leva, nessuno stop —
non c'è un riferimento di rischio in $ senza SL, quindi si usa notional
fisso invece del consueto rischio-dollaro fisso). Fee 0.04%/lato.

Validazione: per-anno dal primo run, holdout 2025-2026 genuino, Monte Carlo
i.i.d.+block, DSR family N=8 (8 varianti testate, corregge per selection
bias su hold×filtro).
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

from src.strategy.data_fetcher import fetch_binance_vision_klines
from src.strategy.monte_carlo import (
    run_monte_carlo, run_monte_carlo_block, deflated_sharpe_ratio_family,
)

SEP = "═" * 78
START_YEAR = 2020
INIT_CAP = 100_000.0
NOTIONAL_FRAC = 1.0
FEE = 0.0004
CUTOFF = pd.Timestamp("2025-01-01")
N_SIMS = 1_000   # kept modest: MC's per-trade Python loop doesn't scale to this script's huge trade counts

HOLD_MIN_TESTS = [1, 5, 10, 15]
VOLUME_FILTER_TESTS = [False, True]
VOL_AVG_WINDOW = 20
SLIPPAGE_TESTS_BPS = [0, 2, 5, 10]

report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

w(SEP)
w("Candle Fade (short green / long red, volume filter, 1-15min hold) — Validation Pipeline")
w(SEP)

# ── DATA ─────────────────────────────────────────────────────────────────
t0 = time.time()
print("\n[DATA] Loading 1m …")
df1m = fetch_binance_vision_klines("1m", start_year=START_YEAR, start_month=1,
                                    workers=6, verbose=False)
IDX = df1m.index
N = len(df1m)
print(f"  1m: {N:,} bars  ({IDX[0].date()} → {IDX[-1].date()})  (loaded in {time.time()-t0:.0f}s)")

OP = df1m["open"].values.astype(float)
CL = df1m["close"].values.astype(float)
VOL = df1m["volume"].values.astype(float)
YEAR = np.array([t.year for t in IDX], dtype=int)

green = CL > OP
red = CL < OP
vol_avg_prior = pd.Series(VOL).rolling(VOL_AVG_WINDOW).mean().shift(1).values
vol_hi = VOL > vol_avg_prior

print(f"[SIGNALS] green={green.sum():,} ({green.mean():.1%})  red={red.sum():,} ({red.mean():.1%})  "
      f"vol_hi={np.nansum(vol_hi):,.0f}")


def run_bt(hold_min, volume_filter, slippage_pct=0.0):
    """Sequential, non-overlapping backtest. Returns dict with trades list.
    FADE: short after green (fade the up move), long after red (fade the down move)."""
    fade_short_sig = green & (vol_hi if volume_filter else True)
    fade_long_sig = red & (vol_hi if volume_filter else True)

    trades = []
    i = VOL_AVG_WINDOW + 1
    last_valid = N - hold_min - 2
    while i < last_valid:
        d = 0
        if fade_long_sig[i]:
            d = 1
        elif fade_short_sig[i]:
            d = -1
        if d == 0:
            i += 1
            continue
        entry_i = i + 1
        exit_i = entry_i + hold_min
        entry_price = OP[entry_i]
        exit_price = OP[exit_i]
        fill_ep = entry_price * (1 + d * slippage_pct)
        fill_xp = exit_price * (1 - d * slippage_pct)
        notional = INIT_CAP * NOTIONAL_FRAC
        units = notional / entry_price
        pnl = units * (fill_xp - fill_ep) * d - FEE * 2 * notional
        trades.append(dict(entry_i=entry_i, exit_i=exit_i, direction=d, pnl=pnl,
                            year=int(YEAR[i])))
        i = exit_i + 1   # cooldown: skip past the holding period, no overlap

    if not trades:
        return dict(n=0, wr=0.0, ret=0.0, mdd=0.0, net_pnls=[], trades=[])
    cap = INIT_CAP; peak = cap; mdd = 0.0; wins = 0; net_pnls = []
    for t in trades:
        cap += t["pnl"]
        peak = max(peak, cap)
        mdd = min(mdd, (cap - peak) / peak)
        wins += int(t["pnl"] > 0)
        net_pnls.append(t["pnl"])
    n = len(trades); wr = wins / n
    return dict(n=n, wr=wr, ret=(cap / INIT_CAP - 1) * 100, mdd=mdd * 100,
                net_pnls=net_pnls, trades=trades)


MC_MAX_TRADES = 1_500   # subsample for MC when n is huge — run_monte_carlo's per-trade Python
                         # loop is O(n_sims x n_trades), doesn't scale to this script's trade counts

def _maybe_subsample(pnls, seed=42):
    if len(pnls) <= MC_MAX_TRADES:
        return pnls
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pnls), size=MC_MAX_TRADES, replace=False)
    return [pnls[i] for i in idx]

def mc_summary(pnls):
    if len(pnls) < 5: return dict(p_profit=0.0, p_ruin=1.0)
    sub = _maybe_subsample(pnls)
    mc = run_monte_carlo(pd.DataFrame({"net_pnl": sub}), INIT_CAP, N_SIMS)
    return dict(p_profit=float(mc.get("p_profit", 0.0)), p_ruin=float(mc.get("p_ruin", 1.0)))

def mc_block_summary(pnls, block_size=50):
    if len(pnls) < 5: return dict(p_profit=0.0, p_ruin=1.0)
    sub = _maybe_subsample(pnls)
    mc = run_monte_carlo_block(pd.DataFrame({"net_pnl": sub}), INIT_CAP, N_SIMS, block_size=block_size)
    return dict(p_profit=float(mc.get("p_profit", 0.0)), p_ruin=float(mc.get("p_ruin", 1.0)))


results = {}
dsr_full = []
dsr_holdout = []

print("\n[RUN] Backtesting 8 varianti (hold x filtro volume) …")
t1 = time.time()
for hold_min in HOLD_MIN_TESTS:
    for volume_filter in VOLUME_FILTER_TESTS:
        key = (hold_min, volume_filter)
        res = run_bt(hold_min, volume_filter)
        results[key] = res
        print(f"  hold={hold_min:>2}min  vol_filter={str(volume_filter):>5}  n={res['n']:>7}  "
              f"wr={res['wr']:.1%}  ret={res['ret']:+.1f}%")
print(f"  done in {time.time()-t1:.0f}s")

for hold_min in HOLD_MIN_TESTS:
    for volume_filter in VOLUME_FILTER_TESTS:
        key = (hold_min, volume_filter)
        res = results[key]
        vf_label = "con filtro volume" if volume_filter else "senza filtro volume"

        w(f"\n{SEP}")
        w(f"HOLD = {hold_min} min   ({vf_label})")
        w(SEP)

        n = res["n"]
        wins = int(round(res["wr"] * n))
        pval = st.binomtest(wins, n, 0.5, alternative="greater").pvalue if n else 1.0
        mean_pnl_pct = (np.mean(res["net_pnls"]) / INIT_CAP * 100) if n else 0.0

        mc = mc_summary(res["net_pnls"])
        mc_blk = mc_block_summary(res["net_pnls"])
        w(f"\n  FULL-SAMPLE: n={n}  wr={res['wr']:.1%} (p={pval:.4f} vs 50%)  "
          f"ret={res['ret']:+.1f}%  mdd={res['mdd']:.1f}%  mean/trade={mean_pnl_pct:.4f}%")
        w(f"    MC i.i.d.  : pp={mc['p_profit']:.3f}  pr={mc['p_ruin']:.3f}")
        w(f"    MC block   : pp={mc_blk['p_profit']:.3f}  pr={mc_blk['p_ruin']:.3f}")

        w(f"\n  Breakdown per anno:")
        w(f"    {'Year':>6}  {'n':>8}  {'Ret%':>8}  {'WR':>6}")
        year_trades: dict[int, list] = {}
        for t in res["trades"]:
            year_trades.setdefault(t["year"], []).append(t)
        for yr in sorted(year_trades):
            yt = year_trades[yr]
            if len(yt) < 5: continue
            ycap = INIT_CAP
            wins_y = 0
            for tr in yt:
                ycap += tr["pnl"]
                wins_y += int(tr["pnl"] > 0)
            yret = (ycap / INIT_CAP - 1) * 100
            w(f"    {yr:>6}  {len(yt):>8}  {yret:>+7.1f}%  {wins_y/len(yt):>5.1%}")

        holdout_trades = [t for t in res["trades"] if IDX[t["entry_i"]] >= CUTOFF]
        if holdout_trades:
            hcap = INIT_CAP
            hwins = 0
            hpnls = []
            for tr in holdout_trades:
                hcap += tr["pnl"]; hwins += int(tr["pnl"] > 0); hpnls.append(tr["pnl"])
            hret = (hcap / INIT_CAP - 1) * 100
            hwr = hwins / len(holdout_trades)
        else:
            hret = 0.0; hwr = 0.0; hpnls = []
        hmc = mc_summary(hpnls)
        w(f"\n  HOLDOUT 2025-2026: n={len(holdout_trades)}  wr={hwr:.1%}  ret={hret:+.1f}%")
        w(f"    MC i.i.d.  : pp={hmc['p_profit']:.3f}  pr={hmc['p_ruin']:.3f}")

        dsr_full.append(dict(key=f"{hold_min}min_{'volfilt' if volume_filter else 'novol'}",
                              net_pnls=res["net_pnls"], ret=res["ret"]))
        dsr_holdout.append(dict(key=f"{hold_min}min_{'volfilt' if volume_filter else 'novol'}",
                                 net_pnls=hpnls, ret=hret))

# ── DSR family (8 varianti) ─────────────────────────────────────────────
w(f"\n{SEP}")
w("DSR — famiglia N=8 (4 hold x 2 filtro volume)")
w(SEP)
deflated_sharpe_ratio_family(dsr_full, sharpe_key="sharpe_hat", dsr_key="dsr", pnls_key="net_pnls")
deflated_sharpe_ratio_family(dsr_holdout, sharpe_key="sharpe_hat", dsr_key="dsr", pnls_key="net_pnls")
w(f"\n  Full-sample:")
w(f"  {'Variante':>20}  {'n':>8}  {'Ret%':>8}  {'Sharpe_hat':>11}  {'DSR':>7}")
for r in dsr_full:
    w(f"  {r['key']:>20}  {len(r['net_pnls']):>8}  {r['ret']:>+7.1f}%  {r['sharpe_hat']:>11.3f}  {r['dsr']:>7.3f}")
w(f"\n  Holdout 2025-2026:")
w(f"  {'Variante':>20}  {'n':>8}  {'Ret%':>8}  {'Sharpe_hat':>11}  {'DSR':>7}")
for r in dsr_holdout:
    w(f"  {r['key']:>20}  {len(r['net_pnls']):>8}  {r['ret']:>+7.1f}%  {r['sharpe_hat']:>11.3f}  {r['dsr']:>7.3f}")

# ── Slippage sensitivity on the best variant (by full-sample DSR) ────────
best = max(dsr_full, key=lambda r: r["dsr"])
best_hold = int(best["key"].split("min_")[0])
best_vf = "volfilt" in best["key"]
w(f"\n{SEP}")
w(f"Slippage sensitivity — variante migliore per DSR = hold={best_hold}min, "
  f"vol_filter={best_vf}")
w(SEP)
w(f"\n  {'Slippage':>10}  {'Scope':>10}  {'n':>8}  {'Ret%':>8}  {'WR':>6}  {'MC pp':>7}  {'MC pr':>7}")
for bps in SLIPPAGE_TESTS_BPS:
    slip = bps / 10_000.0
    res_slip = run_bt(best_hold, best_vf, slippage_pct=slip)
    m = mc_summary(res_slip["net_pnls"])
    holdout_slip = [t["pnl"] for t in res_slip["trades"] if IDX[t["entry_i"]] >= CUTOFF]
    m_h = mc_summary(holdout_slip)
    w(f"  {bps:>7}bps  {'full-sample':>10}  {res_slip['n']:>8}  {res_slip['ret']:>+7.1f}%  "
      f"{res_slip['wr']:>5.1%}  {m['p_profit']:>6.3f}  {m['p_ruin']:>6.3f}")
    hret_slip = (INIT_CAP + sum(holdout_slip)) / INIT_CAP * 100 - 100 if holdout_slip else 0.0
    hwr_slip = (sum(1 for p in holdout_slip if p > 0) / len(holdout_slip)) if holdout_slip else 0.0
    w(f"  {bps:>7}bps  {'holdout':>10}  {len(holdout_slip):>8}  {hret_slip:>+7.1f}%  "
      f"{hwr_slip:>5.1%}  {m_h['p_profit']:>6.3f}  {m_h['p_ruin']:>6.3f}")

w(f"\n{SEP}\n[DONE]\n{SEP}")
out_path = Path("reports/candle_fade.md")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("# Candle Fade (short green / long red, volume filter, 1-15min hold) — Validation Pipeline\n\n```\n" +
                     "\n".join(report_lines) + "\n```\n", encoding="utf-8")
print(f"\n[DONE] {out_path}   (total runtime {time.time()-t0:.0f}s)")
