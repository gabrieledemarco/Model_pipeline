#!/usr/bin/env python3
"""
create_grid_trading_report.py
================================
Grid Trading (spot-style, long-only) sul range della candela precedente
(settimana o giorno): il classico "Grid Trading Bot" retail (Binance/
Bybit/Pionex) — divide il range in N livelli equispaziati, compra ad ogni
livello quando il prezzo scende, vende al livello successivo quando il
prezzo risale, ripetendo indipendentemente per ogni cella del grid finché
il prezzo resta nel range.

Costruzione (causale, nessun look-ahead):
  1. RANGE: [low, high] del periodo PRECEDENTE (settimana o giorno, a
     seconda della variante) applicato come griglia per il periodo
     CORRENTE — il range del periodo N è noto per intero solo a fine
     periodo N, quindi diventa disponibile SOLO a partire dal periodo N+1.
  2. LEVELS: N_GRID+1 livelli equispaziati tra range_lo e range_hi ->
     N_GRID "celle" [L_j, L_j+1].
  3. Per ogni cella, indipendentemente: se il prezzo scende fino a L_j e
     la cella non è già "in posizione" -> BUY a L_j (ordine limite,
     fill esatto al livello). Se il prezzo poi sale fino a L_j+1 e la
     cella è in posizione -> SELL a L_j+1, il trade si chiude, la cella
     torna libera e può ri-comprare a L_j se il prezzo scende di nuovo
     nello stesso periodo.
  4. STOP DI SICUREZZA (invalidazione del grid): se il prezzo rompe sotto
     range_lo - STOP_BUFFER_ATR×ATR, tutte le celle aperte vengono
     chiuse forzatamente a quel prezzo (mercato) e il grid si disattiva
     per il resto del periodo — protegge dal rischio classico del grid
     trading ("comprare un coltello che cade" in un breakdown).
  5. FINE PERIODO: le celle ancora aperte vengono chiuse forzatamente
     alla chiusura dell'ultima barra del periodo (mark-to-market),
     poi si ricalcola un nuovo range dal periodo appena concluso.

Fee: i fill a livello griglia sono ordini LIMITE (maker, 0.02%/lato,
realistico per grid bot). Le chiusure forzate (stop di sicurezza o fine
periodo) sono a mercato (taker, 0.055%/lato). Scenario "tutto taker"
mostrato a parte per confronto conservativo.

Varianti testate: WEEKLY (barre 1H, range = settimana precedente) e
DAILY (barre 15m, range = giorno precedente). Griglia N_GRID scansionata
[5, 10, 20] con DSR family N=3 per variante.

Limite dichiarato: il drawdown/equity curve riflette solo il PnL
realizzato ai fill/chiusure (non mark-to-market delle celle aperte
durante il periodo) — stessa semplificazione usata altrove in sessione
per il conteggio strettamente sequenziale dei trade.
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
CAPITAL_PER_RUNG_PCT = 0.01     # 1% di INIT_CAP per cella, sizing fisso non-compounding
MAX_LEV = 10.0
FEE_MAKER = 0.00020              # fill a livello griglia (ordine limite)
FEE_TAKER = 0.00055              # chiusure forzate (stop / fine periodo)
CUTOFF = pd.Timestamp("2025-01-01")
N_SIMS = 5_000
MC_MAX_TRADES = 1_500   # grid trading produces very large trade counts (>100k for
MC_N_SIMS = 1_000       # fine grids); subsample for MC as done for candle-momentum/VWAP-trend
N_GRID_GRID = [5, 10, 20]
STOP_BUFFER_ATR = 0.25
SLIPPAGE_TESTS_BPS = [0, 2, 5, 10]

report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

w(SEP)
w("Grid Trading su range settimana/giorno precedente — Validation Pipeline")
w(SEP)
w("\nGrid long-only classico: N livelli nel range del periodo precedente,")
w("buy-low/sell-high per cella indipendente. Stop di sicurezza sotto il")
w("range, chiusura forzata a fine periodo. Fee maker su fill griglia,")
w("taker sulle chiusure forzate.")

# ── DATA ─────────────────────────────────────────────────────────────────
t0 = time.time()
print("\n[DATA] Loading 1H + 15M …")
raw = fetch_extended_data(start_year=START_YEAR, start_month=1,
                           fetch_15m=True, fetch_1m=False, fetch_flow=False)
df1h = add_indicators(raw["1H"])
df15 = add_indicators(raw["15M"])
print(f"  1H:  {len(df1h):,} bars  ({df1h.index[0].date()} → {df1h.index[-1].date()})")
print(f"  15M: {len(df15):,} bars  ({df15.index[0].date()} → {df15.index[-1].date()})  "
      f"(loaded in {time.time()-t0:.0f}s)")


def run_variant(df, period_freq_or_norm, label):
    IDX = df.index
    N = len(df)
    CL = df["close"].values.astype(float)
    HI = df["high"].values.astype(float)
    LO = df["low"].values.astype(float)
    ATR = np.where(df["atr_14"].values > 0, df["atr_14"].values, 1.0)

    if period_freq_or_norm == "W":
        period_ids = IDX.to_period("W").values
    else:
        period_ids = IDX.normalize().values

    lo_map: dict = {}
    hi_map: dict = {}
    tmp = pd.DataFrame({"period": period_ids, "hi": HI, "lo": LO})
    g = tmp.groupby("period")
    lo_map = g["lo"].min().to_dict()
    hi_map = g["hi"].max().to_dict()

    def run_bt(n_grid, fee_grid=FEE_MAKER, fee_force=FEE_TAKER, slippage_pct=0.0):
        trades = []  # (bar_idx, pnl)
        holding = np.zeros(n_grid, dtype=bool)
        buy_price = np.zeros(n_grid)
        levels = None
        grid_active = False
        stop_level = -np.inf
        cur_period = None
        capital_notional = INIT_CAP * CAPITAL_PER_RUNG_PCT
        n_buy = n_sl_force = n_period_force = 0

        def close_all(price, bar_i, fee_pct, tag):
            nonlocal n_sl_force, n_period_force
            for j in range(n_grid):
                if holding[j]:
                    units = min(capital_notional / buy_price[j], MAX_LEV * INIT_CAP / buy_price[j])
                    fill_x = price * (1 - slippage_pct)
                    fill_e = buy_price[j] * (1 + slippage_pct)
                    pnl = units * (fill_x - fill_e) - fee_pct * units * fill_e - fee_pct * units * fill_x
                    trades.append((bar_i, pnl))
                    holding[j] = False
                    if tag == "sl": n_sl_force += 1
                    else: n_period_force += 1

        for i in range(N):
            if period_ids[i] != cur_period:
                if cur_period is not None:
                    close_all(CL[i - 1], i - 1, fee_force, "period")
                cur_period = period_ids[i]
                # nuovo range dal periodo appena concluso (quello precedente a cur_period)
                rng_lo = lo_map.get(period_ids[i - 1], np.nan) if i > 0 else np.nan
                rng_hi = hi_map.get(period_ids[i - 1], np.nan) if i > 0 else np.nan
                if i == 0 or not np.isfinite(rng_lo) or not np.isfinite(rng_hi) or rng_hi <= rng_lo:
                    grid_active = False
                    levels = None
                else:
                    levels = np.linspace(rng_lo, rng_hi, n_grid + 1)
                    stop_level = rng_lo - STOP_BUFFER_ATR * ATR[i]
                    grid_active = True
                    holding[:] = False

            if not grid_active or levels is None:
                continue

            if LO[i] < stop_level:
                close_all(stop_level, i, fee_force, "sl")
                grid_active = False
                continue

            for j in range(n_grid):
                if not holding[j] and LO[i] <= levels[j]:
                    holding[j] = True
                    buy_price[j] = levels[j]
                    n_buy += 1
                if holding[j] and HI[i] >= levels[j + 1]:
                    units = min(capital_notional / buy_price[j], MAX_LEV * INIT_CAP / buy_price[j])
                    fill_x = levels[j + 1] * (1 - slippage_pct)
                    fill_e = buy_price[j] * (1 + slippage_pct)
                    pnl = units * (fill_x - fill_e) - fee_grid * units * fill_e - fee_grid * units * fill_x
                    trades.append((i, pnl))
                    holding[j] = False

        if N > 0:
            close_all(CL[N - 1], N - 1, fee_force, "period")

        trades.sort(key=lambda t: t[0])
        net_pnls = [p for _, p in trades]
        bar_idx_of_trade = [b for b, _ in trades]
        cap = INIT_CAP; peak = cap; mdd = 0.0; wins = 0
        for pnl in net_pnls:
            cap += pnl
            peak = max(peak, cap)
            mdd = min(mdd, (cap - peak) / peak)
            wins += int(pnl > 0)
        n = len(net_pnls); wr = wins / n if n else 0.0
        return dict(n=n, wr=wr, ret=(cap / INIT_CAP - 1) * 100, mdd=mdd * 100, net_pnls=net_pnls,
                    bar_idx=bar_idx_of_trade, n_buy=n_buy, n_sl_force=n_sl_force, n_period_force=n_period_force)

    def _maybe_subsample(pnls, seed=42):
        if len(pnls) <= MC_MAX_TRADES:
            return pnls
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pnls), size=MC_MAX_TRADES, replace=False)
        return [pnls[k] for k in idx]

    def mc_summary(pnls):
        if len(pnls) < 5: return dict(p_profit=0.0, p_ruin=1.0)
        sub = _maybe_subsample(pnls)
        mc = run_monte_carlo(pd.DataFrame({"net_pnl": sub}), INIT_CAP, MC_N_SIMS)
        return dict(p_profit=float(mc.get("p_profit", 0.0)), p_ruin=float(mc.get("p_ruin", 1.0)))

    def mc_block_summary(pnls, block_size=10):
        if len(pnls) < 5: return dict(p_profit=0.0, p_ruin=1.0)
        sub = _maybe_subsample(pnls)
        mc = run_monte_carlo_block(pd.DataFrame({"net_pnl": sub}), INIT_CAP, MC_N_SIMS, block_size=block_size)
        return dict(p_profit=float(mc.get("p_profit", 0.0)), p_ruin=float(mc.get("p_ruin", 1.0)))

    w(f"\n{SEP}")
    w(f"VARIANTE: {label}")
    w(SEP)

    dsr_full = []
    dsr_holdout = []
    for n_grid in N_GRID_GRID:
        w(f"\n{'─'*78}")
        w(f"N_GRID = {n_grid}")
        w(f"{'─'*78}")

        res = run_bt(n_grid)
        n = res["n"]
        pval = st.binomtest(int(round(res["wr"] * n)), n, 0.5, alternative="greater").pvalue if n else 1.0
        mc = mc_summary(res["net_pnls"])
        mc_blk = mc_block_summary(res["net_pnls"])
        w(f"\n  FULL-SAMPLE: n={n}  wr={res['wr']:.1%} (p={pval:.4f} vs 50%)  "
          f"ret={res['ret']:+.1f}%  mdd={res['mdd']:.1f}%")
        w(f"    buy={res['n_buy']}  chiusure-stop={res['n_sl_force']}  chiusure-fine-periodo={res['n_period_force']}")
        w(f"    MC i.i.d.  : pp={mc['p_profit']:.3f}  pr={mc['p_ruin']:.3f}")
        w(f"    MC block   : pp={mc_blk['p_profit']:.3f}  pr={mc_blk['p_ruin']:.3f}")

        w(f"\n  Breakdown per anno:")
        w(f"    {'Year':>6}  {'n':>6}  {'Ret%':>8}  {'WR':>6}")
        year_pnls: dict[int, list] = {}
        for b, p in zip(res["bar_idx"], res["net_pnls"]):
            year_pnls.setdefault(IDX[b].year, []).append(p)
        for yr in sorted(year_pnls):
            yp = year_pnls[yr]
            if len(yp) < 5: continue
            ycap = INIT_CAP
            for p in yp: ycap += p
            ywins = sum(1 for p in yp if p > 0)
            w(f"    {yr:>6}  {len(yp):>6}  {(ycap/INIT_CAP-1)*100:>+7.1f}%  {ywins/len(yp):>5.1%}")

        hold_pnls = [p for b, p in zip(res["bar_idx"], res["net_pnls"]) if IDX[b] >= CUTOFF]
        hcap = INIT_CAP
        for p in hold_pnls: hcap += p
        hwins = sum(1 for p in hold_pnls if p > 0)
        hwr = hwins / len(hold_pnls) if hold_pnls else 0.0
        hmc = mc_summary(hold_pnls)
        hmc_blk = mc_block_summary(hold_pnls)
        w(f"\n  HOLDOUT GENUINO 2025-2026: n={len(hold_pnls)}  wr={hwr:.1%}  "
          f"ret={(hcap/INIT_CAP-1)*100:+.1f}%")
        w(f"    MC i.i.d.  : pp={hmc['p_profit']:.3f}  pr={hmc['p_ruin']:.3f}")
        w(f"    MC block   : pp={hmc_blk['p_profit']:.3f}  pr={hmc_blk['p_ruin']:.3f}")

        dsr_full.append(dict(n_grid=n_grid, net_pnls=res["net_pnls"], ret=res["ret"]))
        dsr_holdout.append(dict(n_grid=n_grid, net_pnls=hold_pnls, ret=(hcap/INIT_CAP-1)*100))

    w(f"\n{SEP}")
    w(f"DSR — famiglia N={len(N_GRID_GRID)} (griglia N_GRID) — {label}")
    w(SEP)
    deflated_sharpe_ratio_family(dsr_full, sharpe_key="sharpe_hat", dsr_key="dsr", pnls_key="net_pnls")
    deflated_sharpe_ratio_family(dsr_holdout, sharpe_key="sharpe_hat", dsr_key="dsr", pnls_key="net_pnls")
    w(f"\n  Full-sample:")
    w(f"  {'N_GRID':>7}  {'n':>6}  {'Ret%':>8}  {'Sharpe_hat':>11}  {'DSR':>7}")
    for r in dsr_full:
        w(f"  {r['n_grid']:>7}  {len(r['net_pnls']):>6}  {r['ret']:>+7.1f}%  {r['sharpe_hat']:>11.3f}  {r['dsr']:>7.3f}")
    w(f"\n  Holdout 2025-2026:")
    w(f"  {'N_GRID':>7}  {'n':>6}  {'Ret%':>8}  {'Sharpe_hat':>11}  {'DSR':>7}")
    for r in dsr_holdout:
        w(f"  {r['n_grid']:>7}  {len(r['net_pnls']):>6}  {r['ret']:>+7.1f}%  {r['sharpe_hat']:>11.3f}  {r['dsr']:>7.3f}")

    best_n = max(dsr_full, key=lambda r: r["dsr"])["n_grid"]
    w(f"\n{SEP}")
    w(f"Fee/slippage sensitivity — N_GRID migliore per DSR = {best_n} — {label}")
    w(SEP)
    w(f"\n  {'Scenario':>28}  {'Scope':>10}  {'n':>6}  {'Ret%':>8}  {'WR':>6}  {'MC pp':>7}  {'MC pr':>7}")
    scenarios = [
        ("Mix maker/taker, 0bps slip", FEE_MAKER, FEE_TAKER, 0.0),
        ("Tutto taker, 0bps slip", FEE_TAKER, FEE_TAKER, 0.0),
        ("Mix maker/taker, 2bps slip", FEE_MAKER, FEE_TAKER, 0.0002),
        ("Mix maker/taker, 5bps slip", FEE_MAKER, FEE_TAKER, 0.0005),
        ("Mix maker/taker, 10bps slip", FEE_MAKER, FEE_TAKER, 0.0010),
    ]
    for sname, fg, ff, slip in scenarios:
        r = run_bt(best_n, fee_grid=fg, fee_force=ff, slippage_pct=slip)
        hp = [p for b, p in zip(r["bar_idx"], r["net_pnls"]) if IDX[b] >= CUTOFF]
        for scope_name, pnls_scope, n_scope in [("full-sample", r["net_pnls"], r["n"]), ("holdout", hp, len(hp))]:
            if not pnls_scope:
                w(f"  {sname:>28}  {scope_name:>10}  {0:>6}  {'n/a':>8}  {'n/a':>6}  {'n/a':>7}  {'n/a':>7}")
                continue
            ccap = INIT_CAP
            for p in pnls_scope: ccap += p
            cwins = sum(1 for p in pnls_scope if p > 0)
            m = mc_summary(pnls_scope)
            w(f"  {sname:>28}  {scope_name:>10}  {n_scope:>6}  {(ccap/INIT_CAP-1)*100:>+7.1f}%  "
              f"{cwins/n_scope:>5.1%}  {m['p_profit']:>6.3f}  {m['p_ruin']:>6.3f}")

    return dict(dsr_full=dsr_full, dsr_holdout=dsr_holdout, best_n=best_n)


results_weekly = run_variant(df1h, "W", "WEEKLY (barre 1H, range = settimana precedente)")
results_daily = run_variant(df15, "D", "DAILY (barre 15m, range = giorno precedente)")

w(f"\n{SEP}\n[DONE]\n{SEP}")
out_path = Path("reports/grid_trading.md")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("# Grid Trading su range settimana/giorno precedente — Validation Pipeline\n\n```\n" +
                     "\n".join(report_lines) + "\n```\n", encoding="utf-8")
print(f"\n[DONE] {out_path}   (total runtime {time.time()-t0:.0f}s)")
