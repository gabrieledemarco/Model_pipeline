"""
Full pipeline → HTML report.
Runs: data fetch (cache), indicators, signals (real basis OI + 15m + 1m layers),
backtest, WFO, Monte Carlo (in-sample + OOS), scenarios, leverage grid.
Writes a single self-contained HTML report.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.strategy.data_fetcher   import (fetch_extended_data, fetch_real_funding,
                                          fetch_real_oi, generate_oi)
from src.strategy.indicators     import add_indicators
from src.strategy.signals        import build_signal_matrix
from src.strategy.engine         import run_backtest, INIT_CAP, _compute_kpis
from src.strategy.optimizer      import SCENARIOS, run_comparison, apply_filters
from src.strategy.monte_carlo    import run_monte_carlo
from src.strategy.walk_forward   import run_walk_forward
from src.strategy.leverage_study import run_leverage_grid
from src.strategy.report_html    import generate_html

SCENARIO      = "Session 08-21"
MC_SIMS       = 1000
OUT_PATH      = Path("reports/BTCUSDT_Strategy_Report.html")

t0 = time.time()
print("═" * 66)
print("  BTCUSDT Strategy — HTML Report Pipeline")
print("═" * 66)

# ── 1. Data ──────────────────────────────────────────────────────────────────
print("\n[1/9] Fetching data (from cache) …")
tf_data = fetch_extended_data(fetch_15m=True, fetch_1m=True)

# ── 2. Indicators ────────────────────────────────────────────────────────────
print("\n[2/9] Computing indicators …")
for tf in tf_data:
    if not tf_data[tf].empty:
        tf_data[tf] = add_indicators(tf_data[tf])

df_1h  = tf_data["1H"]
df_15m = tf_data.get("15M", pd.DataFrame())
df_1m  = tf_data.get("1M",  pd.DataFrame())

if not df_15m.empty:
    print(f"  [15M]  {len(df_15m):8,d} bars  "
          f"[{df_15m.index[0].date()} → {df_15m.index[-1].date()}]")
if not df_1m.empty:
    print(f"  [ 1M]  {len(df_1m):8,d} bars  "
          f"[{df_1m.index[0].date()} → {df_1m.index[-1].date()}]")

# ── 3. OI (real basis) + Funding ─────────────────────────────────────────────
print("\n[3/9] Loading real basis (premiumIndexKlines) + funding …")
premium_1h, oi_is_real = fetch_real_oi(df_1h)
oi_source = "real_basis (Binance Vision premiumIndexKlines)" if oi_is_real else "synthetic"
print(f"  OI/Basis : {oi_source}")
if oi_is_real:
    print(f"             {len(premium_1h):,} bars  "
          f"[{premium_1h.index[0].date()} → {premium_1h.index[-1].date()}]  "
          f"range=[{premium_1h.min()*100:.3f}%, {premium_1h.max()*100:.3f}%]")

oi_df = generate_oi(tf_data["1D"]["close"])

funding, is_real = fetch_real_funding(df_1h, tf_data["1D"])
print(f"  Funding  : {'real' if is_real else 'synthetic'}  "
      f"mean={funding.mean()*100:.4f}%")

# ── 4. Signals ────────────────────────────────────────────────────────────────
print("\n[4/9] Building signal matrix (10 components: +15m +1m) …")
signals = build_signal_matrix(
    tf_data,
    oi_df,
    funding,
    premium_1h = premium_1h if oi_is_real else None,
    df_15m     = df_15m if not df_15m.empty else None,
    df_1m      = df_1m  if not df_1m.empty  else None,
)
n_long  = int((signals["signal"] ==  1).sum())
n_short = int((signals["signal"] == -1).sum())
n_flat  = int((signals["signal"] ==  0).sum())
has_15m = bool(signals["has_15m"].iloc[0])
has_1m  = bool(signals["has_1m"].iloc[0])
print(f"  Signals  : long={n_long:,}  short={n_short:,}  flat={n_flat:,}  "
      f"[15m={'✓' if has_15m else '✗'}  1m={'✓' if has_1m else '✗'}]")

# ── 5. Backtest (baseline) ───────────────────────────────────────────────────
print(f"\n[5/9] Backtest ({SCENARIO}) …")
cfg      = SCENARIOS[SCENARIO]
sig_filt = apply_filters(signals, cfg)
common   = df_1h.index.intersection(sig_filt.index)
bt       = run_backtest(df_1h.reindex(common), sig_filt.reindex(common), INIT_CAP)
bt["init_cap"] = INIT_CAP
kpis     = _compute_kpis(bt["equity"], bt["drawdown"], bt["trades"], INIT_CAP)
print(f"  Return={kpis['total_return']*100:+.1f}%  "
      f"Sharpe={kpis['sharpe']:.3f}  "
      f"DD={kpis['max_drawdown']*100:.1f}%  "
      f"Trades={kpis['n_trades']}")

# ── 6. Scenario comparison ───────────────────────────────────────────────────
print("\n[6/9] Scenario comparison …")
scenario_kpis: dict = {}
for sc_name, sc_cfg in SCENARIOS.items():
    try:
        sf    = apply_filters(signals, sc_cfg)
        cm    = df_1h.index.intersection(sf.index)
        bt_sc = run_backtest(df_1h.reindex(cm), sf.reindex(cm), INIT_CAP)
        scenario_kpis[sc_name] = _compute_kpis(
            bt_sc["equity"], bt_sc["drawdown"], bt_sc["trades"], INIT_CAP)
        print(f"  {sc_name:<22} ret={scenario_kpis[sc_name]['total_return']*100:+.1f}%  "
              f"Sharpe={scenario_kpis[sc_name]['sharpe']:.2f}")
    except Exception as e:
        print(f"  {sc_name:<22} FAILED: {e}")

# ── 7. Monte Carlo (in-sample) ───────────────────────────────────────────────
print(f"\n[7/9] Monte Carlo in-sample ({MC_SIMS} sims) …")
trades_is  = bt["trades"]
mc_insample: dict = {}
if not trades_is.empty:
    mc_raw    = run_monte_carlo(trades_is, n_sims=MC_SIMS, initial_capital=INIT_CAP)
    final_ret = mc_raw["total_return"]
    paths_df  = pd.DataFrame(mc_raw["paths"].T)
    mc_insample = {
        "sim_equity":      paths_df,
        "original_equity": bt["equity"],
        "final_returns":   final_ret,
    }
    p_profit = float((final_ret > 0).mean() * 100)
    p_ruin   = float((final_ret < -0.5).mean() * 100)
    print(f"  P(profit)={p_profit:.1f}%  p50={np.percentile(final_ret,50)*100:+.1f}%  "
          f"P(ruin)={p_ruin:.1f}%")

# ── 8. Walk-Forward ──────────────────────────────────────────────────────────
print("\n[8/9] Walk-Forward validation …")
wf_result = run_walk_forward(df_1h, signals, scenario_name=SCENARIO)

# Monte Carlo on OOS trades
mc_oos: dict = {}
oos_trades = wf_result.get("all_oos_trades", pd.DataFrame())
if not oos_trades.empty and len(oos_trades) >= 20:
    print(f"\n  MC on OOS trades ({len(oos_trades)} trades, {MC_SIMS} sims) …")
    mc_oos_raw = run_monte_carlo(oos_trades, n_sims=MC_SIMS, initial_capital=INIT_CAP)
    oos_final  = mc_oos_raw["total_return"]
    oos_paths  = pd.DataFrame(mc_oos_raw["paths"].T)
    mc_oos = {
        "sim_equity":      oos_paths,
        "original_equity": wf_result.get("combined_equity"),
        "final_returns":   oos_final,
    }
    p2 = float((oos_final > 0).mean() * 100)
    print(f"  P(profit)={p2:.1f}%  p50={np.percentile(oos_final,50)*100:+.1f}%")

# ── 9. Leverage grid ─────────────────────────────────────────────────────────
print("\n[9/9] Leverage grid …")
lev_comp_df, lev_equity_store = run_leverage_grid(df_1h, sig_filt)
lev_result = lev_comp_df

# ── HTML report ──────────────────────────────────────────────────────────────
print("\nGenerating HTML report …")
out = generate_html(
    bt_result     = bt,
    kpis          = kpis,
    wf_result     = wf_result,
    mc_insample   = mc_insample,
    mc_oos        = mc_oos,
    scenario_kpis = scenario_kpis,
    lev_result    = lev_result,
    df_1h         = df_1h,
    oi_source     = oi_source,
    scenario_name = SCENARIO,
    out_path      = OUT_PATH,
)
size_mb = out.stat().st_size / 1e6
print(f"  ✓ Saved → {out}  ({size_mb:.1f} MB)")

elapsed = time.time() - t0
print(f"\n{'═'*66}")
print(f"  Done in {elapsed:.1f}s  ·  {len(df_1h):,} bars  ·  {out}")
print(f"{'═'*66}")
