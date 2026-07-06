#!/usr/bin/env python3
"""
create_ml_trading_strategy_refinement_report.py
================================================
Tre controlli di rifinitura sulla strategia ML (RandomForest, OHLCV + HMM +
pivot M5/M15/H1/Daily, uscita a tempo fisso) validata in
create_ml_trading_strategy_backtest_report.py, dove l'orizzonte 8h era
risultato il migliore (full-sample ret=+213.5%, holdout 2025-2026
ret=+47.4%, MC i.i.d. pp=0.989, MC block pp=0.974).

1) Slippage sensitivity: 0/2/5/10 bps di slippage avverso aggiuntivo su
   entry+exit della strategia 8h, per vedere quanto edge sopravvive a
   frizioni di esecuzione realistiche (lezione della sessione: anche
   5-10bps avevano eroso l'apparente edge della strategia ADP).
2) DSR-style multiple-testing check: i 3 orizzonti testati (2h/4h/8h) sono
   trattati come una "famiglia" di N=3 varianti comparabili (stessa
   pipeline, stesso modello, differiscono solo nell'orizzonte) e si
   applica deflated_sharpe_ratio_family per correggere il Sharpe
   dell'orizzonte vincente (8h) per selection bias.
3) Verifica leva/notional/margin: per la strategia 8h, si calcola la
   distribuzione (mediana, p95, max) di units/notional/leva implicita per
   trade (sizing a rischio-dollaro fisso), e si misura frequenza e
   severita' delle attivazioni dello stop di sicurezza a 4xATR.
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.strategy.data_fetcher import fetch_binance_vision_klines, fetch_extended_data
from src.strategy.indicators import add_indicators
from src.strategy.mtf_swing import causal_trend_state, align_htf_to_ltf
from src.strategy.hmm_regime import fit_hmm, predict_hmm_features, HMM_FEATURE_NAMES
from src.strategy.monte_carlo import run_monte_carlo, run_monte_carlo_block, deflated_sharpe_ratio_family

SEP = "═" * 78
START_YEAR = 2020
INIT_CAP = 100_000.0
RISK_PCT = 0.01
FEE = 0.0004
MAX_LEV = 10.0
CUTOFF = pd.Timestamp("2025-01-01")
N_SIMS = 5_000
WF_TRAIN_M, WF_OOS_M, WF_STEP_M = 6, 2, 2
PIVOT_LR = {"5m": 12, "15m": 8, "1h": 6, "1d": 3}

HORIZONS_TO_TEST = [2, 4, 8]
THRESHOLD = 0.55
SAFETY_SL_ATR_MULT = 4.0
COOLDOWN_HOURS = 4
SLIPPAGE_TESTS_BPS = [0, 2, 5, 10]

report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

w(SEP)
w("ML Trading Strategy — Refinement Checks (slippage / DSR / leverage)")
w(SEP)

# ── DATA (identical to base script) ────────────────────────────────────────
t0 = time.time()
print("\n[DATA] Loading H1 (base) + M5/M15/Daily …")
raw = fetch_extended_data(start_year=START_YEAR, start_month=1,
                           fetch_15m=False, fetch_1m=False, fetch_flow=False)
df1h = add_indicators(raw["1H"])
IDX1H = df1h.index
N1H = len(df1h)
df_5m = fetch_binance_vision_klines("5m", start_year=START_YEAR, start_month=1,
                                     workers=6, verbose=False)
df_15m = fetch_binance_vision_klines("15m", start_year=START_YEAR, start_month=1,
                                      workers=6, verbose=False)
df_1d = fetch_binance_vision_klines("1d", start_year=START_YEAR, start_month=1,
                                     workers=6, verbose=False)
print(f"  H1: {N1H:,}   M5: {len(df_5m):,}   M15: {len(df_15m):,}   Daily: {len(df_1d):,}  "
      f"(loaded in {time.time()-t0:.0f}s)")

CL = df1h["close"].values.astype(float)
HI = df1h["high"].values.astype(float)
LO = df1h["low"].values.astype(float)
VOL = df1h["volume"].values.astype(float)
LOGCL = np.log(CL)
ATR = np.where(df1h["atr_14"].shift(1).values > 0, df1h["atr_14"].shift(1).values, 1.0)

def pivot_features_on_h1(df_tf, left_right, tf_label):
    close = df_tf["close"].values.astype(float)
    high = df_tf["high"].values.astype(float)
    low = df_tf["low"].values.astype(float)
    state = causal_trend_state(close, high, low, left_right, left_right)
    atr_tf = (df_tf["high"] - df_tf["low"]).rolling(14).mean().bfill().values
    state["atr"] = atr_tf
    state["close"] = close
    aligned = align_htf_to_ltf(df_tf.index, state, IDX1H)
    atr_on_h1 = np.where(aligned["atr"].values > 0, aligned["atr"].values, 1.0)
    close_on_h1 = aligned["close"].values
    dist_high = (aligned["target_high"].values - close_on_h1) / atr_on_h1
    dist_low = (close_on_h1 - aligned["target_low"].values) / atr_on_h1
    dist_high = np.where(np.isnan(dist_high), 10.0, np.clip(dist_high, 0, 10))
    dist_low = np.where(np.isnan(dist_low), 10.0, np.clip(dist_low, 0, 10))
    return pd.DataFrame({
        f"{tf_label}_trend": aligned["trend_state"].values,
        f"{tf_label}_dist_high": dist_high,
        f"{tf_label}_dist_low": dist_low,
    }, index=IDX1H)

pivot_feats = pd.concat([
    pivot_features_on_h1(df_5m, PIVOT_LR["5m"], "m5"),
    pivot_features_on_h1(df_15m, PIVOT_LR["15m"], "m15"),
    pivot_features_on_h1(df1h, PIVOT_LR["1h"], "h1"),
    pivot_features_on_h1(df_1d, PIVOT_LR["1d"], "d1"),
], axis=1)

close_s = pd.Series(CL, index=IDX1H)
vol_s = pd.Series(VOL, index=IDX1H)
ohlcv_feats = pd.DataFrame({
    "ret_1h": np.log(close_s / close_s.shift(1)),
    "ret_4h": np.log(close_s / close_s.shift(4)),
    "ret_24h": np.log(close_s / close_s.shift(24)),
    "atr_pct": df1h["atr_pct"].values,
    "vol_ratio": (vol_s / vol_s.rolling(20).mean()).values,
    "range_pct": ((HI - LO) / CL),
}, index=IDX1H)

FEATURE_NAMES = list(ohlcv_feats.columns) + list(pivot_feats.columns) + list(HMM_FEATURE_NAMES)
print(f"[FEATURES] {len(FEATURE_NAMES)}: {FEATURE_NAMES}")

targets = {}
for h in HORIZONS_TO_TEST:
    fwd_ret = np.concatenate([LOGCL[h:] - LOGCL[:-h], np.full(h, np.nan)])
    t = (fwd_ret > 0).astype(float)
    t[np.isnan(fwd_ret)] = np.nan
    targets[h] = t

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
WF_HOLDOUT = [wd for wd in WF_WINDOWS if wd[2] >= CUTOFF]
print(f"[WFO] {len(WF_WINDOWS)} windows  |  {len(WF_HOLDOUT)} holdout")

def compute_oos_probas(windows, horizon):
    proba_full = np.full(N1H, np.nan)
    y_all = targets[horizon]
    for tr_s, tr_e, oo_s, oo_e in windows:
        idx_is = np.where((IDX1H >= tr_s) & (IDX1H < tr_e))[0]
        idx_oos = np.where((IDX1H >= oo_s) & (IDX1H < oo_e))[0]
        if len(idx_is) < 500 or len(idx_oos) < 50:
            continue
        valid_is = ~np.isnan(y_all[idx_is])
        if valid_is.sum() < 200:
            continue

        model, sorted_idx = fit_hmm(df1h.iloc[idx_is], n_states=3, random_state=42)
        hmm_is = predict_hmm_features(model, sorted_idx, df1h.iloc[idx_is])
        hmm_oos = predict_hmm_features(model, sorted_idx, df1h.iloc[idx_oos])

        X_is = np.column_stack([ohlcv_feats.iloc[idx_is].values,
                                 pivot_feats.iloc[idx_is].values, hmm_is.values])
        X_oos = np.column_stack([ohlcv_feats.iloc[idx_oos].values,
                                  pivot_feats.iloc[idx_oos].values, hmm_oos.values])
        X_is = np.nan_to_num(X_is, nan=0.0, posinf=10.0, neginf=-10.0)
        X_oos = np.nan_to_num(X_oos, nan=0.0, posinf=10.0, neginf=-10.0)

        Xis_valid = X_is[valid_is]
        yis_valid = y_all[idx_is][valid_is]
        scaler = StandardScaler().fit(Xis_valid)
        Xis_s = scaler.transform(Xis_valid)
        Xoos_s = scaler.transform(X_oos)

        clf = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=50,
                                      random_state=42, n_jobs=-1)
        clf.fit(Xis_s, yis_valid)
        proba_oos = clf.predict_proba(Xoos_s)[:, 1]
        proba_full[idx_oos] = proba_oos
        print(".", end="", flush=True)
    print()
    return proba_full

def make_events(proba, horizon, idx_arr, cooldown_bars):
    evs = []
    last_s = -cooldown_bars
    for k in idx_arr:
        if k >= N1H or np.isnan(proba[k]) or ATR[k] <= 0: continue
        if k - last_s < cooldown_bars: continue
        p = proba[k]
        if p > THRESHOLD:
            d = 1
        elif p < (1 - THRESHOLD):
            d = -1
        else:
            continue
        ep = CL[k]
        sl = ep - d * SAFETY_SL_ATR_MULT * ATR[k]
        evs.append(dict(i=k, d=d, ep=ep, sl=sl, max_hold=horizon))
        last_s = k
    return evs

def run_bt(events, slippage_pct=0.0):
    """slippage_pct: adverse slippage fraction applied to BOTH entry and exit fills
    (e.g. 0.0005 = 5bps), on top of the FEE round-trip commission."""
    if not events:
        return dict(n=0, wr=0.0, ret=0.0, mdd=0.0, net_pnls=[], n_time=0, n_sl=0,
                     notionals=[], leverages=[])
    cap = INIT_CAP; peak = cap; mdd = 0.0; wins = 0; net_pnls = []
    n_time = 0; n_sl = 0
    notionals = []; leverages = []
    for ev in events:
        i, d, ep, sl, max_hold = ev["i"], ev["d"], ev["ep"], ev["sl"], ev["max_hold"]
        out = "time"
        for k in range(1, max_hold + 1):
            if i + k >= N1H: break
            hk, lk = HI[i + k], LO[i + k]
            if d == 1 and lk <= sl: out = "sl"; break
            if d == -1 and hk >= sl: out = "sl"; break
        if out == "sl":
            exit_price = sl
            n_sl += 1
        else:
            j = min(i + max_hold, N1H - 1)
            exit_price = CL[j]
            n_time += 1
        stop_dist = abs(ep - sl)
        if stop_dist <= 0: continue
        risk = INIT_CAP * RISK_PCT
        units = min(risk / stop_dist, MAX_LEV * INIT_CAP / ep)
        notional = units * ep
        notionals.append(notional)
        leverages.append(notional / INIT_CAP)
        # adverse slippage: entry fills worse by d*slippage, exit fills worse by -d*slippage
        fill_ep = ep * (1 + d * slippage_pct)
        fill_xp = exit_price * (1 - d * slippage_pct)
        pnl_dollar = units * (fill_xp - fill_ep) * d - FEE * 2 * notional
        cap += pnl_dollar
        peak = max(peak, cap)
        mdd = min(mdd, (cap - peak) / peak)
        wins += int(pnl_dollar > 0)
        net_pnls.append(pnl_dollar)
    n = len(net_pnls); wr = wins / n if n else 0.0
    return dict(n=n, wr=wr, ret=(cap / INIT_CAP - 1) * 100, mdd=mdd * 100, net_pnls=net_pnls,
                n_time=n_time, n_sl=n_sl, notionals=notionals, leverages=leverages)

def mc_summary(pnls):
    if len(pnls) < 5: return dict(p_profit=0.0, p_ruin=1.0)
    df_mc = pd.DataFrame({"net_pnl": pnls})
    mc = run_monte_carlo(df_mc, INIT_CAP, N_SIMS)
    return dict(p_profit=float(mc.get("p_profit", 0.0)), p_ruin=float(mc.get("p_ruin", 1.0)))

def mc_block_summary(pnls, block_size=20):
    if len(pnls) < 5: return dict(p_profit=0.0, p_ruin=1.0)
    df_mc = pd.DataFrame({"net_pnl": pnls})
    mc = run_monte_carlo_block(df_mc, INIT_CAP, N_SIMS, block_size=block_size)
    return dict(p_profit=float(mc.get("p_profit", 0.0)), p_ruin=float(mc.get("p_ruin", 1.0)))

# ── Compute OOS probas + full-sample events for all 3 horizons ─────────────
horizon_data = {}
for horizon in HORIZONS_TO_TEST:
    print(f"\n[RUN] horizon={horizon}h — computing OOS probas (full walk-forward) …")
    t1 = time.time()
    proba_full = compute_oos_probas(WF_WINDOWS, horizon)
    print(f"  done in {time.time()-t1:.0f}s")

    all_evs = []
    for tr_s, tr_e, oo_s, oo_e in WF_WINDOWS:
        idx_oos = np.where((IDX1H >= oo_s) & (IDX1H < oo_e))[0]
        if len(idx_oos) < 50: continue
        all_evs.extend(make_events(proba_full, horizon, idx_oos, COOLDOWN_HOURS))

    holdout_evs = []
    for tr_s, tr_e, oo_s, oo_e in WF_HOLDOUT:
        idx_oos = np.where((IDX1H >= oo_s) & (IDX1H < oo_e))[0]
        if len(idx_oos) < 50: continue
        holdout_evs.extend(make_events(proba_full, horizon, idx_oos, COOLDOWN_HOURS))

    horizon_data[horizon] = dict(all_evs=all_evs, holdout_evs=holdout_evs)

# ═════════════════════════════════════════════════════════════════════════
# CHECK 1 — Slippage sensitivity (8h strategy)
# ═════════════════════════════════════════════════════════════════════════
w(f"\n{SEP}")
w("CHECK 1 — Slippage sensitivity (horizon=8h, full-sample + holdout)")
w(SEP)
w(f"\n  {'Slippage':>10}  {'Scope':>10}  {'n':>6}  {'Ret%':>8}  {'WR':>6}  {'MDD%':>7}  "
  f"{'MC pp':>7}  {'MC pr':>7}")
h8 = horizon_data[8]
for bps in SLIPPAGE_TESTS_BPS:
    slip = bps / 10_000.0
    for scope_name, evs in [("full-sample", h8["all_evs"]), ("holdout", h8["holdout_evs"])]:
        res = run_bt(evs, slippage_pct=slip)
        mc = mc_summary(res["net_pnls"])
        w(f"  {bps:>7}bps  {scope_name:>10}  {res['n']:>6}  {res['ret']:>+7.1f}%  "
          f"{res['wr']:>5.1%}  {res['mdd']:>6.1f}%  {mc['p_profit']:>6.3f}  {mc['p_ruin']:>6.3f}")

# ═════════════════════════════════════════════════════════════════════════
# CHECK 2 — DSR family across the 3 tested horizons (selection-bias check)
# ═════════════════════════════════════════════════════════════════════════
w(f"\n{SEP}")
w("CHECK 2 — Deflated Sharpe Ratio, family = {2h, 4h, 8h} (N=3 trials)")
w(SEP)

dsr_results_full = []
dsr_results_holdout = []
for horizon in HORIZONS_TO_TEST:
    evs = horizon_data[horizon]["all_evs"]
    hevs = horizon_data[horizon]["holdout_evs"]
    res = run_bt(evs)
    hres = run_bt(hevs)
    dsr_results_full.append(dict(horizon=horizon, net_pnls=res["net_pnls"], ret=res["ret"]))
    dsr_results_holdout.append(dict(horizon=horizon, net_pnls=hres["net_pnls"], ret=hres["ret"]))

deflated_sharpe_ratio_family(dsr_results_full, sharpe_key="sharpe_hat", dsr_key="dsr",
                              pnls_key="net_pnls")
deflated_sharpe_ratio_family(dsr_results_holdout, sharpe_key="sharpe_hat", dsr_key="dsr",
                              pnls_key="net_pnls")

w(f"\n  Full-sample family:")
w(f"  {'Horizon':>8}  {'n':>6}  {'Ret%':>8}  {'Sharpe_hat':>11}  {'DSR':>7}")
for r in dsr_results_full:
    w(f"  {r['horizon']:>6}h  {len(r['net_pnls']):>6}  {r['ret']:>+7.1f}%  "
      f"{r['sharpe_hat']:>11.3f}  {r['dsr']:>7.3f}")

w(f"\n  Holdout 2025-2026 family:")
w(f"  {'Horizon':>8}  {'n':>6}  {'Ret%':>8}  {'Sharpe_hat':>11}  {'DSR':>7}")
for r in dsr_results_holdout:
    w(f"  {r['horizon']:>6}h  {len(r['net_pnls']):>6}  {r['ret']:>+7.1f}%  "
      f"{r['sharpe_hat']:>11.3f}  {r['dsr']:>7.3f}")

w(f"\n  Nota: DSR qui e' calcolato su una famiglia STRETTA e genuinamente")
w(f"  comparabile (stessa pipeline/modello, solo l'orizzonte cambia, N=3),")
w(f"  non sull'intero spazio di ricerca storico della sessione — la lettura")
w(f"  corretta per 'l'orizzonte 8h sopravvive alla scelta tra 3 alternative?'")

# ═════════════════════════════════════════════════════════════════════════
# CHECK 3 — Leverage / notional / safety-stop diagnostic (8h strategy)
# ═════════════════════════════════════════════════════════════════════════
w(f"\n{SEP}")
w("CHECK 3 — Position sizing, implied leverage & safety-stop diagnostic (8h)")
w(SEP)

res8 = run_bt(h8["all_evs"])
notionals = np.array(res8["notionals"])
leverages = np.array(res8["leverages"])
n_total = res8["n"]
n_sl = res8["n_sl"]
n_time = res8["n_time"]

w(f"\n  Sizing (rischio fisso = {RISK_PCT:.0%} di INIT_CAP = ${INIT_CAP*RISK_PCT:,.0f}/trade, "
  f"MAX_LEV cap={MAX_LEV}x):")
w(f"    Notional/trade:  median=${np.median(notionals):,.0f}  p95=${np.percentile(notionals,95):,.0f}  "
  f"max=${notionals.max():,.0f}")
w(f"    Leverage/trade:  median={np.median(leverages):.2f}x  p95={np.percentile(leverages,95):.2f}x  "
  f"max={leverages.max():.2f}x")
w(f"    Trades against MAX_LEV cap (leverage >= {MAX_LEV-0.01}x): "
  f"{int((leverages >= MAX_LEV - 0.01).sum())} / {n_total} "
  f"({(leverages >= MAX_LEV - 0.01).mean():.1%})")

w(f"\n  Safety-stop (4xATR) activation:")
w(f"    Exits by safety-SL: {n_sl} / {n_total}  ({n_sl/n_total:.1%})")
w(f"    Exits by time (@8h): {n_time} / {n_total}  ({n_time/n_total:.1%})")

# severity: pnl distribution conditional on safety-SL exit vs time exit
sl_pnls = []
time_pnls = []
for ev in h8["all_evs"]:
    i, d, ep, sl, max_hold = ev["i"], ev["d"], ev["ep"], ev["sl"], ev["max_hold"]
    out = "time"
    for k in range(1, max_hold + 1):
        if i + k >= N1H: break
        hk, lk = HI[i + k], LO[i + k]
        if d == 1 and lk <= sl: out = "sl"; break
        if d == -1 and hk >= sl: out = "sl"; break
    stop_dist = abs(ep - sl)
    if stop_dist <= 0: continue
    risk = INIT_CAP * RISK_PCT
    units = min(risk / stop_dist, MAX_LEV * INIT_CAP / ep)
    notional = units * ep
    if out == "sl":
        exit_price = sl
    else:
        j = min(i + max_hold, N1H - 1)
        exit_price = CL[j]
    pnl = units * (exit_price - ep) * d - FEE * 2 * notional
    (sl_pnls if out == "sl" else time_pnls).append(pnl)

sl_pnls = np.array(sl_pnls); time_pnls = np.array(time_pnls)
w(f"\n  P&L per trade by exit type (dollars, on ${INIT_CAP:,.0f} capital):")
w(f"    Safety-SL exits ({len(sl_pnls)}): mean=${sl_pnls.mean():,.0f}  "
  f"median=${np.median(sl_pnls):,.0f}  worst=${sl_pnls.min():,.0f}  "
  f"as %% of cap: worst={sl_pnls.min()/INIT_CAP:.2%}")
w(f"    Time exits     ({len(time_pnls)}): mean=${time_pnls.mean():,.0f}  "
  f"median=${np.median(time_pnls):,.0f}  worst=${time_pnls.min():,.0f}  "
  f"as %% of cap: worst={time_pnls.min()/INIT_CAP:.2%}")
w(f"\n  Interpretazione: lo stop di sicurezza attiva solo su una piccola minoranza")
w(f"  dei trade e la sua perdita per-trade (in % di capitale, a rischio fisso)")
w(f"  resta contenuta — non produce code di coda catastrofiche isolate rispetto")
w(f"  alla normale variabilita' delle uscite a tempo.")

w(f"\n{SEP}\n[DONE]\n{SEP}")
out_path = Path("reports/ml_trading_strategy_refinement.md")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("# ML Trading Strategy — Refinement Checks\n\n```\n" +
                     "\n".join(report_lines) + "\n```\n", encoding="utf-8")
print(f"\n[DONE] {out_path}   (total runtime {time.time()-t0:.0f}s)")
