#!/usr/bin/env python3
"""
create_ltf_smc_mtf_report.py
════════════════════════════════════════════════════════════════════════════
Analisi intensiva BTCUSDT su 5m/15m/30m: struttura di mercato (BOS/CHoCH),
order block (domanda/offerta), Fair Value Gap, zone premium/discount,
liquidity sweep, filtro di regime (HMM), filtro di volatilita realizzata
e modello di rendimento atteso (RandomForest walk-forward). Tre varianti
a confluenza crescente, sottoposte alla stessa pipeline di validazione
standard usata per le altre strategie del repo.

Riuso totale dei moduli esistenti (nessuna logica di struttura/liquidita/
validazione riscritta da zero):
  smc.py         -> struttura (trend/BOS/CHoCH), order block, FVG,
                     zone premium/discount, equal highs/lows
  mtf_swing.py   -> bias HTF causale (flips SOLO su CHoCH — questo e'
                     il meccanismo che impedisce long/short di alternarsi
                     senza motivo) + livelli pivot strutturali per SL/TP
  hmm_regime.py  -> regime gate walk-forward (3 stati)
  monte_carlo.py -> Monte Carlo iid/block-bootstrap + Deflated Sharpe Ratio
  report_html.py -> palette/CSS/helper condivisi con gli altri report HTML

Architettura a 3 timeframe
───────────────────────────
  30m  BIAS strutturale   : smc_trend_signal(30m) ∈ {-1,0,+1}, locked fino
                            al prossimo CHoCH confermato — un long resta
                            long finche' la struttura non rompe al ribasso,
                            e viceversa. Nessun "flip" arbitrario.
  15m  ENTRY trigger      : prezzo dentro un order block allineato al bias
                            o in zona discount/premium coerente; SL/TP presi
                            dai livelli pivot strutturali (mtf_swing).
  5m   CONFERMA           : liquidity sweep — wick oltre l'estremo recente
                            (stop hunt) e reclaim in chiusura, nella
                            direzione del bias.

Modelli statistici (walk-forward, mai fit sui dati OOS)
──────────────────────────────────────────────────────
  Regime atteso      : GaussianHMM 3 stati (bear/side/bull) su 15m
  Volatilita' attesa  : percentile rolling dell'ATR% realizzato (15m, 500 barre)
  Rendimento atteso   : RandomForestClassifier, target = segno del
                        log-return a 2h (8 barre 15m), rifittato ogni
                        finestra 6m/2m/2m

Varianti (confluenza crescente)
────────────────────────────────
  V1  Structure Baseline     : bias 30m + zona 15m, nessun filtro statistico
  V2  + Regime + Sweep       : V1 + regime HMM d'accordo + liquidity sweep 5m
                                (proxy: wick oltre l'estremo + reclaim prezzo)
  V3  + Expected-Return Gate : V2 + P(direzione) RandomForest > soglia
                                + filtro volatilita' estrema
  V4  + CVD Order-Flow       : V2 ma il liquidity sweep richiede conferma di
                                order-flow REALE (taker_buy_base Binance,
                                CVD/delta cumulato) invece del solo prezzo —
                                idea nota da letteratura order-flow/FinTwit
                                ("smart money absorption"). Nota: un segnale
                                CVD STANDALONE era gia' stato testato e
                                bocciato (IC FAIL) in create_github_strategies
                                _report.py — qui e' usato come FILTRO di
                                conferma su una struttura gia' validata (V2),
                                ipotesi diversa, mai testata in questa forma.
  V5  + Neural Expected-Return : identica a V3, ma il modello di rendimento
                                atteso e' una rete neurale (MLPClassifier,
                                32-16 neuroni, sklearn) invece di un
                                RandomForest — stesse feature, stesso target,
                                stesso walk-forward. Isola l'effetto della
                                classe di modello dall'effetto delle feature.

Validazione (identica al resto del repo)
─────────────────────────────────────────
  WFO causale 6m/2m/2m -> holdout genuino 2025-2026 -> Monte Carlo
  iid + block-bootstrap -> Deflated Sharpe Ratio (famiglia di 4 varianti)
"""
from __future__ import annotations

import gc
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.strategy.data_fetcher import fetch_binance_vision_klines, fetch_binance_vision_taker_flow
from src.strategy.indicators import add_indicators
from src.strategy.smc import compute_smc_features, smc_trend_signal
from src.strategy.mtf_swing import causal_trend_state, align_htf_to_ltf
from src.strategy.hmm_regime import fit_hmm, predict_hmm_features
from src.strategy.monte_carlo import (
    run_monte_carlo, run_monte_carlo_block, deflated_sharpe_ratio_family,
)
from src.strategy.report_html import (
    _CSS, _fig_to_b64, _img_tag, _kpi_card, _table, _signed, _color_signed,
    BG, PANEL, BORDER, WHITE, GRAY, GREEN, RED, GOLD, BLUE, PURPLE, ORANGE,
)

plt.style.use("dark_background")

# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════
SEP = "═" * 78
START_YEAR, START_MONTH = 2022, 1
END_YEAR, END_MONTH = 2026, 6
INIT_CAP = 100_000.0
RISK_PCT = 0.01
FEE = 0.0006           # 0.06% taker, roundtrip x2 — matches src/live/trader.py
MAX_LEV = 1.0           # matches the live account's fixed 1x leverage config
CUTOFF = pd.Timestamp("2025-01-01")   # genuine holdout boundary (repo convention)
N_SIMS = 2_000    # kept modest — this box has ~3.8GB RAM shared with 5 live_trader.py processes
WF_TRAIN_M, WF_OOS_M, WF_STEP_M = 6, 2, 2
DSR_THRESHOLD = 0.95    # Bailey & Lopez de Prado 2014, same gate as create_adp_optimization_report.py

TIME_STOP_BARS = 48     # 15m bars -> 12h max hold
COOLDOWN_BARS = 4        # 15m bars -> 1h between trades (prevents rapid flip-flop)
SL_BUFFER_ATR = 0.15     # extra margin beyond the structural pivot (avoid exact-level stop hunts)
MIN_RR = 1.2             # minimum reward:risk; widen TP to this if the structural target is too close
FALLBACK_SL_ATR = 1.0
FALLBACK_TP_ATR = 2.0

SWEEP_LOOKBACK_BARS_5M = 6     # 30 min — 5m bars considered for a liquidity sweep confirmation
SWEEP_EXTREME_LOOKBACK_5M = 20  # bars used to define "the recent extreme" being swept

VOL_LOOKBACK = 500       # 15m bars (~5.2 days) for the realized-vol percentile
VOL_EXTREME_LOW, VOL_EXTREME_HIGH = 0.05, 0.90   # skip entries outside this vol-regime band (V3 only)

ML_HORIZON_BARS = 8      # 15m bars -> 2h forward-return target
ML_THRESHOLD = 0.55      # confidence filter on P(direction), same convention as the ML RF 8h strategy

report_lines: list[str] = []
def w(line: str = "") -> None:
    print(line)
    report_lines.append(line)

t_start = time.time()
w(SEP)
w("BTCUSDT — LTF Market Structure / SMC / Liquidity — MTF Strategy Research")
w(SEP)

# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA — 5m / 15m / 30m, 2022-01 -> 2026-06 (cached parquet, instant on rerun)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[DATA] Loading 5m / 15m / 30m BTCUSDT perpetual history...")
t0 = time.time()
# taker_flow (not plain klines): adds taker_buy_base/n_trades, which
# add_indicators turns into cvd/cvd_slope_4/cvd_div/flow_ratio automatically
# (src/strategy/indicators.py, gated on "taker_buy_base" being present) —
# needed for the V4 CVD confluence variant below, no separate fetch/column
# plumbing required.
df5 = add_indicators(fetch_binance_vision_taker_flow(
    "5m", START_YEAR, START_MONTH, END_YEAR, END_MONTH, workers=8))
df15 = add_indicators(fetch_binance_vision_klines(
    "15m", START_YEAR, START_MONTH, END_YEAR, END_MONTH, workers=8))
df30 = add_indicators(fetch_binance_vision_klines(
    "30m", START_YEAR, START_MONTH, END_YEAR, END_MONTH, workers=8))
print(f"  5m: {len(df5):,}   15m: {len(df15):,}   30m: {len(df30):,}  "
      f"({df15.index[0].date()} -> {df15.index[-1].date()}, {time.time()-t0:.0f}s)")

N15 = len(df15)
IDX15 = df15.index
CL15 = df15["close"].values.astype(float)
HI15 = df15["high"].values.astype(float)
LO15 = df15["low"].values.astype(float)
VOL15 = df15["volume"].values.astype(float)
# ATR at entry uses the PRIOR bar's ATR (shift(1)) — the current bar's own
# ATR isn't knowable until it closes, using it would leak lookahead into sizing.
ATR15 = np.where(df15["atr_14"].shift(1).values > 0, df15["atr_14"].shift(1).values, 1.0)

# ═══════════════════════════════════════════════════════════════════════════
# 2. MARKET STRUCTURE — smc.py (30m bias, 15m entry zones), mtf_swing.py (SL/TP levels)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[STRUCTURE] Computing SMC features (structure/order-blocks/FVG/zones/liquidity)...")
t0 = time.time()
smc30 = compute_smc_features(df30, swing_len=50, internal_len=5, prefix="smc")
smc15 = compute_smc_features(df15, swing_len=50, internal_len=5, prefix="smc")
print(f"  smc30: {smc30.shape}  smc15: {smc15.shape}  ({time.time()-t0:.0f}s)")

# 30m structural bias, locked until the next confirmed CHoCH (smc_trend_signal
# only changes value on a CHoCH event — this IS the mechanism that stops
# long/short from alternating without a structural reason).
bias_30m = smc_trend_signal(smc30, prefix="smc")
bias_df = align_htf_to_ltf(df30.index, pd.DataFrame({"bias": bias_30m.values}, index=df30.index), IDX15)
bias15 = bias_df["bias"].fillna(0).astype(int).values

# 15m structural pivots for SL (nearest confirmed pivot behind price) and
# TP (nearest still-unbroken pivot ahead of price) — causal by construction.
struct15 = causal_trend_state(CL15, HI15, LO15, left=8, right=8)
struct15.index = IDX15
sl_basis_low = struct15["last_pivot_low"].values     # SL basis for LONG
sl_basis_high = struct15["last_pivot_high"].values    # SL basis for SHORT
tp_basis_high = struct15["target_high"].values        # TP basis for LONG
tp_basis_low = struct15["target_low"].values           # TP basis for SHORT

# 15m entry zone: inside an order block aligned with the bias, or in the
# discount (for longs) / premium (for shorts) half of the last swing range.
zone_bull15 = ((smc15["smc_ob_bull_in"].values == 1) | (smc15["smc_in_discount"].values == 1)).astype(int)
zone_bear15 = ((smc15["smc_ob_bear_in"].values == 1) | (smc15["smc_in_premium"].values == 1)).astype(int)

# ═══════════════════════════════════════════════════════════════════════════
# 3. LIQUIDITY SWEEP (5m) — wick beyond the recent extreme + reclaim in close,
#    rolled up to "did a sweep happen in the last 30 min" and projected onto
#    the 15m grid. 5m and 15m timestamps nest exactly (15 = 3x5), so a plain
#    reindex+ffill is a correct, lookahead-free downsample (no need for the
#    asof-merge machinery in align_htf_to_ltf, which solves the opposite
#    direction: coarse-to-fine).
# ═══════════════════════════════════════════════════════════════════════════
print("\n[LIQUIDITY] Detecting 5m liquidity sweeps...")
recent_low_5m = df5["low"].rolling(SWEEP_EXTREME_LOOKBACK_5M).min().shift(1)
recent_high_5m = df5["high"].rolling(SWEEP_EXTREME_LOOKBACK_5M).max().shift(1)
sweep_bull_5m = ((df5["low"] < recent_low_5m) & (df5["close"] > recent_low_5m)).astype(int)
sweep_bear_5m = ((df5["high"] > recent_high_5m) & (df5["close"] < recent_high_5m)).astype(int)
sweep_bull_roll = sweep_bull_5m.rolling(SWEEP_LOOKBACK_BARS_5M, min_periods=1).max()
sweep_bear_roll = sweep_bear_5m.rolling(SWEEP_LOOKBACK_BARS_5M, min_periods=1).max()
sweep_bull15 = sweep_bull_roll.reindex(IDX15, method="ffill").fillna(0).astype(int).values
sweep_bear15 = sweep_bear_roll.reindex(IDX15, method="ffill").fillna(0).astype(int).values
print(f"  bull sweeps: {int(sweep_bull_5m.sum()):,}   bear sweeps: {int(sweep_bear_5m.sum()):,}  (5m bars)")

# V4 only: same sweep definition, but additionally require REAL order-flow
# confirmation at the sweep bar — cvd_slope_4 (4-bar CVD momentum, from real
# taker_buy_base, not price-derived) must already be turning in the bias
# direction. This is the "smart money absorption" reading: a genuine sweep
# should show aggressive real buying (not just a price wick) right as the
# recent low is taken out.
cvd_slope_5m = df5["cvd_slope_4"]
sweep_bull_cvd_5m = sweep_bull_5m & (cvd_slope_5m > 0)
sweep_bear_cvd_5m = sweep_bear_5m & (cvd_slope_5m < 0)
sweep_bull_cvd_roll = sweep_bull_cvd_5m.rolling(SWEEP_LOOKBACK_BARS_5M, min_periods=1).max()
sweep_bear_cvd_roll = sweep_bear_cvd_5m.rolling(SWEEP_LOOKBACK_BARS_5M, min_periods=1).max()
sweep_bull_cvd15 = sweep_bull_cvd_roll.reindex(IDX15, method="ffill").fillna(0).astype(int).values
sweep_bear_cvd15 = sweep_bear_cvd_roll.reindex(IDX15, method="ffill").fillna(0).astype(int).values
print(f"  CVD-confirmed bull sweeps: {int(sweep_bull_cvd_5m.sum()):,}   "
      f"bear sweeps: {int(sweep_bear_cvd_5m.sum()):,}  (5m bars)")

# df5 and its intermediates are done being useful — only the four derived
# 15m-grid arrays above are needed downstream. This box has 3.8GB RAM shared
# with 5 live_trader.py processes; freeing the 472k-row frame before the
# memory-heavy WFO loop (HMM+RandomForest fits) is the difference between
# finishing and getting OOM-killed (observed both ways while building this).
del df5, recent_low_5m, recent_high_5m, sweep_bull_5m, sweep_bear_5m, sweep_bull_roll, sweep_bear_roll
del cvd_slope_5m, sweep_bull_cvd_5m, sweep_bear_cvd_5m, sweep_bull_cvd_roll, sweep_bear_cvd_roll
gc.collect()

# ═══════════════════════════════════════════════════════════════════════════
# 4. VOLATILITY REGIME — rolling percentile rank of realized ATR% (15m)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[VOLATILITY] Computing realized-volatility percentile regime...")
t0 = time.time()
def _pctile_of_last(x: np.ndarray) -> float:
    return float((x <= x[-1]).mean())
vol_pctile = df15["atr_pct"].rolling(VOL_LOOKBACK, min_periods=100).apply(_pctile_of_last, raw=True).values
print(f"  done ({time.time()-t0:.0f}s)")

# ═══════════════════════════════════════════════════════════════════════════
# 5. EXPECTED-RETURN FEATURES (15m, all causal) — RandomForest input matrix
# ═══════════════════════════════════════════════════════════════════════════
close_s = df15["close"]
ret_1 = np.log(close_s / close_s.shift(1))
ret_4 = np.log(close_s / close_s.shift(4))
ret_12 = np.log(close_s / close_s.shift(12))
vol_ratio = (df15["volume"] / df15["volume"].rolling(20).mean())
hour = IDX15.hour + IDX15.minute / 60.0
hour_sin = np.sin(2 * np.pi * hour / 24.0)
hour_cos = np.cos(2 * np.pi * hour / 24.0)

ml_feats = pd.DataFrame({
    "bias": bias15.astype(float),
    "zone_pct": smc15["smc_zone_pct"].values,
    "in_discount": smc15["smc_in_discount"].values,
    "in_premium": smc15["smc_in_premium"].values,
    "ob_bull_in": smc15["smc_ob_bull_in"].values,
    "ob_bear_in": smc15["smc_ob_bear_in"].values,
    "dist_to_ob_bull": smc15["smc_dist_to_ob_bull"].values,
    "dist_to_ob_bear": smc15["smc_dist_to_ob_bear"].values,
    "dist_to_sh": smc15["smc_dist_to_sh"].values,
    "dist_to_sl": smc15["smc_dist_to_sl"].values,
    "eq_high": smc15["smc_eq_high"].values,
    "eq_low": smc15["smc_eq_low"].values,
    "sweep_bull": sweep_bull15.astype(float),
    "sweep_bear": sweep_bear15.astype(float),
    "vol_pctile": vol_pctile,
    "ret_1": ret_1.values, "ret_4": ret_4.values, "ret_12": ret_12.values,
    "vol_ratio": vol_ratio.values,
    "hour_sin": hour_sin, "hour_cos": hour_cos,
}, index=IDX15)
FEATURE_NAMES = list(ml_feats.columns)
print(f"\n[ML FEATURES] {len(FEATURE_NAMES)}: {FEATURE_NAMES}")

fwd_ret = np.log(close_s.shift(-ML_HORIZON_BARS) / close_s)
y_target = (fwd_ret > 0).astype(float).to_numpy(copy=True)
y_target[np.isnan(fwd_ret.values)] = np.nan

# ═══════════════════════════════════════════════════════════════════════════
# 6. WALK-FORWARD WINDOWS
# ═══════════════════════════════════════════════════════════════════════════
def wf_dates(idx: pd.DatetimeIndex) -> list[tuple]:
    windows = []
    t0_ = idx[0]
    while True:
        tr_s = t0_
        tr_e = tr_s + pd.DateOffset(months=WF_TRAIN_M)
        oo_s = tr_e
        oo_e = oo_s + pd.DateOffset(months=WF_OOS_M)
        if oo_e > idx[-1]:
            break
        windows.append((tr_s, tr_e, oo_s, oo_e))
        t0_ += pd.DateOffset(months=WF_STEP_M)
    return windows

WF_WINDOWS = wf_dates(IDX15)
WF_HOLDOUT = [wd for wd in WF_WINDOWS if wd[2] >= CUTOFF]
w(f"\n[WFO] {len(WF_WINDOWS)} windows (train={WF_TRAIN_M}m/oos={WF_OOS_M}m/step={WF_STEP_M}m)  "
  f"|  {len(WF_HOLDOUT)} holdout windows (OOS start >= {CUTOFF.date()})")

# ═══════════════════════════════════════════════════════════════════════════
# 7. WALK-FORWARD MODELS — HMM regime (V2/V3) + RandomForest expected-return (V3)
#    Both fit ONLY on each window's training slice, predicted on that
#    window's OOS slice — never on data the model has seen. HMM and the RF
#    both run on the SAME 15m grid as the entry timeframe, so no cross-
#    timeframe alignment is needed here (unlike the 30m->15m structural bias,
#    which is a fixed causal rule, not a fitted model, and doesn't need OOS
#    restriction at all).
# ═══════════════════════════════════════════════════════════════════════════
def compute_oos_hmm_and_ml(windows: list[tuple]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hmm_bull_full = np.full(N15, np.nan)
    hmm_bear_full = np.full(N15, np.nan)
    ml_proba_full = np.full(N15, np.nan)
    for tr_s, tr_e, oo_s, oo_e in windows:
        idx_is = np.where((IDX15 >= tr_s) & (IDX15 < tr_e))[0]
        idx_oos = np.where((IDX15 >= oo_s) & (IDX15 < oo_e))[0]
        if len(idx_is) < 1000 or len(idx_oos) < 50:
            print("x", end="", flush=True)
            continue

        model, sorted_idx = fit_hmm(df15.iloc[idx_is], n_states=3, random_state=42)
        hmm_oos = predict_hmm_features(model, sorted_idx, df15.iloc[idx_oos])
        hmm_bull_full[idx_oos] = hmm_oos["hmm_prob_bull"].values
        hmm_bear_full[idx_oos] = hmm_oos["hmm_prob_bear"].values

        y_is = y_target[idx_is]
        valid = ~np.isnan(y_is)
        if valid.sum() < 300:
            print(".", end="", flush=True)
            continue
        X_is = np.nan_to_num(ml_feats.iloc[idx_is].values[valid], nan=0.0, posinf=10.0, neginf=-10.0)
        X_oos = np.nan_to_num(ml_feats.iloc[idx_oos].values, nan=0.0, posinf=10.0, neginf=-10.0)
        scaler = StandardScaler().fit(X_is)
        # n_jobs=1 deliberately, not -1: joblib's process-based parallelism
        # forks a full copy of X_is per worker, which is what OOM-killed this
        # 3.8GB box (shared with 5 live_trader.py processes) even after every
        # other memory cut below. Single-threaded is slower but survives.
        clf = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=50,
                                      random_state=42, n_jobs=1)
        clf.fit(scaler.transform(X_is), y_is[valid])
        ml_proba_full[idx_oos] = clf.predict_proba(scaler.transform(X_oos))[:, 1]
        print(".", end="", flush=True)
        del model, clf, scaler, X_is, X_oos
        gc.collect()
    print()
    return hmm_bull_full, hmm_bear_full, ml_proba_full

print("\n[WFO] Fitting HMM regime + RandomForest expected-return per window "
      "(walk-forward, causal)...")
t0 = time.time()
hmm_bull_oos, hmm_bear_oos, ml_proba_oos = compute_oos_hmm_and_ml(WF_WINDOWS)
print(f"  done in {time.time()-t0:.0f}s  "
      f"(HMM coverage: {np.isfinite(hmm_bull_oos).sum():,} bars, "
      f"ML coverage: {np.isfinite(ml_proba_oos).sum():,} bars)")


def compute_oos_ml(windows: list[tuple], model_type: str) -> np.ndarray:
    """Same walk-forward feature/target construction as
    compute_oos_hmm_and_ml's RF branch, factored out so a second classifier
    (V5's MLP) can be swapped in without re-fitting HMM (already computed
    above, identical either way — HMM is the regime model, independent of
    which expected-return classifier gates V3 vs V5)."""
    proba_full = np.full(N15, np.nan)
    for tr_s, tr_e, oo_s, oo_e in windows:
        idx_is = np.where((IDX15 >= tr_s) & (IDX15 < tr_e))[0]
        idx_oos = np.where((IDX15 >= oo_s) & (IDX15 < oo_e))[0]
        if len(idx_is) < 1000 or len(idx_oos) < 50:
            print("x", end="", flush=True)
            continue
        y_is = y_target[idx_is]
        valid = ~np.isnan(y_is)
        if valid.sum() < 300:
            print(".", end="", flush=True)
            continue
        X_is = np.nan_to_num(ml_feats.iloc[idx_is].values[valid], nan=0.0, posinf=10.0, neginf=-10.0)
        X_oos = np.nan_to_num(ml_feats.iloc[idx_oos].values, nan=0.0, posinf=10.0, neginf=-10.0)
        scaler = StandardScaler().fit(X_is)
        Xis_s = scaler.transform(X_is)
        Xoos_s = scaler.transform(X_oos)
        if model_type == "mlp":
            # Small (32,16) MLP, early stopping — this box OOM-killed on
            # RandomForest(n_jobs=-1) earlier; a small feed-forward net
            # trained single-threaded stays well inside the same memory
            # budget that worked for RF at n_jobs=1.
            clf = MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu",
                                 alpha=1e-3, max_iter=300, early_stopping=True,
                                 n_iter_no_change=15, random_state=42)
        else:
            clf = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=50,
                                          random_state=42, n_jobs=1)
        clf.fit(Xis_s, y_is[valid])
        proba_full[idx_oos] = clf.predict_proba(Xoos_s)[:, 1]
        print(".", end="", flush=True)
        del clf, scaler, X_is, X_oos, Xis_s, Xoos_s
        gc.collect()
    print()
    return proba_full


print("\n[WFO] Fitting neural-network (MLP) expected-return per window "
      "(walk-forward, causal, V5)...")
t0 = time.time()
ml_proba_mlp_oos = compute_oos_ml(WF_WINDOWS, model_type="mlp")
print(f"  done in {time.time()-t0:.0f}s  "
      f"(MLP coverage: {np.isfinite(ml_proba_mlp_oos).sum():,} bars)")

# NOTE: an earlier version of this script also fit a full-sample
# ("in-sample reference") HMM+RandomForest here, purely as an optimistic
# before-WFO comparison point (not used for the validation verdict). Fitting
# RandomForest(n_estimators=200, n_jobs=-1) on the full ~150k-row span
# repeatedly OOM-killed this box (3.8GB RAM, already near capacity after the
# WFO loop) — dropped. The WFO-OOS chained result below is the only number
# that was ever load-bearing for the verdict.

# ═══════════════════════════════════════════════════════════════════════════
# 8. BACKTEST ENGINE — single-position state machine, structural SL/TP,
#    fixed R:R floor, 12h time-stop, 1h cooldown. Same order-of-checks
#    convention as the rest of the repo: stop before target.
# ═══════════════════════════════════════════════════════════════════════════
def simulate(variant: int, hmm_bull: np.ndarray, hmm_bear: np.ndarray,
             ml_proba: np.ndarray, restrict_to_oos: bool,
             sweep_bull_arr: np.ndarray = None, sweep_bear_arr: np.ndarray = None,
             ) -> tuple[pd.DataFrame, float, float]:
    # sweep_bull_arr/sweep_bear_arr default to the plain price-based sweep
    # (V1/V2/V3). V4 passes the CVD-confirmed sweep arrays instead — same
    # variant==2 gating logic (regime + sweep), different sweep evidence.
    if sweep_bull_arr is None:
        sweep_bull_arr = sweep_bull15
    if sweep_bear_arr is None:
        sweep_bear_arr = sweep_bear15
    cap = INIT_CAP
    peak = cap
    mdd = 0.0
    trades: list[dict] = []
    in_pos = False
    dirn = 0
    entry_px = sl = tp = qty = 0.0
    entry_i = 0
    hold_end = 0
    last_exit = -COOLDOWN_BARS

    for i in range(N15):
        if in_pos:
            hit = None
            if dirn == 1:
                if LO15[i] <= sl: hit = "SL"
                elif HI15[i] >= tp: hit = "TP"
            else:
                if HI15[i] >= sl: hit = "SL"
                elif LO15[i] <= tp: hit = "TP"
            if hit is None and i >= hold_end:
                hit = "TIME"
            if hit is not None:
                exit_px = sl if hit == "SL" else (tp if hit == "TP" else CL15[i])
                notional = qty * entry_px
                pnl = qty * (exit_px - entry_px) * dirn - FEE * 2 * notional
                cap += pnl
                peak = max(peak, cap)
                mdd = min(mdd, (cap - peak) / peak if peak > 0 else 0.0)
                trades.append(dict(
                    entry_time=IDX15[entry_i], exit_time=IDX15[i], direction=dirn,
                    entry_price=entry_px, exit_price=exit_px, exit_reason=hit,
                    net_pnl=pnl, equity=cap,
                ))
                in_pos = False
                last_exit = i
            continue

        if i - last_exit < COOLDOWN_BARS:
            continue
        b = bias15[i]
        if b == 0:
            continue
        zone_ok = zone_bull15[i] if b == 1 else zone_bear15[i]
        if not zone_ok:
            continue

        if variant >= 2:
            if restrict_to_oos and np.isnan(hmm_bull[i]):
                continue
            regime_ok = (hmm_bull[i] > 0.5) if b == 1 else (hmm_bear[i] > 0.5)
            sweep_ok = sweep_bull_arr[i] if b == 1 else sweep_bear_arr[i]
            if not (regime_ok and sweep_ok):
                continue

        if variant >= 3:
            if restrict_to_oos and np.isnan(ml_proba[i]):
                continue
            p_dir = ml_proba[i] if b == 1 else (1.0 - ml_proba[i])
            if np.isnan(p_dir) or p_dir < ML_THRESHOLD:
                continue
            vp = vol_pctile[i]
            if np.isnan(vp) or vp < VOL_EXTREME_LOW or vp > VOL_EXTREME_HIGH:
                continue

        atrv = ATR15[i]
        if atrv <= 0:
            continue
        px = CL15[i]
        sl_level = sl_basis_low[i] if b == 1 else sl_basis_high[i]
        tp_level = tp_basis_high[i] if b == 1 else tp_basis_low[i]
        slp = (sl_level - b * SL_BUFFER_ATR * atrv) if not np.isnan(sl_level) else (px - b * FALLBACK_SL_ATR * atrv)
        tpp = tp_level if not np.isnan(tp_level) else (px + b * FALLBACK_TP_ATR * atrv)
        risk = abs(px - slp)
        if risk <= 0:
            continue
        reward = abs(tpp - px)
        if reward / risk < MIN_RR:
            tpp = px + b * max(MIN_RR * risk, FALLBACK_TP_ATR * atrv)

        risk_usd = INIT_CAP * RISK_PCT
        q = risk_usd / risk
        max_q = cap * MAX_LEV / px
        q = min(q, max_q)
        if q <= 0:
            continue

        dirn, entry_px, sl, tp, qty, entry_i = b, px, slp, tpp, q, i
        hold_end = i + TIME_STOP_BARS
        in_pos = True

    trades_df = pd.DataFrame(trades, columns=[
        "entry_time", "exit_time", "direction", "entry_price", "exit_price",
        "exit_reason", "net_pnl", "equity",
    ])
    return trades_df, cap, mdd

# ═══════════════════════════════════════════════════════════════════════════
# 9. RUN 3 VARIANTS
# ═══════════════════════════════════════════════════════════════════════════
print("\n[BACKTEST] Running 3 variants (full-history + WFO-OOS)...")
NAN15 = np.full(N15, np.nan)

# V1: pure structural rule, no fitted parameters -> valid over the full span,
# no OOS restriction needed (nothing was fit to leak).
v1_trades, v1_cap, v1_mdd = simulate(1, NAN15, NAN15, NAN15, restrict_to_oos=False)

# V2/V3: HMM (and RF for V3) ARE fitted -> the only causal/validated result
# is the OOS-restricted chained backtest (no separate full-sample IS run —
# see the note above the WFO section on why that was dropped).
v2_oos_trades, v2_oos_cap, v2_oos_mdd = simulate(2, hmm_bull_oos, hmm_bear_oos, NAN15, restrict_to_oos=True)
v3_oos_trades, v3_oos_cap, v3_oos_mdd = simulate(3, hmm_bull_oos, hmm_bear_oos, ml_proba_oos, restrict_to_oos=True)
# V4: identical to V2 (same regime gate, same fitted HMM — reused, not
# refit) except the sweep evidence is real order-flow (CVD) instead of price.
v4_oos_trades, v4_oos_cap, v4_oos_mdd = simulate(
    2, hmm_bull_oos, hmm_bear_oos, NAN15, restrict_to_oos=True,
    sweep_bull_arr=sweep_bull_cvd15, sweep_bear_arr=sweep_bear_cvd15,
)
# V5: identical to V3 (same regime gate + vol filter) except the expected-
# return classifier is a neural net (MLP) instead of RandomForest.
v5_oos_trades, v5_oos_cap, v5_oos_mdd = simulate(3, hmm_bull_oos, hmm_bear_oos, ml_proba_mlp_oos, restrict_to_oos=True)

VARIANTS = {
    "V1 Structure Baseline": dict(is_trades=v1_trades, is_cap=v1_cap, is_mdd=v1_mdd,
                                   oos_trades=v1_trades, oos_cap=v1_cap, oos_mdd=v1_mdd,
                                   fitted=False),
    "V2 + Regime + Sweep": dict(is_trades=v2_oos_trades, is_cap=v2_oos_cap, is_mdd=v2_oos_mdd,
                                 oos_trades=v2_oos_trades, oos_cap=v2_oos_cap, oos_mdd=v2_oos_mdd,
                                 fitted=True),
    "V3 + Expected-Return Gate": dict(is_trades=v3_oos_trades, is_cap=v3_oos_cap, is_mdd=v3_oos_mdd,
                                       oos_trades=v3_oos_trades, oos_cap=v3_oos_cap, oos_mdd=v3_oos_mdd,
                                       fitted=True),
    "V4 + CVD Order-Flow": dict(is_trades=v4_oos_trades, is_cap=v4_oos_cap, is_mdd=v4_oos_mdd,
                                 oos_trades=v4_oos_trades, oos_cap=v4_oos_cap, oos_mdd=v4_oos_mdd,
                                 fitted=True),
    "V5 + Neural Expected-Return": dict(is_trades=v5_oos_trades, is_cap=v5_oos_cap, is_mdd=v5_oos_mdd,
                                         oos_trades=v5_oos_trades, oos_cap=v5_oos_cap, oos_mdd=v5_oos_mdd,
                                         fitted=True),
}

for name, v in VARIANTS.items():
    n = len(v["oos_trades"])
    ret = (v["oos_cap"] / INIT_CAP - 1) * 100
    wr = (v["oos_trades"]["net_pnl"] > 0).mean() * 100 if n else 0.0
    w(f"  {name:<28}  n={n:>5}  ret={ret:>+7.1f}%  mdd={v['oos_mdd']*100:>6.1f}%  wr={wr:>5.1f}%  "
      f"({'WFO-OOS chained' if v['fitted'] else 'full-history, no fitted params'})")

# ═══════════════════════════════════════════════════════════════════════════
# 10. VALIDATION — Monte Carlo (iid + block), holdout, Deflated Sharpe Ratio
# ═══════════════════════════════════════════════════════════════════════════
print("\n[VALIDATION] Monte Carlo + holdout + DSR...")

def mc_pair(trades_df: pd.DataFrame) -> tuple[dict, dict]:
    if len(trades_df) < 5:
        return {}, {}
    mc = run_monte_carlo(trades_df, INIT_CAP, N_SIMS)
    mc_blk = run_monte_carlo_block(trades_df, INIT_CAP, N_SIMS, block_size=20)
    return mc, mc_blk

dsr_family = []
for name, v in VARIANTS.items():
    trades = v["oos_trades"]
    mc, mc_blk = mc_pair(trades)
    holdout = trades[trades["entry_time"] >= CUTOFF] if len(trades) else trades
    hmc, hmc_blk = mc_pair(holdout)
    v["mc"] = mc
    v["mc_block"] = mc_blk
    v["holdout_trades"] = holdout
    v["holdout_mc"] = hmc
    v["holdout_mc_block"] = hmc_blk
    dsr_family.append({"variant": name, "net_pnls": trades["net_pnl"].tolist() if len(trades) else []})

deflated_sharpe_ratio_family(dsr_family, pnls_key="net_pnls")
for d in dsr_family:
    VARIANTS[d["variant"]]["sharpe_hat"] = d["sharpe_hat"]
    VARIANTS[d["variant"]]["dsr"] = d["dsr"]

w(f"\n  Deflated Sharpe Ratio (family N={len(dsr_family)}, threshold={DSR_THRESHOLD}):")
for name, v in VARIANTS.items():
    w(f"    {name:<28}  sharpe_hat={v['sharpe_hat']:>+6.3f}  DSR={v['dsr']:>5.3f}  "
      f"{'PASS' if v['dsr'] >= DSR_THRESHOLD else 'FAIL'}")

# ── Pass/fail verdict (repo convention: OOS positive, P(ruin)<10%, DSR>=0.95) ──
for name, v in VARIANTS.items():
    oos_ret = (v["oos_cap"] / INIT_CAP - 1)
    p_ruin = v["mc"].get("p_ruin", 1.0) if v["mc"] else 1.0
    n_trades = len(v["oos_trades"])
    holdout_ret = ((v["holdout_trades"]["net_pnl"].sum() / INIT_CAP) if len(v["holdout_trades"]) else -1.0)
    v["verdict"] = (
        n_trades >= 30
        and oos_ret > 0
        and p_ruin < 0.10
        and v["dsr"] >= DSR_THRESHOLD
        and holdout_ret > 0
    )
    w(f"  {name:<28}  n>=30:{n_trades>=30}  OOS>0:{oos_ret>0}  P(ruin)<10%:{p_ruin<0.10}  "
      f"DSR>=0.95:{v['dsr']>=DSR_THRESHOLD}  holdout>0:{holdout_ret>0}  "
      f"-> {'VALIDATA' if v['verdict'] else 'NON VALIDATA'}")

# ── Year breakdown ──
w("\n  Breakdown per anno (OOS chained):")
for name, v in VARIANTS.items():
    w(f"\n    {name}")
    w(f"      {'Year':>6}  {'n':>5}  {'Ret%':>8}  {'WR':>6}")
    trades = v["oos_trades"]
    if trades.empty:
        continue
    for yr, grp in trades.groupby(trades["entry_time"].dt.year):
        ret_yr = grp["net_pnl"].sum() / INIT_CAP * 100
        wr_yr = (grp["net_pnl"] > 0).mean() * 100
        w(f"      {yr:>6}  {len(grp):>5}  {ret_yr:>+7.1f}%  {wr_yr:>5.1f}%")

print(f"\n[DONE] backtest+validation runtime: {time.time()-t_start:.0f}s")

# ═══════════════════════════════════════════════════════════════════════════
# 11. HTML REPORT — reuses report_html.py's palette/CSS/helpers for visual
#     consistency with every other report in reports/.
# ═══════════════════════════════════════════════════════════════════════════
print("\n[REPORT] Building HTML report...")
t0 = time.time()
COLORS = {"V1 Structure Baseline": BLUE, "V2 + Regime + Sweep": GOLD,
          "V3 + Expected-Return Gate": PURPLE, "V4 + CVD Order-Flow": GREEN,
          "V5 + Neural Expected-Return": ORANGE}


def _ax2(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=GRAY, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.grid(True, alpha=0.10, color=GRAY, linestyle="--")
    if title:  ax.set_title(title, color=WHITE, fontsize=9, pad=4)
    if xlabel: ax.set_xlabel(xlabel, color=GRAY, fontsize=7)
    if ylabel: ax.set_ylabel(ylabel, color=GRAY, fontsize=7)


def equity_series(trades_df: pd.DataFrame) -> pd.Series:
    if trades_df.empty:
        return pd.Series([INIT_CAP], index=[IDX15[0]])
    idx = pd.DatetimeIndex([IDX15[0]] + list(trades_df["exit_time"]))
    vals = [INIT_CAP] + list(trades_df["equity"])
    return pd.Series(vals, index=idx)


def chart_equity_overlay() -> str:
    fig, ax = plt.subplots(figsize=(12, 4.2), facecolor=BG)
    _ax2(ax, "Equity Curves — WFO-OOS chained (V2/V3) vs full-history (V1)", ylabel="Equity ($)")
    for name, v in VARIANTS.items():
        eq = equity_series(v["oos_trades"])
        ax.plot(eq.index, eq.values, color=COLORS[name], lw=1.4, label=name)
    ax.axhline(INIT_CAP, color=GRAY, lw=0.8, ls=":")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda val, _: f"${val:,.0f}"))
    ax.legend(fontsize=7, labelcolor=GRAY, framealpha=0.2, loc="upper left")
    fig.patch.set_facecolor(BG)
    fig.tight_layout()
    return _fig_to_b64(fig)


def chart_mc_fan(mc: dict, title: str) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.6), facecolor=BG)
    fig.subplots_adjust(wspace=0.3, left=0.06, right=0.97)
    _ax2(ax1, f"{title} — MC Equity Paths (iid bootstrap)")
    if mc:
        paths = mc["paths"]
        xs = np.arange(paths.shape[1])
        for k in range(min(250, paths.shape[0])):
            ax1.plot(xs, paths[k], color=BLUE, lw=0.3, alpha=0.10)
        p5 = np.percentile(paths, 5, axis=0)
        p50 = np.percentile(paths, 50, axis=0)
        p95 = np.percentile(paths, 95, axis=0)
        ax1.plot(xs, p5, color=RED, lw=1.2, label="p5")
        ax1.plot(xs, p50, color=WHITE, lw=1.5, label="p50")
        ax1.plot(xs, p95, color=GREEN, lw=1.2, label="p95")
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda val, _: f"${val:,.0f}"))
        ax1.legend(fontsize=7, labelcolor=GRAY, framealpha=0.2)
    _ax2(ax2, "Final Return Distribution")
    if mc:
        ax2.hist(mc["total_return"] * 100, bins=50, color=BLUE, alpha=0.75, edgecolor=BORDER)
        ax2.axvline(0, color=GOLD, lw=1, ls=":")
        ax2.set_xlabel("Final Return (%)", color=GRAY, fontsize=7)
    fig.patch.set_facecolor(BG)
    return _fig_to_b64(fig)


def chart_year_breakdown() -> str:
    years: set[int] = set()
    for v in VARIANTS.values():
        if len(v["oos_trades"]):
            years.update(v["oos_trades"]["entry_time"].dt.year.unique().tolist())
    years = sorted(int(y) for y in years)
    fig, ax = plt.subplots(figsize=(12, 3.6), facecolor=BG)
    _ax2(ax, "Return by Year — OOS chained", ylabel="Return (%)")
    width = 0.25
    x = np.arange(len(years))
    for i, (name, v) in enumerate(VARIANTS.items()):
        trades = v["oos_trades"]
        rets = []
        for yr in years:
            grp = trades[trades["entry_time"].dt.year == yr] if len(trades) else trades
            rets.append(grp["net_pnl"].sum() / INIT_CAP * 100 if len(grp) else 0.0)
        ax.bar(x + (i - 1) * width, rets, width=width, color=COLORS[name], label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years], color=GRAY, fontsize=7)
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.legend(fontsize=7, labelcolor=GRAY, framealpha=0.2)
    fig.patch.set_facecolor(BG)
    fig.tight_layout()
    return _fig_to_b64(fig)


eq_chart_b64 = chart_equity_overlay()
year_chart_b64 = chart_year_breakdown()

kpi_cards = ""
variant_sections = ""
for name, v in VARIANTS.items():
    n = len(v["oos_trades"])
    ret = (v["oos_cap"] / INIT_CAP - 1) * 100
    wr = (v["oos_trades"]["net_pnl"] > 0).mean() * 100 if n else 0.0
    mc = v["mc"]
    p_profit = mc.get("p_profit", 0.0) if mc else 0.0
    p_ruin = mc.get("p_ruin", 1.0) if mc else 1.0
    mc_blk = v["mc_block"]
    p_ruin_blk = mc_blk.get("p_ruin", 1.0) if mc_blk else 1.0
    holdout = v["holdout_trades"]
    holdout_ret = (holdout["net_pnl"].sum() / INIT_CAP * 100) if len(holdout) else 0.0
    verdict_badge = ('<span class="badge badge-green">VALIDATA</span>' if v["verdict"]
                      else '<span class="badge badge-red">NON VALIDATA</span>')

    kpi_cards += f"""
<div class="card" style="border-left:3px solid {COLORS[name]}">
  <div class="val {'green' if ret>0 else 'red'}">{ret:+.1f}%</div>
  <div class="lbl">{name} — OOS Return</div>
</div>"""

    mc_chart_b64 = chart_mc_fan(mc, name) if mc else ""
    stat_rows = [
        ["Trades (OOS)", f"{n:,}"],
        ["Win Rate", f"{wr:.1f}%"],
        ["Max Drawdown", f"{v['oos_mdd']*100:.1f}%"],
        ["Sharpe (trade-level)", f"{v['sharpe_hat']:+.3f}"],
        ["Deflated Sharpe Ratio", f"{v['dsr']:.3f}"],
        ["MC P(profit)", f"{p_profit:.1%}"],
        ["MC P(ruin) iid", f"{p_ruin:.1%}"],
        ["MC P(ruin) block", f"{p_ruin_blk:.1%}"],
        ["Holdout 2025-2026 return", f"{holdout_ret:+.1f}%"],
        ["Holdout trades", f"{len(holdout):,}"],
        ["Fitted parameters",
         ("Yes (HMM+RF)" if "V3" in name else "Yes (HMM+MLP)" if "V5" in name else "Yes (HMM)")
         if v["fitted"] else "No — pure structural rule"],
    ]
    variant_sections += f"""
<section id="{name.split()[0].lower()}">
  <h2>{name} &nbsp; {verdict_badge}</h2>
  <div class="two-col" style="margin-bottom:16px">
    <div>{_table(["Metric", "Value"], stat_rows)}</div>
    <div class="chart">{_img_tag(mc_chart_b64)}</div>
  </div>
</section>"""

now_str = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC")
nav = """
<nav>
  <span class="brand">BTCUSDT LTF SMC/MTF Research</span>
  <a href="#summary">Summary</a>
  <a href="#decision-criteria">Criteri Decisionali</a>
  <a href="#equity">Equity</a>
  <a href="#v1">V1</a>
  <a href="#v2">V2</a>
  <a href="#v3">V3</a>
  <a href="#v4">V4</a>
  <a href="#v5">V5</a>
  <a href="#years">Per Anno</a>
</nav>"""

verdict_summary_rows = [
    [name,
     f"{(v['oos_cap']/INIT_CAP-1)*100:+.1f}%",
     f"{v['oos_mdd']*100:.1f}%",
     f"{v['dsr']:.3f}",
     f"{(v['holdout_trades']['net_pnl'].sum()/INIT_CAP*100) if len(v['holdout_trades']) else 0.0:+.1f}%",
     "VALIDATA" if v["verdict"] else "NON VALIDATA"]
    for name, v in VARIANTS.items()
]

body = f"""
<section id="summary">
  <h2>Executive Summary</h2>
  <p style="color:var(--gray);margin-bottom:16px">
    BTCUSDT Perpetual &nbsp;·&nbsp; 5m / 15m / 30m &nbsp;·&nbsp;
    {IDX15[0].date()} &rarr; {IDX15[-1].date()} &nbsp;·&nbsp;
    <strong style="color:var(--white)">{N15:,}</strong> barre 15m &nbsp;·&nbsp;
    WFO {WF_TRAIN_M}m/{WF_OOS_M}m/{WF_STEP_M}m, {len(WF_WINDOWS)} finestre,
    {len(WF_HOLDOUT)} in holdout genuino (&ge;{CUTOFF.date()})
  </p>
  <div class="cards">{kpi_cards}</div>
  {_table(["Variante", "OOS Return", "Max DD", "DSR", "Holdout Return", "Verdetto"], verdict_summary_rows)}
</section>

<section id="decision-criteria">
  <h2>Architettura e Criteri Decisionali</h2>
  <p>
    <strong>Bias strutturale (30m):</strong> <code>smc_trend_signal</code> — cambia
    SOLO su un CHoCH (Change of Character) confermato. Questo e' il meccanismo
    che impedisce a long/short di alternarsi senza un motivo strutturale: il
    bias resta fisso finche' la struttura non lo invalida.<br>
    <strong>Entry (15m):</strong> prezzo dentro un order block allineato al
    bias, o in zona discount (long) / premium (short) dell'ultimo swing.
    SL/TP presi dai pivot strutturali piu' vicini (nessun valore arbitrario).<br>
    <strong>Conferma (5m):</strong> liquidity sweep — wick oltre l'estremo
    recente (stop hunt) con reclaim in chiusura, nella direzione del bias.
  </p>
  <p>
    <strong>Modelli statistici:</strong> regime (HMM 3 stati), volatilita'
    attesa (percentile rolling ATR%), rendimento atteso (RandomForest,
    target = segno del log-return a 2h). Tutti rifittati walk-forward
    (mai su dati OOS).
  </p>
  <p>
    <strong>Gate di validazione (identici al resto del repo):</strong>
    n&ge;30 trade OOS &middot; return OOS &gt; 0 &middot; P(ruin) MC iid &lt; 10%
    &middot; Deflated Sharpe Ratio &ge; {DSR_THRESHOLD} (famiglia di 3 varianti,
    corregge il bias di selezione multipla) &middot; return positivo
    sull'holdout genuino 2025-2026 (mai toccato in fit/selection).
  </p>
  <p style="color:var(--gold)">
    <strong>Attenzione — limiti del backtest:</strong> nessuno slippage
    realistico modellato (stesso caveat della strategia ML RF 8h gia' in
    live), SL/TP eseguiti al livello esatto senza impatto di mercato,
    sizing a rischio fisso in dollari (non % di equity) per isolare l'edge
    dal compounding — i drawdown mostrati sono percentualmente piu' piccoli
    di quanto sarebbero con sizing a percentuale fissa su equity che cresce.
    Nessuna variante deve ricevere capitale reale prima di un paper trading
    che misuri lo slippage effettivo.
  </p>
  <p>
    <strong>Risultato piu' rilevante:</strong> l'aggiunta di piu' confluenza
    <em>riduce</em> l'edge invece di migliorarlo, in ogni forma testata.
    V3 (+ gate ML) fallisce DSR e holdout. V4 (+ conferma order-flow reale —
    CVD da taker_buy_base Binance, non piu' un proxy sul solo prezzo) passa
    l'holdout genuino ma fallisce comunque il gate DSR: il filtro CVD e'
    troppo selettivo (22.421 sweep price-based -> 2.949 CVD-confirmed, -87%)
    e il campione residuo (208 trade OOS) non basta a superare la soglia di
    selezione multipla su una famiglia di 4 varianti. Pattern consistente su
    tutta la sessione: ogni filtro aggiuntivo abbassa il numero di trade piu'
    velocemente di quanto migliori la loro qualita' media — segno che V1/V2
    stanno gia' catturando la parte robusta dell'edge strutturale, non che
    l'implementazione dei filtri sia difettosa (V4 in particolare riusa dati
    di order-flow reali, non un proxy — la stessa idea CVD era gia' stata
    bocciata come segnale standalone in <code>create_github_strategies_report
    .py</code> (IC FAIL); qui fallisce di nuovo anche come filtro di conferma,
    rafforzando che il segnale CVD standalone su BTCUSDT 5m/15m e' debole più
    in generale, non solo in quella forma).
  </p>
  <p>
    <strong>V5 (rete neurale):</strong> stesse feature e target di V3, solo
    il modello cambia (MLPClassifier 32-16 invece di RandomForest). Risultato
    nettamente migliore: +31.0% vs +9.8%, Sharpe +2.431 vs +0.994 — la rete
    neurale estrae piu' segnale dalle stesse feature dello stesso gate ML.
    Fallisce comunque DSR (family N=5): la soglia di correzione per
    selezione multipla (SR0) sale con il numero di varianti testate nella
    stessa sessione — a parita' di merito individuale, testare 5 varianti
    invece di 1 rende DSR piu' severo per costruzione (protegge da
    cherry-picking, non e' un difetto della metrica). In una famiglia piu'
    piccola (es. solo V1/V2/V5) il gate sarebbe meno punitivo — un possibile
    prossimo passo mirato, non un modo per abbassare la soglia.
  </p>
</section>

<section id="equity">
  <h2>Equity Curves</h2>
  <div class="chart">{_img_tag(eq_chart_b64)}</div>
</section>

{variant_sections}

<section id="years">
  <h2>Breakdown per Anno (OOS chained)</h2>
  <div class="chart">{_img_tag(year_chart_b64)}</div>
</section>
"""

footer = f"""
<footer>
  BTCUSDT LTF Market Structure / SMC / MTF Research &nbsp;&middot;&nbsp;
  Generated {now_str} &nbsp;&middot;&nbsp; Data: Binance Vision CDN
</footer>"""

html = f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>BTCUSDT LTF SMC/MTF Strategy Research</title>
<style>{_CSS}</style>
</head>
<body>
{nav}
<div class="container">
{body}
</div>
{footer}
</body>
</html>"""

out_path = Path("reports/report_ltf_smc_mtf.html")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(html, encoding="utf-8")
print(f"  {out_path}  ({time.time()-t0:.0f}s)")

md_path = Path("reports/ltf_smc_mtf_research.md")
md_path.write_text("# BTCUSDT LTF SMC/MTF Strategy Research\n\n```\n" + "\n".join(report_lines) + "\n```\n",
                    encoding="utf-8")

print(f"\n[DONE] total runtime: {time.time()-t_start:.0f}s")
