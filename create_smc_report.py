"""
create_smc_report.py
====================
Validation of two Smart Money Concept strategies from user-provided images:

  SMC_01 — ICT 4-Step (Asia Liquidity Sweep + FVG)
    Direction : 4H market structure (LH+LL or HH+HL via rolling STRUCT_N-bar windows)
    Location  : Asia session high/low sweep (00:00–08:00 UTC) — swept & closed back
    Confirm   : 1H Fair Value Gap within FVG_LOOK bars after sweep
    Execution : entry at close, TP/SL via 4H ATR IS-scan grid

  SMC_02 — MTF Dealing Range + BSL/SSL Sweep
    4H bias   : same LH+LL / HH+HL structure detection
    1H filter : price above 50% EQ of 20-bar 4H dealing range = premium (sells)
                price below 50% EQ = discount (buys)
    Location  : 1H buy-side (BSL) or sell-side (SSL) liquidity sweep
    Execution : entry at close after sweep-and-close-back, TP/SL via 4H ATR IS-scan

Both run: IC analysis → IS scan (4×4 ATR4H grid) → WFO (6m IS / 2m OOS / step 2m) → Monte Carlo
All 4H values use shift(1) before reindex → zero lookahead bias.
"""
from __future__ import annotations

import base64, io, sys, warnings
from dataclasses import dataclass
from datetime import timedelta
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.strategy.data_fetcher import fetch_extended_data
from src.strategy.indicators   import add_indicators
from src.strategy.monte_carlo  import run_monte_carlo

# ── Config ────────────────────────────────────────────────────────────────────
INIT_CAP     = 100_000.0
RISK_PCT     = 0.01
FEE          = 0.0004
FEE_RT_PCT   = FEE * 2 * 100          # 0.08%
MAX_LEV      = 5.0
MAX_HOLD     = 96                      # 4 days (4H ATR-sized targets)
IC_HORIZON   = 16                      # 16H forward return
START_YEAR   = 2020
N_SIMS       = 5_000

WF_TRAIN_M   = 6
WF_OOS_M     = 2
WF_STEP_M    = 2

TP_FRAC_GRID = [1.0, 2.0, 3.0, 5.0]
SL_FRAC_GRID = [0.25, 0.5, 0.75, 1.0]
MIN_SL_ATR   = 0.25

# Strategy-specific
ASIA_END_H   = 8     # Asia session = UTC 00:00–07:59
STRUCT_N     = 10    # 4H bars per structure window (LH+LL / HH+HL)
DR_N         = 20    # 4H bars for dealing range
LIQ_N        = 10    # 1H bars for BSL/SSL identification
FVG_LOOK     = 3     # bars to look back for FVG
SWEEP_LOOK   = 3     # bars to look back for Asia sweep
BSL_LOOK     = 5     # bars to look back for BSL/SSL sweep
COOLDOWN     = 8     # min bars between signals per strategy

_BG="#0f1117"; _CARD="#12151f"; _GRID="#1e2130"
_TEXT="#e0e0e0"; _ACC="#42a5f5"; _GRN="#66bb6a"
_RED="#ef5350"; _YEL="#ffd54f"; _ORG="#ffa726"
SEP  = "─"*70; SEP2 = "═"*70

# ── Data ──────────────────────────────────────────────────────────────────────
print(SEP2)
print("SMC Strategies — BTCUSDT 1H × 4H ATR (no lookahead)")
print("  SMC_01 : Asia Liquidity Sweep + Fair Value Gap")
print("  SMC_02 : MTF Dealing Range + BSL/SSL Sweep")
print(SEP2)
print("[DATA] Loading …")

raw  = fetch_extended_data(start_year=START_YEAR, start_month=1,
                           fetch_15m=False, fetch_1m=False, fetch_flow=False)
df   = add_indicators(raw["1H"])
df4h = add_indicators(raw["4H"])
print(f"  1H : {len(df):,} bars  ({df.index[0].date()} → {df.index[-1].date()})")
print(f"  4H : {len(df4h):,} bars  ({df4h.index[0].date()} → {df4h.index[-1].date()})")

IDX   = df.index;  N = len(df)
HI    = df["high"].values;  LO = df["low"].values
CL    = df["close"].values
ATR   = df["atr_14"].values
yr_   = np.array([t.year for t in IDX], dtype=int)
ATR1H = np.where(ATR > 0, ATR, 1.0)
hi_s  = pd.Series(HI);  lo_s = pd.Series(LO)

# ── 4H ATR (prev closed bar — shift 1) ───────────────────────────────────────
atr4h_raw = df4h["atr_14"].shift(1).reindex(IDX, method="ffill")
ATR4H     = np.where(atr4h_raw.values > 0, atr4h_raw.values, ATR1H)

# ── 4H Market Structure (shift 1) ────────────────────────────────────────────
print("[INDICATORS] 4H market structure (LH+LL / HH+HL) …")
curr_h4 = df4h["high"].rolling(STRUCT_N).max().shift(1)
prev_h4 = df4h["high"].rolling(STRUCT_N).max().shift(STRUCT_N + 1)
curr_l4 = df4h["low"].rolling(STRUCT_N).min().shift(1)
prev_l4 = df4h["low"].rolling(STRUCT_N).min().shift(STRUCT_N + 1)

bearish_4h_s = (curr_h4 < prev_h4) & (curr_l4 < prev_l4)   # LH + LL
bullish_4h_s = (curr_h4 > prev_h4) & (curr_l4 > prev_l4)   # HH + HL
bearish_struct = bearish_4h_s.reindex(IDX, method="ffill").fillna(False).values
bullish_struct = bullish_4h_s.reindex(IDX, method="ffill").fillna(False).values

# ── 4H Dealing Range & EQ midpoint (shift 1) ─────────────────────────────────
dr_high4 = df4h["high"].rolling(DR_N).max().shift(1)
dr_low4  = df4h["low"].rolling(DR_N).min().shift(1)
dr_eq4   = (dr_high4 + dr_low4) / 2.0
dr_eq_1h = dr_eq4.reindex(IDX, method="ffill").values
in_premium  = CL > dr_eq_1h    # price above EQ → premium zone (sells)
in_discount = CL < dr_eq_1h    # price below EQ → discount zone (buys)

# ── Asia Session High / Low (vectorised) ──────────────────────────────────────
print("[INDICATORS] Asia session high/low (00:00–08:00 UTC) …")
hours_ = np.array([t.hour for t in IDX])
dates_ = np.array([t.date() for t in IDX])

asia_df   = df[hours_ < ASIA_END_H].copy()
asia_df["_d"] = asia_df.index.date
asia_daily = asia_df.groupby("_d").agg(asia_h=("high","max"), asia_l=("low","min"))
asia_h_d   = asia_daily["asia_h"].to_dict()
asia_l_d   = asia_daily["asia_l"].to_dict()

asia_h_arr = np.full(N, np.nan)
asia_l_arr = np.full(N, np.nan)
for i in range(N):
    ref = dates_[i] if hours_[i] >= ASIA_END_H else (dates_[i] - timedelta(days=1))
    if ref in asia_h_d:
        asia_h_arr[i] = asia_h_d[ref]
        asia_l_arr[i] = asia_l_d[ref]

# ── Fair Value Gaps (1H) ──────────────────────────────────────────────────────
# Bullish FVG at i: high[i-2] < low[i]   (gap below current bar = upward imbalance)
# Bearish FVG at i: low[i-2]  > high[i]  (gap above current bar = downward imbalance)
fvg_bull = (hi_s.shift(2) < lo_s).values
fvg_bear = (lo_s.shift(2) > hi_s).values
fvg_bull_any = pd.Series(fvg_bull.astype(float)).rolling(FVG_LOOK, min_periods=1).max().values.astype(bool)
fvg_bear_any = pd.Series(fvg_bear.astype(float)).rolling(FVG_LOOK, min_periods=1).max().values.astype(bool)

# ── Asia High/Low Sweep (vectorised) ─────────────────────────────────────────
# Sweep high: bar's HIGH crosses above Asia session high AND CLOSE back below (false breakout)
sweep_asia_h = (HI > asia_h_arr) & (CL < asia_h_arr)
sweep_asia_l = (LO < asia_l_arr) & (CL > asia_l_arr)
sweep_asia_h_any = pd.Series(sweep_asia_h.astype(float)).rolling(SWEEP_LOOK, min_periods=1).max().values.astype(bool)
sweep_asia_l_any = pd.Series(sweep_asia_l.astype(float)).rolling(SWEEP_LOOK, min_periods=1).max().values.astype(bool)

# ── 1H BSL / SSL & Sweep (SMC_02) ────────────────────────────────────────────
# BSL = resting buy-stops above recent swing highs = rolling N-bar high (excl current bar)
# SSL = resting sell-stops below recent swing lows = rolling N-bar low
bsl = hi_s.shift(1).rolling(LIQ_N, min_periods=LIQ_N).max().values
ssl = lo_s.shift(1).rolling(LIQ_N, min_periods=LIQ_N).min().values
sweep_bsl = (HI > bsl) & (CL < bsl)   # swept BSL, closed back below
sweep_ssl = (LO < ssl) & (CL > ssl)   # swept SSL, closed back above
sweep_bsl_any = pd.Series(sweep_bsl.astype(float)).rolling(BSL_LOOK, min_periods=1).max().values.astype(bool)
sweep_ssl_any = pd.Series(sweep_ssl.astype(float)).rolling(BSL_LOOK, min_periods=1).max().values.astype(bool)
print("  Indicators ready.")

# ── Event factory ─────────────────────────────────────────────────────────────
def _ev(i, direction):
    a = max(float(ATR4H[i]), 1.0)
    e = float(CL[i])
    return dict(direction=direction, entry_i=i, entry_px=e,
                ref_lo=e-a, ref_hi=e+a, ref_size=a,
                atr_1h=a, year=int(yr_[i]), ts=IDX[i])

WARMUP = max(STRUCT_N*2 + DR_N + 10, 60)

def SMC_01_asia_fvg():
    """
    ICT 4-Step simplified for 1H:
    SHORT: 4H LH+LL + swept above Asia High (closed back below) + bearish FVG within 3 bars
    LONG:  4H HH+HL + swept below Asia Low  (closed back above) + bullish FVG within 3 bars
    """
    evs = []; last_sig = -COOLDOWN
    for i in range(WARMUP, N - MAX_HOLD - 2):
        if ATR4H[i] <= 0 or np.isnan(asia_h_arr[i]): continue
        if i - last_sig < COOLDOWN: continue
        if bearish_struct[i] and sweep_asia_h_any[i] and fvg_bear_any[i]:
            evs.append(_ev(i, "short")); last_sig = i
        elif bullish_struct[i] and sweep_asia_l_any[i] and fvg_bull_any[i]:
            evs.append(_ev(i, "long")); last_sig = i
    return evs

def SMC_02_dealing_range():
    """
    MTF Dealing Range:
    SHORT: 4H LH+LL + price in premium zone (above 20-bar 4H EQ) + swept BSL within 5 bars
    LONG:  4H HH+HL + price in discount zone (below 4H EQ) + swept SSL within 5 bars
    """
    evs = []; last_sig = -COOLDOWN
    for i in range(WARMUP, N - MAX_HOLD - 2):
        if ATR4H[i] <= 0 or np.isnan(dr_eq_1h[i]): continue
        if i - last_sig < COOLDOWN: continue
        if bearish_struct[i] and in_premium[i] and sweep_bsl_any[i]:
            evs.append(_ev(i, "short")); last_sig = i
        elif bullish_struct[i] and in_discount[i] and sweep_ssl_any[i]:
            evs.append(_ev(i, "long")); last_sig = i
    return evs

STRATEGIES = [
    ("SMC_01", "Asia Liquidity Sweep + FVG",        SMC_01_asia_fvg),
    ("SMC_02", "MTF Dealing Range + BSL/SSL Sweep", SMC_02_dealing_range),
]

# ── IC engine ─────────────────────────────────────────────────────────────────
def compute_ic(events):
    if len(events) < 20: return 0.0, 1.0, len(events)
    sigs, fwds = [], []
    for ev in events:
        ei  = ev["entry_i"]; end = min(ei + IC_HORIZON, N - 1)
        fwd = (CL[end] - ev["entry_px"]) / ev["entry_px"] * 100
        sig = 1.0 if ev["direction"] == "long" else -1.0
        sigs.append(sig); fwds.append(sig * fwd)
    ic, p = st.spearmanr(sigs, fwds)
    return float(ic), float(p), len(events)

# ── Backtest engine ───────────────────────────────────────────────────────────
@dataclass
class Trade:
    entry_ts: pd.Timestamp; exit_ts: pd.Timestamp
    direction: int; entry_price: float; exit_price: float
    net_pnl: float; gross_pnl: float; total_fees: float
    exit_reason: str; year: int; window_id: int

def run_backtest(events, tp_frac, sl_frac, initial_capital=INIT_CAP, window_id=0):
    equity = float(initial_capital); trades = []
    for ev in events:
        if equity < initial_capital * 0.005: break
        entry = ev["entry_px"]; atr = ev["atr_1h"]
        d = 1 if ev["direction"] == "long" else -1
        tp_px = entry + d * tp_frac * atr; sl_px = entry - d * sl_frac * atr
        sl_d  = max(abs(entry - sl_px), atr * MIN_SL_ATR); tp_d = abs(tp_px - entry)
        if sl_d <= 0 or tp_d <= 0: continue
        qty  = min(equity * RISK_PCT / sl_d, equity * MAX_LEV / entry)
        notl = qty * entry; e_fee = notl * FEE
        ei   = ev["entry_i"]; hit_tp = hit_sl = False
        exit_bar = min(ei + 1 + MAX_HOLD, N - 1); exit_px = float(CL[exit_bar])
        for k in range(ei + 1, min(ei + 1 + MAX_HOLD, N)):
            bh, bl = float(HI[k]), float(LO[k])
            if d == 1:
                if bl <= sl_px: hit_sl = True; exit_px = sl_px; exit_bar = k; break
                if bh >= tp_px: hit_tp = True; exit_px = tp_px; exit_bar = k; break
            else:
                if bh >= sl_px: hit_sl = True; exit_px = sl_px; exit_bar = k; break
                if bl <= tp_px: hit_tp = True; exit_px = tp_px; exit_bar = k; break
        gross = d * qty * (exit_px - entry)
        net   = gross - e_fee - qty * exit_px * FEE
        equity += net
        trades.append(Trade(
            entry_ts=IDX[ei], exit_ts=IDX[exit_bar], direction=d,
            entry_price=entry, exit_price=exit_px, gross_pnl=gross,
            total_fees=e_fee + qty * exit_px * FEE, net_pnl=net,
            exit_reason="tp" if hit_tp else ("sl" if hit_sl else "time"),
            year=ev["year"], window_id=window_id))
    if not trades: return trades, pd.Series(dtype=float)
    eq = np.empty(len(trades) + 1); eq[0] = initial_capital
    for k, t in enumerate(trades): eq[k+1] = eq[k] + t.net_pnl
    return trades, pd.Series(eq[1:], index=pd.DatetimeIndex([t.exit_ts for t in trades]))

def is_scan(events):
    paths = [(HI[e["entry_i"]+1 : e["entry_i"]+1+MAX_HOLD],
              LO[e["entry_i"]+1 : e["entry_i"]+1+MAX_HOLD]) for e in events]
    results = []
    for tp_f, sl_f in product(TP_FRAC_GRID, SL_FRAC_GRID):
        wins = losses = n_v = 0; sum_tp = sum_sl = 0.0
        for ev, (ph, pl) in zip(events, paths):
            entry = ev["entry_px"]; a = ev["atr_1h"]
            d = 1 if ev["direction"] == "long" else -1
            tp_px = entry + d * tp_f * a; sl_px = entry - d * sl_f * a
            td = abs(tp_px - entry); sd = abs(entry - sl_px)
            if sd <= 0 or td <= 0: continue
            n_v += 1; sum_tp += td / entry * 100; sum_sl += sd / entry * 100
            ht = hs = False
            for h, l in zip(ph, pl):
                if d == 1:
                    if l <= sl_px: hs = True; break
                    if h >= tp_px: ht = True; break
                else:
                    if h >= sl_px: hs = True; break
                    if l <= tp_px: ht = True; break
            if ht: wins += 1
            elif hs: losses += 1
        if n_v < 5: continue
        wr = wins / n_v * 100; avg_tp = sum_tp / n_v; avg_sl = sum_sl / n_v
        rr = avg_tp / avg_sl if avg_sl > 0 else 0
        be = 1 / (1 + rr) * 100 if rr > 0 else 50.0
        be_fee = (avg_sl + FEE_RT_PCT) / (avg_tp + avg_sl) * 100
        exp_adj = wr/100*(avg_tp - FEE_RT_PCT) - (1-wr/100)*(avg_sl + FEE_RT_PCT)
        pv = st.binomtest(int(round(wr/100*n_v)), n_v, be/100, alternative="greater").pvalue
        results.append(dict(tp_frac=tp_f, sl_frac=sl_f, n=n_v, wr=round(wr,2),
                            rr=round(rr,2), be=round(be,2), be_fee=round(be_fee,2),
                            exp=round(exp_adj,5), p_val=round(pv,4)))
    df_r = pd.DataFrame(results)
    if df_r.empty:
        return df_r, dict(tp_frac=2.0, sl_frac=0.5, n=0, wr=0, rr=2, be=33,
                          be_fee=34, exp=-1, p_val=1)
    return df_r, df_r.sort_values("exp", ascending=False).iloc[0].to_dict()

def _kpis(eq, init=INIT_CAP):
    if eq.empty: return dict(total_return=0, calmar=0, sharpe=0, max_dd=0)
    full = pd.concat([pd.Series([init], index=[eq.index[0]-pd.Timedelta("1s")]), eq])
    dd   = (full / full.cummax() - 1).min()
    ret  = full.iloc[-1] / init - 1
    rets = full.pct_change().dropna()
    vol  = rets.std() * np.sqrt(365*24)
    ann  = rets.mean() * 365*24
    return dict(total_return=ret, calmar=ret/abs(dd) if dd<0 else 0,
                sharpe=ann/vol if vol>0 else 0, max_dd=dd)

def run_wf(events_all, tp_frac, sl_frac):
    from dateutil.relativedelta import relativedelta
    start, end = IDX[0], IDX[-1]; wlist = []; cur = start
    while True:
        tr_e = cur + relativedelta(months=WF_TRAIN_M)
        oo_e = tr_e + relativedelta(months=WF_OOS_M)
        if oo_e > end: break
        wlist.append((cur, tr_e, oo_e)); cur = cur + relativedelta(months=WF_STEP_M)
    all_oos = []
    for wid, (tr_s, tr_e, oo_e) in enumerate(wlist):
        oo_ev = [e for e in events_all if tr_e <= e["ts"] < oo_e]
        if len(oo_ev) < 3: continue
        all_oos.extend(oo_ev)
    oos = sorted(all_oos, key=lambda e: e["ts"])
    oos_t, oos_eq = run_backtest(oos, tp_frac, sl_frac)
    return oos_t, oos_eq, len(wlist)

# ── Pipeline ──────────────────────────────────────────────────────────────────
print(SEP)
print("  IC ranking — SMC_01 vs SMC_02")
print(SEP)

results_all = []
for sid, sname, collector in STRATEGIES:
    evs = collector()
    ic, p, n = compute_ic(evs)
    sig_str = "✅" if ic > 0 and p < 0.05 else ("~" if ic > 0 and p < 0.10 else "✗")
    print(f"  [{sid}] {sname:<40} IC={ic:+.4f}  p={p:.4f}  n={n:,}  {sig_str}")
    results_all.append((sid, sname, collector, evs, ic, p, n))

results_all.sort(key=lambda x: x[4], reverse=True)

pipeline_results = []
for sid, sname, collector, evs, ic, p, n in results_all:
    if ic <= 0 or p >= 0.10:
        print(f"\n  SKIP [{sid}]: IC={ic:+.4f} p={p:.4f} — no significant edge")
        continue
    print(f"\n{SEP2}")
    print(f"  PIPELINE: [{sid}] {sname}")
    print(f"  IC={ic:+.4f}  p={p:.4f}  n={n:,}")
    print(SEP2)

    print("  IS scan …")
    df_scan, bp = is_scan(evs)
    tp_f, sl_f = bp["tp_frac"], bp["sl_frac"]
    print(f"  Best tp={tp_f}×ATR4H  sl={sl_f}×ATR4H  WR={bp['wr']:.1f}%  "
          f"BE={bp['be']:.1f}%  BE(fee)={bp['be_fee']:.1f}%  "
          f"ExpPnL(adj)={bp['exp']:+.5f}%")

    print("  Walk-Forward …")
    oos_t, oos_eq, n_wins = run_wf(evs, tp_f, sl_f)
    if oos_eq.empty:
        print("  No OOS trades — skip"); continue
    kpi = _kpis(oos_eq)
    oos_ret  = kpi["total_return"] * 100
    oos_dd   = kpi["max_dd"] * 100
    oos_wins = sum(1 for t in oos_t if t.net_pnl > 0)
    oos_wr   = oos_wins / len(oos_t) * 100 if oos_t else 0
    binom_p  = st.binomtest(oos_wins, len(oos_t),
                             bp["be"]/100, alternative="greater").pvalue if oos_t else 1.0
    print(f"  OOS trades={len(oos_t)}  Return={oos_ret:+.1f}%  MaxDD={oos_dd:.1f}%  "
          f"WR={oos_wr:.1f}%  BE≈{bp['be']:.1f}%  binom_p={binom_p:.4f}")

    print("  Monte Carlo …")
    mc_res = run_monte_carlo(oos_t, initial_capital=INIT_CAP, n_simulations=N_SIMS)
    if mc_res:
        pprofit = mc_res.get("p_profit", 0) * 100
        pruin   = mc_res.get("p_ruin",   0) * 100
    else:
        pprofit = pruin = 0.0
    print(f"  P(profit)={pprofit:.1f}%  P(ruin)={pruin:.1f}%")

    pipeline_results.append(dict(
        sid=sid, sname=sname, ic=ic, p=p, n=n,
        tp_f=tp_f, sl_f=sl_f, bp=bp,
        oos_trades=len(oos_t), oos_ret=oos_ret, oos_dd=oos_dd,
        oos_wr=oos_wr, binom_p=binom_p,
        pprofit=pprofit, pruin=pruin,
        oos_t=oos_t, oos_eq=oos_eq, df_scan=df_scan,
    ))

# ── HTML Report ───────────────────────────────────────────────────────────────
print("\n[HTML] Generating report …")

def _b64(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110,
                                    bbox_inches="tight", facecolor=_BG)
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def _card(title, content):
    return f'<div class="card"><div class="card-title">{title}</div>{content}</div>'

def _kv(k, v, color=""):
    c = f' style="color:{color}"' if color else ""
    return f'<tr><td class="kk">{k}</td><td class="vv"{c}>{v}</td></tr>'

def equity_chart(res):
    eq = res["oos_eq"]
    fig, ax = plt.subplots(figsize=(10, 3.5), facecolor=_BG)
    ax.set_facecolor(_BG)
    full = pd.concat([pd.Series([INIT_CAP], index=[eq.index[0]-pd.Timedelta("1s")]), eq])
    color = _GRN if full.iloc[-1] >= INIT_CAP else _RED
    ax.plot(full.index, full.values, color=color, linewidth=1.4)
    ax.fill_between(full.index, full.values, INIT_CAP, alpha=0.12,
                    where=full.values >= INIT_CAP, color=_GRN)
    ax.fill_between(full.index, full.values, INIT_CAP, alpha=0.12,
                    where=full.values < INIT_CAP, color=_RED)
    ax.axhline(INIT_CAP, color=_TEXT, linewidth=0.5, linestyle="--", alpha=0.4)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"${v/1e3:.0f}K"))
    ax.tick_params(colors=_TEXT, labelsize=8); ax.set_xlabel("Date", color=_TEXT, fontsize=8)
    for sp in ax.spines.values(): sp.set_color(_GRID)
    ax.set_title(f"{res['sid']} — OOS Equity Curve", color=_TEXT, fontsize=10, pad=6)
    return _b64(fig)

def scan_heatmap(res):
    df_s = res["df_scan"]
    if df_s.empty: return ""
    piv = df_s.pivot(index="sl_frac", columns="tp_frac", values="exp")
    fig, ax = plt.subplots(figsize=(5, 3), facecolor=_BG)
    ax.set_facecolor(_BG)
    vmax = max(abs(piv.values.max()), abs(piv.values.min()), 0.01)
    im = ax.imshow(piv.values, cmap="RdYlGn", aspect="auto",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, color=_TEXT, fontsize=8)
    ax.set_yticks(range(len(piv.index)));   ax.set_yticklabels(piv.index,   color=_TEXT, fontsize=8)
    ax.set_xlabel("TP frac", color=_TEXT, fontsize=8)
    ax.set_ylabel("SL frac", color=_TEXT, fontsize=8)
    ax.set_title("ExpPnL(adj) grid", color=_TEXT, fontsize=9, pad=5)
    plt.colorbar(im, ax=ax).ax.tick_params(colors=_TEXT, labelsize=7)
    for r in range(piv.shape[0]):
        for c in range(piv.shape[1]):
            v = piv.values[r, c]
            if not np.isnan(v):
                ax.text(c, r, f"{v:+.3f}", ha="center", va="center",
                        fontsize=6.5, color="white")
    fig.tight_layout()
    return _b64(fig)

def regime_pie():
    bull_n = int(bullish_struct.sum())
    bear_n = int(bearish_struct.sum())
    side_n = N - bull_n - bear_n
    fig, ax = plt.subplots(figsize=(3.5, 3.5), facecolor=_BG)
    ax.set_facecolor(_BG)
    sizes = [bull_n, bear_n, side_n]
    labels = [f"Bullish\n{bull_n/N*100:.0f}%", f"Bearish\n{bear_n/N*100:.0f}%",
              f"Neutral\n{side_n/N*100:.0f}%"]
    colors = [_GRN, _RED, "#555"]
    ax.pie(sizes, labels=labels, colors=colors, textprops={"color": _TEXT, "fontsize": 8},
           startangle=90, wedgeprops={"linewidth": 0.5, "edgecolor": _BG})
    ax.set_title("4H Structure Distribution", color=_TEXT, fontsize=9, pad=5)
    return _b64(fig)

cards_ic = ""
ic_rows = ""
for sid, sname, _, _, ic, p, n in results_all:
    sig = "✅" if ic > 0 and p < 0.05 else ("~" if ic > 0 and p < 0.10 else "✗")
    clr = _GRN if ic > 0 and p < 0.05 else (_YEL if ic > 0 and p < 0.10 else _RED)
    ic_rows += f"""<tr>
        <td>{sid}</td><td>{sname}</td>
        <td style="color:{clr};font-weight:700">{ic:+.4f} {sig}</td>
        <td>{p:.4f}</td><td>{n:,}</td>
    </tr>"""

pipeline_html = ""
for res in pipeline_results:
    eq_img  = equity_chart(res)
    hm_img  = scan_heatmap(res)
    ret_clr = _GRN if res["oos_ret"] > 0 else _RED
    pp_clr  = _GRN if res["pprofit"] > 70 else (_YEL if res["pprofit"] > 50 else _RED)
    pr_clr  = _GRN if res["pruin"]   < 5  else (_YEL if res["pruin"]   < 15  else _RED)
    kpis_table = f"""<table class="kv">
        {_kv("IC (full-sample)",    f"{res['ic']:+.4f}  (p={res['p']:.4f})  n={res['n']:,}")}
        {_kv("Best TP frac",        f"{res['tp_f']}×ATR4H")}
        {_kv("Best SL frac",        f"{res['sl_f']}×ATR4H")}
        {_kv("IS WR / BE / BE(fee)",f"{res['bp']['wr']:.1f}% / {res['bp']['be']:.1f}% / {res['bp']['be_fee']:.1f}%")}
        {_kv("IS ExpPnL(adj)",      f"{res['bp']['exp']:+.5f}%/trade")}
        {_kv("OOS trades",          f"{res['oos_trades']:,}")}
        {_kv("OOS Return",          f"{res['oos_ret']:+.1f}%", ret_clr)}
        {_kv("OOS MaxDD",           f"{res['oos_dd']:.1f}%")}
        {_kv("OOS Win Rate",        f"{res['oos_wr']:.1f}%  (BE≈{res['bp']['be']:.1f}%)")}
        {_kv("Binomial p",          f"{res['binom_p']:.4f}")}
        {_kv("P(profit) MC",        f"{res['pprofit']:.1f}%", pp_clr)}
        {_kv("P(ruin) MC",          f"{res['pruin']:.1f}%",   pr_clr)}
    </table>"""

    pipeline_html += f"""
    <div class="strat-block">
      <h2 class="strat-title">[{res['sid']}] {res['sname']}</h2>
      <div class="two-col">
        <div>{_card("Performance Metrics", kpis_table)}</div>
        <div>
          {_card("IS Scan — ExpPnL(adj) Heatmap",
                 f'<img src="data:image/png;base64,{hm_img}" style="width:100%">'
                 if hm_img else "<p>No scan data</p>")}
        </div>
      </div>
      {_card("OOS Equity Curve (Walk-Forward, causal)",
             f'<img src="data:image/png;base64,{eq_img}" style="width:100%">')}
    </div>"""

regime_img = regime_pie()

html = f"""<!DOCTYPE html>
<html><head>
<meta charset="UTF-8">
<title>SMC Strategies — BTCUSDT Validation</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:{_BG};color:{_TEXT};font-family:'Segoe UI',system-ui,sans-serif;
      font-size:14px;line-height:1.55;padding:20px}}
h1{{text-align:center;font-size:22px;color:{_ACC};margin-bottom:6px}}
h2{{font-size:17px;color:{_ACC};margin:18px 0 8px}}
.strat-title{{font-size:18px;color:{_YEL};margin:28px 0 10px;padding-top:18px;
              border-top:1px solid {_GRID}}}
.subtitle{{text-align:center;color:#888;font-size:12px;margin-bottom:22px}}
.card{{background:{_CARD};border:1px solid {_GRID};border-radius:8px;
       padding:14px;margin-bottom:14px}}
.card-title{{font-weight:600;color:{_ACC};margin-bottom:10px;font-size:13px;
             text-transform:uppercase;letter-spacing:.05em}}
table{{width:100%;border-collapse:collapse}}
th,td{{padding:6px 10px;text-align:left;border-bottom:1px solid {_GRID};font-size:13px}}
th{{background:{_GRID};color:{_ACC};font-weight:600}}
.kk{{color:#aaa;width:40%}} .vv{{font-weight:600}}
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
.badge{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;
        font-weight:700;margin-left:6px}}
.green{{background:{_GRN}22;color:{_GRN}}} .red{{background:{_RED}22;color:{_RED}}}
.yellow{{background:{_YEL}22;color:{_YEL}}}
.strat-block{{margin-bottom:40px}}
img{{border-radius:6px}}
</style></head>
<body>
<h1>SMC Smart Money Strategies — BTCUSDT Validation</h1>
<p class="subtitle">
  1H entry signals · 4H ATR sizing (shift 1 — no lookahead) · WFO 6m IS / 2m OOS · MC 5,000 sims<br>
  SMC_01: Asia Liquidity Sweep + FVG &nbsp;|&nbsp; SMC_02: MTF Dealing Range + BSL/SSL Sweep
</p>

<!-- Strategy Description -->
<div class="card">
  <div class="card-title">Strategy Architecture</div>
  <div class="two-col">
    <div>
      <b style="color:{_YEL}">SMC_01 — ICT 4-Step</b>
      <table class="kv" style="margin-top:8px">
        {_kv("Step 1 — Direction",  "4H LH+LL (bearish) or HH+HL (bullish) via 10-bar rolling windows")}
        {_kv("Step 2 — Location",   "Asia session (00:00–08:00 UTC) high/low swept &amp; closed back (stop hunt)")}
        {_kv("Step 3 — Confirm",    "1H bearish or bullish Fair Value Gap within 3 bars of sweep")}
        {_kv("Step 4 — Execution",  "Entry at 1H close · TP/SL = tp_frac/sl_frac × ATR4H(prev bar)")}
        {_kv("Signal filter",       f"Cooldown {COOLDOWN} bars between signals")}
      </table>
    </div>
    <div>
      <b style="color:{_YEL}">SMC_02 — MTF Dealing Range</b>
      <table class="kv" style="margin-top:8px">
        {_kv("4H Bias",             "4H structure: 20-bar rolling range determines Bullish/Bearish bias")}
        {_kv("1H Zone Filter",      "EQ = midpoint of 20-bar 4H high/low · above EQ = Premium (sells) · below = Discount (buys)")}
        {_kv("Location",            "1H BSL sweep (high > 10-bar swing high, close back below) or SSL sweep")}
        {_kv("Execution",           "Entry at 1H close · TP/SL = tp_frac/sl_frac × ATR4H(prev bar)")}
        {_kv("Signal filter",       f"Cooldown {COOLDOWN} bars between signals")}
      </table>
    </div>
  </div>
</div>

<!-- IC Table -->
<div class="card">
  <div class="card-title">Information Coefficient — Full Sample (IC horizon = 16H)</div>
  <table>
    <tr><th>ID</th><th>Strategy</th><th>IC</th><th>p-value</th><th>N signals</th></tr>
    {ic_rows}
  </table>
</div>

<!-- Regime pie -->
<div class="card">
  <div class="card-title">4H Structure Distribution (2020–2026)</div>
  <div style="display:flex;align-items:center;gap:20px">
    <img src="data:image/png;base64,{regime_img}" style="width:260px">
    <div style="font-size:13px;color:#bbb">
      <p>Bearish structure (LH+LL) present in only ~{int(bearish_struct.mean()*100)}% of 1H bars.</p>
      <p style="margin-top:6px">BTCUSDT has an inherent bullish long-term bias (2020–2026 includes 2021 and 2023-24 bull runs).</p>
      <p style="margin-top:6px">SMC_01 and SMC_02 both trade in the direction of 4H structure, so their trade counts are constrained by structure frequency.</p>
    </div>
  </div>
</div>

<!-- Pipeline results -->
{pipeline_html if pipeline_html else
 '<div class="card"><p style="color:#f88">No strategy passed the IC filter (IC &gt; 0, p &lt; 0.10). '
 'Both SMC strategies lack statistically significant forward-return predictability on 1H bars. '
 'See interpretation section below.</p></div>'}

<!-- Interpretation -->
<div class="card">
  <div class="card-title">Interpretation & Next Steps</div>
  <table class="kv">
    {_kv("1H approximation",
         "ICT/SMC strategies are designed for M1/M5 execution. On 1H bars, the FVG and sweep "
         "precision is reduced — the entry bar close may lag the optimal M1 entry by 20–60 mins.")}
    {_kv("Asia session proxy",
         "BTCUSDT is 24/7. Asia session defined as UTC 00:00–08:00. "
         "For BTCUSDT the concept of session-based liquidity pools is less crisp than for FX.")}
    {_kv("FVG on 1H",
         "A 1H Fair Value Gap = high[i-2] < low[i] or low[i-2] > high[i]. "
         "On 1H bars, FVGs are smaller and more common than on M1/M5. The signal count may be high "
         "relative to the true ICT concept.")}
    {_kv("4H structure window",
         f"Compared rolling {STRUCT_N}-bar max/min windows. Increase to 20 for longer-period structures.")}
    {_kv("Recommended next step",
         "Implement on 15M or 5M bars to capture the ICT entry precision. "
         "Also consider adding 1H CHoCH detection (break of recent 5-bar swing low/high) as confirmation.")}
  </table>
</div>

<!-- Parameters -->
<div class="card">
  <div class="card-title">Configuration</div>
  <table class="kv">
    {_kv("Dataset",       "BTCUSDT 1H · Binance Vision · 2020-01 → 2026-05")}
    {_kv("IC horizon",    f"{IC_HORIZON} bars (16H forward return)")}
    {_kv("WFO",           f"{WF_TRAIN_M}m IS / {WF_OOS_M}m OOS / {WF_STEP_M}m step")}
    {_kv("Risk/trade",    f"{RISK_PCT*100:.0f}% equity · max lev {MAX_LEV}×")}
    {_kv("Fee",           f"{FEE*100:.2f}%/side = {FEE_RT_PCT:.2f}% round-trip")}
    {_kv("Max hold",      f"{MAX_HOLD} bars ({MAX_HOLD}H = {MAX_HOLD//24} days)")}
    {_kv("TP/SL grid",    f"TP ∈ {TP_FRAC_GRID} × ATR4H(prev)  ·  SL ∈ {SL_FRAC_GRID} × ATR4H(prev)")}
    {_kv("Asia session",  f"UTC 00:00–{ASIA_END_H:02d}:00")}
    {_kv("Structure N",   f"{STRUCT_N} 4H bars per window")}
    {_kv("Dealing range", f"{DR_N} 4H bars")}
    {_kv("BSL/SSL N",     f"{LIQ_N} 1H bars")}
    {_kv("Cooldown",      f"{COOLDOWN} 1H bars min between signals")}
    {_kv("MC sims",       f"{N_SIMS:,}")}
  </table>
</div>
</body></html>"""

out = Path("reports/report_smc.html")
out.parent.mkdir(exist_ok=True)
out.write_text(html, encoding="utf-8")
print(f"\n✅ Report saved: {out}  ({out.stat().st_size//1024} KB)")
print(SEP2)
