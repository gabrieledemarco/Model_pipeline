"""
test_orb_tp_scan.py
===================
Scan sistematico per trovare i parametri TP/SL ottimali per la strategia
ICT Reversion con MSS proxy su BTCUSDT 15M.

Setup base (invariato):
  1. London KZ (07:00-09:59 UTC)
  2. Sweep: bar.low < Asian_Low AND bar.close >= Asian_Low (LONG)
             bar.high > Asian_High AND bar.close <= Asian_High (SHORT)
  3. MSS proxy: primo bar successivo con higher-high (LONG) / lower-low (SHORT)
  4. Entry: open del bar MSS + 1
  5. SL: swept extreme - sl_buffer × ATR

Parametri scansionati:
  tp_frac     ∈ {0.20, 0.25, 0.33, 0.40, 0.50, 0.60, 0.75, 1.00}
               TP = entry + tp_frac × asian_range (verso il lato opposto)
  sl_buf      ∈ {0.00, 0.10, 0.20, 0.50, 1.00, 1.50, 2.00}  (× ATR oltre lo swept extreme)
               0.00 = SL esattamente al minimo/massimo dello sweep bar
               2.00 = SL molto largo per evitare stop prematuri
  range_atr_max ∈ {inf, 2.0, 1.5, 1.2}
               Filtra sessioni asiatiche "strette" (range/ATR < soglia)

Output:
  Tabella completa: WR%, R:R, Expected P&L%, N trade, Breakeven WR%
  Mappa termica per anno delle combinazioni profitable
"""
from __future__ import annotations

import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from src.strategy.data_fetcher import fetch_extended_data
from src.strategy.indicators   import add_indicators
from src.strategy.orb_ict      import (
    build_asian_range, LONDON_START, LONDON_END, ASIA_START, ASIA_END,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
START_YEAR    = 2020
TP_FRAC_GRID  = [0.20, 0.25, 0.33, 0.40, 0.50, 0.60, 0.75, 1.00]
SL_BUF_GRID   = [0.00, 0.10, 0.20, 0.50, 1.00, 1.50, 2.00]
RANGE_ATR_MAX = [999, 2.0, 1.5, 1.2]  # 999 = no filter
MAX_HOLD_BARS = 32  # ~8h in 15M bars
MSS_LOOKBACK  = 8   # max bars to find MSS confirmation
SEP  = "─" * 78
SEP2 = "═" * 78

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
print(SEP2)
print("ORB TP/SL Scan — Ottimizzazione parametri ICT Reversion")
print(SEP2)
print("\n[1/3] Caricamento dati …")
raw    = fetch_extended_data(start_year=START_YEAR, start_month=1,
                              fetch_15m=True, fetch_1m=False, fetch_flow=False)
df_1h  = add_indicators(raw["1H"])
df_15m = add_indicators(raw["15M"])
print(f"  1H : {len(df_1h):,} bar")
print(f"  15M: {len(df_15m):,} bar")

print("\n[2/3] Asian Range + range/ATR …")
asian_daily = build_asian_range(df_1h)

mask_as     = (df_1h.index.hour >= ASIA_START) & (df_1h.index.hour < ASIA_END)
atr_by_date = (df_1h[mask_as].copy()
               .assign(date=lambda x: x.index.date)
               .groupby("date")["atr_14"].mean())
atr_by_date.index = pd.to_datetime(atr_by_date.index)
asian_daily["atr_1h_mean"] = atr_by_date.reindex(asian_daily.index)
asian_daily["range_to_atr"] = (
    asian_daily["asian_range"] / asian_daily["atr_1h_mean"].clip(lower=1))

# Arrays
IDX = df_15m.index
H   = IDX.hour
HI  = df_15m["high"].values
LO  = df_15m["low"].values
CL  = df_15m["close"].values
ATR = df_15m["atr_14"].clip(lower=1.0).values
N   = len(df_15m)

date_idx   = IDX.normalize()
ah_map     = asian_daily["asian_high"].to_dict()
al_map     = asian_daily["asian_low"].to_dict()
ar_map     = asian_daily["asian_range"].to_dict()
rta_map    = asian_daily["range_to_atr"].to_dict()

ah_arr  = np.array([ah_map.get(d, np.nan)  for d in date_idx], dtype=float)
al_arr  = np.array([al_map.get(d, np.nan)  for d in date_idx], dtype=float)
ar_arr  = np.array([ar_map.get(d, np.nan)  for d in date_idx], dtype=float)
rta_arr = np.array([rta_map.get(d, np.nan) for d in date_idx], dtype=float)
yr_arr  = np.array([d.year for d in date_idx], dtype=int)

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: collect all sweep + MSS events with raw geometry
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/3] Raccolta sweep + MSS events …")

# Each record: (direction, entry_i, swept_extreme, asian_high, asian_low,
#               asian_range, atr_at_sweep, range_to_atr, year)
events = []

for i in range(N - MAX_HOLD_BARS - MSS_LOOKBACK - 2):
    if not (LONDON_START <= H[i] < LONDON_END):
        continue
    ah  = ah_arr[i]
    al  = al_arr[i]
    ar  = ar_arr[i]
    rta = rta_arr[i]
    if np.isnan(ah) or np.isnan(al) or ar < 1.0:
        continue

    atr_i = ATR[i]

    # ── LONG sweep ──────────────────────────────────────────────────────
    if LO[i] < al and CL[i] >= al:
        swept_ext  = LO[i]          # swept low
        sweep_hi   = HI[i]          # high of sweep bar (used as MSS trigger)

        mss_bar = None
        for j in range(i + 1, min(i + MSS_LOOKBACK + 1, N)):
            if HI[j] > sweep_hi:
                mss_bar = j
                break

        if mss_bar is None or mss_bar + MAX_HOLD_BARS >= N:
            continue

        # entry = open of bar AFTER mss_bar (next bar open, zero look-ahead)
        entry_i  = mss_bar + 1
        entry_px = CL[mss_bar]  # use close of MSS bar as entry price

        events.append({
            "direction":   "long",
            "sweep_i":     i,
            "entry_i":     entry_i,
            "entry_px":    entry_px,
            "swept_ext":   swept_ext,
            "ah":          ah,
            "al":          al,
            "ar":          ar,
            "atr_sweep":   atr_i,
            "rta":         rta,
            "year":        yr_arr[i],
        })

    # ── SHORT sweep ─────────────────────────────────────────────────────
    elif HI[i] > ah and CL[i] <= ah:
        swept_ext = HI[i]           # swept high
        sweep_lo  = LO[i]

        mss_bar = None
        for j in range(i + 1, min(i + MSS_LOOKBACK + 1, N)):
            if LO[j] < sweep_lo:
                mss_bar = j
                break

        if mss_bar is None or mss_bar + MAX_HOLD_BARS >= N:
            continue

        entry_i  = mss_bar + 1
        entry_px = CL[mss_bar]

        events.append({
            "direction":   "short",
            "sweep_i":     i,
            "entry_i":     entry_i,
            "entry_px":    entry_px,
            "swept_ext":   swept_ext,
            "ah":          ah,
            "al":          al,
            "ar":          ar,
            "atr_sweep":   atr_i,
            "rta":         rta,
            "year":        yr_arr[i],
        })

print(f"  Sweep + MSS events raccolti: {len(events)}")
print(f"    Long: {sum(1 for e in events if e['direction']=='long')}")
print(f"    Short: {sum(1 for e in events if e['direction']=='short')}")

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: scan parameters
# ─────────────────────────────────────────────────────────────────────────────
# For each event, pre-compute: price sequence after entry
# Then for each (tp_frac, sl_buf), quickly check hit TP or SL

# Pre-cache price paths
print(f"\n  Pre-caching price paths …")
event_paths = []
for ev in events:
    ei = ev["entry_i"]
    path_hi = HI[ei : ei + MAX_HOLD_BARS]
    path_lo = LO[ei : ei + MAX_HOLD_BARS]
    event_paths.append((path_hi, path_lo))

print(f"\n  Scanning {len(TP_FRAC_GRID)}×{len(SL_BUF_GRID)}×{len(RANGE_ATR_MAX)} = "
      f"{len(TP_FRAC_GRID)*len(SL_BUF_GRID)*len(RANGE_ATR_MAX)} combinazioni "
      f"(SL da 0.00 a 2.00×ATR) …\n")

results = []

for tp_frac, sl_buf, rta_max in product(TP_FRAC_GRID, SL_BUF_GRID, RANGE_ATR_MAX):
    wins = 0
    losses = 0
    total_tp_dist = 0.0
    total_sl_dist = 0.0
    n_valid = 0
    yearly = {}

    for ev, (ph, pl) in zip(events, event_paths):
        # Apply range_to_atr filter
        if ev["rta"] > rta_max:
            continue

        entry  = ev["entry_px"]
        ar     = ev["ar"]
        atr_sw = ev["atr_sweep"]

        if ev["direction"] == "long":
            tp_px = entry + tp_frac * ar
            sl_px = ev["swept_ext"] - sl_buf * atr_sw
            if sl_px >= entry or tp_px <= entry:
                continue
            sl_dist = entry - sl_px
            tp_dist = tp_px - entry
            # scan forward
            hit_tp = hit_sl = False
            for ph_bar, pl_bar in zip(ph, pl):
                if ph_bar >= tp_px and not hit_sl:
                    hit_tp = True; break
                if pl_bar <= sl_px:
                    hit_sl = True; break
        else:  # short
            tp_px = entry - tp_frac * ar
            sl_px = ev["swept_ext"] + sl_buf * atr_sw
            if sl_px <= entry or tp_px >= entry:
                continue
            sl_dist = sl_px - entry
            tp_dist = entry - tp_px
            hit_tp = hit_sl = False
            for ph_bar, pl_bar in zip(ph, pl):
                if pl_bar <= tp_px and not hit_sl:
                    hit_tp = True; break
                if ph_bar >= sl_px:
                    hit_sl = True; break

        if sl_dist <= 0 or tp_dist <= 0:
            continue

        n_valid += 1
        total_tp_dist += tp_dist / entry * 100.0
        total_sl_dist += sl_dist / entry * 100.0
        yr = ev["year"]
        if yr not in yearly:
            yearly[yr] = {"w": 0, "n": 0}
        yearly[yr]["n"] += 1

        if hit_tp:
            wins += 1
            yearly[yr]["w"] += 1
        elif hit_sl:
            losses += 1

    if n_valid < 10:
        continue

    wr  = wins / n_valid * 100
    rr  = (total_tp_dist / n_valid) / (total_sl_dist / n_valid) if total_sl_dist > 0 else 0
    be  = 1 / (1 + rr) * 100 if rr > 0 else 50.0
    avg_tp = total_tp_dist / n_valid
    avg_sl = total_sl_dist / n_valid
    exp_pnl = (wr / 100 * avg_tp) - ((1 - wr / 100) * avg_sl)

    results.append({
        "tp_frac": tp_frac,
        "sl_buf":  sl_buf,
        "rta_max": rta_max,
        "n":       n_valid,
        "wr":      round(wr, 2),
        "rr":      round(rr, 2),
        "be_wr":   round(be, 2),
        "margin":  round(wr - be, 2),
        "exp_pnl": round(exp_pnl, 4),
        "yearly":  yearly,
    })

df_res = pd.DataFrame(results)

# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────
print(SEP2)
print("RISULTATI SCAN — tutte le combinazioni (no filtro range/ATR)")
print(SEP2)
base = df_res[df_res.rta_max == 999].sort_values("exp_pnl", ascending=False)
print(f"\n{'tp_frac':>8} {'sl_buf':>7} {'N':>6} {'WR%':>7} {'R:R':>6} "
      f"{'BE%':>6} {'Margin':>8} {'ExpPnL%':>9}")
print(SEP)
for _, r in base.iterrows():
    marker = " ◄ POSITIVO" if r.exp_pnl > 0 else ""
    print(f"  {r.tp_frac:>6.2f}   {r.sl_buf:>5.2f}   {r.n:>5}  "
          f"{r.wr:>6.1f}%  {r.rr:>5.2f}  {r.be_wr:>5.1f}%  "
          f"{r.margin:>+7.2f}pp  {r.exp_pnl:>+8.4f}%{marker}")

# Best combos per rta_max filter
print(f"\n{SEP2}")
print("TOP 5 COMBINAZIONI PER EXPECTED P&L — per filtro range/ATR")
print(SEP2)

for rta in RANGE_ATR_MAX:
    sub = df_res[df_res.rta_max == rta].sort_values("exp_pnl", ascending=False).head(5)
    label = f"range/ATR < {rta}" if rta < 999 else "nessun filtro"
    print(f"\n  Filtro: {label}  (N eventi disponibili: {sub['n'].max() if len(sub)>0 else 0})")
    print(f"  {'tp_frac':>8} {'sl_buf':>7} {'N':>6} {'WR%':>7} {'R:R':>6} "
          f"{'ExpPnL%':>9} {'Sig':>14}")
    print(f"  {SEP}")
    for _, r in sub.iterrows():
        # binomial test vs breakeven
        k = int(round(r.wr / 100 * r.n))
        be_p = r.be_wr / 100
        res_be = st.binomtest(k, int(r.n), p=be_p, alternative="greater")
        sig = ("*** p<.001" if res_be.pvalue < 0.001 else
               "**  p<.01"  if res_be.pvalue < 0.01  else
               "*   p<.05"  if res_be.pvalue < 0.05  else
               f"    p={res_be.pvalue:.3f}")
        print(f"  {r.tp_frac:>8.2f}  {r.sl_buf:>6.2f}  {r.n:>6}  "
              f"{r.wr:>6.1f}%  {r.rr:>5.2f}  {r.exp_pnl:>+8.4f}%  {sig:>14}")

# Best overall
best = df_res.sort_values("exp_pnl", ascending=False).head(1).iloc[0]
print(f"\n{SEP2}")
print(f"MIGLIOR COMBINAZIONE ASSOLUTA:")
print(f"  tp_frac={best.tp_frac}  sl_buf={best.sl_buf}  rta_max={best.rta_max}")
print(f"  N={best.n}  WR={best.wr:.1f}%  R:R={best.rr:.2f}  "
      f"ExpPnL={best.exp_pnl:+.4f}%  Margin={best.margin:+.2f}pp")

# Year-by-year for best combo
print(f"\n  Stabilita' per anno (best combo):")
print(f"  {'Anno':>6} {'N':>6} {'WR%':>7} {'Exp%':>9}")
print(f"  {SEP[:50]}")
yearly_data = best.yearly
for yr in sorted(yearly_data.keys()):
    yd = yearly_data[yr]
    n_yr = yd["n"]
    wr_yr = yd["w"] / n_yr * 100 if n_yr > 0 else 0
    # approximate exp pnl per year using same avg tp/sl
    # (reuse global best.rr to approximate)
    avg_tp_approx = best.exp_pnl / (best.wr/100 - (1-best.wr/100)/best.rr) if best.rr > 0 else 0
    print(f"  {yr:>6}  {n_yr:>6}  {wr_yr:>6.1f}%")

# Positive Expected P&L summary
pos = df_res[df_res.exp_pnl > 0].sort_values("exp_pnl", ascending=False)
print(f"\n{SEP2}")
print(f"RIEPILOGO: {len(pos)} combinazioni con Expected P&L > 0 su {len(df_res)} totali")
if len(pos) > 0:
    print(f"  Range tp_frac: {pos.tp_frac.min():.2f} – {pos.tp_frac.max():.2f}")
    print(f"  Range sl_buf:  {pos.sl_buf.min():.2f} – {pos.sl_buf.max():.2f}")
    print(f"  Range rta_max: {pos.rta_max.min():.1f} – {pos.rta_max.max():.1f}")
    print(f"  WR medio:      {pos.wr.mean():.1f}%")
    print(f"  R:R medio:     {pos.rr.mean():.2f}")
    print(f"  ExpPnL max:    {pos.exp_pnl.max():+.4f}%")
    print()
    print(f"  Parametri ottimali → usare come base per WF backtest:")
    best_pos = pos.iloc[0]
    print(f"    tp_frac  = {best_pos.tp_frac}")
    print(f"    sl_buf   = {best_pos.sl_buf} × ATR")
    print(f"    rta_max  = {best_pos.rta_max} (filtro range/ATR asiatico)")
    print(f"    WR       = {best_pos.wr:.1f}%  R:R = {best_pos.rr:.2f}  N = {best_pos.n}")
else:
    print("  NESSUNA combinazione con Expected P&L positivo trovata.")


# ─────────────────────────────────────────────────────────────────────────────
# Analisi effetto sl_buf: per ogni tp_frac fisso al best, varia sl_buf
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP2}")
print("EFFETTO SL_BUF — WR e ExpPnL al variare dello stop (nessun filtro rta)")
print("(mostra come allargare SL impatta WR e R:R)")
print(SEP2)

best_tp = df_res[df_res.rta_max == 999].sort_values("exp_pnl", ascending=False).iloc[0].tp_frac
sub_tp = df_res[(df_res.rta_max == 999) & (df_res.tp_frac == best_tp)].sort_values("sl_buf")

print(f"\n  tp_frac fisso = {best_tp:.2f}")
print(f"\n  {'sl_buf':>8} {'N':>6} {'WR%':>7} {'R:R':>6} {'BE%':>6} "
      f"{'Margin':>8} {'ExpPnL%':>9}")
print(f"  {SEP}")
for _, r in sub_tp.iterrows():
    sl_label = "← exact swept low" if r.sl_buf == 0.0 else ""
    marker   = " ◄ POSITIVO" if r.exp_pnl > 0 else ""
    print(f"  {r.sl_buf:>6.2f}×ATR  {r.n:>6}  {r.wr:>6.1f}%  {r.rr:>5.2f}  "
          f"{r.be_wr:>5.1f}%  {r.margin:>+7.2f}pp  {r.exp_pnl:>+8.4f}%"
          f"{marker}  {sl_label}")

# Anche la tabella completa ordinata per sl_buf (no rta filter, top tp_fracs)
print(f"\n{SEP2}")
print("MATRICE SL_BUF × TP_FRAC — Expected P&L% (nessun filtro rta)")
print(SEP2)
pivot_data = df_res[df_res.rta_max == 999].pivot_table(
    index="sl_buf", columns="tp_frac", values="exp_pnl", aggfunc="first"
)
header = "  sl_buf\\tp_frac"
for col in pivot_data.columns:
    header += f"  {col:.2f}"
print(header)
print(f"  {SEP}")
for idx_val, row in pivot_data.iterrows():
    line = f"  {idx_val:.2f}×ATR     "
    for val in row.values:
        marker = "*" if val > 0 else " "
        line += f"  {val:+6.3f}{marker}"
    print(line)
print("  (* = positivo)")

print(f"\n{SEP2}\n")
