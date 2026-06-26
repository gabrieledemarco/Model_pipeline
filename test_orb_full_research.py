"""
test_orb_full_research.py
=========================
Test statistici estesi per capire la vera struttura dell'edge ORB su BTCUSDT.

Copre 3 ipotesi che NON sono state testate prima:

IPOTESI A — London Breakout CONTINUATION (non reversion)
  Quando London rompe l'Asian High/Low, il prezzo CONTINUA nella
  direzione del breakout nelle successive 1-4h?
  (Questo è il classico London Breakout: momentum, non reversion)

IPOTESI B — ICT Reversion con MSS confirmation
  Dopo un sweep (low<AL, close>=AL), aspettiamo che il BAR SUCCESSIVO
  faccia un higher high (proxy per Market Structure Shift su 15M).
  Entry su quel bar, non sul bar di sweep.
  TP = Asian High. SL = sotto il swept low.
  Win rate e R:R effettivo?

IPOTESI C — Asian session character filter
  La natura della sessione asiatica cambia il tipo di signal:
  - Asia ranging (range < 0.5 × ATR_1h_medio) → reversion in London
  - Asia trending (range > 1.0 × ATR_1h_medio) → continuation in London
  Esiste differenza statistica tra questi due scenari?

Per ogni ipotesi: Binomial test, Mann-Whitney, statistiche per anno.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from src.strategy.data_fetcher import fetch_extended_data
from src.strategy.indicators   import add_indicators
from src.strategy.orb_ict      import (
    build_asian_range,
    LONDON_START, LONDON_END,
    ASIA_START, ASIA_END,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
HORIZONS   = [1, 4, 8, 16]
HOR_LABELS = ["15m", "1h", "2h", "4h"]
START_YEAR = 2020
SEP  = "─" * 72
SEP2 = "═" * 72

def sig_str(p):
    if p < 0.001: return "*** p<.001"
    if p < 0.01:  return "**  p<.01"
    if p < 0.05:  return "*   p<.05"
    if p < 0.10:  return "~   p<.10"
    return f"    p={p:.3f}"

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
print(SEP2)
print("ORB FULL RESEARCH — Test Statistici Estesi")
print(SEP2)
print("\n[1/5] Caricamento dati …")
raw   = fetch_extended_data(start_year=START_YEAR, start_month=1,
                             fetch_15m=True, fetch_1m=False, fetch_flow=False)
df_1h  = add_indicators(raw["1H"])
df_15m = add_indicators(raw["15M"])
print(f"  1H : {len(df_1h):,} bar  ({df_1h.index[0].date()} → {df_1h.index[-1].date()})")
print(f"  15M: {len(df_15m):,} bar")

print("\n[2/5] Asian Range + character …")
asian_daily = build_asian_range(df_1h)

# Asian session character: trending vs ranging
# ATR proxy: mean ATR(14) of the 1H bars during Asian session
mask_as = (df_1h.index.hour >= ASIA_START) & (df_1h.index.hour < ASIA_END)
atr_by_date = (df_1h[mask_as].copy()
               .assign(date=lambda x: x.index.date)
               .groupby("date")["atr_14"].mean())
atr_by_date.index = pd.to_datetime(atr_by_date.index)
asian_daily["atr_1h_mean"] = atr_by_date.reindex(asian_daily.index)
asian_daily["range_to_atr"] = (
    asian_daily["asian_range"] / asian_daily["atr_1h_mean"].clip(lower=1))

# Asian "ranging": range < 0.5× ATR (tight consolidation)
# Asian "trending": range > 1.0× ATR (directional)
asian_daily["asia_char"] = "neutral"
asian_daily.loc[asian_daily["range_to_atr"] < 0.5, "asia_char"] = "ranging"
asian_daily.loc[asian_daily["range_to_atr"] > 1.0, "asia_char"] = "trending"

print(f"  Giorni completi: {len(asian_daily)}")
for ch in ["ranging", "neutral", "trending"]:
    n = (asian_daily["asia_char"] == ch).sum()
    print(f"    Asia {ch}: {n} ({n/len(asian_daily)*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# Prepare arrays
# ─────────────────────────────────────────────────────────────────────────────
IDX  = df_15m.index
H    = IDX.hour
HI   = df_15m["high"].values
LO   = df_15m["low"].values
CL   = df_15m["close"].values
OP   = df_15m["open"].values
ATR  = df_15m["atr_14"].clip(lower=1.0).values if "atr_14" in df_15m.columns else np.ones(len(df_15m))
N    = len(df_15m)

date_idx  = IDX.normalize()
ah_map    = asian_daily["asian_high"].to_dict()
al_map    = asian_daily["asian_low"].to_dict()
ar_map    = asian_daily["asian_range"].to_dict()
ac_map    = asian_daily["asia_char"].to_dict()

ah_arr = np.array([ah_map.get(d, np.nan) for d in date_idx], dtype=float)
al_arr = np.array([al_map.get(d, np.nan) for d in date_idx], dtype=float)
ar_arr = np.array([ar_map.get(d, np.nan) for d in date_idx], dtype=float)
ac_arr = np.array([ac_map.get(d, "neutral") for d in date_idx], dtype=object)

MAX_H = max(HORIZONS)

# ─────────────────────────────────────────────────────────────────────────────
# IPOTESI A — London Breakout CONTINUATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/5] Ipotesi A — London Breakout CONTINUATION …")

# For each day, find the FIRST bar in London KZ that breaks outside the
# Asian Range (either above AH or below AL) and STAYS outside (close beyond).
# Direction = direction of the breakout (not reversion).
# Question: does price continue in that direction for the next 1-4h?

cont_records = []

# Group by date
dates = sorted(set(IDX.date))
for date in dates:
    mask_day = np.where(IDX.date == date)[0]
    if len(mask_day) == 0:
        continue

    # First index in London KZ for this date
    london_idx = [i for i in mask_day
                  if LONDON_START <= H[i] < LONDON_END]
    if not london_idx:
        continue

    i0 = london_idx[0]
    ah = ah_arr[i0]
    al = al_arr[i0]
    ar = ar_arr[i0]
    ac = ac_arr[i0]

    if np.isnan(ah) or np.isnan(al) or ar < 1.0:
        continue

    # Find first bar that breaks + closes outside the range
    first_break_up   = None
    first_break_down = None

    for i in london_idx:
        if first_break_up is None and CL[i] > ah:
            first_break_up = i
        if first_break_down is None and CL[i] < al:
            first_break_down = i
        if first_break_up and first_break_down:
            break

    # Evaluate each break direction
    for direction, first_i, correct_fwd in [
        ("up",   first_break_up,   lambda r: r > 0),   # up break → expect positive return
        ("down", first_break_down, lambda r: r < 0),   # down break → expect negative return
    ]:
        if first_i is None:
            continue
        if first_i + MAX_H >= N:
            continue

        cl_entry = CL[first_i]
        fwd = {}
        for n, lbl in zip(HORIZONS, HOR_LABELS):
            fwd[lbl] = (CL[first_i + n] - cl_entry) / cl_entry * 100.0

        signed = {}
        dir_ok = {}
        for lbl, r in fwd.items():
            signed[lbl] = r if direction == "up" else -r
            dir_ok[lbl] = 1 if correct_fwd(r) else 0

        rec = {"date": date, "direction": direction,
               "asia_char": ac, "ar": ar, "atr_entry": ATR[first_i]}
        rec.update({f"fwd_{l}": fwd[l] for l in HOR_LABELS})
        rec.update({f"signed_{l}": signed[l] for l in HOR_LABELS})
        rec.update({f"dir_ok_{l}": dir_ok[l] for l in HOR_LABELS})
        cont_records.append(rec)

cont = pd.DataFrame(cont_records)
print(f"  Break events: {len(cont)}  "
      f"(up={len(cont[cont.direction=='up'])}, "
      f"down={len(cont[cont.direction=='down'])})")

# ─────────────────────────────────────────────────────────────────────────────
# IPOTESI B — ICT Reversion con MSS proxy (Higher High after sweep)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/5] Ipotesi B — ICT Reversion con MSS proxy …")
# After a sweep (lo < AL, close >= AL), wait for the FIRST subsequent bar
# (within 8 bars) that makes a higher high than the sweep bar.
# Entry = close of that confirmation bar (or next bar open).
# TP = Asian High (opposite side of range).
# SL = swept low - small buffer (0.2 × ATR).
# Win: does price reach AH before the SL?

mss_records = []

for i in range(N - MAX_H - 10):
    if not (LONDON_START <= H[i] < LONDON_END):
        continue
    ah = ah_arr[i]
    al = al_arr[i]
    ar = ar_arr[i]
    ac = ac_arr[i]

    if np.isnan(ah) or np.isnan(al) or ar < 1.0:
        continue

    # ── LONG setup: sweep below Asian Low ────────────────────────────────
    if LO[i] < al and CL[i] >= al:
        swept_low = LO[i]
        sweep_high = HI[i]

        # Look for MSS proxy: first bar in next 8 bars with HIGH > sweep bar HIGH
        mss_bar = None
        for j in range(i + 1, min(i + 9, N)):
            if HI[j] > sweep_high:
                mss_bar = j
                break

        if mss_bar is None:
            continue
        if mss_bar + MAX_H >= N:
            continue

        entry = CL[mss_bar]
        sl_price = swept_low - 0.2 * ATR[i]
        tp_price = ah  # opposite side of Asian range

        sl_dist = entry - sl_price
        tp_dist = tp_price - entry
        if sl_dist <= 0 or tp_dist <= 0:
            continue
        rr_actual = tp_dist / sl_dist

        # Check outcome: scan bars after entry for TP or SL hit
        hit_tp = False
        hit_sl = False
        for k in range(mss_bar + 1, min(mss_bar + 33, N)):  # ~8h max hold
            if HI[k] >= tp_price:
                hit_tp = True
                break
            if LO[k] <= sl_price:
                hit_sl = True
                break

        win = hit_tp and not hit_sl
        # If both hit same bar: worst-case (sl first) → loss
        if hit_tp and hit_sl:
            win = False

        # Also capture simple forward return
        fwd = {}
        for n, lbl in zip(HORIZONS, HOR_LABELS):
            if mss_bar + n < N:
                fwd[lbl] = (CL[mss_bar + n] - entry) / entry * 100.0
            else:
                fwd[lbl] = np.nan

        mss_records.append({
            "ts": IDX[i], "setup": "long",
            "rr_actual": round(rr_actual, 2),
            "sl_dist_pct": round(sl_dist / entry * 100, 4),
            "tp_dist_pct": round(tp_dist / entry * 100, 4),
            "hit_tp": hit_tp, "hit_sl": hit_sl, "win": int(win),
            "asia_char": ac,
            **{f"fwd_{l}": fwd.get(l, np.nan) for l in HOR_LABELS},
        })

    # ── SHORT setup: sweep above Asian High ──────────────────────────────
    elif HI[i] > ah and CL[i] <= ah:
        swept_high = HI[i]
        sweep_low = LO[i]

        # MSS proxy: first bar with LOW < sweep bar LOW
        mss_bar = None
        for j in range(i + 1, min(i + 9, N)):
            if LO[j] < sweep_low:
                mss_bar = j
                break

        if mss_bar is None:
            continue
        if mss_bar + MAX_H >= N:
            continue

        entry = CL[mss_bar]
        sl_price = swept_high + 0.2 * ATR[i]
        tp_price = al  # Asian Low

        sl_dist = sl_price - entry
        tp_dist = entry - tp_price
        if sl_dist <= 0 or tp_dist <= 0:
            continue
        rr_actual = tp_dist / sl_dist

        hit_tp = False
        hit_sl = False
        for k in range(mss_bar + 1, min(mss_bar + 33, N)):
            if LO[k] <= tp_price:
                hit_tp = True
                break
            if HI[k] >= sl_price:
                hit_sl = True
                break

        win = hit_tp and not hit_sl
        if hit_tp and hit_sl:
            win = False

        fwd = {}
        for n, lbl in zip(HORIZONS, HOR_LABELS):
            if mss_bar + n < N:
                fwd[lbl] = -(CL[mss_bar + n] - entry) / entry * 100.0
            else:
                fwd[lbl] = np.nan

        mss_records.append({
            "ts": IDX[i], "setup": "short",
            "rr_actual": round(rr_actual, 2),
            "sl_dist_pct": round(sl_dist / entry * 100, 4),
            "tp_dist_pct": round(tp_dist / entry * 100, 4),
            "hit_tp": hit_tp, "hit_sl": hit_sl, "win": int(win),
            "asia_char": ac,
            **{f"fwd_{l}": fwd.get(l, np.nan) for l in HOR_LABELS},
        })

mss = pd.DataFrame(mss_records)
print(f"  MSS-confirmed events: {len(mss)}")
if len(mss) > 0:
    print(f"    Long: {len(mss[mss.setup=='long'])}  "
          f"Short: {len(mss[mss.setup=='short'])}")
    print(f"    R:R medio effettivo: {mss['rr_actual'].mean():.2f}")
    print(f"    SL medio %: {mss['sl_dist_pct'].mean():.3f}%")
    print(f"    TP medio %: {mss['tp_dist_pct'].mean():.3f}%")

print(f"\n[5/5] Test statistici …\n")

# ─────────────────────────────────────────────────────────────────────────────
# RISULTATI — IPOTESI A: London Continuation Breakout
# ─────────────────────────────────────────────────────────────────────────────
print(SEP2)
print("IPOTESI A — London Breakout CONTINUATION")
print("  (trade nella DIREZIONE del primo breakout dell'Asian Range)")
print(SEP2)

print(f"\n{'Orizzonte':<10} {'N':>6} {'Dir%':>7} {'MedRet%':>9} {'Binomial p':>12} {'Sig':>12}")
print(SEP)
for lbl in HOR_LABELS:
    col = f"dir_ok_{lbl}"
    if col not in cont.columns:
        continue
    s = cont[col].dropna().astype(int)
    n = len(s)
    if n < 10:
        continue
    k = s.sum()
    acc = k / n * 100
    med = cont[f"signed_{lbl}"].dropna().median()
    res = st.binomtest(k, n, p=0.5, alternative="greater")
    print(f"  {lbl:<10} {n:>6} {acc:>6.1f}% {med:>8.3f}% {res.pvalue:>12.4f} {sig_str(res.pvalue):>12}")

# Split by Asia character
print(f"\n  Split per carattere sessione asiatica (orizzonte 4h):")
print(SEP)
lbl = "4h"
col = f"dir_ok_{lbl}"
if col in cont.columns:
    print(f"  {'Asia char':<12} {'N':>6} {'Dir%':>7} {'Binomial p':>12} {'Sig':>12}")
    for ch in ["ranging", "neutral", "trending"]:
        sub = cont[cont.asia_char == ch][col].dropna().astype(int)
        n = len(sub)
        if n < 5:
            continue
        k = sub.sum()
        acc = k / n * 100
        res = st.binomtest(k, n, p=0.5, alternative="greater")
        print(f"  {ch:<12} {n:>6} {acc:>6.1f}% {res.pvalue:>12.4f} {sig_str(res.pvalue):>12}")

# Up vs Down breaks
print(f"\n  Split per direzione break (orizzonte 4h):")
print(SEP)
for direction in ["up", "down"]:
    sub = cont[cont.direction == direction][col].dropna().astype(int) if col in cont.columns else pd.Series(dtype=int)
    n = len(sub)
    if n < 5:
        continue
    k = sub.sum()
    acc = k / n * 100
    res = st.binomtest(k, n, p=0.5, alternative="greater")
    print(f"  Break {direction:>4}:  N={n:>5}  Dir={acc:>5.1f}%  p={res.pvalue:.4f}  {sig_str(res.pvalue)}")

# Year breakdown
print(f"\n  Stabilita' per anno (orizzonte 4h):")
print(SEP)
if "date" in cont.columns:
    cont["year"] = pd.to_datetime(cont["date"]).dt.year
    for yr, grp in cont.groupby("year"):
        s = grp[col].dropna().astype(int) if col in grp.columns else pd.Series(dtype=int)
        n = len(s)
        if n < 5: continue
        k = s.sum()
        acc = k / n * 100
        res = st.binomtest(k, n, p=0.5, alternative="greater")
        med = grp[f"signed_{lbl}"].dropna().median()
        print(f"  {yr}  N={n:>4}  dir={acc:>5.1f}%  med={med:>6.3f}%  {sig_str(res.pvalue)}")

# ─────────────────────────────────────────────────────────────────────────────
# RISULTATI — IPOTESI B: ICT Reversion con MSS
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP2}")
print("IPOTESI B — ICT Reversion con MSS Confirmation")
print("  (entry sul primo higher-high/lower-low dopo sweep, TP=Asian opposite)")
print(SEP2)

if len(mss) > 0:
    wr = mss["win"].mean() * 100
    n_tp = mss["hit_tp"].sum()
    n_sl = mss["hit_sl"].sum()
    n_neither = len(mss) - n_tp - n_sl  # still open at 8h exit
    mean_rr = mss["rr_actual"].mean()

    print(f"\n  Totale setup: {len(mss)}")
    print(f"  Win rate (TP hit first): {wr:.1f}%")
    print(f"  TP hit: {n_tp} ({n_tp/len(mss)*100:.1f}%)")
    print(f"  SL hit: {n_sl} ({n_sl/len(mss)*100:.1f}%)")
    print(f"  Nessuno (> 8h): {n_neither} ({n_neither/len(mss)*100:.1f}%)")
    print(f"  R:R medio effettivo: {mean_rr:.2f}:1")

    # Breakeven WR for this R:R
    be_wr = 1 / (1 + mean_rr) * 100
    print(f"  WR breakeven per questo R:R: {be_wr:.1f}%")
    print(f"  Margin sopra breakeven: {wr - be_wr:+.1f}%")

    # Binomial test
    k = mss["win"].sum()
    n = len(mss)
    be_p = 1 / (1 + mean_rr)  # breakeven probability
    res_be = st.binomtest(k, n, p=be_p, alternative="greater")
    res_50 = st.binomtest(k, n, p=0.5, alternative="greater")
    print(f"\n  Binomial test (H0: WR = breakeven {be_wr:.1f}%): p = {res_be.pvalue:.4f} {sig_str(res_be.pvalue)}")
    print(f"  Binomial test (H0: WR = 50%):                    p = {res_50.pvalue:.4f} {sig_str(res_50.pvalue)}")

    # Expected P&L per trade
    avg_win_pct  = mss["tp_dist_pct"].mean()
    avg_loss_pct = mss["sl_dist_pct"].mean()
    exp_pnl = (wr / 100 * avg_win_pct) - ((1 - wr / 100) * avg_loss_pct)
    print(f"\n  Avg TP dist (win): {avg_win_pct:.3f}%  |  Avg SL dist (loss): {avg_loss_pct:.3f}%")
    print(f"  Expected P&L per trade: {exp_pnl:+.4f}%  "
          f"({'POSITIVO' if exp_pnl > 0 else 'NEGATIVO'})")

    # Split by Asia char
    print(f"\n  Split per carattere Asia:")
    print(SEP)
    print(f"  {'Asia char':<12} {'N':>6} {'WR%':>7} {'R:R':>6} {'ExpPnL%':>9}")
    for ch in ["ranging", "neutral", "trending"]:
        sub = mss[mss.asia_char == ch]
        if len(sub) < 5:
            continue
        wr_ch = sub["win"].mean() * 100
        rr_ch = sub["rr_actual"].mean()
        wr_pct = sub["tp_dist_pct"].mean()
        sl_pct = sub["sl_dist_pct"].mean()
        exp = (wr_ch / 100 * wr_pct) - ((1 - wr_ch / 100) * sl_pct)
        print(f"  {ch:<12} {len(sub):>6} {wr_ch:>6.1f}% {rr_ch:>6.2f} {exp:>+8.4f}%")

    # Year breakdown
    print(f"\n  Stabilita' per anno:")
    print(SEP)
    mss["year"] = pd.to_datetime(mss["ts"]).dt.year
    for yr, grp in mss.groupby("year"):
        wr_yr = grp["win"].mean() * 100
        n_yr = len(grp)
        rr_yr = grp["rr_actual"].mean()
        be_yr = 1 / (1 + rr_yr) * 100
        exp_yr = (wr_yr / 100 * grp["tp_dist_pct"].mean()) - \
                 ((1 - wr_yr / 100) * grp["sl_dist_pct"].mean())
        print(f"  {yr}  N={n_yr:>4}  WR={wr_yr:>5.1f}%  R:R={rr_yr:.2f}  "
              f"BE={be_yr:.1f}%  ExpPnL={exp_yr:+.4f}%")

# ─────────────────────────────────────────────────────────────────────────────
# IPOTESI C — Confronto diretto: Continuation vs Reversion per Asia char
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP2}")
print("IPOTESI C — CONFRONTO: Continuation vs Reversion per tipo Asia")
print(SEP2)

# For this comparison, we use the London continuation (Ipotesi A) directional
# accuracy at 4h broken by Asia character, and compare to the sweep reversion
# (original test_ict_correlation results — loaded inline here for reference).

print(f"\n  London Continuation (4h) per Asia character:")
lbl = "4h"
col = f"dir_ok_{lbl}"
if col in cont.columns:
    for ch in ["ranging", "neutral", "trending"]:
        sub = cont[cont.asia_char == ch]
        s = sub[col].dropna().astype(int)
        n = len(s)
        if n < 5: continue
        k = s.sum()
        acc = k / n * 100
        res = st.binomtest(k, n, p=0.5, alternative="greater")
        print(f"    Asia={ch:<10} N={n:>4}  Continuation dir={acc:.1f}%  {sig_str(res.pvalue)}")

# Recompute sweep+reversion per Asia character
print(f"\n  London Reversion (sweep+close-back) (4h) per Asia character:")
rev_by_char = {}
for i in range(N - MAX_H - 1):
    if not (LONDON_START <= H[i] < LONDON_END):
        continue
    ah = ah_arr[i]; al = al_arr[i]; ar = ar_arr[i]; ac = ac_arr[i]
    if np.isnan(ah) or np.isnan(al) or ar < 1.0:
        continue
    if i + 16 >= N:
        continue

    fwd_4h = (CL[i + 16] - CL[i]) / CL[i] * 100.0

    # Long sweep
    if LO[i] < al and CL[i] >= al:
        dir_ok = 1 if fwd_4h > 0 else 0
        rev_by_char.setdefault(ac, []).append(dir_ok)
    # Short sweep
    elif HI[i] > ah and CL[i] <= ah:
        dir_ok = 1 if fwd_4h < 0 else 0
        rev_by_char.setdefault(ac, []).append(dir_ok)

for ch in ["ranging", "neutral", "trending"]:
    arr = rev_by_char.get(ch, [])
    n = len(arr)
    if n < 5: continue
    k = sum(arr)
    acc = k / n * 100
    res = st.binomtest(k, n, p=0.5, alternative="greater")
    print(f"    Asia={ch:<10} N={n:>4}  Reversion dir={acc:.1f}%    {sig_str(res.pvalue)}")

# ─────────────────────────────────────────────────────────────────────────────
# RIEPILOGO FINALE
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP2}")
print("RIEPILOGO — QUALE APPROCCIO HA EDGE?")
print(SEP2)

print("""
  IPOTESI A — London Continuation Breakout:
    Testa se il prezzo CONTINUA nella direzione del primo breakout
    dell'Asian Range durante la London KZ (strategia classica).

  IPOTESI B — ICT Reversion con MSS:
    Testa il setup ICT CORRETTO: sweep + conferma higher-high (MSS proxy)
    + TP = lato opposto Asian Range, SL = sotto swept low.
    Include il R:R reale e l'Expected P&L per trade.

  IPOTESI C — Filtro Asia ranging vs trending:
    Verifica se il carattere della sessione asiatica determina
    quale strategia usare (continuation vs reversion).
""")
