"""
test_ict_correlation.py
=======================
Test statistico puro della correlazione ICT documentata:

Ipotesi nulla (H0): dopo un evento di sweep+reversion dell'Asian Range
nella London Kill Zone, il rendimento forward NON è diverso dal caso
(media = 0, distribuzione identica ai bar di controllo).

Ipotesi alternativa (H1): il rendimento forward è statisticamente
significativo nella direzione predetta (LONG dopo down-sweep, SHORT
dopo up-sweep).

Test eseguiti:
  1. Accuratezza direzionale (% volte che il prezzo va nella direzione
     predetta) a 1, 4, 8, 16 bar (15m, 1h, 2h, 4h)
  2. Binomial test: è l'accuratezza significativamente > 50%?
  3. Mann-Whitney U: la distribuzione dei return post-sweep è diversa
     da quella dei bar di controllo (stessa ora, stesso giorno, no sweep)?
  4. Spearman: correlazione tra penetration_depth/ATR e return forward
  5. Spearman: correlazione tra compression_ratio e |return forward|
  6. NR vs non-NR: Mann-Whitney sulla differenza di distribuzione
  7. London KZ vs NY KZ: confronto accuratezza direzionale
  8. Effetto temporale: performance per anno (2020-2026)
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
    LONDON_START, LONDON_END, NY_START, NY_END,
    ASIA_START, ASIA_END,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
HORIZONS   = [1, 4, 8, 16]   # bars (15m × n = 15m, 1h, 2h, 4h)
HOR_LABELS = ["15m", "1h", "2h", "4h"]
START_YEAR = 2020
NR_LOOKBACK = 20

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("ICT Asian Range Sweep — Test di Correlazione Statistica")
print("=" * 70)
print("\n[1/4] Caricamento dati …")
data = fetch_extended_data(
    start_year=START_YEAR, start_month=1,
    fetch_15m=True, fetch_1m=False, fetch_flow=False)

df_1h  = add_indicators(data["1H"])
df_15m = add_indicators(data["15M"])
print(f"  1H : {len(df_1h):,} bar  ({df_1h.index[0].date()} → {df_1h.index[-1].date()})")
print(f"  15M: {len(df_15m):,} bar")

# ─────────────────────────────────────────────────────────────────────────────
# Asian Range
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/4] Costruzione Asian Range …")
asian_daily = build_asian_range(df_1h, nr_lookback=NR_LOOKBACK)
print(f"  Giorni con range asiatico completo: {len(asian_daily)}")
print(f"  NR days (Q25): {asian_daily['is_nr'].sum()} "
      f"({asian_daily['is_nr'].mean()*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────────────────
# Raccolta eventi sweep + return forward
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/4] Identificazione eventi sweep e return forward …")

CLOSE = df_15m["close"].values
HIGH  = df_15m["high"].values
LOW   = df_15m["low"].values
ATR   = df_15m["atr_14"].clip(lower=1.0).values if "atr_14" in df_15m.columns else np.ones(len(df_15m))
IDX   = df_15m.index

# Pre-map daily levels to 15M index
date_idx  = IDX.normalize()
ah_map    = asian_daily["asian_high"].to_dict()
al_map    = asian_daily["asian_low"].to_dict()
ar_map    = asian_daily["asian_range"].to_dict()
nr_map    = asian_daily["is_nr"].fillna(False).to_dict()
cr_map    = asian_daily["compression_ratio"].fillna(1.0).to_dict()

ah_arr = np.array([ah_map.get(d, np.nan) for d in date_idx], dtype=float)
al_arr = np.array([al_map.get(d, np.nan) for d in date_idx], dtype=float)
ar_arr = np.array([ar_map.get(d, np.nan) for d in date_idx], dtype=float)
nr_arr = np.array([nr_map.get(d, False) for d in date_idx], dtype=bool)
cr_arr = np.array([cr_map.get(d, 1.0) for d in date_idx], dtype=float)

N = len(df_15m)
sweep_records = []   # evento sweep
ctrl_records  = []   # bar di controllo (stessa KZ window, stesso giorno, NO sweep)

for i in range(N - max(HORIZONS) - 1):
    h = IDX[i].hour
    ah = ah_arr[i]
    al = al_arr[i]
    ar = ar_arr[i]

    if np.isnan(ah) or np.isnan(al) or ar < 1.0:
        continue

    # Finestre KZ
    in_london = (h >= LONDON_START) and (h < LONDON_END)
    in_ny     = (h >= NY_START)     and (h < NY_END)
    if not (in_london or in_ny):
        continue

    kz_name = "London" if in_london else "NY"
    hi_i = HIGH[i]
    lo_i = LOW[i]
    cl_i = CLOSE[i]

    # Return forward a diversi orizzonti (dal CLOSE del bar i al CLOSE del bar i+n)
    fwd_rets = {}
    for n, lbl in zip(HORIZONS, HOR_LABELS):
        if i + n < N:
            fwd_rets[lbl] = (CLOSE[i + n] - cl_i) / cl_i * 100.0
        else:
            fwd_rets[lbl] = np.nan

    # ── LONG sweep: bar.low < AL e bar.close >= AL ───────────────────────
    if lo_i < al and cl_i >= al:
        pen = al - lo_i
        depth_atr = pen / ATR[i]
        rec = {
            "ts": IDX[i], "type": "long", "kz": kz_name,
            "is_nr": nr_arr[i], "cr": cr_arr[i],
            "pen": pen, "ar": ar, "depth_atr": depth_atr,
        }
        rec.update(fwd_rets)
        # Signed return: LONG predice mossa POSITIVA
        for lbl in HOR_LABELS:
            if not np.isnan(fwd_rets.get(lbl, np.nan)):
                rec[f"signed_{lbl}"] = fwd_rets[lbl]   # positivo = buono
                rec[f"dir_ok_{lbl}"] = 1 if fwd_rets[lbl] > 0 else 0
        sweep_records.append(rec)

    # ── SHORT sweep: bar.high > AH e bar.close <= AH ────────────────────
    elif hi_i > ah and cl_i <= ah:
        pen = hi_i - ah
        depth_atr = pen / ATR[i]
        rec = {
            "ts": IDX[i], "type": "short", "kz": kz_name,
            "is_nr": nr_arr[i], "cr": cr_arr[i],
            "pen": pen, "ar": ar, "depth_atr": depth_atr,
        }
        rec.update(fwd_rets)
        for lbl in HOR_LABELS:
            if not np.isnan(fwd_rets.get(lbl, np.nan)):
                rec[f"signed_{lbl}"] = -fwd_rets[lbl]  # negativo = buono per short
                rec[f"dir_ok_{lbl}"] = 1 if fwd_rets[lbl] < 0 else 0
        sweep_records.append(rec)

    # ── Bar di CONTROLLO: in KZ, no sweep ───────────────────────────────
    else:
        rec = {"ts": IDX[i], "kz": kz_name, "is_nr": nr_arr[i]}
        rec.update(fwd_rets)
        ctrl_records.append(rec)

sw = pd.DataFrame(sweep_records)
ct = pd.DataFrame(ctrl_records)
print(f"  Sweep events totali: {len(sw)}")
if len(sw) > 0:
    print(f"    London LONG : {len(sw[(sw.kz=='London') & (sw.type=='long')])}")
    print(f"    London SHORT: {len(sw[(sw.kz=='London') & (sw.type=='short')])}")
    print(f"    NY LONG     : {len(sw[(sw.kz=='NY') & (sw.type=='long')])}")
    print(f"    NY SHORT    : {len(sw[(sw.kz=='NY') & (sw.type=='short')])}")
    print(f"  Bar di controllo (KZ, no sweep): {len(ct)}")

# ─────────────────────────────────────────────────────────────────────────────
# Helper: tabella risultati
# ─────────────────────────────────────────────────────────────────────────────
SEP  = "─" * 70
SEP2 = "═" * 70

def sig_str(p):
    if p < 0.001: return "*** p<.001"
    if p < 0.01:  return "**  p<.01"
    if p < 0.05:  return "*   p<.05"
    if p < 0.10:  return "~   p<.10"
    return f"    p={p:.3f}"

# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/4] Test statistici\n")
print(SEP2)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — Accuratezza direzionale + Binomial test
# ─────────────────────────────────────────────────────────────────────────────
print("\n▶ TEST 1 — Accuratezza Direzionale (H0: 50/50 casualità)")
print(SEP)
header = f"{'Orizzonte':<10} {'N':>6} {'Dir%':>7} {'Mean ret%':>10} {'Binomial p':>12} {'Sig':>12}"
print(header)
print(SEP)

for lbl in HOR_LABELS:
    col = f"dir_ok_{lbl}"
    if col not in sw.columns:
        continue
    sub = sw[col].dropna().astype(int)
    n   = len(sub)
    if n == 0:
        continue
    k   = sub.sum()
    acc = k / n * 100
    mean_ret = sw[f"signed_{lbl}"].dropna().mean()
    # Binomial test: k successi su n, p_H0=0.5, alternativa > 0.5
    res = st.binomtest(k, n, p=0.5, alternative="greater")
    p   = res.pvalue
    print(f"{'  '+lbl:<10} {n:>6} {acc:>6.1f}% {mean_ret:>9.3f}% {p:>12.4f} {sig_str(p):>12}")

print(SEP)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — London vs NY accuratezza
# ─────────────────────────────────────────────────────────────────────────────
print("\n▶ TEST 2 — London KZ vs NY KZ Accuratezza Direzionale")
print(SEP)
header2 = f"{'KZ':<10} {'Orizzonte':<10} {'N':>6} {'Dir%':>7} {'Binomial p':>12} {'Sig':>12}"
print(header2)
print(SEP)

for kz in ["London", "NY"]:
    sub_kz = sw[sw.kz == kz]
    for lbl in HOR_LABELS:
        col = f"dir_ok_{lbl}"
        if col not in sub_kz.columns:
            continue
        s = sub_kz[col].dropna().astype(int)
        n = len(s)
        if n < 10:
            continue
        k   = s.sum()
        acc = k / n * 100
        res = st.binomtest(k, n, p=0.5, alternative="greater")
        p   = res.pvalue
        print(f"{'  '+kz:<10} {'  '+lbl:<10} {n:>6} {acc:>6.1f}% {p:>12.4f} {sig_str(p):>12}")

print(SEP)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — Mann-Whitney U: sweep vs controllo
# ─────────────────────────────────────────────────────────────────────────────
print("\n▶ TEST 3 — Mann-Whitney U: distribuzione return SWEEP vs CONTROLLO")
print("  (H0: distribuzione identica → sweep non ha effetto direzionale)")
print(SEP)
header3 = f"{'Orizzonte':<10} {'N sweep':>8} {'N ctrl':>8} {'MedSW%':>9} {'MedCT%':>9} {'MWU p':>12} {'Sig':>12}"
print(header3)
print(SEP)

for lbl in HOR_LABELS:
    # Signed return: per long = fwd_ret, per short = -fwd_ret
    if f"signed_{lbl}" not in sw.columns:
        continue
    sw_ret = sw[f"signed_{lbl}"].dropna().values
    ct_ret = ct[lbl].dropna().values if lbl in ct.columns else np.array([])

    if len(sw_ret) < 10 or len(ct_ret) < 10:
        continue

    med_sw = np.median(sw_ret)
    med_ct = np.median(ct_ret)
    # Mann-Whitney one-sided: sweep returns > control returns
    stat, p = st.mannwhitneyu(sw_ret, ct_ret, alternative="greater")
    print(f"{'  '+lbl:<10} {len(sw_ret):>8} {len(ct_ret):>8} {med_sw:>8.3f}% {med_ct:>8.3f}% {p:>12.4f} {sig_str(p):>12}")

print(SEP)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — Spearman: depth/ATR vs signed return
# ─────────────────────────────────────────────────────────────────────────────
print("\n▶ TEST 4 — Spearman: penetration_depth/ATR vs return_forward")
print("  (H0: nessuna correlazione tra profondità sweep e intensità mossa)")
print(SEP)
header4 = f"{'Orizzonte':<10} {'N':>6} {'Spearman r':>12} {'p':>12} {'Sig':>12}"
print(header4)
print(SEP)

for lbl in HOR_LABELS:
    if f"signed_{lbl}" not in sw.columns:
        continue
    sub = sw[["depth_atr", f"signed_{lbl}"]].dropna()
    if len(sub) < 10:
        continue
    r, p = st.spearmanr(sub["depth_atr"], sub[f"signed_{lbl}"])
    print(f"{'  '+lbl:<10} {len(sub):>6} {r:>12.4f} {p:>12.4f} {sig_str(p):>12}")

print(SEP)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 — Spearman: compression_ratio vs |signed return|
# ─────────────────────────────────────────────────────────────────────────────
print("\n▶ TEST 5 — Spearman: compression_ratio vs |return_forward|")
print("  (H0: range compresso oggi NON predice mossa più ampia dopo sweep)")
print(SEP)
header5 = f"{'Orizzonte':<10} {'N':>6} {'Spearman r':>12} {'p':>12} {'Sig':>12}"
print(header5)
print(SEP)

for lbl in HOR_LABELS:
    if f"signed_{lbl}" not in sw.columns:
        continue
    sub = sw[["cr", f"signed_{lbl}"]].dropna()
    if len(sub) < 10:
        continue
    abs_ret = sub[f"signed_{lbl}"].abs()
    r, p = st.spearmanr(sub["cr"], abs_ret)
    print(f"{'  '+lbl:<10} {len(sub):>6} {r:>12.4f} {p:>12.4f} {sig_str(p):>12}")

print(SEP)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 6 — NR vs non-NR: Mann-Whitney sulla direzione
# ─────────────────────────────────────────────────────────────────────────────
print("\n▶ TEST 6 — NR days vs non-NR: differenza distribuzione return")
print("  (H0: il filtro NR NON migliora il return forward)")
print(SEP)
header6 = f"{'Orizzonte':<10} {'N_NR':>7} {'N_noNR':>8} {'Med_NR%':>9} {'Med_noNR%':>10} {'MWU p':>10} {'Sig':>12}"
print(header6)
print(SEP)

sw_nr   = sw[sw.is_nr == True]
sw_nonr = sw[sw.is_nr == False]
for lbl in HOR_LABELS:
    if f"signed_{lbl}" not in sw.columns:
        continue
    nr_ret   = sw_nr[f"signed_{lbl}"].dropna().values
    nonr_ret = sw_nonr[f"signed_{lbl}"].dropna().values
    if len(nr_ret) < 5 or len(nonr_ret) < 5:
        continue
    med_nr   = np.median(nr_ret)
    med_nonr = np.median(nonr_ret)
    stat, p  = st.mannwhitneyu(nr_ret, nonr_ret, alternative="greater")
    print(f"{'  '+lbl:<10} {len(nr_ret):>7} {len(nonr_ret):>8} {med_nr:>8.3f}% {med_nonr:>9.3f}% {p:>10.4f} {sig_str(p):>12}")

print(SEP)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 7 — Accuratezza per anno (stability over time)
# ─────────────────────────────────────────────────────────────────────────────
print("\n▶ TEST 7 — Stabilità nel tempo: accuratezza direzionale per anno (orizzonte 1h)")
print(SEP)
lbl = "1h"
col = f"dir_ok_{lbl}"
if col in sw.columns and "ts" in sw.columns:
    sw["year"] = pd.to_datetime(sw["ts"]).dt.year
    for yr, grp in sw.groupby("year"):
        s = grp[col].dropna().astype(int)
        n = len(s)
        if n < 5:
            continue
        k   = s.sum()
        acc = k / n * 100
        res = st.binomtest(k, n, p=0.5, alternative="greater")
        p   = res.pvalue
        signed_med = grp[f"signed_{lbl}"].dropna().median()
        print(f"  {yr}  N={n:>4}  dir={acc:>5.1f}%  med_ret={signed_med:>6.3f}%  {sig_str(p)}")

print(SEP)

# ─────────────────────────────────────────────────────────────────────────────
# RIEPILOGO FINALE
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP2}")
print("RIEPILOGO — LA CORRELAZIONE ICT ESISTE NEI DATI?")
print(SEP2)

results_summary = {}
for lbl in HOR_LABELS:
    col = f"dir_ok_{lbl}"
    if col not in sw.columns:
        continue
    sub = sw[col].dropna().astype(int)
    if len(sub) == 0:
        continue
    k = sub.sum()
    n = len(sub)
    acc = k / n * 100
    res = st.binomtest(k, n, p=0.5, alternative="greater")
    results_summary[lbl] = (acc, res.pvalue, n)

print(f"\n  Accuratezza direzionale TOTALE (LONG dopo down-sweep, SHORT dopo up-sweep):")
any_sig = False
for lbl, (acc, p, n) in results_summary.items():
    sig = "SIGNIFICATIVO" if p < 0.05 else "non significativo"
    print(f"    {lbl:>4}: {acc:.1f}% su {n} eventi  (p={p:.4f})  → {sig}")
    if p < 0.05:
        any_sig = True

print()
if any_sig:
    print("  ✓ La correlazione ICT ESISTE statisticamente ad almeno un orizzonte.")
    print("    Il problema del backtest precedente è nella STRUTTURA DI ENTRY/SL/TP,")
    print("    non nell'assenza del segnale sottostante.")
else:
    print("  ✗ La correlazione ICT NON è statisticamente significativa su BTCUSDT.")
    print("    Il sweep+reversion non predice la direzione futura in modo affidabile.")

print(f"\n{SEP2}\n")
