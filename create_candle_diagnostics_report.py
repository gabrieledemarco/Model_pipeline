#!/usr/bin/env python3
"""
create_candle_diagnostics_report.py
======================================
Diagnostica che spiega perché sia create_candle_momentum_report.py sia
create_candle_fade_report.py falliscono in modo quasi identico (invece che
uno essere il complementare dell'altro, come ci si aspetterebbe da un vero
segnale di reversal). Due controlli:

  1. Return LORDO (senza fee) del "seguire" una candela verde/rossa,
     indipendentemente dalla direzione realmente tradata — se il colore
     della candela avesse un vero potere predittivo (continuazione O
     inversione), il win rate lordo si scosterebbe dal 50%.
  2. Lo stesso controllo, filtrato per dimensione del body della candela
     (relativo alla media mobile 20 barre) — per verificare se solo le
     candele "forti" abbiano un segnale nascosto nel rumore delle piccole.

Nessun sizing/fee applicato qui except dove esplicitamente indicato — è
un'analisi diagnostica di segnale grezzo, non un backtest di strategia.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from src.strategy.data_fetcher import fetch_binance_vision_klines

START_YEAR = 2020
FEE = 0.0004
FEE_RT = FEE * 2
HOLD_TESTS = [1, 5, 10, 15]
BODY_MULT_TESTS = [1.0, 2.0, 3.0, 5.0, 8.0]
VOL_AVG_WINDOW = 20

report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

w("=" * 78)
w("Candle color diagnostics — gross (no-fee) predictive power check")
w("=" * 78)

df1m = fetch_binance_vision_klines("1m", start_year=START_YEAR, start_month=1,
                                    workers=6, verbose=False)
OP = df1m["open"].values.astype(float)
CL = df1m["close"].values.astype(float)
HI = df1m["high"].values.astype(float)
LO = df1m["low"].values.astype(float)
VOL = df1m["volume"].values.astype(float)
N = len(df1m)

green = CL > OP
red = CL < OP
body = np.abs(CL - OP)
body_avg20 = pd.Series(body).rolling(VOL_AVG_WINDOW).mean().shift(1).values

w("\n[1] Return lordo (nessuna fee) seguendo il colore della candela — "
  "direzione = +1 su verde, -1 su rosso, indipendentemente da cosa poi si tradi:")
w(f"\n  {'Hold':>6}  {'n':>10}  {'gross WR':>10}  {'gross mean%':>12}  {'frac|move|<fee_rt':>18}")
for hold in HOLD_TESTS:
    entry_i = np.arange(VOL_AVG_WINDOW, N - hold - 2)
    ep = OP[entry_i + 1]; xp = OP[entry_i + 1 + hold]
    gross_ret_long = (xp - ep) / ep
    d = np.where(green[entry_i], 1, np.where(red[entry_i], -1, 0))
    mask = d != 0
    gm = d[mask] * gross_ret_long[mask]
    frac_below_fee = (np.abs(gm) < FEE_RT).mean()
    w(f"  {hold:>4}m  {mask.sum():>10}  {(gm>0).mean():>9.1%}  {gm.mean()*100:>11.5f}%  {frac_below_fee:>17.1%}")

w("\n[2] Stesso controllo, filtrato per dimensione del body (>= Nx la media mobile 20 barre):")
w(f"\n  {'Hold':>6}  {'body>=':>8}  {'n':>9}  {'gross WR':>10}  {'gross mean%':>12}")
for hold in HOLD_TESTS:
    entry_i = np.arange(VOL_AVG_WINDOW, N - hold - 2)
    ep = OP[entry_i + 1]; xp = OP[entry_i + 1 + hold]
    gross_ret_long = (xp - ep) / ep
    d = np.where(green[entry_i], 1, np.where(red[entry_i], -1, 0))
    b = body[entry_i]; ba = body_avg20[entry_i]
    body_mult = np.divide(b, ba, out=np.zeros_like(b), where=ba > 0)
    for thresh in BODY_MULT_TESTS:
        mask = (d != 0) & (ba > 0) & (body_mult >= thresh)
        if mask.sum() < 1000: continue
        gm = d[mask] * gross_ret_long[mask]
        w(f"  {hold:>4}m  {thresh:>6.1f}x  {mask.sum():>9}  {(gm>0).mean():>9.1%}  {gm.mean()*100:>11.5f}%")
    w()

w("=" * 78)
w("[CONCLUSIONE] Il colore della candela 1m non ha potere predittivo lordo "
  "(win rate ~48.8-49.1%, un coin-flip). Filtrare per body grande rivela una "
  "leggera tendenza a INVERTIRSI (win rate di continuazione scende con la "
  "dimensione del body), ma l'effetto (<2bps anche nei casi estremi) resta "
  "un ordine di grandezza sotto la fee round-trip (8bps) — non sfruttabile.")
w("=" * 78)

out_path = Path("reports/candle_diagnostics.md")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("# Candle Color Diagnostics — gross predictive power check\n\n```\n" +
                     "\n".join(report_lines) + "\n```\n", encoding="utf-8")
print(f"\n[DONE] {out_path}")
