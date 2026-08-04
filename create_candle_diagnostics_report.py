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
w("[CONCLUSIONE parziale] Il colore della candela 1m non ha potere predittivo "
  "lordo (win rate ~48.8-49.1%, un coin-flip). Filtrare per body grande rivela "
  "una leggera tendenza a INVERTIRSI (win rate di continuazione scende con la "
  "dimensione del body), ma l'effetto (<2bps anche nei casi estremi) resta "
  "un ordine di grandezza sotto la fee round-trip (8bps) — non sfruttabile.")
w("=" * 78)

# ── [3] "Big trades hold information for the next 20 minutes" — claim specifica
# citata dall'utente (fonte: reel Instagram @matfinog). Testata con una
# soglia di volume molto più selettiva (percentile su finestra scorrevole di
# 24h, non la media mobile 20 barre usata sopra) e orizzonti fino a 8 ore,
# sia in direzione MOMENTUM (segui il big trade) sia FADE.
w("\n[3] Claim specifica: 'big trades hold information for the next 20 minutes' "
  "(volume = top X% della finestra scorrevole 24h, causale)")

WIN_BIG = 1440
roll_rank = pd.Series(VOL).rolling(WIN_BIG).apply(lambda x: (x[-1] > x[:-1]).mean(), raw=True).values
HOLDS_EXT = [1, 2, 5, 10, 15, 20, 30, 60, 120, 240, 480]

for pct_thresh in [0.99, 0.999]:
    w(f"\n  Soglia: volume nel top {(1-pct_thresh)*100:.2f}% della finestra 24h — "
      f"direzione FADE (short big-green, long big-red):")
    w(f"  {'Hold':>6}  {'n':>8}  {'fade WR':>9}  {'fade gross%':>13}  {'fade net%':>11}")
    for hold in HOLDS_EXT:
        entry_i = np.arange(WIN_BIG, N - hold - 2)
        ep = OP[entry_i + 1]; xp = OP[entry_i + 1 + hold]
        gross_ret_long = (xp - ep) / ep
        d_mom = np.where(green[entry_i], 1, np.where(red[entry_i], -1, 0))
        d_fade = -d_mom
        big = roll_rank[entry_i] >= pct_thresh
        mask = (d_mom != 0) & big & ~np.isnan(roll_rank[entry_i])
        if mask.sum() < 100: continue
        gm = d_fade[mask] * gross_ret_long[mask]
        w(f"  {hold:>4}m  {mask.sum():>8}  {(gm>0).mean():>8.1%}  {gm.mean()*100:>12.5f}%  "
          f"{(gm.mean()-FEE_RT)*100:>10.5f}%")

w(f"\n{'=' * 78}")
w("[CONCLUSIONE FINALE] Anche isolando le candele con volume genuinamente "
  "estremo (top 1% e top 0.1% di una finestra di 24h, non solo 'sopra la "
  "media 20 barre') e testando orizzonti fino a 8 ore, l'effetto lordo "
  "massimo osservato è ~2.2 bps (fade, top 1%, hold 30min) — un quarto "
  "della fee round-trip (8bps) — e decade verso il rumore su orizzonti più "
  "lunghi (240-480min). La claim 'big trades hold information for the next "
  "20 minutes' non è verificabile con dati OHLCV aggregati a 1 minuto: "
  "servirebbero dati order-flow/Level 2 (aggressore reale, bid/ask "
  "imbalance) per isolare i 'big trade' veri, di cui il volume di barra 1m "
  "è solo un proxy molto rumoroso.")
w("=" * 78)

out_path = Path("reports/candle_diagnostics.md")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("# Candle Color Diagnostics — gross predictive power check\n\n```\n" +
                     "\n".join(report_lines) + "\n```\n", encoding="utf-8")
print(f"\n[DONE] {out_path}")
