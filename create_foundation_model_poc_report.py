#!/usr/bin/env python3
"""
create_foundation_model_poc_report.py
=======================================
POC: Chronos-Bolt (Amazon, zero-shot time-series foundation model) applicato
a BTC/USDT 1H come segnale di forecast direzionale.

Approccio zero-shot: nessun fitting sui nostri dati, quindi nessun rischio di
overfitting/lookahead nella SELEZIONE del modello (il modello è pretrained
esternamente da Amazon su un corpus enorme e generico di serie temporali).
Il rischio è diverso: verificare se ha comunque un potere predittivo REALE
su BTC, non un artefatto di un singolo anno — per questo il breakdown
per-anno è incluso fin dal primo report, non aggiunto dopo (lezione della
validazione ADP).

Metodologia:
  - context = ultime `CTX_LEN` ore di log-price, ancore ogni `STEP_H` ore
  - forecast quantili (10/50/90) a orizzonte fino a 48h in un'unica chiamata
  - metriche a h=4,16,24,48: hit-rate direzionale, correlazione forecast/
    realized return, calibrazione quantili (P(actual in [q10,q90]) atteso ~80%)
  - tutto riportato per anno solare, non solo in aggregato
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from chronos import BaseChronosPipeline

from src.strategy.data_fetcher import fetch_extended_data
from src.strategy.indicators import add_indicators

SEP = "═" * 78
START_YEAR = 2020
CTX_LEN    = 512     # hours of context fed to the model (~21 days)
STEP_H     = 4        # anchor spacing in hours
HORIZONS   = [4, 16, 24, 48]
MODEL_NAME = "amazon/chronos-bolt-tiny"
BATCH_SIZE = 128

print(SEP)
print("Foundation Model POC — Chronos-Bolt zero-shot su BTC/USDT 1H")
print(SEP)

# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n[DATA] Loading 1H OHLCV …")
raw  = fetch_extended_data(start_year=START_YEAR, start_month=1,
                            fetch_15m=False, fetch_1m=False, fetch_flow=False)
df1h = add_indicators(raw["1H"])
IDX1H = df1h.index
N1H   = len(df1h)
CL    = df1h["close"].values.astype(float)
LOGCL = np.log(CL)
print(f"  {N1H:,} bars  ({IDX1H[0].date()} → {IDX1H[-1].date()})")

# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[MODEL] Loading {MODEL_NAME} (CPU, zero-shot, no fine-tuning) …")
pipeline = BaseChronosPipeline.from_pretrained(
    MODEL_NAME, device_map="cpu", torch_dtype=torch.float32,
)
max_h = max(HORIZONS)

# ══════════════════════════════════════════════════════════════════════════════
# ANCHORS
# ══════════════════════════════════════════════════════════════════════════════
anchors = list(range(CTX_LEN, N1H - max_h, STEP_H))
print(f"[ANCHORS] {len(anchors):,} points every {STEP_H}h "
      f"(context={CTX_LEN}h, max horizon={max_h}h)")

# ══════════════════════════════════════════════════════════════════════════════
# ROLLING ZERO-SHOT FORECAST
# ══════════════════════════════════════════════════════════════════════════════
records = []
print("[RUN] Batched inference", end="", flush=True)
for b_start in range(0, len(anchors), BATCH_SIZE):
    batch_anchors = anchors[b_start:b_start + BATCH_SIZE]
    inputs = [torch.tensor(LOGCL[i - CTX_LEN:i], dtype=torch.float32) for i in batch_anchors]
    quantiles, _ = pipeline.predict_quantiles(
        inputs=inputs, prediction_length=max_h, quantile_levels=[0.1, 0.5, 0.9],
    )
    quantiles = quantiles.numpy()  # (batch, max_h, 3)

    for j, i in enumerate(batch_anchors):
        last_logp = LOGCL[i - 1]
        row = dict(anchor=i, ts=IDX1H[i - 1], year=IDX1H[i - 1].year)
        for h in HORIZONS:
            q10, q50, q90 = quantiles[j, h - 1]
            actual = LOGCL[i - 1 + h]
            row[f"fc_ret_{h}h"]  = q50 - last_logp
            row[f"real_ret_{h}h"] = actual - last_logp
            row[f"in_band_{h}h"] = float(q10 <= actual <= q90)
        records.append(row)
    print(".", end="", flush=True)
print()

res = pd.DataFrame(records)

# ══════════════════════════════════════════════════════════════════════════════
# METRICS — overall + per year
# ══════════════════════════════════════════════════════════════════════════════
def compute_metrics(sub: pd.DataFrame) -> dict:
    out = {}
    for h in HORIZONS:
        fc = sub[f"fc_ret_{h}h"].values
        rl = sub[f"real_ret_{h}h"].values
        hit = float(np.mean(np.sign(fc) == np.sign(rl)))
        corr = float(np.corrcoef(fc, rl)[0, 1]) if len(sub) > 2 else float("nan")
        band = float(sub[f"in_band_{h}h"].mean())
        out[h] = dict(n=len(sub), hit_rate=hit, corr=corr, band_coverage=band)
    return out

report_lines = []
def w(line=""):
    print(line)
    report_lines.append(line)

w(f"\n{SEP}")
w("RISULTATI AGGREGATI (tutto il periodo)")
w(SEP)
overall = compute_metrics(res)
w(f"  {'Horizon':>8}  {'n':>7}  {'Hit rate':>9}  {'Corr(fc,real)':>14}  {'Coverage q10-90 (atteso ~80%)':>30}")
for h in HORIZONS:
    m = overall[h]
    w(f"  {h:>6}h  {m['n']:>7}  {m['hit_rate']:>8.1%}  {m['corr']:>+13.3f}  {m['band_coverage']:>29.1%}")

w(f"\n{SEP}")
w("RISULTATI PER ANNO (verifica di stabilità — lezione dalla validazione ADP)")
w(SEP)
years = sorted(res["year"].unique())
for h in HORIZONS:
    w(f"\n  Horizon {h}h:")
    w(f"    {'Year':>6}  {'n':>6}  {'Hit rate':>9}  {'Corr':>8}  {'Coverage':>9}")
    for yr in years:
        sub = res[res["year"] == yr]
        if len(sub) < 10:
            continue
        m = compute_metrics(sub)[h]
        w(f"    {yr:>6}  {m['n']:>6}  {m['hit_rate']:>8.1%}  {m['corr']:>+7.3f}  {m['band_coverage']:>8.1%}")

w(f"\n{SEP}")
w("INTERPRETAZIONE")
w(SEP)
w("  Hit rate: 50% = random walk (nessun potere predittivo). Valori consistentemente")
w("  >52-53% su TUTTI gli anni (non solo alcuni) sarebbero un segnale genuino;")
w("  valori vicini al 50% o instabili anno per anno indicano nessun edge reale.")
w("  Coverage q10-90: se il modello è ben calibrato, l'80% degli outcome reali")
w("  dovrebbe cadere nella banda [q10,q90]. Coverage molto diverso da 80% indica")
w("  intervalli di incertezza mal calibrati per questo asset/orizzonte.")

out_path = Path("reports/foundation_model_poc.md")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text("# Foundation Model POC — Chronos-Bolt zero-shot su BTC/USDT\n\n```\n" +
                     "\n".join(report_lines) + "\n```\n", encoding="utf-8")
res.to_csv("reports/foundation_model_poc_raw.csv", index=False)
print(f"\n[DONE] {out_path}")
print(f"[DONE] reports/foundation_model_poc_raw.csv  ({len(res)} righe)")
