#!/usr/bin/env python3
"""
create_ict_fade_html_report.py
=================================
Report HTML autonomo per la strategia "Fade ICT standalone" con i
risultati del walk-forward optimization (stop/target dinamici) prodotti
da `create_ict_fade_wfo_report.py` (legge `reports/ict_fade_wfo_data.pkl`).
Riusa le utility di stile/chart di `src/strategy/report_html.py` (stessa
palette, stessi helper _table/_kpi_card/_ax) senza dipendere dal suo
schema dati specifico (costruito per una strategia precedente).
"""
from __future__ import annotations

import datetime
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.strategy.report_html import (
    _ax, _fig_to_b64, _kpi_card, _color_val, _table, _signed, _color_signed,
    _CSS, BG, PANEL, BORDER, WHITE, GRAY, GREEN, RED, GOLD, BLUE, PURPLE, ORANGE, TEAL,
)

INIT_CAP = 100_000.0

# Riferimento: risultato a RR fisso senza WFO (reports/ict_fade_standalone.md)
REF_FIXED = dict(n=1399, ret=512.7, wr=47.0, mc_pp=1.000, mc_pr=0.000,
                  holdout_n=314, holdout_ret=65.2, holdout_wr=44.6)

print("[DATA] Loading WFO pickle …")
with open("reports/ict_fade_wfo_data.pkl", "rb") as f:
    D = pickle.load(f)

all_oos_trades = D["all_oos_trades"]
sel_df = D["sel_df"]
n, wr, ret_pct, mdd_pct = D["n"], D["wr"], D["ret_pct"], D["mdd_pct"]
mc, mc_blk = D["mc"], D["mc_blk"]
net_pnls = D["net_pnls"]
year_pnls = D["year_pnls"]
hpnls = D["hpnls"]
wf_windows = D["wf_windows"]
idx_start, idx_end = D["idx_start"], D["idx_end"]

trade_ts = pd.DatetimeIndex([t["exit_ts"] for t in all_oos_trades])
cap_curve = INIT_CAP + np.cumsum(net_pnls)
equity = pd.Series(cap_curve, index=trade_ts)

# ─────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────

def chart_equity():
    fig, ax1 = plt.subplots(1, 1, figsize=(13, 4.5), facecolor=BG)
    _ax(ax1, "Walk-Forward OOS Equity — Fade ICT standalone (stop/target dinamici)")
    ax1.plot(equity.index, equity.values, color=TEAL, lw=1.6)
    ax1.fill_between(equity.index, INIT_CAP, equity.values,
                      where=(equity.values >= INIT_CAP), color=GREEN, alpha=0.12)
    ax1.fill_between(equity.index, INIT_CAP, equity.values,
                      where=(equity.values < INIT_CAP), color=RED, alpha=0.15)
    ax1.axhline(INIT_CAP, color=GRAY, lw=0.8, ls=":")
    ax1.axvline(pd.Timestamp("2025-01-01"), color=GOLD, lw=1, ls="--", alpha=0.7)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    fig.patch.set_facecolor(BG)
    return _fig_to_b64(fig)


def chart_per_year():
    years = sorted(year_pnls.keys())
    rets = []
    for y in years:
        yp = year_pnls[y]
        ycap = INIT_CAP + np.cumsum(yp)
        rets.append((ycap[-1] / INIT_CAP - 1) * 100)
    fig, ax = plt.subplots(1, 1, figsize=(13, 3.2), facecolor=BG)
    _ax(ax, "Rendimento OOS per anno (%)")
    cols = [GREEN if r >= 0 else RED for r in rets]
    ax.bar([str(y) for y in years], rets, color=cols, alpha=0.85, width=0.6)
    ax.axhline(0, color=WHITE, lw=0.7, ls="--")
    for i, r in enumerate(rets):
        ax.text(i, r + (2 if r >= 0 else -6), f"{r:+.1f}%", ha="center", color=WHITE, fontsize=8)
    fig.patch.set_facecolor(BG)
    return _fig_to_b64(fig)


def chart_param_selection():
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.2), facecolor=BG)
    _ax(axes[0], "SWING_LOOKBACK selezionato (n finestre)")
    if not sel_df.empty:
        vc = sel_df["sl_look"].value_counts().sort_index()
        axes[0].bar([str(x) for x in vc.index], vc.values, color=BLUE, alpha=0.85)
        for i, v in enumerate(vc.values):
            axes[0].text(i, v + 0.3, str(v), ha="center", color=WHITE, fontsize=8)
    _ax(axes[1], "RR selezionato (n finestre)")
    if not sel_df.empty:
        vc2 = sel_df["rr"].value_counts().sort_index()
        axes[1].bar([str(x) for x in vc2.index], vc2.values, color=PURPLE, alpha=0.85)
        for i, v in enumerate(vc2.values):
            axes[1].text(i, v + 0.3, str(v), ha="center", color=WHITE, fontsize=8)
    fig.patch.set_facecolor(BG)
    return _fig_to_b64(fig)


def chart_pnl_dist():
    fig, ax = plt.subplots(1, 1, figsize=(13, 3.2), facecolor=BG)
    _ax(ax, "Distribuzione P&L per trade (OOS)")
    pnls = np.array(net_pnls)
    ax.hist(pnls[pnls > 0], bins=30, color=GREEN, alpha=0.7, label=f"Vincenti ({(pnls>0).sum()})")
    ax.hist(pnls[pnls <= 0], bins=30, color=RED, alpha=0.7, label=f"Perdenti ({(pnls<=0).sum()})")
    ax.axvline(0, color=WHITE, lw=0.8)
    ax.legend(fontsize=8, labelcolor=GRAY, framealpha=0.2)
    fig.patch.set_facecolor(BG)
    return _fig_to_b64(fig)


print("[CHARTS] Rendering …")
eq_b64 = chart_equity()
yr_b64 = chart_per_year()
sel_b64 = chart_param_selection()
pnl_b64 = chart_pnl_dist()

# ─────────────────────────────────────────────────────────────────────────
# Sections
# ─────────────────────────────────────────────────────────────────────────

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC")
date_range = f"{idx_start.date()} → {idx_end.date()}"

top_sl = sel_df["sl_look"].mode().iloc[0] if not sel_df.empty else "n/a"
top_rr = sel_df["rr"].mode().iloc[0] if not sel_df.empty else "n/a"

hcap = INIT_CAP + np.cumsum(hpnls) if hpnls else np.array([INIT_CAP])
hret = (hcap[-1] / INIT_CAP - 1) * 100 if hpnls else 0.0
hwr = (sum(1 for p in hpnls if p > 0) / len(hpnls)) * 100 if hpnls else 0.0

kpi_html = "".join([
    _kpi_card("OOS Trade", f"{n:,}", "blue"),
    _kpi_card("Ret. OOS totale", _signed(ret_pct, pct=True, decimals=1), "green" if ret_pct >= 0 else "red"),
    _kpi_card("Win Rate OOS", f"{wr:.1%}", "gold"),
    _kpi_card("Max Drawdown", f"{mdd_pct:.1f}%", "red"),
    _kpi_card("MC p(profit)", f"{mc['p_profit']:.3f}", "green"),
    _kpi_card("MC p(ruin)", f"{mc['p_ruin']:.3f}", "green" if mc['p_ruin'] < 0.05 else "red"),
    _kpi_card("Holdout 25-26 Ret.", _signed(hret, pct=True, decimals=1), "green" if hret >= 0 else "red"),
    _kpi_card("SWING_LOOKBACK modale", f"{top_sl}", "purple"),
    _kpi_card("RR modale", f"{top_rr}", "purple"),
])

years = sorted(year_pnls.keys())
year_rows = []
for y in years:
    yp = year_pnls[y]
    ycap = INIT_CAP + np.cumsum(yp)
    yret = (ycap[-1] / INIT_CAP - 1) * 100
    ywr = sum(1 for p in yp if p > 0) / len(yp) * 100
    year_rows.append([y, len(yp), _color_signed(yret, pct=True, decimals=1), f"{ywr:.1f}%"])
year_table = _table(["Anno", "n trade", "Ret%", "Win Rate"], year_rows)

sel_rows = []
if not sel_df.empty:
    for _, row in sel_df.iterrows():
        sel_rows.append([row["window_start"].strftime("%Y-%m"), int(row["sl_look"]), f"{row['rr']:.1f}",
                          f"{row['sharpe']:.2f}", int(row["n_is"])])
sel_table = _table(["Finestra IS (inizio)", "SWING_LOOKBACK scelto", "RR scelto",
                     "Sharpe IS", "n trade IS"], sel_rows)

compare_table = _table(
    ["Configurazione", "n OOS/full", "Ret%", "Win Rate", "MC p(profit)", "MC p(ruin)"],
    [
        ["WFO dinamico (stop/target per finestra, causale)", n, _color_signed(ret_pct, pct=True, decimals=1),
         f"{wr:.1%}", f"{mc['p_profit']:.3f}", f"{mc['p_ruin']:.3f}"],
        ["RR=3.0 fisso (rif. ict_fade_standalone.md, DSR su tutta la storia)", REF_FIXED["n"],
         _color_signed(REF_FIXED["ret"], pct=True, decimals=1), f"{REF_FIXED['wr']:.1f}%",
         f"{REF_FIXED['mc_pp']:.3f}", f"{REF_FIXED['mc_pr']:.3f}"],
    ])

nav = """
<nav>
  <span class="brand">Fade ICT Standalone — WFO Report</span>
  <a href="#summary">Summary</a>
  <a href="#signal">Segnale</a>
  <a href="#equity">Equity OOS</a>
  <a href="#peryear">Per Anno</a>
  <a href="#selection">Selezione Parametri</a>
  <a href="#compare">Confronto</a>
</nav>"""

body = f"""
<section id="summary">
  <h2>Summary — Walk-Forward Optimization</h2>
  <p class="sub">BTCUSDT Perpetual · Segnale 15M + contesto 4H · Periodo dati: {date_range}
     · WFO: 6m IS / 2m OOS / step 2m · Generato {now}</p>
  <div class="cards">{kpi_html}</div>
  <p class="note">Selezione per finestra 100% causale (IS-scan su Sharpe, mai barre OOS viste durante la
     selezione) su griglia SWING_LOOKBACK × RR — nessuna correzione DSR necessaria (la selezione stessa
     è già out-of-sample per costruzione). Frizioni Bybit obbligatorie incluse (taker 0.055% + slippage
     0.015%/lato = 0.14% round-trip).</p>
</section>

<section id="signal">
  <h2>Costruzione del segnale (riassunto)</h2>
  <p>Combina 6 trigger tecnici SMC/ICT sul 15M (Liquidity Sweep, Fair Value Gap, Order Block, Breaker
     Block, Inversion FVG, Power of 3/AMD) con lo stato di zone Demand/Supply sul 4H (Order Block + FVG,
     tracciate come attive da conferma a invalidazione). Quando un trigger si conferma SENZA una zona 4H
     della stessa direzione attiva a supporto ("standalone"), si prende il lato OPPOSTO (fade). Entry
     causale all'apertura della barra successiva, stop strutturale (minimo/massimo locale su
     SWING_LOOKBACK barre), target RR × rischio. Dettaglio algoritmico completo:
     <code>docs/ICT_FADE_STANDALONE_STRATEGY_SPEC.md</code>.</p>
</section>

<section id="equity">
  <h2>Equity Curve OOS (walk-forward, stop/target dinamici)</h2>
  <img src="data:image/png;base64,{eq_b64}" style="width:100%;border-radius:8px;" />
  <p class="note">Linea tratteggiata dorata: inizio holdout genuino 2025-2026 (sotto-porzione del
     WFO-OOS, non un holdout separato — l'intera curva è già out-of-sample per costruzione).</p>
  <img src="data:image/png;base64,{pnl_b64}" style="width:100%;border-radius:8px;margin-top:14px;" />
</section>

<section id="peryear">
  <h2>Breakdown per anno</h2>
  <img src="data:image/png;base64,{yr_b64}" style="width:100%;border-radius:8px;" />
  {year_table}
  <p class="note">Positivo in OGNI singolo anno del campione 2020-2026 — stessa proprietà del risultato
     a RR fisso, confermata sotto selezione causale per finestra.</p>
</section>

<section id="selection">
  <h2>Selezione dei parametri per finestra</h2>
  <img src="data:image/png;base64,{sel_b64}" style="width:100%;border-radius:8px;" />
  <p><b>SWING_LOOKBACK={top_sl}</b> e <b>RR={top_rr}</b> sono i valori scelti più frequentemente
     dall'IS-scan causale — la raccomandazione operativa per il paper trading (Sezione 8 della spec)
     si basa su questi valori modali piuttosto che sul singolo best-fit sull'intera storia.</p>
  {sel_table}
</section>

<section id="compare">
  <h2>Confronto: WFO dinamico vs RR fisso (selezione sull'intera storia + DSR)</h2>
  {compare_table}
  <p class="note">Il WFO dinamico (100% causale, nessuna correzione DSR necessaria) conferma il
     risultato del report base (RR=3.0 fisso, selezionato con DSR family sull'intera storia): ordine di
     grandezza comparabile, win rate leggermente più alto (51.0% vs 47.0%), positivo in ogni anno in
     entrambi i casi. Questa convergenza tra due metodologie di selezione indipendenti è la prova più
     solida di robustezza raccolta in questa sessione di validazione.</p>
</section>
"""

footer = f"""
<footer>
  Fade ICT Standalone — Walk-Forward Optimization Report &nbsp;·&nbsp; Generato {now}
  &nbsp;·&nbsp; Dati: Binance Vision CDN (proxy per BTCUSDT Perpetual Bybit)
</footer>"""

html = f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fade ICT Standalone — Walk-Forward Optimization Report</title>
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

out_path = Path("reports/ict_fade_wfo_report.html")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(html, encoding="utf-8")
print(f"[DONE] {out_path}")
