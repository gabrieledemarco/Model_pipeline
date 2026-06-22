"""
Interactive HTML comparison report — all strategy versions tested.

Phases covered:
  Phase 1  — Baseline (8 signals, no risk mgmt)
  Phase 2  — Extended (+15m +1m signals, no risk mgmt)
  Phase 3  — 13 risk-management variants (all on extended signals)

Walk-forward run on: Baseline, Extended, and best risk combo.

Output: reports/comparison_report.html  (self-contained except CDN JS/CSS)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.strategy.data_fetcher import (fetch_extended_data, fetch_real_funding,
                                        fetch_real_oi, generate_oi)
from src.strategy.indicators   import add_indicators
from src.strategy.signals      import build_signal_matrix
from src.strategy.engine       import run_backtest, INIT_CAP, _compute_kpis
from src.strategy.optimizer    import SCENARIOS, apply_filters
from src.strategy.walk_forward import run_walk_forward

SCENARIO = "Session 08-21"
OUT_PATH = Path("reports/comparison_report.html")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ─── 1. Data ──────────────────────────────────────────────────────────────────
print("Loading data …")
tf_data = fetch_extended_data(fetch_15m=True, fetch_1m=True)
for tf in tf_data:
    if not tf_data[tf].empty:
        tf_data[tf] = add_indicators(tf_data[tf])

df_1h  = tf_data["1H"]
df_15m = tf_data.get("15M", pd.DataFrame())
df_1m  = tf_data.get("1M",  pd.DataFrame())

funding, _         = fetch_real_funding(tf_data["1H"], tf_data["1D"])
premium_1h, oi_real = fetch_real_oi(df_1h)
oi_df              = generate_oi(tf_data["1D"]["close"])
print(f"  1H bars: {len(df_1h):,}  [{df_1h.index[0].date()} → {df_1h.index[-1].date()}]")


# ─── 2. Signal matrices ───────────────────────────────────────────────────────
print("Building signal matrices …")
sig_baseline = build_signal_matrix(
    tf_data, oi_df, funding,
    premium_1h=premium_1h if oi_real else None,
    df_15m=None, df_1m=None,
)
sig_extended = build_signal_matrix(
    tf_data, oi_df, funding,
    premium_1h=premium_1h if oi_real else None,
    df_15m=df_15m if not df_15m.empty else None,
    df_1m=df_1m  if not df_1m.empty  else None,
)

cfg = SCENARIOS[SCENARIO]
sf_base_filt = apply_filters(sig_baseline, cfg)
sf_ext_filt  = apply_filters(sig_extended, cfg)

common_base = df_1h.index.intersection(sf_base_filt.index)
common_ext  = df_1h.index.intersection(sf_ext_filt.index)
df_base_bt  = df_1h.reindex(common_base)
sf_base_bt  = sf_base_filt.reindex(common_base)
df_ext_bt   = df_1h.reindex(common_ext)
sf_ext_bt   = sf_ext_filt.reindex(common_ext)


# ─── 3. Phase 1 & 2 backtests ────────────────────────────────────────────────
print("Backtesting Phase 1 (Baseline) …")
bt1 = run_backtest(df_base_bt, sf_base_bt, INIT_CAP)
k1  = bt1["kpis"]

print("Backtesting Phase 2 (Extended +15m +1m) …")
bt2 = run_backtest(df_ext_bt, sf_ext_bt, INIT_CAP)
k2  = bt2["kpis"]


# ─── 4. Phase 3: risk management variants ────────────────────────────────────
RISK_VARIANTS = [
    # (label,                  max_notional, vol_target, dd_halt, min_score)
    ("Baseline",               1.00, None,  None,  None),
    ("Notional cap 20%",       0.20, None,  None,  None),
    ("Notional cap 10%",       0.10, None,  None,  None),
    ("Vol target 20%",         1.00, 0.20,  None,  None),
    ("Vol target 15%",         1.00, 0.15,  None,  None),
    ("Circuit breaker 15%",    1.00, None,  0.15,  None),
    ("Circuit breaker 20%",    1.00, None,  0.20,  None),
    ("Strong signals (≥8)",    1.00, None,  None,  8.0),
    ("Cap20 + Vol20",          0.20, 0.20,  None,  None),
    ("Cap20 + CB15",           0.20, None,  0.15,  None),
    ("Cap20 + Vol20 + CB15",   0.20, 0.20,  0.15,  None),
    ("Cap10 + Vol15 + CB15",   0.10, 0.15,  0.15,  None),
    ("Cap20+Vol20+CB15+Str",   0.20, 0.20,  0.15,  8.0),
]

print("Phase 3: running 13 risk variants …")
risk_rows = []
equity_store: dict[str, pd.Series] = {
    "Baseline (8 sig)":    bt1["equity"],
    "Extended (+15m+1m)":  bt2["equity"],
}

for label, notional, vol_t, dd_h, score in RISK_VARIANTS:
    bt = run_backtest(df_ext_bt, sf_ext_bt, INIT_CAP,
                      max_notional_pct=notional,
                      vol_target=vol_t,
                      dd_halt_pct=dd_h,
                      min_score=score)
    k = bt["kpis"]
    equity_store[f"Risk:{label}"] = bt["equity"]
    risk_rows.append({
        "label":         label,
        "max_notional":  notional,
        "vol_target":    vol_t,
        "dd_halt":       dd_h,
        "min_score":     score,
        "return":        k["total_return"],
        "sharpe":        k["sharpe"],
        "max_dd":        k["max_drawdown"],
        "calmar":        k["calmar"],
        "win_rate":      k["win_rate"],
        "profit_factor": k.get("profit_factor", 0),
        "n_trades":      k["n_trades"],
        "equity":        bt["equity"],
    })
    print(f"  {label:<30}: {k['total_return']*100:+.1f}%  DD={k['max_drawdown']*100:.1f}%  "
          f"Calmar={k['calmar']:.2f}  Trades={k['n_trades']}")

df_risk = pd.DataFrame(risk_rows)

# Best combo by Calmar (multi-technique only)
combo_mask = df_risk["label"].str.contains(r"\+")
best       = df_risk[combo_mask].sort_values("calmar", ascending=False).iloc[0]
best_label = best["label"]
print(f"\nBest combo by Calmar: '{best_label}' ({best['calmar']:.2f})\n")


# ─── 5. Walk-forward (baseline, extended, best combo) ────────────────────────
print("Walk-forward: Baseline …")
wf1 = run_walk_forward(df_1h, sig_baseline, scenario_name=SCENARIO)

print("\nWalk-forward: Extended …")
wf2 = run_walk_forward(df_1h, sig_extended, scenario_name=SCENARIO)

print(f"\nWalk-forward: Best combo ({best_label}) …")
wf3 = run_walk_forward(
    df_1h, sig_extended, scenario_name=SCENARIO,
    backtest_kwargs=dict(
        max_notional_pct=float(best["max_notional"]),
        vol_target=best["vol_target"] if not pd.isna(best["vol_target"] or 0) else None,
        dd_halt_pct=best["dd_halt"]   if not pd.isna(best["dd_halt"] or 0)   else None,
        min_score=best["min_score"]   if not pd.isna(best["min_score"] or 0)  else None,
    ),
)


# ─── 6. Serialise chart data ──────────────────────────────────────────────────
CHART_COLORS = [
    "#58a6ff", "#f0883e", "#2ea043", "#f85149", "#e3b341",
    "#bc8cff", "#ff7eb6", "#00d2ff", "#a8ff3e", "#c471ed",
    "#ffb347", "#f8e71c", "#12c2e9", "#ff6b6b", "#4ecdc4",
]

def _series_to_daily(s: pd.Series) -> list[dict]:
    d = s.resample("D").last().dropna()
    return [{"x": str(ts.date()), "y": round(float(v), 2)} for ts, v in d.items()]

def _dd_series_to_daily(s: pd.Series) -> list[dict]:
    dd = (s / s.cummax() - 1) * 100
    d  = dd.resample("D").min().dropna()
    return [{"x": str(ts.date()), "y": round(float(v), 2)} for ts, v in d.items()]

# Equity overlay (key curves)
KEY_EQUITY = [
    ("Baseline (8 sig)",    bt1["equity"]),
    ("Extended (+15m+1m)",  bt2["equity"]),
    ("Risk: Baseline",      equity_store["Risk:Baseline"]),
    ("Risk: Notional 20%",  equity_store["Risk:Notional cap 20%"]),
    ("Risk: Cap20+Vol20+CB15", equity_store["Risk:Cap20 + Vol20 + CB15"]),
    (f"Risk: {best_label}", equity_store[f"Risk:{best_label}"]),
]
# deduplicate
seen: set[str] = set()
KEY_EQUITY_DEDUPED = []
for name, eq in KEY_EQUITY:
    if name not in seen:
        seen.add(name)
        KEY_EQUITY_DEDUPED.append((name, eq))

equity_chart_datasets = []
dd_chart_datasets = []
for i, (name, eq) in enumerate(KEY_EQUITY_DEDUPED):
    c = CHART_COLORS[i % len(CHART_COLORS)]
    equity_chart_datasets.append({
        "label": name,
        "data":  _series_to_daily(eq),
        "borderColor": c,
        "backgroundColor": c + "22",
        "borderWidth": 2 if i < 2 else 1.5,
        "pointRadius": 0,
        "tension": 0.1,
    })
    dd_chart_datasets.append({
        "label": name,
        "data":  _dd_series_to_daily(eq),
        "borderColor": c,
        "backgroundColor": c + "22",
        "borderWidth": 2 if i < 2 else 1.5,
        "fill": True,
        "pointRadius": 0,
        "tension": 0.1,
    })

# WFO OOS equity
wfo_eq_datasets = []
for i, (wf, label, color) in enumerate([
    (wf1, "Baseline",  CHART_COLORS[0]),
    (wf2, "Extended",  CHART_COLORS[1]),
    (wf3, f"Best: {best_label}", CHART_COLORS[2]),
]):
    oos = wf["combined_equity"]
    if not oos.empty:
        wfo_eq_datasets.append({
            "label": label,
            "data":  _series_to_daily(oos),
            "borderColor": color,
            "backgroundColor": color + "22",
            "borderWidth": 2,
            "pointRadius": 0,
            "tension": 0.1,
        })

# WFO per-window bars
def _wfo_window_bars(wf, label):
    wdf = wf["windows"]
    if wdf.empty:
        return None
    rets = wdf["OOS Return (%)"].tolist()
    return {
        "label": label,
        "returns": rets,
        "sharpes": wdf["Sharpe"].tolist(),
        "trades": wdf["# Trades"].tolist(),
        "win_rates": wdf["Win Rate (%)"].tolist(),
    }

wfo_windows = {
    "baseline": _wfo_window_bars(wf1, "Baseline"),
    "extended": _wfo_window_bars(wf2, "Extended"),
    "best":     _wfo_window_bars(wf3, f"Best: {best_label}"),
}

# Risk scatter: Return vs MaxDD
scatter_pts = [
    {"x": round(k1["max_drawdown"] * 100, 2), "y": round(k1["total_return"] * 100, 2),
     "label": "Baseline (8 sig)", "phase": 1,
     "calmar": round(k1["calmar"], 2), "sharpe": round(k1["sharpe"], 2)},
    {"x": round(k2["max_drawdown"] * 100, 2), "y": round(k2["total_return"] * 100, 2),
     "label": "Extended (+15m+1m)", "phase": 2,
     "calmar": round(k2["calmar"], 2), "sharpe": round(k2["sharpe"], 2)},
]
for r in risk_rows:
    scatter_pts.append({
        "x": round(r["max_dd"] * 100, 2),
        "y": round(r["return"] * 100, 2),
        "label": r["label"],
        "phase": 3,
        "calmar": round(r["calmar"], 2),
        "sharpe": round(r["sharpe"], 2),
    })

# All variants table rows
def _fmt_pct(v, digits=1):
    if v is None:
        return "—"
    return f"{v:+.{digits}f}%"

def _fmt_f(v, digits=2):
    if v is None:
        return "—"
    return f"{v:.{digits}f}"

def _risk_badge(r):
    parts = []
    if r["max_notional"] < 1.0:
        parts.append(f"Cap {r['max_notional']*100:.0f}%")
    if r["vol_target"]:
        parts.append(f"Vol {r['vol_target']*100:.0f}%")
    if r["dd_halt"]:
        parts.append(f"CB {r['dd_halt']*100:.0f}%")
    if r["min_score"]:
        parts.append(f"Score≥{r['min_score']:.0f}")
    return " · ".join(parts) if parts else "No controls"

table_rows = []
# Phase 1
table_rows.append({
    "phase": "1", "label": "Baseline (8 signals)",
    "controls": "No controls",
    "return": _fmt_pct(k1["total_return"] * 100),
    "sharpe": _fmt_f(k1["sharpe"]),
    "max_dd": _fmt_pct(k1["max_drawdown"] * 100),
    "calmar": _fmt_f(k1["calmar"]),
    "win_rate": _fmt_pct(k1["win_rate"] * 100, 0),
    "n_trades": str(k1["n_trades"]),
    "oos_return": _fmt_pct(wf1["full_kpis"]["total_return"] * 100),
    "oos_sharpe": _fmt_f(wf1["full_kpis"]["sharpe"]),
    "oos_dd": _fmt_pct(wf1["full_kpis"]["max_drawdown"] * 100),
    "wfo_pct": f"{wf1['pct_profitable']:.0f}%",
    "raw_dd": k1["max_drawdown"] * 100,
    "raw_return": k1["total_return"] * 100,
})
# Phase 2
table_rows.append({
    "phase": "2", "label": "Extended (+15m +1m)",
    "controls": "No controls",
    "return": _fmt_pct(k2["total_return"] * 100),
    "sharpe": _fmt_f(k2["sharpe"]),
    "max_dd": _fmt_pct(k2["max_drawdown"] * 100),
    "calmar": _fmt_f(k2["calmar"]),
    "win_rate": _fmt_pct(k2["win_rate"] * 100, 0),
    "n_trades": str(k2["n_trades"]),
    "oos_return": _fmt_pct(wf2["full_kpis"]["total_return"] * 100),
    "oos_sharpe": _fmt_f(wf2["full_kpis"]["sharpe"]),
    "oos_dd": _fmt_pct(wf2["full_kpis"]["max_drawdown"] * 100),
    "wfo_pct": f"{wf2['pct_profitable']:.0f}%",
    "raw_dd": k2["max_drawdown"] * 100,
    "raw_return": k2["total_return"] * 100,
})
# Phase 3
for r in risk_rows:
    is_best = r["label"] == best_label
    wfo_return = _fmt_pct(wf3["full_kpis"]["total_return"] * 100) if is_best else "—"
    wfo_sharpe = _fmt_f(wf3["full_kpis"]["sharpe"]) if is_best else "—"
    wfo_dd     = _fmt_pct(wf3["full_kpis"]["max_drawdown"] * 100) if is_best else "—"
    wfo_pct    = f"{wf3['pct_profitable']:.0f}%" if is_best else "—"
    table_rows.append({
        "phase": "3", "label": r["label"],
        "controls": _risk_badge(r),
        "return": _fmt_pct(r["return"] * 100),
        "sharpe": _fmt_f(r["sharpe"]),
        "max_dd": _fmt_pct(r["max_dd"] * 100),
        "calmar": _fmt_f(r["calmar"]),
        "win_rate": _fmt_pct(r["win_rate"] * 100, 0),
        "n_trades": str(r["n_trades"]),
        "oos_return": wfo_return,
        "oos_sharpe": wfo_sharpe,
        "oos_dd": wfo_dd,
        "wfo_pct": wfo_pct,
        "raw_dd": r["max_dd"] * 100,
        "raw_return": r["return"] * 100,
        "is_best": is_best,
    })

# Serialise everything to JSON for embedding
chart_json = json.dumps({
    "equity":      equity_chart_datasets,
    "drawdown":    dd_chart_datasets,
    "wfo_equity":  wfo_eq_datasets,
    "wfo_windows": wfo_windows,
    "scatter":     scatter_pts,
}, separators=(",", ":"))


# ─── 7. Generate HTML ─────────────────────────────────────────────────────────
def _badge(phase):
    cls = {"1": "bg-primary", "2": "bg-warning text-dark", "3": "bg-success"}
    return f'<span class="badge {cls.get(phase,"bg-secondary")}">P{phase}</span>'

def _cell(val, is_dd=False, is_return=False):
    """Return <td> with conditional colour."""
    if val == "—":
        return f'<td class="text-muted">—</td>'
    raw_class = ""
    try:
        num = float(val.replace("%", "").replace("+", ""))
        if is_dd:
            raw_class = "text-success" if num > -15 else ("text-warning" if num > -30 else "text-danger")
        elif is_return:
            raw_class = "text-success" if num > 0 else "text-danger"
    except Exception:
        pass
    return f'<td class="{raw_class} text-end">{val}</td>'

table_html = ""
for row in table_rows:
    best_row = row.get("is_best", False)
    highlight = 'class="table-active"' if best_row else ""
    best_icon = ' <i class="bi bi-star-fill text-warning" title="Best Calmar"></i>' if best_row else ""
    table_html += f"""
    <tr {highlight}>
      <td>{_badge(row['phase'])}</td>
      <td>{row['label']}{best_icon}</td>
      <td><small class="text-muted">{row['controls']}</small></td>
      {_cell(row['return'], is_return=True)}
      {_cell(row['sharpe'])}
      {_cell(row['max_dd'], is_dd=True)}
      {_cell(row['calmar'])}
      {_cell(row['win_rate'])}
      <td class="text-end">{row['n_trades']}</td>
      {_cell(row['oos_return'], is_return=True)}
      {_cell(row['oos_sharpe'])}
      {_cell(row['oos_dd'], is_dd=True)}
      <td class="text-end">{row['wfo_pct']}</td>
    </tr>"""

# Summary cards
def _card(title, val, sub="", color="primary"):
    return f"""
    <div class="col">
      <div class="card bg-dark border-{color} h-100">
        <div class="card-body text-center">
          <div class="text-{color} fw-bold fs-4">{val}</div>
          <div class="card-title small text-white mb-1">{title}</div>
          <div class="text-muted" style="font-size:.75rem">{sub}</div>
        </div>
      </div>
    </div>"""

def _summary_cards(k, wf, phase_label):
    ret   = k["total_return"] * 100
    dd    = k["max_drawdown"] * 100
    oos_r = wf["full_kpis"]["total_return"] * 100 if wf else None
    oos_d = wf["full_kpis"]["max_drawdown"] * 100 if wf else None
    c1 = "success" if ret > 0 else "danger"
    c2 = "success" if dd > -20 else ("warning" if dd > -40 else "danger")
    cards = [
        _card("IS Return", f"{ret:+.1f}%", f"Sharpe {k['sharpe']:.2f}", c1),
        _card("IS Max DD", f"{dd:+.1f}%", f"Calmar {k['calmar']:.2f}", c2),
        _card("# Trades", str(k["n_trades"]), f"Win rate {k['win_rate']*100:.0f}%", "info"),
    ]
    if wf:
        c3 = "success" if oos_r > 0 else "danger"
        c4 = "success" if oos_d > -20 else ("warning" if oos_d > -40 else "danger")
        cards += [
            _card("OOS Return", f"{oos_r:+.1f}%", f"WFO Sharpe {wf['full_kpis']['sharpe']:.2f}", c3),
            _card("OOS Max DD", f"{oos_d:+.1f}%", f"{wf['pct_profitable']:.0f}% win windows", c4),
        ]
    return '<div class="row row-cols-2 row-cols-md-5 g-2">' + "".join(cards) + "</div>"

cards1 = _summary_cards(k1, wf1, "Phase 1")
cards2 = _summary_cards(k2, wf2, "Phase 2")
best_risk = risk_rows[df_risk[df_risk["label"] == best_label].index[0]]
k_best_risk = {
    "total_return": best_risk["return"], "max_drawdown": best_risk["max_dd"],
    "sharpe": best_risk["sharpe"], "calmar": best_risk["calmar"],
    "win_rate": best_risk["win_rate"], "n_trades": best_risk["n_trades"],
}
cards3 = _summary_cards(k_best_risk, wf3, "Phase 3 best")

generated_at = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

html = f"""<!DOCTYPE html>
<html lang="en" data-bs-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BTCUSDT Strategy — Version Comparison</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
<style>
  body {{ background: #0d1117; color: #e6edf3; }}
  .card {{ background: #161b22 !important; }}
  .section-title {{ border-left: 4px solid #58a6ff; padding-left: .75rem; }}
  .table-active {{ background: rgba(88,166,255,.08) !important; }}
  .table {{ --bs-table-bg: #161b22; --bs-table-striped-bg: #1a2030; }}
  .nav-pills .nav-link.active {{ background: #58a6ff; color: #0d1117; }}
  .nav-pills .nav-link {{ color: #8b949e; }}
  .chart-container {{ position: relative; height: 320px; }}
  .phase-pill-1 {{ background: #1f3d7a; }}
  .phase-pill-2 {{ background: #7a4e1f; }}
  .phase-pill-3 {{ background: #1f5c2e; }}
  .legend-dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:4px; }}
  .sticky-top-nav {{ position: sticky; top: 0; z-index: 1020; background: #0d1117cc;
                     backdrop-filter: blur(8px); border-bottom: 1px solid #30363d; }}
  .tooltip-inner {{ max-width: 300px; }}
  @media (max-width: 768px) {{ .chart-container {{ height: 220px; }} }}
</style>
</head>
<body>

<!-- ── Top nav ── -->
<nav class="navbar sticky-top-nav px-3 py-2">
  <span class="navbar-brand text-white fw-bold">
    <i class="bi bi-graph-up-arrow text-primary me-2"></i>BTCUSDT Strategy Comparison
  </span>
  <span class="text-muted small">Generated {generated_at}</span>
</nav>

<div class="container-xl py-4">

  <!-- ── Section nav ── -->
  <ul class="nav nav-pills mb-4 gap-1" id="sectionNav">
    <li class="nav-item"><a class="nav-link active" href="#overview">Overview</a></li>
    <li class="nav-item"><a class="nav-link" href="#equity">Equity</a></li>
    <li class="nav-item"><a class="nav-link" href="#drawdown">Drawdown</a></li>
    <li class="nav-item"><a class="nav-link" href="#risk-scatter">Risk/Return</a></li>
    <li class="nav-item"><a class="nav-link" href="#wfo">Walk-Forward</a></li>
    <li class="nav-item"><a class="nav-link" href="#full-table">Full Table</a></li>
  </ul>


  <!-- ══════════════════════════════════════════════════════════════ -->
  <!-- SECTION 1 — Overview cards                                    -->
  <!-- ══════════════════════════════════════════════════════════════ -->
  <section id="overview" class="mb-5">
    <h4 class="section-title mb-3">Strategy Evolution</h4>

    <!-- phase timeline -->
    <div class="row g-3 mb-4">
      <div class="col-md-4">
        <div class="card h-100 border-primary">
          <div class="card-header phase-pill-1 text-white fw-bold small">
            <i class="bi bi-1-circle me-1"></i>Phase 1 — Baseline (8 signals)
          </div>
          <div class="card-body small text-muted">
            Weekly momentum · Daily trend · 4H structure · 1H entry ·
            Open Interest · Funding rate · Volume surge · BTC cycle
          </div>
        </div>
      </div>
      <div class="col-md-4">
        <div class="card h-100 border-warning">
          <div class="card-header phase-pill-2 text-white fw-bold small">
            <i class="bi bi-2-circle me-1"></i>Phase 2 — Extended (+15m +1m)
          </div>
          <div class="card-body small text-muted">
            Phase 1 + <strong>15-min precision entry</strong> (RSI-7 zone + MACD histogram)
            and <strong>1-min microstructure</strong> (high-volume bar direction).
            Backtest stays on 1H; sub-1H signals forward-filled.
          </div>
        </div>
      </div>
      <div class="col-md-4">
        <div class="card h-100 border-success">
          <div class="card-header phase-pill-3 text-white fw-bold small">
            <i class="bi bi-3-circle me-1"></i>Phase 3 — Risk Management (13 variants)
          </div>
          <div class="card-body small text-muted">
            Notional cap · Volatility targeting · Drawdown circuit breaker ·
            Strong-signal filter — tested individually and in combination on
            the Phase 2 extended signal matrix.
          </div>
        </div>
      </div>
    </div>

    <!-- Phase 1 summary -->
    <h6 class="text-primary mb-2"><i class="bi bi-1-square me-1"></i>Phase 1 · Baseline</h6>
    {cards1}

    <div class="my-3 border-top border-secondary"></div>

    <!-- Phase 2 summary -->
    <h6 class="text-warning mb-2"><i class="bi bi-2-square me-1"></i>Phase 2 · Extended (+15m +1m)</h6>
    {cards2}

    <div class="my-3 border-top border-secondary"></div>

    <!-- Phase 3 best summary -->
    <h6 class="text-success mb-2">
      <i class="bi bi-3-square me-1"></i>Phase 3 · Best Risk Combo
      <span class="badge bg-success ms-1">{best_label}</span>
    </h6>
    {cards3}
  </section>


  <!-- ══════════════════════════════════════════════════════════════ -->
  <!-- SECTION 2 — Equity curves                                     -->
  <!-- ══════════════════════════════════════════════════════════════ -->
  <section id="equity" class="mb-5">
    <h4 class="section-title mb-3">Equity Curves</h4>
    <div class="card">
      <div class="card-body">
        <p class="text-muted small mb-2">
          In-sample equity (daily resampled). Hover for values.
          Solid lines = Phase 1/2; dashed = risk-managed variants.
        </p>
        <div class="chart-container" style="height:380px">
          <canvas id="equityChart"></canvas>
        </div>
      </div>
    </div>
  </section>


  <!-- ══════════════════════════════════════════════════════════════ -->
  <!-- SECTION 3 — Drawdown                                          -->
  <!-- ══════════════════════════════════════════════════════════════ -->
  <section id="drawdown" class="mb-5">
    <h4 class="section-title mb-3">Drawdown Analysis</h4>
    <div class="card">
      <div class="card-body">
        <p class="text-muted small mb-2">
          Daily drawdown from peak (%). Shallower curves = better downside protection.
        </p>
        <div class="chart-container" style="height:340px">
          <canvas id="drawdownChart"></canvas>
        </div>
      </div>
    </div>
  </section>


  <!-- ══════════════════════════════════════════════════════════════ -->
  <!-- SECTION 4 — Risk / Return scatter                             -->
  <!-- ══════════════════════════════════════════════════════════════ -->
  <section id="risk-scatter" class="mb-5">
    <h4 class="section-title mb-3">Risk / Return Map</h4>
    <div class="card">
      <div class="card-body">
        <p class="text-muted small mb-2">
          Each point = one variant. X-axis: Max Drawdown (negative = worse).
          Y-axis: Total Return. Upper-right = best.
          <span class="legend-dot" style="background:#58a6ff"></span>Phase 1
          <span class="legend-dot ms-2" style="background:#f0883e"></span>Phase 2
          <span class="legend-dot ms-2" style="background:#2ea043"></span>Phase 3
        </p>
        <div class="chart-container" style="height:360px">
          <canvas id="scatterChart"></canvas>
        </div>
      </div>
    </div>
  </section>


  <!-- ══════════════════════════════════════════════════════════════ -->
  <!-- SECTION 5 — Walk-Forward                                      -->
  <!-- ══════════════════════════════════════════════════════════════ -->
  <section id="wfo" class="mb-5">
    <h4 class="section-title mb-3">Walk-Forward Out-of-Sample</h4>
    <div class="row g-3 mb-3">
      <div class="col-md-4">
        <div class="card text-center">
          <div class="card-body">
            <div class="text-muted small">Configuration</div>
            <div class="fw-bold">6-month warm-up + 2-month OOS</div>
            <div class="text-muted small">Step = 2 months (non-overlapping)</div>
          </div>
        </div>
      </div>
      <div class="col-md-4">
        <div class="card text-center">
          <div class="card-body">
            <div class="text-muted small">Baseline WFO</div>
            <div class="fw-bold">{wf1['pct_profitable']:.0f}% profitable windows</div>
            <div class="text-muted small">OOS Sharpe {wf1['full_kpis']['sharpe']:.3f}</div>
          </div>
        </div>
      </div>
      <div class="col-md-4">
        <div class="card text-center">
          <div class="card-body">
            <div class="text-muted small">Best Combo WFO</div>
            <div class="fw-bold text-success">{wf3['pct_profitable']:.0f}% profitable windows</div>
            <div class="text-muted small">OOS Sharpe {wf3['full_kpis']['sharpe']:.3f}</div>
          </div>
        </div>
      </div>
    </div>

    <!-- WFO equity overlay -->
    <div class="card mb-3">
      <div class="card-body">
        <h6 class="text-muted mb-2">Chained OOS Equity (compounded)</h6>
        <div class="chart-container" style="height:300px">
          <canvas id="wfoEqChart"></canvas>
        </div>
      </div>
    </div>

    <!-- Per-window bar charts -->
    <div class="card">
      <div class="card-body">
        <h6 class="text-muted mb-3">OOS Return per 2-Month Window</h6>
        <ul class="nav nav-pills mb-3" id="wfoPills">
          <li class="nav-item">
            <button class="nav-link active" data-wfo="baseline">Baseline</button>
          </li>
          <li class="nav-item">
            <button class="nav-link" data-wfo="extended">Extended</button>
          </li>
          <li class="nav-item">
            <button class="nav-link" data-wfo="best">Best Combo</button>
          </li>
        </ul>
        <div class="chart-container" style="height:260px">
          <canvas id="wfoWindowChart"></canvas>
        </div>
      </div>
    </div>
  </section>


  <!-- ══════════════════════════════════════════════════════════════ -->
  <!-- SECTION 6 — Full comparison table                             -->
  <!-- ══════════════════════════════════════════════════════════════ -->
  <section id="full-table" class="mb-5">
    <h4 class="section-title mb-3">Full Comparison Table</h4>
    <p class="text-muted small">
      OOS columns populated only for variants with WFO run (Baseline, Extended, Best Combo).
      <i class="bi bi-star-fill text-warning"></i> = best Calmar among multi-technique combos.
    </p>
    <div class="table-responsive">
      <table class="table table-sm table-striped table-hover align-middle" style="font-size:.82rem">
        <thead class="table-dark">
          <tr>
            <th>Ph</th><th>Variant</th><th>Controls</th>
            <th class="text-end">IS Return</th>
            <th class="text-end">IS Sharpe</th>
            <th class="text-end">IS Max DD</th>
            <th class="text-end">IS Calmar</th>
            <th class="text-end">Win Rate</th>
            <th class="text-end">Trades</th>
            <th class="text-end">OOS Return</th>
            <th class="text-end">OOS Sharpe</th>
            <th class="text-end">OOS Max DD</th>
            <th class="text-end">WFO Win%</th>
          </tr>
        </thead>
        <tbody>
          {table_html}
        </tbody>
      </table>
    </div>
  </section>

  <footer class="text-muted text-center small py-4 border-top border-secondary mt-4">
    BTCUSDT Quant Strategy — Research Report · {generated_at} ·
    Scenario: {SCENARIO} · Data: Binance Vision
  </footer>
</div>

<!-- ══ Scripts ══ -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
<script>
const DATA = {chart_json};

// ── Chart defaults ──
Chart.defaults.color = "#8b949e";
Chart.defaults.borderColor = "#30363d";
Chart.defaults.font.family = "system-ui, -apple-system, sans-serif";
Chart.defaults.font.size = 11;

const DARK_BG = "#161b22";

function timeAxis(suggestedMinY, suggestedMaxY) {{
  return {{
    x: {{
      type: "time",
      time: {{ unit: "month", tooltipFormat: "MMM yyyy" }},
      grid: {{ color: "#30363d" }},
    }},
    y: {{
      grid: {{ color: "#30363d" }},
      suggestedMin: suggestedMinY,
      suggestedMax: suggestedMaxY,
    }}
  }};
}}

// ── 1. Equity chart ──
new Chart(document.getElementById("equityChart"), {{
  type: "line",
  data: {{ datasets: DATA.equity }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    interaction: {{ mode: "index", intersect: false }},
    plugins: {{
      legend: {{ position: "top", labels: {{ color: "#e6edf3", boxWidth: 12, padding: 10 }} }},
      tooltip: {{
        callbacks: {{
          label: ctx => ` ${{ctx.dataset.label}}: ${{ctx.parsed.y.toLocaleString("en-US", {{style:"currency",currency:"USD",minimumFractionDigits:0}})}}`,
        }}
      }}
    }},
    scales: timeAxis(),
  }}
}});

// ── 2. Drawdown chart ──
new Chart(document.getElementById("drawdownChart"), {{
  type: "line",
  data: {{ datasets: DATA.drawdown }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    interaction: {{ mode: "index", intersect: false }},
    plugins: {{
      legend: {{ position: "top", labels: {{ color: "#e6edf3", boxWidth: 12, padding: 10 }} }},
      tooltip: {{
        callbacks: {{
          label: ctx => ` ${{ctx.dataset.label}}: ${{ctx.parsed.y.toFixed(1)}}%`,
        }}
      }}
    }},
    scales: {{
      x: {{ type: "time", time: {{ unit: "month", tooltipFormat: "MMM yyyy" }},
            grid: {{ color: "#30363d" }} }},
      y: {{ grid: {{ color: "#30363d" }}, suggestedMax: 0,
            ticks: {{ callback: v => v + "%" }} }}
    }},
  }}
}});

// ── 3. Scatter chart ──
(function() {{
  const phaseColors = {{ 1: "#58a6ff", 2: "#f0883e", 3: "#2ea043" }};
  const datasets = [1, 2, 3].map(ph => ({{
    label: "Phase " + ph,
    data: DATA.scatter.filter(p => p.phase === ph).map(p => ({{
      x: p.x, y: p.y, label: p.label, calmar: p.calmar, sharpe: p.sharpe
    }})),
    backgroundColor: phaseColors[ph] + "cc",
    borderColor: phaseColors[ph],
    pointRadius: ph < 3 ? 10 : 7,
    pointHoverRadius: ph < 3 ? 13 : 10,
  }}));

  new Chart(document.getElementById("scatterChart"), {{
    type: "scatter",
    data: {{ datasets }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        legend: {{ position: "top", labels: {{ color: "#e6edf3" }} }},
        tooltip: {{
          callbacks: {{
            label: ctx => [
              ctx.raw.label,
              `Return: ${{ctx.parsed.y.toFixed(1)}}%`,
              `Max DD: ${{ctx.parsed.x.toFixed(1)}}%`,
              `Calmar: ${{ctx.raw.calmar}}  Sharpe: ${{ctx.raw.sharpe}}`,
            ]
          }}
        }}
      }},
      scales: {{
        x: {{ grid: {{ color: "#30363d" }},
              title: {{ display: true, text: "Max Drawdown (%)", color: "#8b949e" }},
              ticks: {{ callback: v => v + "%" }} }},
        y: {{ grid: {{ color: "#30363d" }},
              title: {{ display: true, text: "Total Return (%)", color: "#8b949e" }},
              ticks: {{ callback: v => v + "%" }} }},
      }}
    }}
  }});
}})();

// ── 4. WFO equity chart ──
new Chart(document.getElementById("wfoEqChart"), {{
  type: "line",
  data: {{ datasets: DATA.wfo_equity }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    interaction: {{ mode: "index", intersect: false }},
    plugins: {{
      legend: {{ position: "top", labels: {{ color: "#e6edf3", boxWidth: 12 }} }},
      tooltip: {{
        callbacks: {{
          label: ctx => ` ${{ctx.dataset.label}}: ${{ctx.parsed.y.toLocaleString("en-US", {{style:"currency",currency:"USD",minimumFractionDigits:0}})}}`,
        }}
      }}
    }},
    scales: timeAxis(),
  }}
}});

// ── 5. WFO per-window bars (tabbed) ──
let wfoWindowChart = null;

function buildWfoWindowChart(key) {{
  const d = DATA.wfo_windows[key];
  if (!d) return;
  const rets = d.returns;
  const colors = rets.map(v => v >= 0 ? "#2ea04388" : "#f8514988");
  const borders = rets.map(v => v >= 0 ? "#2ea043" : "#f85149");
  const labels = rets.map((_, i) => "W" + (i + 1));
  const median = rets.slice().sort((a,b)=>a-b)[Math.floor(rets.length/2)];

  if (wfoWindowChart) wfoWindowChart.destroy();
  wfoWindowChart = new Chart(document.getElementById("wfoWindowChart"), {{
    type: "bar",
    data: {{
      labels,
      datasets: [{{
        label: d.label + " OOS Return",
        data: rets,
        backgroundColor: colors,
        borderColor: borders,
        borderWidth: 1,
      }}]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          callbacks: {{
            label: ctx => ` ${{ctx.parsed.y.toFixed(2)}}%  (trades: ${{d.trades[ctx.dataIndex]}})`,
          }}
        }},
        annotation: {{ /* median line would need plugin */ }}
      }},
      scales: {{
        x: {{ grid: {{ color: "#30363d" }}, ticks: {{ color: "#8b949e", font: {{ size: 9 }} }} }},
        y: {{
          grid: {{ color: "#30363d" }},
          ticks: {{ callback: v => v + "%" }},
        }}
      }}
    }}
  }});
}}

buildWfoWindowChart("baseline");

document.querySelectorAll("[data-wfo]").forEach(btn => {{
  btn.addEventListener("click", function() {{
    document.querySelectorAll("[data-wfo]").forEach(b => b.classList.remove("active"));
    this.classList.add("active");
    buildWfoWindowChart(this.dataset.wfo);
  }});
}});

// ── Smooth scroll for nav pills ──
document.querySelectorAll("#sectionNav a").forEach(a => {{
  a.addEventListener("click", function(e) {{
    e.preventDefault();
    document.querySelectorAll("#sectionNav a").forEach(x => x.classList.remove("active"));
    this.classList.add("active");
    document.querySelector(this.getAttribute("href")).scrollIntoView({{behavior:"smooth"}});
  }});
}});
</script>
</body>
</html>
"""

OUT_PATH.write_text(html, encoding="utf-8")
print(f"\nReport saved → {OUT_PATH}  ({OUT_PATH.stat().st_size // 1024} KB)")
print("Done.")
