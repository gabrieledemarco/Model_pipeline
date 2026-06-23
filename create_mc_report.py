"""
Monte Carlo report — 3 validated configurations.

Configurations:
  A. Baseline            : composite ±3, session 08-21, no ML gate
  B. ML Gate P≥0.50      : binary walk-forward gate, best Calmar (1.009)
  C. ML Gate P≥0.50 +    : same gate + vol regime filter (0.5 ≤ rvol_ratio ≤ 2.0)
     Vol Regime Filter

Method: bootstrap resampling of realized trade returns (5,000 paths).
Output → reports/mc_report.html
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.strategy.data_fetcher import fetch_extended_data, generate_oi, generate_funding
from src.strategy.indicators   import add_indicators
from src.strategy.signals      import build_signal_matrix
from src.strategy.optimizer    import apply_filters, ScenarioConfig
from src.strategy.engine       import run_backtest, INIT_CAP
from src.strategy.ml_features  import build_feature_matrix
from src.strategy.ml_gate      import (
    walk_forward_binary_gate,
    run_gated_backtest,
)
from src.strategy.monte_carlo  import run_monte_carlo

SESSION_CFG = ScenarioConfig(
    "Session 08-21",
    session_hours=(8, 21),
    long_threshold=3.0,
    short_threshold=-3.0,
)

N_SIMS = 5_000
SEED   = 42


# ─────────────────────────────────────────────────────────────────────────────
# Vol mask helper (mirrors create_threshold_report.py)
# ─────────────────────────────────────────────────────────────────────────────

def _rvol_ratio(df_1h: pd.DataFrame) -> pd.Series:
    if "rvol_ratio" in df_1h.columns:
        return df_1h["rvol_ratio"]
    log_ret = df_1h.get("log_ret", np.log(df_1h["close"]).diff())
    rv20  = log_ret.rolling(20,  min_periods=1).std() * np.sqrt(8760)
    rv120 = rv20.rolling(120, min_periods=10).mean()
    return (rv20 / rv120.replace(0, np.nan)).fillna(1.0)


def _apply_vol_regime(df_1h, gate, signals, low=0.5, high=2.0):
    """Re-apply vol regime mask on top of an existing gated signal."""
    ratio     = _rvol_ratio(df_1h)
    in_regime = (ratio >= low) & (ratio <= high)
    gated2    = gate.gated_signal.copy()
    gated2[~in_regime.reindex(gated2.index, fill_value=False)] = 0
    sig_copy           = signals.copy()
    sig_copy["signal"] = gated2
    bt = run_backtest(df_1h, sig_copy)
    return bt


# ─────────────────────────────────────────────────────────────────────────────
# MC stats helpers
# ─────────────────────────────────────────────────────────────────────────────

def _percentile_table(mc: dict) -> list[dict]:
    rows = []
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        rows.append({
            "pct":      p,
            "ret":      round(float(np.percentile(mc["total_return"] * 100, p)), 2),
            "dd":       round(float(np.percentile(mc["max_drawdown"] * 100, p)), 2),
            "sharpe":   round(float(np.percentile(mc["sharpe"], p)), 3),
            "equity":   round(float(np.percentile(mc["final_equity"], p)), 0),
        })
    return rows


def _fan_series(mc: dict, n_points: int = 300) -> dict:
    """Downsample equity paths to p10/p25/p50/p75/p90 bands."""
    paths = mc["paths"]
    n_trades = paths.shape[1]
    idx = np.linspace(0, n_trades - 1, min(n_points, n_trades), dtype=int)
    sub = paths[:, idx]
    return {
        "x":   list(range(len(idx))),
        "p10": np.percentile(sub, 10, axis=0).tolist(),
        "p25": np.percentile(sub, 25, axis=0).tolist(),
        "p50": np.percentile(sub, 50, axis=0).tolist(),
        "p75": np.percentile(sub, 75, axis=0).tolist(),
        "p90": np.percentile(sub, 90, axis=0).tolist(),
    }


def _ret_histogram(mc: dict, bins: int = 60) -> dict:
    tr = mc["total_return"] * 100
    counts, edges = np.histogram(tr, bins=bins)
    centers = ((edges[:-1] + edges[1:]) / 2).tolist()
    return {"x": centers, "y": counts.tolist()}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n══ Monte Carlo Report ({N_SIMS:,} simulations) ══════════════════════")

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("\n[1/6] Loading data …")
    raw = fetch_extended_data(start_year=2022, start_month=1,
                              fetch_1m=False, fetch_flow=True)
    tf_ind = {}
    for tf in ["1W", "1D", "4H", "1H", "15M"]:
        df = raw.get(tf, pd.DataFrame())
        tf_ind[tf] = add_indicators(df) if not df.empty and len(df) > 20 else df

    df_1h   = tf_ind["1H"]
    df_15m  = tf_ind["15M"]
    oi_df   = generate_oi(tf_ind["1D"]["close"])
    funding = generate_funding(tf_ind["1D"]["close"])

    # ── 2. Signals ────────────────────────────────────────────────────────────
    print("\n[2/6] Building signals (±3, session 08-21) …")
    raw_sig = build_signal_matrix(
        tf_data=tf_ind, oi_df=oi_df, funding=funding,
        premium_1h=None, df_15m=df_15m, df_1m=None,
    )
    signals = apply_filters(raw_sig, SESSION_CFG)

    # ── 3. Feature matrix + ML gate (single walk-forward pass) ───────────────
    print("\n[3/6] Feature matrix + walk-forward binary gate (P≥0.50) …")
    feat_df = build_feature_matrix(tf_ind, signals)
    gate = walk_forward_binary_gate(
        df_1h, signals, feat_df,
        gate_threshold=0.50, use_feat_sel=True, verbose=True,
    )

    # ── 4. Run the 3 backtests ────────────────────────────────────────────────
    print("\n[4/6] Running 3 backtests …")

    bt_baseline = run_backtest(df_1h, signals)
    bt_ml       = run_gated_backtest(df_1h, signals, gate)
    bt_vol      = _apply_vol_regime(df_1h, gate, signals, low=0.5, high=2.0)

    def _print_bt(label, bt):
        k = bt["kpis"]
        print(f"  {label}: ret={k.get('total_return',0)*100:+.1f}%  "
              f"DD={k.get('max_drawdown',0)*100:.1f}%  "
              f"Calmar={k.get('calmar',0):.3f}  "
              f"Trades={k.get('n_trades',0)}")

    _print_bt("A. Baseline",              bt_baseline)
    _print_bt("B. ML P≥0.50",             bt_ml)
    _print_bt("C. ML P≥0.50 + VolRegime", bt_vol)

    # ── 5. Monte Carlo ────────────────────────────────────────────────────────
    print(f"\n[5/6] Monte Carlo ({N_SIMS:,} simulations per config) …")
    configs = [
        ("A. Baseline",               bt_baseline),
        ("B. ML Gate P≥0.50",         bt_ml),
        ("C. ML Gate P≥0.50 + Vol",   bt_vol),
    ]

    mc_results = {}
    for name, bt in configs:
        trades = bt.get("trades")
        if trades is None or trades.empty:
            print(f"  {name}: no trades — skipping")
            mc_results[name] = {}
            continue
        mc = run_monte_carlo(trades, initial_capital=INIT_CAP,
                             n_sims=N_SIMS, seed=SEED)
        mc_results[name] = mc
        print(f"  {name}: n_trades={mc['n_trades']}  "
              f"P(profit)={mc['p_profit']*100:.1f}%  "
              f"P(ruin)={mc['p_ruin']*100:.2f}%  "
              f"median_ret={np.median(mc['total_return']*100):+.1f}%")

    # ── 6. HTML report ────────────────────────────────────────────────────────
    print("\n[6/6] Generating HTML report …")
    html = _build_html(mc_results, bt_baseline, bt_ml, bt_vol)
    out  = Path("reports/mc_report.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"  Saved → {out}")
    print("\n══ Done ══════════════════════════════════════════════════════════")


# ─────────────────────────────────────────────────────────────────────────────
# HTML builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_html(mc_results: dict, bt_baseline, bt_ml, bt_vol) -> str:

    COLORS = {
        "A. Baseline":             "#64b5f6",
        "B. ML Gate P≥0.50":       "#81c784",
        "C. ML Gate P≥0.50 + Vol": "#ffb74d",
    }

    # ── Serialise data for JS ─────────────────────────────────────────────────
    fan_data, hist_data, pct_tables, summary_rows = {}, {}, {}, []

    for name, mc in mc_results.items():
        if not mc:
            continue
        fan_data[name]   = _fan_series(mc)
        hist_data[name]  = _ret_histogram(mc)
        pct_tables[name] = _percentile_table(mc)
        summary_rows.append({
            "name":      name,
            "n_trades":  mc["n_trades"],
            "p_profit":  round(mc["p_profit"]  * 100, 1),
            "p_ruin":    round(mc["p_ruin"]    * 100, 2),
            "med_ret":   round(float(np.median(mc["total_return"] * 100)), 2),
            "p5_ret":    round(float(np.percentile(mc["total_return"] * 100,  5)), 2),
            "p95_ret":   round(float(np.percentile(mc["total_return"] * 100, 95)), 2),
            "med_dd":    round(float(np.median(mc["max_drawdown"]  * 100)), 2),
            "worst_dd":  round(float(np.percentile(mc["max_drawdown"] * 100,  5)), 2),
            "med_sharpe":round(float(np.median(mc["sharpe"])), 3),
            "color":     COLORS.get(name, "#aaa"),
        })

    realized = {
        "A. Baseline":             bt_baseline["kpis"],
        "B. ML Gate P≥0.50":       bt_ml["kpis"],
        "C. ML Gate P≥0.50 + Vol": bt_vol["kpis"],
    }

    data_json = json.dumps({
        "fan":     fan_data,
        "hist":    hist_data,
        "pct":     pct_tables,
        "summary": summary_rows,
        "colors":  COLORS,
        "init_cap": INIT_CAP,
        "n_sims":   N_SIMS,
        "realized": {k: {
            "ret":    round(v.get("total_return", 0) * 100, 2),
            "dd":     round(v.get("max_drawdown",  0) * 100, 2),
            "calmar": round(v.get("calmar",         0), 3),
            "sharpe": round(v.get("sharpe",         0), 3),
            "trades": v.get("n_trades", 0),
        } for k, v in realized.items()},
    }, default=str)

    # ── Color helper for table cells ──────────────────────────────────────────
    def _summary_table_rows():
        rows_html = ""
        for r in summary_rows:
            def cc(v, good_high=True):
                if good_high:
                    cls = "pos" if v > 0 else "neg"
                else:
                    cls = "neg" if v < 0 else "pos"
                return f'<td class="{cls}">{v}</td>'

            rz = realized.get(r["name"], {})
            rows_html += f"""
            <tr>
              <td><span class="dot" style="background:{r['color']}"></span>{r['name']}</td>
              <td>{r['n_trades']}</td>
              {cc(rz.get('ret', 0))}
              {cc(rz.get('dd',  0), good_high=False)}
              {cc(rz.get('calmar', 0))}
              {cc(r['p_profit'])}
              {cc(r['p_ruin'], good_high=False)}
              {cc(r['med_ret'])}
              {cc(r['p5_ret'])}
              {cc(r['p95_ret'])}
              {cc(r['med_dd'], good_high=False)}
              {cc(r['worst_dd'], good_high=False)}
              {cc(r['med_sharpe'])}
            </tr>"""
        return rows_html

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Monte Carlo Report — BTCUSDT Strategy</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0d1117; color: #c9d1d9; font-family: 'Segoe UI', sans-serif; padding: 24px; }}
  h1 {{ color: #58a6ff; margin-bottom: 6px; font-size: 1.6rem; }}
  h2 {{ color: #8b949e; font-size: 1.1rem; margin: 28px 0 12px; border-bottom: 1px solid #21262d; padding-bottom: 6px; }}
  .subtitle {{ color: #8b949e; font-size: 0.9rem; margin-bottom: 24px; }}
  .card {{ background: #161b22; border: 1px solid #21262d; border-radius: 10px; padding: 20px; margin-bottom: 20px; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }}
  canvas {{ max-height: 320px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
  th {{ background: #21262d; color: #8b949e; padding: 8px 10px; text-align: right; font-weight: 600; }}
  th:first-child {{ text-align: left; }}
  td {{ padding: 7px 10px; text-align: right; border-bottom: 1px solid #21262d; }}
  td:first-child {{ text-align: left; }}
  tr:hover td {{ background: #1c2128; }}
  .pos {{ color: #3fb950; }}
  .neg {{ color: #f85149; }}
  .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 14px; }}
  .stat-box {{ background: #0d1117; border: 1px solid #21262d; border-radius: 8px; padding: 14px; text-align: center; }}
  .stat-val {{ font-size: 1.4rem; font-weight: 700; }}
  .stat-lbl {{ font-size: 0.75rem; color: #8b949e; margin-top: 4px; }}
  .section-note {{ font-size: 0.78rem; color: #6e7681; margin-bottom: 10px; }}
</style>
</head>
<body>
<h1>Monte Carlo Stress-Test — BTCUSDT Perpetual Futures</h1>
<p class="subtitle">
  Bootstrap resampling of realized trade returns &nbsp;|&nbsp;
  {N_SIMS:,} simulations per configuration &nbsp;|&nbsp;
  Composite ±3, Session 08–21 UTC &nbsp;|&nbsp;
  Walk-forward 6m/2m, 23 windows (2022–2026)
</p>

<script>
const DATA = {data_json};
const COLORS = DATA.colors;
const INIT_CAP = DATA.init_cap;
</script>

<!-- ── Summary stats ────────────────────────────────────────────────────── -->
<h2>1. Realized OOS Performance (single path)</h2>
<div class="card">
<div class="stat-grid" id="stat-grid"></div>
</div>

<!-- ── Fan charts ───────────────────────────────────────────────────────── -->
<h2>2. Equity Path Distribution (p10 / p25 / p50 / p75 / p90)</h2>
<p class="section-note">Each band shows the range of simulated equity paths. The median path (p50) is the thick line.</p>
<div class="grid-3" id="fan-charts"></div>

<!-- ── Return histograms ─────────────────────────────────────────────────── -->
<h2>3. Distribution of Final Returns (%)</h2>
<p class="section-note">Frequency distribution across {N_SIMS:,} bootstrap paths. Dashed vertical = realized OOS return.</p>
<div class="grid-3" id="hist-charts"></div>

<!-- ── Percentile tables ─────────────────────────────────────────────────── -->
<h2>4. Percentile Breakdown</h2>
<div class="grid-3" id="pct-tables"></div>

<!-- ── Comparison table ──────────────────────────────────────────────────── -->
<h2>5. Full Comparison</h2>
<div class="card">
<table>
  <thead>
    <tr>
      <th>Configuration</th>
      <th>Trades</th>
      <th>OOS Ret%</th>
      <th>OOS DD%</th>
      <th>Calmar</th>
      <th>P(Profit)%</th>
      <th>P(Ruin)%</th>
      <th>Med Ret%</th>
      <th>p5 Ret%</th>
      <th>p95 Ret%</th>
      <th>Med DD%</th>
      <th>Worst DD p5%</th>
      <th>Med Sharpe</th>
    </tr>
  </thead>
  <tbody>{_summary_table_rows()}</tbody>
</table>
</div>

<script>
// ── Stat boxes ──────────────────────────────────────────────────────────────
const statGrid = document.getElementById('stat-grid');
DATA.summary.forEach(r => {{
  const rz = DATA.realized[r.name] || {{}};
  statGrid.innerHTML += `
    <div class="stat-box">
      <div class="stat-val ${{rz.ret >= 0 ? 'pos' : 'neg'}}" style="color:${{r.color}}">${{rz.ret >= 0 ? '+' : ''}}${{rz.ret}}%</div>
      <div class="stat-lbl">${{r.name}}</div>
      <div style="font-size:0.72rem;color:#8b949e;margin-top:4px">
        DD: ${{rz.dd}}% &nbsp;|&nbsp; Calmar: ${{rz.calmar}}
      </div>
    </div>
    <div class="stat-box">
      <div class="stat-val ${{r.p_profit >= 50 ? 'pos' : 'neg'}}">${{r.p_profit}}%</div>
      <div class="stat-lbl">P(Profit) — ${{r.name.split(' ')[0]}}</div>
    </div>
    <div class="stat-box">
      <div class="stat-val ${{r.p_ruin <= 5 ? 'pos' : 'neg'}}">${{r.p_ruin}}%</div>
      <div class="stat-lbl">P(Ruin &lt;50%) — ${{r.name.split(' ')[0]}}</div>
    </div>`;
}});

// ── Fan charts ──────────────────────────────────────────────────────────────
const fanContainer = document.getElementById('fan-charts');
Object.entries(DATA.fan).forEach(([name, f]) => {{
  const color = COLORS[name] || '#aaa';
  const div   = document.createElement('div');
  div.className = 'card';
  div.innerHTML = `<p style="font-weight:600;margin-bottom:10px;color:${{color}}">${{name}}</p><canvas id="fan_${{name.replace(/[^a-z0-9]/gi,'_')}}"></canvas>`;
  fanContainer.appendChild(div);

  const rz  = DATA.realized[name] || {{}};
  const ctx  = div.querySelector('canvas').getContext('2d');
  new Chart(ctx, {{
    type: 'line',
    data: {{
      labels: f.x,
      datasets: [
        {{ label: 'p90', data: f.p90, borderColor: 'transparent', backgroundColor: color + '22', fill: '+1', pointRadius: 0 }},
        {{ label: 'p75', data: f.p75, borderColor: 'transparent', backgroundColor: color + '33', fill: '+1', pointRadius: 0 }},
        {{ label: 'p50', data: f.p50, borderColor: color,         backgroundColor: 'transparent', borderWidth: 2, fill: false, pointRadius: 0, label: 'Median' }},
        {{ label: 'p25', data: f.p25, borderColor: 'transparent', backgroundColor: color + '33', fill: '-1', pointRadius: 0 }},
        {{ label: 'p10', data: f.p10, borderColor: 'transparent', backgroundColor: color + '22', fill: '-1', pointRadius: 0 }},
      ],
    }},
    options: {{
      responsive: true,
      animation: false,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{ callbacks: {{ label: c => '$' + c.parsed.y.toFixed(0) }} }},
      }},
      scales: {{
        x: {{ display: false }},
        y: {{ ticks: {{ color: '#8b949e', callback: v => '$'+v.toLocaleString() }}, grid: {{ color: '#21262d' }} }},
      }},
    }},
  }});
}});

// ── Histogram charts ─────────────────────────────────────────────────────────
const histContainer = document.getElementById('hist-charts');
Object.entries(DATA.hist).forEach(([name, h]) => {{
  const color = COLORS[name] || '#aaa';
  const rz    = DATA.realized[name] || {{}};
  const div   = document.createElement('div');
  div.className = 'card';
  div.innerHTML = `<p style="font-weight:600;margin-bottom:10px;color:${{color}}">${{name}}</p><canvas id="hist_${{name.replace(/[^a-z0-9]/gi,'_')}}"></canvas>`;
  histContainer.appendChild(div);

  const ctx = div.querySelector('canvas').getContext('2d');
  const barColors = h.x.map(v => v >= 0 ? color + 'cc' : '#f8514966');
  new Chart(ctx, {{
    type: 'bar',
    data: {{
      labels: h.x.map(v => v.toFixed(1) + '%'),
      datasets: [{{
        data: h.y,
        backgroundColor: barColors,
        borderWidth: 0,
      }}],
    }},
    options: {{
      responsive: true,
      animation: false,
      plugins: {{
        legend: {{ display: false }},
        annotation: rz.ret !== undefined ? {{
          annotations: {{
            realized: {{
              type: 'line',
              xMin: rz.ret, xMax: rz.ret,
              borderColor: '#fff',
              borderWidth: 1.5,
              borderDash: [4, 3],
            }},
          }},
        }} : {{}},
      }},
      scales: {{
        x: {{
          ticks: {{
            color: '#8b949e', maxRotation: 45,
            callback: (v, i) => i % 10 === 0 ? h.x[i].toFixed(0) + '%' : '',
          }},
          grid: {{ display: false }},
        }},
        y: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }},
      }},
    }},
  }});
}});

// ── Percentile tables ────────────────────────────────────────────────────────
const pctContainer = document.getElementById('pct-tables');
Object.entries(DATA.pct).forEach(([name, rows]) => {{
  const color = COLORS[name] || '#aaa';
  const div   = document.createElement('div');
  div.className = 'card';
  let html = `<p style="font-weight:600;margin-bottom:10px;color:${{color}}">${{name}}</p>
    <table>
      <thead><tr><th>Pct</th><th>Ret%</th><th>Max DD%</th><th>Sharpe</th><th>Equity $</th></tr></thead>
      <tbody>`;
  rows.forEach(r => {{
    const rc = v => `<td class="${{v>=0?'pos':'neg'}}">${{v}}</td>`;
    const dc = v => `<td class="${{v<=0?'pos':'neg'}}">${{v}}</td>`;
    html += `<tr><td><b>p${{r.pct}}</b></td>${{rc(r.ret)}}${{dc(r.dd)}}${{rc(r.sharpe)}}<td>$${{r.equity.toLocaleString()}}</td></tr>`;
  }});
  html += '</tbody></table>';
  div.innerHTML = html;
  pctContainer.appendChild(div);
}});
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
