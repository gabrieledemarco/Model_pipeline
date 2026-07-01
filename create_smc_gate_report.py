"""
SMC-augmented ML gate report.

Tests whether adding multi-timeframe Smart Money Concepts (SMC) features
to the ML gate's feature matrix improves walk-forward performance.

Configurations
──────────────
  REF  : Baseline 1H full-sample backtest (no gate, for reference)
  A    : Walk-forward ML gate — existing features (~75 cols)
  B    : Walk-forward ML gate — existing features + multi-TF SMC features

SMC features added (per timeframe: 1W / 1D / 4H / 1H)
  – Structural trend (+1 / -1)
  – bars_since_choch_bull/bear_{1-5}   (last 5 CHoCH events)
  – bars_since_bos_bull/bear_{1-5}     (last 5 BOS events)
  – bars_since_fvg_bull/bear_{1-5}     (last 5 FVG formation bars)
  – ob_bull_in / ob_bear_in            (price inside active order block)
  – fvg_bull_active / fvg_bear_active  (price inside unfilled FVG)
  – in_premium / in_discount / zone_pct
  – dist_to_sh / dist_to_sl            (distance to swing level in ATR)
  – eq_high / eq_low
  Plus internal structure (n=5) for same-TF (1H): int_trend, bars_since_int_*

Total new features: ≈ 130 (selected down to top-20 by walk-forward feature selection)

Feature importance analysis shows which SMC features the model finds most useful.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.strategy.data_fetcher  import fetch_extended_data, generate_oi, generate_funding
from src.strategy.indicators    import add_indicators
from src.strategy.signals       import build_signal_matrix
from src.strategy.optimizer     import apply_filters, ScenarioConfig
from src.strategy.engine        import run_backtest, INIT_CAP
from src.strategy.ml_features   import build_feature_matrix, SMC_N_EVENTS, SMC_SWING_LENGTHS
from src.strategy.ml_gate       import (
    walk_forward_binary_gate,
    run_gated_backtest,
    MLGateResult,
)

SESSION_CFG = ScenarioConfig(
    "Session 08-21",
    session_hours=(8, 21),
    long_threshold=3.0,
    short_threshold=-3.0,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def kpi_row(name: str, bt: dict, extra: dict | None = None) -> dict:
    k = bt["kpis"]
    r = {
        "name":          name,
        "total_return":  round(k.get("total_return",  0) * 100, 2),
        "max_dd":        round(k.get("max_drawdown",  0) * 100, 2),
        "sharpe":        round(k.get("sharpe",        0), 3),
        "calmar":        round(k.get("calmar",        0), 3),
        "win_rate":      round(k.get("win_rate",      0) * 100, 1),
        "profit_factor": round(k.get("profit_factor", 0), 2),
        "n_trades":      k.get("n_trades", 0),
        "expectancy":    round(k.get("expectancy",    0), 0),
        "final_equity":  round(k.get("final_equity",  0), 0),
        "equity":        bt["equity"].tolist(),
        "drawdown":      bt["drawdown"].tolist(),
        "index":         [str(t) for t in bt["equity"].index],
    }
    if extra:
        r.update(extra)
    return r


def _impact(row: dict, ref: dict) -> str:
    dr = row["total_return"] - ref["total_return"]
    dd = row["max_dd"]       - ref["max_dd"]
    dc = row["calmar"]       - ref["calmar"]
    ds = row["sharpe"]       - ref["sharpe"]
    s  = lambda v: ("+" if v >= 0 else "")
    return (f"ΔReturn {s(dr)}{dr:.1f}%p &nbsp; "
            f"ΔDD {s(dd)}{dd:.1f}%p &nbsp; "
            f"ΔCalmar {s(dc)}{dc:.3f} &nbsp; "
            f"ΔSharpe {s(ds)}{ds:.3f}")


def _importance_table(imp_df: pd.DataFrame | None, top_n: int = 25) -> str:
    if imp_df is None or imp_df.empty:
        return "<p>No feature importance data.</p>"
    top = imp_df.nlargest(top_n, "importance")
    rows = "".join(
        f"<tr><td>{r['feature']}</td><td>{r['importance']:.4f}</td></tr>"
        for _, r in top.iterrows()
    )
    return f"""
<table class="metrics">
<thead><tr><th>Feature</th><th>Avg Importance</th></tr></thead>
<tbody>{rows}</tbody>
</table>"""


def _wf_window_table(stats: list[dict]) -> str:
    if not stats:
        return "<p>No window stats available.</p>"
    ths = "<tr><th>Window</th><th>OOS Return</th><th>OOS DD</th><th>OOS Sharpe</th><th># Trades</th><th>WR%</th><th>PF</th></tr>"
    rows = ""
    for i, w in enumerate(stats):
        ret = round(w.get("oos_ret", 0) * 100, 2)
        dd  = round(w.get("oos_dd", 0)  * 100, 2)
        sh  = round(w.get("oos_sharpe", 0), 3)
        nt  = w.get("oos_n", 0)
        wr  = round(w.get("oos_wr", 0) * 100, 1)
        pf  = round(w.get("oos_pf", 0), 2)
        clr = "color:#4caf50" if ret > 0 else "color:#f44336"
        rows += f'<tr><td>W{i+1:02d}</td><td style="{clr}">{ret:+.2f}%</td><td>{dd:.2f}%</td><td>{sh:.3f}</td><td>{nt}</td><td>{wr:.1f}%</td><td>{pf:.2f}</td></tr>\n'
    return f'<table class="metrics"><thead>{ths}</thead><tbody>{rows}</tbody></table>'


_PALETTE = ["#2196f3", "#f44336", "#4caf50", "#ff9800", "#9c27b0"]


def _equity_chart(rows: list[dict], chart_id: str) -> str:
    datasets = []
    for i, r in enumerate(rows):
        c = _PALETTE[i % len(_PALETTE)]
        datasets.append({
            "label": r["name"],
            "data":  [{"x": t, "y": round(e, 2)}
                      for t, e in zip(r["index"], r["equity"])],
            "borderColor": c, "backgroundColor": c + "22",
            "borderWidth": 2, "pointRadius": 0, "fill": False,
        })
    return f"""
<div class="chart-wrap"><canvas id="{chart_id}" height="300"></canvas></div>
<script>
(function() {{
  new Chart(document.getElementById('{chart_id}').getContext('2d'), {{
    type: 'line',
    data: {{ datasets: {json.dumps(datasets)} }},
    options: {{
      responsive: true, animation: false,
      interaction: {{ mode: 'index', intersect: false }},
      plugins: {{
        legend: {{ position: 'top', labels: {{ color: '#ccc' }} }},
        tooltip: {{ callbacks: {{ label: ctx => ctx.dataset.label + ': $' + ctx.parsed.y.toLocaleString() }} }}
      }},
      scales: {{
        x: {{ type: 'time', time: {{ unit: 'month' }}, ticks: {{ color: '#aaa' }}, grid: {{ color: '#333' }} }},
        y: {{ ticks: {{ color: '#aaa', callback: v => '$' + v.toLocaleString() }}, grid: {{ color: '#333' }} }}
      }}
    }}
  }});
}})();
</script>"""


def _metrics_table(rows: list[dict], ref: dict) -> str:
    cols    = ["name", "total_return", "max_dd", "sharpe", "calmar",
               "win_rate", "profit_factor", "n_trades", "expectancy", "final_equity"]
    headers = ["Config", "Return %", "Max DD %", "Sharpe", "Calmar",
               "Win %", "PF", "# Trades", "Expectancy $", "Final Equity $"]
    ths = "".join(f"<th>{h}</th>" for h in headers)
    trs = ""
    for i, r in enumerate(rows):
        extra = ""
        if r["name"] != ref["name"]:
            extra = f'<br><small style="color:#888">{_impact(r, ref)}</small>'
        tds = ""
        for j, c in enumerate(cols):
            v = r.get(c, "")
            tds += f"<td><b>{v}</b>{extra if j == 0 else ''}</td>"
        bg = "background:#1e3a1e" if r["name"] == ref["name"] else ""
        trs += f'<tr style="{bg}">{tds}</tr>\n'
    return f'<table class="metrics"><thead><tr>{ths}</tr></thead><tbody>{trs}</tbody></table>'


def _smc_config_table() -> str:
    rows = ""
    for tf, n in SMC_SWING_LENGTHS.items():
        rows += f"<tr><td>{tf}</td><td>{n} bars</td><td>~{n} × bar_dur on each side</td></tr>\n"
    return f"""<table class="metrics">
<thead><tr><th>Timeframe</th><th>Swing Length</th><th>Structure lookback</th></tr></thead>
<tbody>{rows}</tbody>
</table>"""


def _html(rows: list[dict], ref: dict,
          gate_a: MLGateResult, gate_b: MLGateResult,
          feat_a_shape: tuple, feat_b_shape: tuple) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SMC-Augmented ML Gate Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
<style>
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ background:#121212; color:#e0e0e0; font-family:'Segoe UI',sans-serif; padding:24px; }}
h1  {{ color:#fff; margin-bottom:8px; }}
h2  {{ color:#90caf9; margin:28px 0 12px; font-size:1.2rem; border-bottom:1px solid #333; padding-bottom:6px; }}
h3  {{ color:#aaa; margin:18px 0 8px; font-size:1rem; }}
p   {{ color:#bbb; margin:6px 0; line-height:1.6; }}
table.metrics {{ width:100%; border-collapse:collapse; font-size:0.82rem; margin-bottom:16px; }}
table.metrics th {{ background:#1e2a3a; color:#90caf9; padding:7px 10px; text-align:right; border:1px solid #2a3a4a; }}
table.metrics th:first-child {{ text-align:left; }}
table.metrics td {{ padding:6px 10px; border:1px solid #2a2a2a; text-align:right; }}
table.metrics td:first-child {{ text-align:left; color:#e0e0e0; }}
table.metrics tbody tr:hover {{ background:#1a2030; }}
.chart-wrap {{ background:#1a1a2e; border-radius:8px; padding:16px; margin:16px 0; }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
.info  {{ background:#1e2a3a; border-left:3px solid #90caf9; padding:10px 14px; margin:12px 0; border-radius:0 6px 6px 0; }}
</style>
</head>
<body>
<h1>SMC-Augmented ML Gate — Impact Analysis</h1>
<p>Walk-forward comparison: ML gate with vs without multi-timeframe SMC features.</p>
<p>Setup: 6-month train / 2-month OOS / rolling windows / feature selection top-20.</p>

<h2>Performance Summary</h2>
{_metrics_table(rows, ref)}
{_equity_chart(rows, "chart_main")}

<h2>SMC Feature Configuration</h2>
{_smc_config_table()}
<div class="info">
  <b>Feature matrix sizes:</b>
  Config A (no SMC): {feat_a_shape[0]:,} rows × {feat_a_shape[1]} features &nbsp;|&nbsp;
  Config B (+ SMC): {feat_b_shape[0]:,} rows × {feat_b_shape[1]} features
  (Config B adds ≈ {feat_b_shape[1] - feat_a_shape[1]} SMC features; feature selection keeps top-20)
</div>

<h2>Config A — Top Feature Importances (baseline, no SMC)</h2>
{_importance_table(gate_a.importances)}

<h2>Config B — Top Feature Importances (+ multi-TF SMC)</h2>
{_importance_table(gate_b.importances)}

<h2>Walk-Forward Window Detail — Config A</h2>
{_wf_window_table(gate_a.window_stats)}

<h2>Walk-Forward Window Detail — Config B (+ SMC)</h2>
{_wf_window_table(gate_b.window_stats)}

</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n══ SMC-Augmented ML Gate Report ═══════════════════════════════")

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("\n[1/6] Loading data …")
    raw = fetch_extended_data(start_year=2022, start_month=1,
                              fetch_1m=False, fetch_flow=True)
    tf_ind = {}
    for tf in ["1W", "1D", "4H", "1H", "15M"]:
        df = raw.get(tf, pd.DataFrame())
        tf_ind[tf] = add_indicators(df) if not df.empty and len(df) > 20 else df

    df_1h  = tf_ind["1H"]
    oi_df  = generate_oi(tf_ind["1D"]["close"])
    funding = generate_funding(tf_ind["1D"]["close"])
    print(f"  1H bars: {len(df_1h):,}")

    # ── 2. Baseline signal ────────────────────────────────────────────────────
    print("\n[2/6] Building signals …")
    raw_sig = build_signal_matrix(
        tf_data=tf_ind, oi_df=oi_df, funding=funding,
        premium_1h=None, df_15m=tf_ind.get("15M"), df_1m=None,
    )
    sig     = apply_filters(raw_sig, SESSION_CFG)
    n_sig   = int((sig["signal"] != 0).sum())
    print(f"  Signal bars: {n_sig:,}")

    # ── 3. Feature matrices ───────────────────────────────────────────────────
    print("\n[3/6] Building feature matrices …")
    print("  A. Baseline features (no SMC) …")
    feat_a = build_feature_matrix(tf_ind, sig, base_tf="1H", include_smc=False)
    print(f"     Shape: {feat_a.shape[0]:,} × {feat_a.shape[1]}")

    print(f"  B. + multi-TF SMC features (n_events={SMC_N_EVENTS}, "
          f"swing_len: {SMC_SWING_LENGTHS}) …")
    feat_b = build_feature_matrix(tf_ind, sig, base_tf="1H", include_smc=True)
    print(f"     Shape: {feat_b.shape[0]:,} × {feat_b.shape[1]}")
    n_new = feat_b.shape[1] - feat_a.shape[1]
    print(f"     → +{n_new} SMC features added")

    # List new SMC feature groups
    smc_cols = [c for c in feat_b.columns if c not in feat_a.columns]
    tfs_seen = sorted({c.split("_")[1] for c in smc_cols if c.startswith("smc_")})
    for tf in tfs_seen:
        n_tf = sum(1 for c in smc_cols if c.startswith(f"smc_{tf.lower()}_"))
        print(f"       smc_{tf.lower()}_*: {n_tf} features")

    # ── 4. REF: full-sample backtest (no gate) ────────────────────────────────
    print("\n[4/6] REF — full-sample backtest (no gate) …")
    bt_ref  = run_backtest(df_1h, sig)
    row_ref = kpi_row("REF — no gate (full-sample)", bt_ref,
                      extra={"n_sig": n_sig})
    print(f"  → Return={row_ref['total_return']:+.1f}%  DD={row_ref['max_dd']:.1f}%  "
          f"Trades={row_ref['n_trades']}  Calmar={row_ref['calmar']:.3f}")

    # ── 5. Config A: WF ML gate, no SMC ──────────────────────────────────────
    print("\n[5/6] Config A — WF ML gate (baseline features) …")
    gate_a = walk_forward_binary_gate(
        df_1h, sig, feat_a,
        gate_threshold=0.50,
        use_feat_sel=True,
        verbose=True,
    )
    bt_a  = run_gated_backtest(df_1h, sig, gate_a)
    n_a   = int((gate_a.gated_signal != 0).sum())
    row_a = kpi_row("A — ML gate (no SMC)", bt_a,
                    extra={
                        "n_sig": n_a,
                        "filter_rate": round((1 - n_a / n_sig) * 100, 1),
                        "window_stats": gate_a.window_stats,
                    })
    print(f"  → Return={row_a['total_return']:+.1f}%  DD={row_a['max_dd']:.1f}%  "
          f"Trades={row_a['n_trades']}  Calmar={row_a['calmar']:.3f}  "
          f"Filtered={row_a['filter_rate']:.0f}%")

    # ── 6. Config B: WF ML gate + SMC features ────────────────────────────────
    print("\n[6/6] Config B — WF ML gate (+ multi-TF SMC features) …")
    gate_b = walk_forward_binary_gate(
        df_1h, sig, feat_b,
        gate_threshold=0.50,
        use_feat_sel=True,
        verbose=True,
    )
    bt_b  = run_gated_backtest(df_1h, sig, gate_b)
    n_b   = int((gate_b.gated_signal != 0).sum())
    row_b = kpi_row("B — ML gate (+ SMC features)", bt_b,
                    extra={
                        "n_sig": n_b,
                        "filter_rate": round((1 - n_b / n_sig) * 100, 1),
                        "window_stats": gate_b.window_stats,
                    })
    print(f"  → Return={row_b['total_return']:+.1f}%  DD={row_b['max_dd']:.1f}%  "
          f"Trades={row_b['n_trades']}  Calmar={row_b['calmar']:.3f}  "
          f"Filtered={row_b['filter_rate']:.0f}%")

    # ── Top SMC features found by model B ─────────────────────────────────────
    if gate_b.importances is not None:
        top_smc = gate_b.importances[gate_b.importances["feature"].str.startswith("smc_")]
        print("\n  Top SMC features (by avg importance across WF windows):")
        for _, r in top_smc.head(15).iterrows():
            print(f"    {r['feature']:<45s}  {r['importance']:.4f}")

    # ── Write report ──────────────────────────────────────────────────────────
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    html_path = out_dir / "report_smc_gate.html"
    html_path.write_text(
        _html(
            rows=[row_ref, row_a, row_b],
            ref=row_ref,
            gate_a=gate_a,
            gate_b=gate_b,
            feat_a_shape=feat_a.shape,
            feat_b_shape=feat_b.shape,
        ),
        encoding="utf-8",
    )
    print(f"\n  → {html_path.resolve()}")
    print("══ Done ══")


if __name__ == "__main__":
    main()
