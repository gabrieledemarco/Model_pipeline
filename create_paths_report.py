"""
Paths A / B / C exploration report.

Path A – Long / Short Asymmetry
  REF : Baseline 1H (session 08-21, threshold ±3)
  A1  : Short-only   (zero out all long signals)
  A2  : Long-only    (zero out all short signals)
  A3  : Asymmetric threshold  (LONG ≥ +7, SHORT ≤ -3)
  A4  : SMC-aligned  (only trade when SMC structure matches signal direction)

Path B – Smart Money Concepts as composite component
  B1  : Baseline + SMC score (weight = 2) added to composite
  B2  : Baseline + funding extreme block (|funding| > 0.1 % blocks over-extended trades)
  B3  : B1 + B2 combined

Path C – Simplified trend following
  C1  : 4H EMA-50/200 crossover signal (crossing-only)
  C2  : C1 + 1H RSI timing filter (RSI > 50 for longs, < 50 for shorts)
  C3  : SMC CHoCH / BOS structure-only signal

SMC event analysis
  – Distribution of BOS / CHoCH events on 1H data
  – Forward return statistics after each event type
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
from src.strategy.smc          import compute_smc_features, smc_composite_score

SESSION_CFG = ScenarioConfig(
    "Session 08-21",
    session_hours=(8, 21),
    long_threshold=3.0,
    short_threshold=-3.0,
)


# ─────────────────────────────────────────────────────────────────────────────
# KPI helpers
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


# ─────────────────────────────────────────────────────────────────────────────
# Path A helpers
# ─────────────────────────────────────────────────────────────────────────────

def _short_only(sig: pd.DataFrame) -> pd.DataFrame:
    out = sig.copy()
    out.loc[out["signal"] == 1, "signal"] = 0
    return out


def _long_only(sig: pd.DataFrame) -> pd.DataFrame:
    out = sig.copy()
    out.loc[out["signal"] == -1, "signal"] = 0
    return out


def _asymmetric_threshold(raw_sig: pd.DataFrame,
                           long_thresh: float = 7.0,
                           short_thresh: float = -3.0) -> pd.DataFrame:
    """Raise the long entry bar while keeping short entry unchanged."""
    out = raw_sig.copy()
    out["signal"] = np.where(
        out["composite"] >= long_thresh,  1,
        np.where(out["composite"] <= short_thresh, -1, 0)
    ).astype(int)
    # Re-apply session filter
    if SESSION_CFG.session_hours is not None:
        h_s, h_e = SESSION_CFG.session_hours
        not_session = ~((out.index.hour >= h_s) & (out.index.hour < h_e))
        out.loc[not_session, "signal"] = 0
    return out


def _smc_aligned(sig: pd.DataFrame, smc_trend: pd.Series) -> pd.DataFrame:
    """Keep signal only when SMC structural trend agrees with trade direction."""
    out = sig.copy()
    trend = smc_trend.reindex(out.index, method="ffill").fillna(0)
    out.loc[(out["signal"] == 1)  & (trend <= 0), "signal"] = 0
    out.loc[(out["signal"] == -1) & (trend >= 0), "signal"] = 0
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Path B helpers
# ─────────────────────────────────────────────────────────────────────────────

def _add_smc_to_composite(raw_sig: pd.DataFrame,
                           smc_score: pd.Series,
                           weight: float = 2.0) -> pd.DataFrame:
    """Add SMC composite score to the existing composite, then re-threshold."""
    out = raw_sig.copy()
    sc  = smc_score.reindex(out.index, fill_value=0.0)
    out["composite"] = out["composite"] + weight * sc
    out["signal"] = np.where(
        out["composite"] >= SESSION_CFG.long_threshold,  1,
        np.where(out["composite"] <= SESSION_CFG.short_threshold, -1, 0)
    ).astype(int)
    if SESSION_CFG.session_hours is not None:
        h_s, h_e = SESSION_CFG.session_hours
        not_session = ~((out.index.hour >= h_s) & (out.index.hour < h_e))
        out.loc[not_session, "signal"] = 0
    return out


def _funding_extreme_block(sig: pd.DataFrame,
                            funding_1h: pd.Series,
                            threshold: float = 0.001) -> pd.DataFrame:
    """
    Block over-extended entries when funding rate is extreme.
    Extreme positive funding (crowded longs) → block new longs.
    Extreme negative funding (crowded shorts) → block new shorts.
    """
    out = sig.copy()
    fr  = funding_1h.reindex(out.index, method="ffill").fillna(0.0)
    out.loc[(out["signal"] == 1)  & (fr >  threshold), "signal"] = 0
    out.loc[(out["signal"] == -1) & (fr < -threshold), "signal"] = 0
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Path C helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_ema_crossover_signal(df_4h: pd.DataFrame,
                                 df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Simple 4H EMA-50 / EMA-200 crossover signal, aligned to 1H base index.
    Signal fires ONLY at the crossing bar (not on every bar of the trend).
    """
    base = df_1h.index

    # 4H crossings (at close of the 4H bar)
    bull_4h = df_4h["ema_50"] > df_4h["ema_200"]
    cross_up = bull_4h & (~bull_4h.shift(1).fillna(False))
    cross_dn = (~bull_4h) & (bull_4h.shift(1).fillna(False))

    sig_4h = pd.Series(0, index=df_4h.index)
    sig_4h[cross_up]  =  1
    sig_4h[cross_dn]  = -1

    # Align to 1H: shift 4H index by +4H (bar closes 4H after its open)
    src_ts = (df_4h.index + pd.Timedelta(hours=4)).astype("datetime64[s]")
    tgt_ts = base.astype("datetime64[s]")
    src_df = pd.DataFrame({"ts": src_ts, "v": sig_4h.values}).sort_values("ts")
    tgt_df = pd.DataFrame({"ts": tgt_ts})
    merged = pd.merge_asof(tgt_df, src_df, on="ts", direction="backward")
    sig_1h = pd.Series(merged["v"].fillna(0).values, index=base, dtype=int)

    out = pd.DataFrame(index=base)
    out["composite"] = sig_1h.astype(float)
    out["signal"]    = sig_1h

    # Session filter
    if SESSION_CFG.session_hours is not None:
        h_s, h_e = SESSION_CFG.session_hours
        not_sess = ~((out.index.hour >= h_s) & (out.index.hour < h_e))
        out.loc[not_sess, "signal"] = 0

    return out


def _add_rsi_timing(sig_c1: pd.DataFrame, df_1h: pd.DataFrame) -> pd.DataFrame:
    """C2: keep C1 entries only when 1H RSI confirms the direction."""
    out = sig_c1.copy()
    rsi = df_1h["rsi_14"].reindex(out.index, fill_value=50.0)
    out.loc[(out["signal"] == 1)  & (rsi < 50), "signal"] = 0
    out.loc[(out["signal"] == -1) & (rsi > 50), "signal"] = 0
    return out


def _build_smc_only_signal(smc_feats: pd.DataFrame,
                            df_1h: pd.DataFrame,
                            entry_bars: int = 5) -> pd.DataFrame:
    """
    C3: Signal fires when a CHoCH or BOS event occurred within the last
    `entry_bars` bars AND aligns with the structural trend.
    """
    base  = df_1h.index
    trend = smc_feats["smc_trend"].reindex(base, fill_value=0)
    b_choch_bull = smc_feats["smc_bars_since_choch_bull"].reindex(base, fill_value=9999)
    b_choch_bear = smc_feats["smc_bars_since_choch_bear"].reindex(base, fill_value=9999)
    b_bos_bull   = smc_feats["smc_bars_since_bos_bull"].reindex(base, fill_value=9999)
    b_bos_bear   = smc_feats["smc_bars_since_bos_bear"].reindex(base, fill_value=9999)

    long_event  = (b_choch_bull <= entry_bars) | (b_bos_bull <= entry_bars)
    short_event = (b_choch_bear <= entry_bars) | (b_bos_bear <= entry_bars)

    sig = np.zeros(len(base), dtype=int)
    sig[(trend > 0)  & long_event]  =  1
    sig[(trend < 0) & short_event] = -1

    out = pd.DataFrame({"composite": sig.astype(float), "signal": sig}, index=base)

    if SESSION_CFG.session_hours is not None:
        h_s, h_e = SESSION_CFG.session_hours
        not_sess = ~((out.index.hour >= h_s) & (out.index.hour < h_e))
        out.loc[not_sess, "signal"] = 0

    return out


# ─────────────────────────────────────────────────────────────────────────────
# SMC event analysis
# ─────────────────────────────────────────────────────────────────────────────

def _smc_event_stats(df_1h: pd.DataFrame, smc: pd.DataFrame) -> dict:
    """Forward-return statistics after each SMC event type."""
    c = df_1h["close"]
    horizons = [1, 4, 12, 24, 48]

    def _fwd_ret(event_mask: pd.Series, h: int) -> float:
        fwd = c.shift(-h) / c - 1
        vals = fwd[event_mask & ~fwd.isna()]
        return float(vals.mean()) * 100 if len(vals) > 0 else 0.0

    events = {
        "CHoCH Bull": smc["smc_choch_bull"].astype(bool),
        "CHoCH Bear": smc["smc_choch_bear"].astype(bool),
        "BOS Bull":   smc["smc_bos_bull"].astype(bool),
        "BOS Bear":   smc["smc_bos_bear"].astype(bool),
        "Baseline":   pd.Series(True, index=df_1h.index),
    }

    stats = {}
    for name, mask in events.items():
        n   = int(mask.sum())
        row = {"n": n}
        for h in horizons:
            row[f"fwd_{h}h"] = _fwd_ret(mask, h)
        stats[name] = row

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# HTML generation
# ─────────────────────────────────────────────────────────────────────────────

_PALETTE = [
    "#2196f3", "#f44336", "#4caf50", "#ff9800", "#9c27b0",
    "#00bcd4", "#e91e63", "#795548", "#607d8b", "#ffeb3b",
]


def _impact_delta(row: dict, ref: dict) -> str:
    dr = row["total_return"] - ref["total_return"]
    dd = row["max_dd"]       - ref["max_dd"]
    dc = row["calmar"]       - ref["calmar"]
    ds = row["sharpe"]       - ref["sharpe"]
    sign = lambda v: ("+" if v >= 0 else "")
    return (f"ΔReturn {sign(dr)}{dr:.1f}%p&nbsp;&nbsp;"
            f"ΔDD {sign(dd)}{dd:.1f}%p&nbsp;&nbsp;"
            f"ΔCalmar {sign(dc)}{dc:.3f}&nbsp;&nbsp;"
            f"ΔSharpe {sign(ds)}{ds:.3f}")


def _table(rows: list[dict], ref: dict | None = None) -> str:
    cols = ["name", "total_return", "max_dd", "sharpe", "calmar",
            "win_rate", "profit_factor", "n_trades", "expectancy", "final_equity"]
    headers = ["Config", "Return %", "Max DD %", "Sharpe", "Calmar",
               "Win %", "Profit Factor", "# Trades", "Expectancy $", "Final Equity $"]

    ths = "".join(f"<th>{h}</th>" for h in headers)
    trs = ""
    for i, r in enumerate(rows):
        extra = ""
        if ref is not None and r["name"] != ref["name"]:
            extra = f'<br><small style="color:#888">{_impact_delta(r, ref)}</small>'
        tds = ""
        for j, c in enumerate(cols):
            v = r.get(c, "")
            if j == 0:
                tds += f"<td><b>{v}</b>{extra}</td>"
            else:
                tds += f"<td>{v}</td>"
        bg = "#1e3a1e" if (ref and r["name"] == ref["name"]) else ""
        style = f' style="background:{bg}"' if bg else ""
        trs += f"<tr{style}>{tds}</tr>\n"

    return f"""
<table class="metrics">
<thead><tr>{ths}</tr></thead>
<tbody>{trs}</tbody>
</table>"""


def _equity_chart(rows: list[dict], chart_id: str) -> str:
    datasets = []
    for i, r in enumerate(rows):
        color = _PALETTE[i % len(_PALETTE)]
        datasets.append({
            "label": r["name"],
            "data":  [{"x": t, "y": round(e, 2)}
                      for t, e in zip(r["index"], r["equity"])],
            "borderColor": color,
            "backgroundColor": color + "22",
            "borderWidth": 2,
            "pointRadius": 0,
            "fill": False,
        })
    data_json = json.dumps(datasets)
    return f"""
<div class="chart-wrap"><canvas id="{chart_id}" height="300"></canvas></div>
<script>
(function() {{
  var ctx = document.getElementById('{chart_id}').getContext('2d');
  new Chart(ctx, {{
    type: 'line',
    data: {{ datasets: {data_json} }},
    options: {{
      responsive: true,
      animation: false,
      interaction: {{ mode: 'index', intersect: false }},
      plugins: {{
        legend: {{ position: 'top', labels: {{ color: '#ccc' }} }},
        tooltip: {{
          callbacks: {{
            label: function(ctx) {{
              return ctx.dataset.label + ': $' + ctx.parsed.y.toLocaleString();
            }}
          }}
        }}
      }},
      scales: {{
        x: {{
          type: 'time', time: {{ unit: 'month' }},
          ticks: {{ color: '#aaa' }}, grid: {{ color: '#333' }}
        }},
        y: {{
          ticks: {{ color: '#aaa', callback: v => '$' + v.toLocaleString() }},
          grid: {{ color: '#333' }}
        }}
      }}
    }}
  }});
}})();
</script>"""


def _smc_event_table(stats: dict) -> str:
    horizons = [1, 4, 12, 24, 48]
    hs = "".join(f"<th>+{h}H avg%</th>" for h in horizons)
    thead = f"<tr><th>Event</th><th>Count</th>{hs}</tr>"
    rows = ""
    for name, s in stats.items():
        n = s["n"]
        cols = "".join(f'<td>{s[f"fwd_{h}h"]:+.3f}%</td>' for h in horizons)
        rows += f"<tr><td><b>{name}</b></td><td>{n:,}</td>{cols}</tr>\n"
    return f'<table class="metrics"><thead>{thead}</thead><tbody>{rows}</tbody></table>'


def _html(path_a: list, path_b: list, path_c: list,
          ref: dict, smc_stats: dict, smc_counts: dict) -> str:
    all_rows  = [ref] + path_a[1:] + path_b[1:] + path_c[1:]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Paths A / B / C Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: #121212; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; padding: 24px; }}
h1 {{ color: #fff; margin-bottom: 8px; }}
h2 {{ color: #90caf9; margin: 28px 0 12px; font-size: 1.2rem; border-bottom: 1px solid #333; padding-bottom: 6px; }}
h3 {{ color: #aaa; margin: 18px 0 8px; font-size: 1rem; }}
p  {{ color: #bbb; margin: 6px 0; line-height: 1.6; }}
table.metrics {{ width:100%; border-collapse:collapse; font-size:0.82rem; margin-bottom:16px; }}
table.metrics th {{ background:#1e2a3a; color:#90caf9; padding:7px 10px; text-align:right; border:1px solid #2a3a4a; }}
table.metrics th:first-child {{ text-align:left; }}
table.metrics td {{ padding:6px 10px; border:1px solid #2a2a2a; text-align:right; }}
table.metrics td:first-child {{ text-align:left; color:#e0e0e0; }}
table.metrics tbody tr:hover {{ background:#1a2030; }}
.chart-wrap {{ background:#1a1a2e; border-radius:8px; padding:16px; margin:16px 0; }}
.grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
.card {{ background:#1a1a2e; border-radius:8px; padding:16px; margin:8px 0; }}
.badge {{ display:inline-block; padding:2px 8px; border-radius:4px; font-size:0.75rem;
          background:#1e3a1e; color:#66bb6a; margin-left:8px; }}
.info  {{ background:#1e2a3a; border-left:3px solid #90caf9; padding:10px 14px; margin:12px 0; border-radius:0 6px 6px 0; }}
</style>
</head>
<body>
<h1>Strategy Exploration: Paths A / B / C</h1>
<p>Full-sample backtests comparing three strategic directions for improving the BTCUSDT strategy.</p>
<p>Reference: <b>Baseline 1H</b> — session 08–21 UTC, threshold ±3, ATR-based TP/SL.</p>

<!-- ───── SMC ANALYSIS ───── -->
<h2>Smart Money Concepts (SMC) — 1H Event Analysis</h2>
<div class="info">
  <b>Swing detection</b>: external n=50 bars, internal n=5 bars.
  BOS/CHoCH detected when close crosses the confirmed swing level.
  Forward returns shown as average % gain/loss over next N hours.
</div>
<p><b>Event counts on 1H data:</b>
  CHoCH Bull: {smc_counts.get('choch_bull', 0):,} &nbsp;|&nbsp;
  CHoCH Bear: {smc_counts.get('choch_bear', 0):,} &nbsp;|&nbsp;
  BOS Bull:   {smc_counts.get('bos_bull', 0):,} &nbsp;|&nbsp;
  BOS Bear:   {smc_counts.get('bos_bear', 0):,}
</p>
{_smc_event_table(smc_stats)}

<!-- ───── PATH A ───── -->
<h2>Path A — Long / Short Asymmetry</h2>
<div class="info">
  Diagnostic finding: long trades contribute PF=1.01 (near-random), short trades PF=1.15 with 6× more P&amp;L.
  This path tests whether separating or filtering by direction improves risk-adjusted returns.
</div>
{_table(path_a, ref=ref)}
{_equity_chart(path_a, "chart_a")}

<!-- ───── PATH B ───── -->
<h2>Path B — SMC as New Signal Component</h2>
<div class="info">
  Adding Smart Money Concepts (CHoCH, OBs, zones) as an additional composite signal component.
  B1 adds the SMC score with weight=2 to the existing composite.
  B2 filters out over-extended entries using the funding rate.
  B3 combines both.
</div>
{_table(path_b, ref=ref)}
{_equity_chart(path_b, "chart_b")}

<!-- ───── PATH C ───── -->
<h2>Path C — Simplified Trend Following</h2>
<div class="info">
  Replace the complex 9-component composite with simpler signals.
  C1: 4H EMA-50/200 crossover (structural trend change entry).
  C2: C1 + 1H RSI momentum alignment.
  C3: SMC CHoCH / BOS structure-only signal (pure market structure).
</div>
{_table(path_c, ref=ref)}
{_equity_chart(path_c, "chart_c")}

<!-- ───── ALL CONFIGS ───── -->
<h2>Full Comparison — All Configs</h2>
{_table(all_rows, ref=ref)}
{_equity_chart(all_rows, "chart_all")}

</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n══ Paths A / B / C Exploration ══════════════════════════════════")

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("\n[1/5] Loading data …")
    raw = fetch_extended_data(start_year=2022, start_month=1,
                              fetch_1m=False, fetch_flow=True)
    tf_ind = {}
    for tf in ["1W", "1D", "4H", "1H", "15M"]:
        df = raw.get(tf, pd.DataFrame())
        tf_ind[tf] = add_indicators(df) if not df.empty and len(df) > 20 else df

    df_1h = tf_ind["1H"]
    df_4h = tf_ind["4H"]
    oi_df   = generate_oi(tf_ind["1D"]["close"])
    funding = generate_funding(tf_ind["1D"]["close"])
    print(f"  1H bars: {len(df_1h):,}   |   4H bars: {len(df_4h):,}")

    # ── 2. Baseline signal ────────────────────────────────────────────────────
    print("\n[2/5] Building baseline signal …")
    raw_sig = build_signal_matrix(
        tf_data=tf_ind, oi_df=oi_df, funding=funding,
        premium_1h=None, df_15m=tf_ind.get("15M"), df_1m=None,
    )
    sig_ref = apply_filters(raw_sig, SESSION_CFG)
    n_ref = int((sig_ref["signal"] != 0).sum())
    print(f"  Baseline signals: {n_ref:,}")

    # ── 3. SMC features ───────────────────────────────────────────────────────
    print("\n[3/5] Computing SMC features (1H, swing_len=50) …")
    smc_1h = compute_smc_features(df_1h, swing_len=50, internal_len=5, prefix="smc")
    smc_score_1h = smc_composite_score(smc_1h, prefix="smc")

    smc_counts = {
        "choch_bull": int(smc_1h["smc_choch_bull"].sum()),
        "choch_bear": int(smc_1h["smc_choch_bear"].sum()),
        "bos_bull":   int(smc_1h["smc_bos_bull"].sum()),
        "bos_bear":   int(smc_1h["smc_bos_bear"].sum()),
    }
    print(f"  CHoCH Bull: {smc_counts['choch_bull']:,}   CHoCH Bear: {smc_counts['choch_bear']:,}"
          f"   BOS Bull: {smc_counts['bos_bull']:,}   BOS Bear: {smc_counts['bos_bear']:,}")

    smc_stats = _smc_event_stats(df_1h, smc_1h)
    print("  Forward-return analysis complete.")

    # ── 4. Funding rate aligned to 1H ─────────────────────────────────────────
    # funding is daily → align to 1H via ffill
    funding_1h = funding.reindex(df_1h.index, method="ffill").fillna(0.0)

    # ── 5. Run all configs ────────────────────────────────────────────────────
    print("\n[4/5] Running backtests …")

    # ── REF ─────────────────────────────────────────────────────────────────
    print("  REF: Baseline 1H …")
    bt_ref  = run_backtest(df_1h, sig_ref)
    row_ref = kpi_row("REF Baseline 1H", bt_ref)
    print(f"    → Return={row_ref['total_return']:+.1f}%  DD={row_ref['max_dd']:.1f}%  "
          f"Trades={row_ref['n_trades']}  Calmar={row_ref['calmar']:.3f}")

    # ── PATH A ───────────────────────────────────────────────────────────────
    print("\n  Path A …")

    sig_a1 = _short_only(sig_ref)
    bt_a1  = run_backtest(df_1h, sig_a1)
    row_a1 = kpi_row("A1. Short-only", bt_a1,
                     extra={"n_sig": int((sig_a1["signal"] != 0).sum())})
    print(f"    A1 Short-only → Return={row_a1['total_return']:+.1f}%  "
          f"Calmar={row_a1['calmar']:.3f}  Trades={row_a1['n_trades']}")

    sig_a2 = _long_only(sig_ref)
    bt_a2  = run_backtest(df_1h, sig_a2)
    row_a2 = kpi_row("A2. Long-only", bt_a2,
                     extra={"n_sig": int((sig_a2["signal"] != 0).sum())})
    print(f"    A2 Long-only  → Return={row_a2['total_return']:+.1f}%  "
          f"Calmar={row_a2['calmar']:.3f}  Trades={row_a2['n_trades']}")

    sig_a3 = _asymmetric_threshold(raw_sig, long_thresh=7.0, short_thresh=-3.0)
    bt_a3  = run_backtest(df_1h, sig_a3)
    row_a3 = kpi_row("A3. Asymmetric threshold (L≥7, S≤-3)", bt_a3,
                     extra={"n_sig": int((sig_a3["signal"] != 0).sum())})
    print(f"    A3 Asymmetric → Return={row_a3['total_return']:+.1f}%  "
          f"Calmar={row_a3['calmar']:.3f}  Trades={row_a3['n_trades']}")

    smc_trend = smc_1h["smc_trend"]
    sig_a4 = _smc_aligned(sig_ref, smc_trend)
    bt_a4  = run_backtest(df_1h, sig_a4)
    row_a4 = kpi_row("A4. SMC-aligned (structure agrees)", bt_a4,
                     extra={"n_sig": int((sig_a4["signal"] != 0).sum()),
                            "filter_rate": round((1 - (sig_a4["signal"] != 0).sum() / n_ref) * 100, 1)})
    print(f"    A4 SMC-align  → Return={row_a4['total_return']:+.1f}%  "
          f"Calmar={row_a4['calmar']:.3f}  Trades={row_a4['n_trades']}  "
          f"Filtered={row_a4['filter_rate']:.0f}%")

    path_a = [row_ref, row_a1, row_a2, row_a3, row_a4]

    # ── PATH B ───────────────────────────────────────────────────────────────
    print("\n  Path B …")

    sig_b1 = _add_smc_to_composite(raw_sig, smc_score_1h, weight=2.0)
    bt_b1  = run_backtest(df_1h, sig_b1)
    row_b1 = kpi_row("B1. + SMC score (w=2)", bt_b1,
                     extra={"n_sig": int((sig_b1["signal"] != 0).sum())})
    print(f"    B1 + SMC      → Return={row_b1['total_return']:+.1f}%  "
          f"Calmar={row_b1['calmar']:.3f}  Trades={row_b1['n_trades']}")

    sig_b2 = _funding_extreme_block(sig_ref, funding_1h, threshold=0.001)
    bt_b2  = run_backtest(df_1h, sig_b2)
    row_b2 = kpi_row("B2. + Funding extreme block", bt_b2,
                     extra={"n_sig": int((sig_b2["signal"] != 0).sum()),
                            "filter_rate": round((1 - (sig_b2["signal"] != 0).sum() / n_ref) * 100, 1)})
    print(f"    B2 + Funding  → Return={row_b2['total_return']:+.1f}%  "
          f"Calmar={row_b2['calmar']:.3f}  Trades={row_b2['n_trades']}  "
          f"Filtered={row_b2['filter_rate']:.0f}%")

    sig_b3_tmp = _add_smc_to_composite(raw_sig, smc_score_1h, weight=2.0)
    sig_b3     = _funding_extreme_block(sig_b3_tmp, funding_1h, threshold=0.001)
    bt_b3  = run_backtest(df_1h, sig_b3)
    row_b3 = kpi_row("B3. SMC + Funding block", bt_b3,
                     extra={"n_sig": int((sig_b3["signal"] != 0).sum())})
    print(f"    B3 Combined   → Return={row_b3['total_return']:+.1f}%  "
          f"Calmar={row_b3['calmar']:.3f}  Trades={row_b3['n_trades']}")

    path_b = [row_ref, row_b1, row_b2, row_b3]

    # ── PATH C ───────────────────────────────────────────────────────────────
    print("\n  Path C …")

    sig_c1 = _build_ema_crossover_signal(df_4h, df_1h)
    bt_c1  = run_backtest(df_1h, sig_c1)
    row_c1 = kpi_row("C1. 4H EMA-50/200 crossover", bt_c1,
                     extra={"n_sig": int((sig_c1["signal"] != 0).sum())})
    print(f"    C1 EMA cross  → Return={row_c1['total_return']:+.1f}%  "
          f"Calmar={row_c1['calmar']:.3f}  Trades={row_c1['n_trades']}")

    sig_c2 = _add_rsi_timing(sig_c1, df_1h)
    bt_c2  = run_backtest(df_1h, sig_c2)
    row_c2 = kpi_row("C2. EMA crossover + 1H RSI", bt_c2,
                     extra={"n_sig": int((sig_c2["signal"] != 0).sum())})
    print(f"    C2 + RSI gate → Return={row_c2['total_return']:+.1f}%  "
          f"Calmar={row_c2['calmar']:.3f}  Trades={row_c2['n_trades']}")

    sig_c3 = _build_smc_only_signal(smc_1h, df_1h, entry_bars=5)
    bt_c3  = run_backtest(df_1h, sig_c3)
    row_c3 = kpi_row("C3. SMC CHoCH/BOS structure only", bt_c3,
                     extra={"n_sig": int((sig_c3["signal"] != 0).sum())})
    print(f"    C3 SMC-only   → Return={row_c3['total_return']:+.1f}%  "
          f"Calmar={row_c3['calmar']:.3f}  Trades={row_c3['n_trades']}")

    path_c = [row_ref, row_c1, row_c2, row_c3]

    # ── 6. Report ─────────────────────────────────────────────────────────────
    print("\n[5/5] Writing report …")
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    html_path = out_dir / "report_paths.html"
    html_path.write_text(
        _html(path_a, path_b, path_c, row_ref, smc_stats, smc_counts),
        encoding="utf-8",
    )
    print(f"  → {html_path.resolve()}")
    print("\n══ Done ══")


if __name__ == "__main__":
    main()
