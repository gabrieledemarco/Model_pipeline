"""
15M signal resolution experiment — v3 (impact analysis).

Two improvements tested over v2 baseline:

  1. 1H directional filter
     Only take 15M signals aligned with the current 1H composite direction.
     Applied post-hoc to both the raw signal and any gated signal — no retraining.
     Impact: E vs C (baseline), F vs D (ML trade-label gate).

  2. 4-bar bar-level ML label
     Train LightGBM on ALL 15M bars predicting close[T+4] > close[T].
     4 bars × 15M = 1H forward horizon — matches the 1H ATR stop/target sizing.
     Uses walk_forward_bar_level_gate (30× more training samples than trade gate).
     Impact: G vs D (no dir filter), H vs F (with dir filter).

Strategies compared
───────────────────
  A. Baseline 1H                    reference
  B. ML Gate 1H P≥0.50              best known 1H config
  C. Baseline 15M v2                crossing + 1H ATR
  D. ML Gate 15M (trade-label)      v2 trade-level binary gate
  E. Baseline 15M + 1H dir filter   dir filter impact (no ML)
  F. ML Gate 15M + dir filter       D signals + dir filter post-hoc
  G. ML Gate 15M (4-bar bar-label)  bar-level gate, 1H horizon, no dir filter
  H. G + 1H dir filter              G signals + dir filter post-hoc

Impact grid
───────────
                   no dir filter   + dir filter
  trade-label      D               F
  4-bar bar-label  G               H
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
from src.strategy.signals      import build_signal_matrix, build_signal_matrix_15m
from src.strategy.optimizer    import apply_filters, ScenarioConfig
from src.strategy.engine       import run_backtest, INIT_CAP
from src.strategy.ml_features  import build_feature_matrix
from src.strategy.ml_gate      import (
    walk_forward_binary_gate,
    walk_forward_bar_level_gate,
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

def kpi_row(name: str, bt: dict, extra: dict = None) -> dict:
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


def _build_df15m_with_1h_atr(df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df_15m where atr_14 is replaced by the last-closed 1H
    bar's ATR (aligned via merge_asof + shift +1H).
    """
    src_ts  = (df_1h.index + pd.Timedelta(hours=1)).astype("datetime64[s]")
    tgt_ts  = df_15m.index.astype("datetime64[s]")
    src_df  = pd.DataFrame({"ts": src_ts, "v": df_1h["atr_14"].values}).sort_values("ts")
    tgt_df  = pd.DataFrame({"ts": tgt_ts})
    merged  = pd.merge_asof(tgt_df, src_df, on="ts", direction="backward")
    atr_1h  = pd.Series(merged["v"].ffill().fillna(0.0).values, index=df_15m.index)
    out = df_15m.copy()
    out["atr_14"] = atr_1h
    return out


def _apply_1h_dir_filter(sig_15m: pd.DataFrame, sig_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only 15M signals aligned with the 1H composite direction.
    Long signal dropped if 1H composite < 0.
    Short signal dropped if 1H composite > 0.
    1H composite is forward-filled to every 15M bar.
    """
    comp_at_15m = (sig_1h["composite"]
                   .reindex(sig_15m.index, method="ffill")
                   .fillna(0.0))
    out = sig_15m.copy()
    out.loc[(out["signal"] == 1)  & (comp_at_15m < 0), "signal"] = 0
    out.loc[(out["signal"] == -1) & (comp_at_15m > 0), "signal"] = 0
    return out


def _apply_dir_filter_to_gate(gate: MLGateResult,
                               sig_1h: pd.DataFrame) -> MLGateResult:
    """
    Post-hoc: apply 1H dir filter to an existing gated_signal without retraining.
    Returns a new MLGateResult with the filtered signal and updated window_stats.
    """
    comp = (sig_1h["composite"]
            .reindex(gate.gated_signal.index, method="ffill")
            .fillna(0.0))
    filtered = gate.gated_signal.copy()
    filtered[(filtered == 1)  & (comp < 0)] = 0
    filtered[(filtered == -1) & (comp > 0)] = 0
    return MLGateResult(
        oos_pred      = gate.oos_pred,
        gated_signal  = filtered,
        window_stats  = gate.window_stats,
        feature_names = gate.feature_names,
        importances   = gate.importances,
    )


# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n══ 15M Signal Resolution — v3 Impact Analysis ═══════════════════")

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("\n[1/8] Loading data …")
    raw = fetch_extended_data(start_year=2022, start_month=1,
                              fetch_1m=False, fetch_flow=True)
    tf_ind = {}
    for tf in ["1W", "1D", "4H", "1H", "15M"]:
        df = raw.get(tf, pd.DataFrame())
        tf_ind[tf] = add_indicators(df) if not df.empty and len(df) > 20 else df

    df_1h  = tf_ind["1H"]
    df_15m = tf_ind["15M"]
    oi_df   = generate_oi(tf_ind["1D"]["close"])
    funding = generate_funding(tf_ind["1D"]["close"])

    # df_15m_bt: 15M OHLCV with 1H ATR for stop/target sizing
    df_15m_bt = _build_df15m_with_1h_atr(df_15m, df_1h)
    print(f"  1H bars : {len(df_1h):,}   |   15M bars: {len(df_15m):,}")

    # ── 2. Signals ────────────────────────────────────────────────────────────
    print("\n[2/8] Building signals …")
    raw_sig_1h  = build_signal_matrix(
        tf_data=tf_ind, oi_df=oi_df, funding=funding,
        premium_1h=None, df_15m=df_15m, df_1m=None,
    )
    raw_sig_15m = build_signal_matrix_15m(
        tf_data=tf_ind, oi_df=oi_df, funding=funding,
        premium_1h=None, crossing_only=True,
    )
    sig_1h  = apply_filters(raw_sig_1h,  SESSION_CFG)
    sig_15m = apply_filters(raw_sig_15m, SESSION_CFG)

    # 1H dir-filtered 15M signal (post-hoc, no retrain)
    sig_15m_filt = _apply_1h_dir_filter(sig_15m, sig_1h)

    n_1h       = int((sig_1h["signal"]       != 0).sum())
    n_15m      = int((sig_15m["signal"]      != 0).sum())
    n_15m_filt = int((sig_15m_filt["signal"] != 0).sum())
    print(f"  1H  signal bars:              {n_1h:,}")
    print(f"  15M signal bars (crossings):  {n_15m:,}")
    print(f"  15M signal bars (+ dir filt): {n_15m_filt:,}  "
          f"({100*(1 - n_15m_filt/n_15m):.0f}% removed by dir filter)")

    # ── 3. A. Baseline 1H ─────────────────────────────────────────────────────
    print("\n[3/8] A. Baseline 1H …")
    bt_a  = run_backtest(df_1h, sig_1h)
    row_a = kpi_row("A. Baseline 1H", bt_a, extra={"n_sig": n_1h, "base_tf": "1H"})
    print(f"  → Return={row_a['total_return']:+.1f}%  DD={row_a['max_dd']:.1f}%  "
          f"Trades={row_a['n_trades']}  Calmar={row_a['calmar']:.3f}")

    # ── 4. B. ML Gate 1H P≥0.50 ──────────────────────────────────────────────
    print("\n[4/8] B. ML Gate 1H P≥0.50 …")
    feat_1h  = build_feature_matrix(tf_ind, sig_1h, base_tf="1H")
    print(f"  Feature matrix: {feat_1h.shape[0]:,} × {feat_1h.shape[1]}")
    gate_1h  = walk_forward_binary_gate(
        df_1h, sig_1h, feat_1h, gate_threshold=0.50,
        use_feat_sel=True, verbose=True,
    )
    bt_b = run_gated_backtest(df_1h, sig_1h, gate_1h)
    n_b  = int((gate_1h.gated_signal != 0).sum())
    row_b = kpi_row("B. ML Gate 1H P≥0.50", bt_b,
                    extra={"n_sig": n_b,
                           "filter_rate": round((1 - n_b / n_1h) * 100, 1),
                           "base_tf": "1H",
                           "window_stats": gate_1h.window_stats,
                           "importances": gate_1h.importances.head(15).to_dict("records")
                                          if gate_1h.importances is not None else []})
    print(f"  → Return={row_b['total_return']:+.1f}%  DD={row_b['max_dd']:.1f}%  "
          f"Trades={row_b['n_trades']}  Calmar={row_b['calmar']:.3f}  "
          f"Filtered={row_b['filter_rate']:.0f}%")

    # ── 5. C. Baseline 15M v2 (crossing + 1H ATR) ────────────────────────────
    print("\n[5/8] C. Baseline 15M v2 (crossing + 1H ATR) …")
    bt_c  = run_backtest(df_15m_bt, sig_15m)
    row_c = kpi_row("C. Baseline 15M", bt_c, extra={"n_sig": n_15m, "base_tf": "15M"})
    print(f"  → Return={row_c['total_return']:+.1f}%  DD={row_c['max_dd']:.1f}%  "
          f"Trades={row_c['n_trades']}  Calmar={row_c['calmar']:.3f}")

    # ── 5b. E. Baseline 15M + 1H dir filter (post-hoc, no ML) ────────────────
    print("       E. + 1H directional filter …")
    bt_e  = run_backtest(df_15m_bt, sig_15m_filt)
    n_e   = int((sig_15m_filt["signal"] != 0).sum())
    row_e = kpi_row("E. Baseline 15M + dir filter", bt_e,
                    extra={"n_sig": n_e,
                           "filter_rate": round((1 - n_e / n_15m) * 100, 1),
                           "base_tf": "15M"})
    print(f"  → Return={row_e['total_return']:+.1f}%  DD={row_e['max_dd']:.1f}%  "
          f"Trades={row_e['n_trades']}  Calmar={row_e['calmar']:.3f}  "
          f"Filtered={row_e['filter_rate']:.0f}%")

    # ── 6. D. ML Gate 15M P≥0.50 (trade-label, v2) ───────────────────────────
    print("\n[6/8] D. ML Gate 15M P≥0.50 (trade-label) …")
    feat_15m = build_feature_matrix(tf_ind, sig_15m, base_tf="15M")
    print(f"  Feature matrix: {feat_15m.shape[0]:,} × {feat_15m.shape[1]}")
    gate_15m_trade = walk_forward_binary_gate(
        df_15m_bt, sig_15m, feat_15m, gate_threshold=0.50,
        use_feat_sel=True, verbose=True,
    )
    bt_d = run_gated_backtest(df_15m_bt, sig_15m, gate_15m_trade)
    n_d  = int((gate_15m_trade.gated_signal != 0).sum())
    row_d = kpi_row("D. ML Gate 15M (trade-label)", bt_d,
                    extra={"n_sig": n_d,
                           "filter_rate": round((1 - n_d / n_15m) * 100, 1),
                           "base_tf": "15M",
                           "window_stats": gate_15m_trade.window_stats,
                           "importances": gate_15m_trade.importances.head(15).to_dict("records")
                                          if gate_15m_trade.importances is not None else []})
    print(f"  → Return={row_d['total_return']:+.1f}%  DD={row_d['max_dd']:.1f}%  "
          f"Trades={row_d['n_trades']}  Calmar={row_d['calmar']:.3f}  "
          f"Filtered={row_d['filter_rate']:.0f}%")

    # ── 6b. F. ML Gate 15M + dir filter (post-hoc from D) ────────────────────
    print("       F. + 1H directional filter (post-hoc from D) …")
    gate_15m_trade_filt = _apply_dir_filter_to_gate(gate_15m_trade, sig_1h)
    bt_f  = run_gated_backtest(df_15m_bt, sig_15m, gate_15m_trade_filt)
    n_f   = int((gate_15m_trade_filt.gated_signal != 0).sum())
    row_f = kpi_row("F. ML Gate 15M trade-label + dir filter", bt_f,
                    extra={"n_sig": n_f,
                           "filter_rate": round((1 - n_f / n_15m) * 100, 1),
                           "base_tf": "15M"})
    print(f"  → Return={row_f['total_return']:+.1f}%  DD={row_f['max_dd']:.1f}%  "
          f"Trades={row_f['n_trades']}  Calmar={row_f['calmar']:.3f}  "
          f"Filtered={row_f['filter_rate']:.0f}%")

    # ── 7. G. ML Gate 15M 4-bar bar-level label ──────────────────────────────
    print("\n[7/8] G. ML Gate 15M 4-bar bar-level label …")
    print(f"  Feature matrix: {feat_15m.shape[0]:,} × {feat_15m.shape[1]} (reusing feat_15m)")
    gate_15m_bar = walk_forward_bar_level_gate(
        df_15m_bt, sig_15m, feat_15m,
        gate_threshold=0.55,   # direction-aware: LONG if P(up)≥0.55, SHORT if P(up)≤0.45
        forward_bars=4,        # 4 × 15M = 1H forward horizon
        use_feat_sel=True, verbose=True,
    )
    bt_g = run_gated_backtest(df_15m_bt, sig_15m, gate_15m_bar)
    n_g  = int((gate_15m_bar.gated_signal != 0).sum())
    row_g = kpi_row("G. ML Gate 15M (4-bar bar-label)", bt_g,
                    extra={"n_sig": n_g,
                           "filter_rate": round((1 - n_g / n_15m) * 100, 1),
                           "base_tf": "15M",
                           "window_stats": gate_15m_bar.window_stats,
                           "importances": gate_15m_bar.importances.head(15).to_dict("records")
                                          if gate_15m_bar.importances is not None else []})
    print(f"  → Return={row_g['total_return']:+.1f}%  DD={row_g['max_dd']:.1f}%  "
          f"Trades={row_g['n_trades']}  Calmar={row_g['calmar']:.3f}  "
          f"Filtered={row_g['filter_rate']:.0f}%")

    # ── 7b. H. G + 1H dir filter (post-hoc) ──────────────────────────────────
    print("       H. + 1H directional filter (post-hoc from G) …")
    gate_15m_bar_filt = _apply_dir_filter_to_gate(gate_15m_bar, sig_1h)
    bt_h  = run_gated_backtest(df_15m_bt, sig_15m, gate_15m_bar_filt)
    n_h   = int((gate_15m_bar_filt.gated_signal != 0).sum())
    row_h = kpi_row("H. ML Gate 15M 4-bar + dir filter", bt_h,
                    extra={"n_sig": n_h,
                           "filter_rate": round((1 - n_h / n_15m) * 100, 1),
                           "base_tf": "15M"})
    print(f"  → Return={row_h['total_return']:+.1f}%  DD={row_h['max_dd']:.1f}%  "
          f"Trades={row_h['n_trades']}  Calmar={row_h['calmar']:.3f}  "
          f"Filtered={row_h['filter_rate']:.0f}%")

    results = [row_a, row_b, row_c, row_d, row_e, row_f, row_g, row_h]

    # ── 8. HTML ───────────────────────────────────────────────────────────────
    print("\n[8/8] Generating report …")
    html = _build_html(results)
    out  = Path("reports/report_15m.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"  Saved → {out}")
    print("\n══ Done ══════════════════════════════════════════════════════════")

    # ── Console impact summary ────────────────────────────────────────────────
    print("\n── Impact Analysis ────────────────────────────────────────────")
    hdr = f"  {'Config':<42} {'Return%':>8} {'DD%':>7} {'Calmar':>8} {'Trades':>7}"
    print(hdr)
    print("  " + "─" * 76)
    for r in results:
        pf = r.get("filter_rate", "—")
        flt = f"  (filt {pf}%)" if isinstance(pf, float) else ""
        print(f"  {r['name']:<42} {r['total_return']:>+8.1f} {r['max_dd']:>7.1f} "
              f"{r['calmar']:>8.3f} {r['n_trades']:>7}{flt}")
    print()
    # Dir filter delta on baseline
    delta_e_c = row_e["calmar"] - row_c["calmar"]
    delta_f_d = row_f["calmar"] - row_d["calmar"]
    delta_g_d = row_g["calmar"] - row_d["calmar"]
    delta_h_f = row_h["calmar"] - row_f["calmar"]
    print(f"  Dir filter impact (baseline) E−C : ΔCalmar={delta_e_c:+.3f}")
    print(f"  Dir filter impact (ML trade) F−D : ΔCalmar={delta_f_d:+.3f}")
    print(f"  Bar-label impact (no filter) G−D : ΔCalmar={delta_g_d:+.3f}")
    print(f"  Bar-label+filter combined    H−F : ΔCalmar={delta_h_f:+.3f}")
    print("─" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# HTML builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_html(results: list) -> str:

    COLORS = {
        "A. Baseline 1H":                           "#8b949e",
        "B. ML Gate 1H P≥0.50":                     "#81c784",
        "C. Baseline 15M":                           "#4fc3f7",
        "D. ML Gate 15M (trade-label)":              "#ff8a65",
        "E. Baseline 15M + dir filter":              "#29b6f6",
        "F. ML Gate 15M trade-label + dir filter":   "#ffa726",
        "G. ML Gate 15M (4-bar bar-label)":          "#ce93d8",
        "H. ML Gate 15M 4-bar + dir filter":         "#ab47bc",
    }

    def _cc(v, good_high=True):
        cls = ("pos" if v > 0 else "neg") if good_high else ("neg" if v < 0 else "pos")
        return f'<td class="{cls}">{v}</td>'

    def _summary_rows():
        rows = ""
        groups = [
            ("1H Reference",           ["A. Baseline 1H", "B. ML Gate 1H P≥0.50"]),
            ("15M Baseline",           ["C. Baseline 15M", "E. Baseline 15M + dir filter"]),
            ("15M ML (trade-label)",   ["D. ML Gate 15M (trade-label)",
                                        "F. ML Gate 15M trade-label + dir filter"]),
            ("15M ML (4-bar bar-label)",["G. ML Gate 15M (4-bar bar-label)",
                                         "H. ML Gate 15M 4-bar + dir filter"]),
        ]
        res_by_name = {r["name"]: r for r in results}
        for grp_name, names in groups:
            rows += f'<tr><td colspan="9" style="background:#21262d;color:#8b949e;font-size:0.76rem;padding:5px 10px">{grp_name}</td></tr>'
            for name in names:
                r = res_by_name.get(name)
                if not r:
                    continue
                color = COLORS.get(name, "#aaa")
                pf = r.get("filter_rate", "—")
                flt = f"{pf}%" if isinstance(pf, float) else "—"
                rows += f"""
            <tr>
              <td><span class="dot" style="background:{color}"></span>{r['name']}</td>
              <td>{r['n_trades']}</td>
              {_cc(r['total_return'])}
              {_cc(r['max_dd'], False)}
              {_cc(r['calmar'])}
              {_cc(r['sharpe'])}
              {_cc(r['win_rate'])}
              {_cc(r['profit_factor'])}
              <td>{flt}</td>
            </tr>"""
        return rows

    def _impact_rows():
        res = {r["name"]: r for r in results}
        pairs = [
            ("Dir filter on baseline",  "E. Baseline 15M + dir filter",              "C. Baseline 15M"),
            ("Dir filter on ML trade",  "F. ML Gate 15M trade-label + dir filter",   "D. ML Gate 15M (trade-label)"),
            ("4-bar label (no filter)", "G. ML Gate 15M (4-bar bar-label)",           "D. ML Gate 15M (trade-label)"),
            ("4-bar label + filter",    "H. ML Gate 15M 4-bar + dir filter",          "F. ML Gate 15M trade-label + dir filter"),
        ]
        rows = ""
        for label, new_name, base_name in pairs:
            n = res.get(new_name); b = res.get(base_name)
            if not n or not b:
                continue
            dr = round(n["total_return"] - b["total_return"], 1)
            dd = round(n["max_dd"]       - b["max_dd"],       1)
            dc = round(n["calmar"]       - b["calmar"],       3)
            ds = round(n["sharpe"]       - b["sharpe"],       3)
            dt = n["n_trades"]           - b["n_trades"]
            c_ret = "pos" if dr > 0 else "neg"
            c_dd  = "pos" if dd > 0 else "neg"
            c_cal = "pos" if dc > 0 else "neg"
            rows += f"""
            <tr>
              <td>{label}</td>
              <td class="{c_ret}">{dr:+.1f}%</td>
              <td class="{c_dd}">{dd:+.1f}%</td>
              <td class="{c_cal}">{dc:+.3f}</td>
              <td class="{c_cal}">{ds:+.3f}</td>
              <td>{dt:+d}</td>
            </tr>"""
        return rows

    data = {
        "results": [{
            "name":   r["name"],
            "equity": r["equity"],
            "dd":     r["drawdown"],
            "index":  r["index"],
            "color":  COLORS.get(r["name"], "#aaa"),
        } for r in results],
        "window_stats": {
            r["name"]: r.get("window_stats", [])
            for r in results if r.get("window_stats")
        },
        "importances": {
            r["name"]: r.get("importances", [])
            for r in results if r.get("importances")
        },
    }
    data_json = json.dumps(data, default=str)

    ml_names_json = json.dumps([
        "B. ML Gate 1H P≥0.50",
        "D. ML Gate 15M (trade-label)",
        "G. ML Gate 15M (4-bar bar-label)",
    ])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>15M Impact Analysis — BTCUSDT</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0d1117; color: #c9d1d9; font-family: 'Segoe UI', sans-serif; padding: 24px; }}
  h1  {{ color: #58a6ff; margin-bottom: 6px; font-size: 1.6rem; }}
  h2  {{ color: #8b949e; font-size: 1.05rem; margin: 28px 0 10px; border-bottom: 1px solid #21262d; padding-bottom: 6px; }}
  .subtitle {{ color: #8b949e; font-size: 0.88rem; margin-bottom: 24px; }}
  .card {{ background: #161b22; border: 1px solid #21262d; border-radius: 10px; padding: 20px; margin-bottom: 20px; }}
  canvas {{ max-height: 340px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
  th  {{ background: #21262d; color: #8b949e; padding: 8px 10px; text-align: right; font-weight: 600; }}
  th:first-child {{ text-align: left; }}
  td  {{ padding: 7px 10px; text-align: right; border-bottom: 1px solid #21262d; }}
  td:first-child {{ text-align: left; }}
  tr:hover td {{ background: #1c2128; }}
  .pos {{ color: #3fb950; }} .neg {{ color: #f85149; }}
  .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }}
  .note {{ font-size: 0.78rem; color: #6e7681; margin-bottom: 10px; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 600; margin-left: 6px; }}
  .b1h  {{ background: #1f3a1f; color: #81c784; }}
  .b15m {{ background: #1a2d3d; color: #64b5f6; }}
  .legend-item {{ display: inline-flex; align-items: center; margin: 4px 10px; font-size: 0.78rem; }}
</style>
</head>
<body>

<h1>15M Signal Resolution — Impact Analysis v3</h1>
<p class="subtitle">
  BTCUSDT Perpetual Futures &nbsp;|&nbsp; Composite ±3, Session 08–21 UTC &nbsp;|&nbsp;
  Walk-forward 6m/2m, 23 windows (2022–2026) &nbsp;|&nbsp;
  Two improvements over v2: <strong>1H directional filter</strong> + <strong>4-bar bar-level ML label</strong>
</p>

<script>const DATA = {data_json};</script>
<script>const ML_NAMES = {ml_names_json};</script>

<!-- ── Summary table ──────────────────────────────────────────────────────── -->
<h2>1. Performance Summary</h2>
<div class="card">
<table>
  <thead>
    <tr>
      <th>Strategy</th><th>Trades</th><th>Return%</th><th>Max DD%</th>
      <th>Calmar</th><th>Sharpe</th><th>Win%</th><th>Profit Factor</th>
      <th>Filtered%</th>
    </tr>
  </thead>
  <tbody>{_summary_rows()}</tbody>
</table>
</div>

<!-- ── Impact analysis ────────────────────────────────────────────────────── -->
<h2>2. Incremental Impact of Each Improvement</h2>
<div class="card">
<p class="note">Each row shows the delta from applying one change relative to its control. Positive ΔCalmar = improvement.</p>
<table>
  <thead>
    <tr>
      <th>Change Applied</th><th>ΔReturn%</th><th>ΔMax DD%</th>
      <th>ΔCalmar</th><th>ΔSharpe</th><th>ΔTrades</th>
    </tr>
  </thead>
  <tbody>{_impact_rows()}</tbody>
</table>
</div>

<!-- ── Equity curves ──────────────────────────────────────────────────────── -->
<h2>3. Equity Curves — 15M Strategies</h2>
<div class="card"><canvas id="equity_15m"></canvas></div>

<h2>4. Equity Curves — 1H Reference</h2>
<div class="card"><canvas id="equity_1h"></canvas></div>

<!-- ── Drawdown ───────────────────────────────────────────────────────────── -->
<h2>5. Drawdown — 15M Strategies</h2>
<div class="card"><canvas id="dd_15m"></canvas></div>

<!-- ── Walk-forward window stats ─────────────────────────────────────────── -->
<h2>6. Walk-Forward val_acc per Window (ML models)</h2>
<div class="grid-3" id="wf_grid"></div>

<!-- ── Feature importance ─────────────────────────────────────────────────── -->
<h2>7. Top Feature Importances (ML models)</h2>
<div class="grid-3" id="imp_grid"></div>

<script>
// ── Data helpers ─────────────────────────────────────────────────────────────
const refIdx = DATA.results.find(r => r.name.includes("Baseline 1H")).index;

function makeLineChart(canvasId, datasets, options = {{}}) {{
  new Chart(document.getElementById(canvasId).getContext('2d'), {{
    type: 'line',
    data: {{ labels: refIdx, datasets }},
    options: {{
      responsive: true, animation: false,
      plugins: {{ legend: {{ labels: {{ color: '#8b949e' }} }} }},
      scales: {{
        x: {{ ticks: {{ color: '#8b949e', maxTicksLimit: 12 }}, grid: {{ color: '#21262d' }} }},
        y: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }},
      }},
      ...options,
    }},
  }});
}}

// ── 15M equity (C, D, E, F, G, H) ────────────────────────────────────────────
const names15m = ['C.','D.','E.','F.','G.','H.'];
makeLineChart('equity_15m',
  DATA.results
    .filter(r => names15m.some(p => r.name.startsWith(p)))
    .map(r => ({{ label: r.name, data: r.equity, borderColor: r.color,
                 borderWidth: r.name.includes('H.') ? 2.5 : 1.6,
                 pointRadius: 0, fill: false }})),
  {{ scales: {{ y: {{ ticks: {{ callback: v => '$' + v.toLocaleString() }} }} }} }}
);

// ── 1H equity (A, B) ─────────────────────────────────────────────────────────
makeLineChart('equity_1h',
  DATA.results
    .filter(r => r.name.startsWith('A.') || r.name.startsWith('B.'))
    .map(r => ({{ label: r.name, data: r.equity, borderColor: r.color,
                 borderWidth: 2, pointRadius: 0, fill: false }})),
  {{ scales: {{ y: {{ ticks: {{ callback: v => '$' + v.toLocaleString() }} }} }} }}
);

// ── 15M drawdown ─────────────────────────────────────────────────────────────
makeLineChart('dd_15m',
  DATA.results
    .filter(r => names15m.some(p => r.name.startsWith(p)))
    .map(r => ({{ label: r.name, data: r.dd.map(v => v * 100),
                 borderColor: r.color, borderWidth: 1.5,
                 pointRadius: 0, fill: false }})),
  {{ scales: {{ y: {{ ticks: {{ callback: v => v.toFixed(0) + '%' }} }} }} }}
);

// ── Walk-forward charts ──────────────────────────────────────────────────────
function makeWfChart(container, name, color) {{
  const ws = DATA.window_stats[name];
  if (!ws || !ws.length) return;
  const div = document.createElement('div');
  div.className = 'card';
  div.innerHTML = `<p style="font-weight:600;color:${{color}};margin-bottom:10px">${{name}}</p><canvas id="wf_${{name.replace(/[^a-z0-9]/gi,'_')}}"></canvas>`;
  container.appendChild(div);
  const labels = ws.map(w => w.oos_start ? w.oos_start.slice(0,7) : 'W'+w.window);
  new Chart(div.querySelector('canvas').getContext('2d'), {{
    type: 'bar',
    data: {{ labels, datasets: [{{ label: 'val_acc %', data: ws.map(w => w.val_acc),
              backgroundColor: color + 'cc', yAxisID: 'y' }}] }},
    options: {{
      responsive: true, animation: false,
      plugins: {{ legend: {{ labels: {{ color: '#8b949e' }} }} }},
      scales: {{
        x: {{ ticks: {{ color: '#8b949e', maxRotation: 45 }}, grid: {{ display: false }} }},
        y: {{ min: 30, max: 85, ticks: {{ color: '#8b949e', callback: v => v+'%' }}, grid: {{ color: '#21262d' }} }},
      }},
    }},
  }});
}}

const wfGrid = document.getElementById('wf_grid');
const COLORS_MAP = Object.fromEntries(DATA.results.map(r => [r.name, r.color]));
ML_NAMES.forEach(name => makeWfChart(wfGrid, name, COLORS_MAP[name] || '#aaa'));

// ── Feature importance charts ─────────────────────────────────────────────────
function makeImpChart(container, name, color) {{
  const imp = DATA.importances[name];
  if (!imp || !imp.length) return;
  const div = document.createElement('div');
  div.className = 'card';
  div.innerHTML = `<p style="font-weight:600;color:${{color}};margin-bottom:10px">${{name}}</p><canvas id="imp_${{name.replace(/[^a-z0-9]/gi,'_')}}" style="max-height:280px"></canvas>`;
  container.appendChild(div);
  new Chart(div.querySelector('canvas').getContext('2d'), {{
    type: 'bar',
    data: {{ labels: imp.map(r => r.feature),
             datasets: [{{ data: imp.map(r => r.importance),
               backgroundColor: color + 'cc', borderWidth: 0 }}] }},
    options: {{
      indexAxis: 'y', responsive: true, animation: false,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{ ticks: {{ color: '#8b949e' }}, grid: {{ color: '#21262d' }} }},
        y: {{ ticks: {{ color: '#8b949e', font: {{ size: 10 }} }}, grid: {{ display: false }} }},
      }},
    }},
  }});
}}

const impGrid = document.getElementById('imp_grid');
ML_NAMES.forEach(name => makeImpChart(impGrid, name, COLORS_MAP[name] || '#aaa'));
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
