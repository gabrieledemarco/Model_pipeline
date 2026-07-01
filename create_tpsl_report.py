"""
Walk-forward TP/SL optimisation experiment.

For each training window, grid-searches (atr_sl, atr_tp1) to maximise
in-sample Calmar, then applies the best pair to the OOS window.
TP2 and TP3 are derived automatically: TP2 = 2×TP1, TP3 = 3×TP1.

Grid
────
  atr_sl  : 1.0 / 1.5 / 2.0 / 2.5 / 3.0   (5 values)
  atr_tp1 : 1.0 / 1.5 / 2.0 / 3.0 / 4.0   (5 values)
  Total   : 25 combinations per window

Walk-forward scheme
───────────────────
  Train : 6 months   OOS : 2 months   Step : 2 months   Windows : 23

Configs evaluated
─────────────────
  A.  Baseline 1H (fixed TP/SL)
  A'. Baseline 1H + WF TP/SL opt
  B.  ML Gate 1H P≥0.50 (fixed TP/SL)
  B'. ML Gate 1H P≥0.50 + WF TP/SL opt
  E.  Baseline 15M + 1H dir filter (fixed TP/SL)
  E'. Baseline 15M + 1H dir filter + WF TP/SL opt

Output → reports/report_tpsl.html
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.strategy.data_fetcher import fetch_extended_data, generate_oi, generate_funding
from src.strategy.indicators   import add_indicators
from src.strategy.signals      import build_signal_matrix, build_signal_matrix_15m
from src.strategy.optimizer    import apply_filters, ScenarioConfig
from src.strategy.engine       import run_backtest, INIT_CAP
from src.strategy.ml_features  import build_feature_matrix
from src.strategy.ml_gate      import walk_forward_binary_gate, run_gated_backtest

SESSION_CFG = ScenarioConfig(
    "Session 08-21",
    session_hours=(8, 21),
    long_threshold=3.0,
    short_threshold=-3.0,
)

# ── Walk-forward parameters ───────────────────────────────────────────────────
TRAIN_MONTHS = 6
OOS_MONTHS   = 2
STEP_MONTHS  = 2

# ── TP/SL grid ────────────────────────────────────────────────────────────────
SL_GRID  = [1.0, 1.5, 2.0, 2.5, 3.0]
TP1_GRID = [1.0, 1.5, 2.0, 3.0, 4.0]


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


def _wf_windows(index: pd.DatetimeIndex) -> List[Tuple]:
    start, end = index[0], index[-1]
    windows, cur = [], start
    while True:
        tr_end = cur + pd.DateOffset(months=TRAIN_MONTHS)
        oo_s   = tr_end
        oo_e   = oo_s + pd.DateOffset(months=OOS_MONTHS)
        if oo_e > end:
            break
        windows.append((cur, tr_end, oo_s, oo_e))
        cur = cur + pd.DateOffset(months=STEP_MONTHS)
    return windows


def _build_df15m_with_1h_atr(df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> pd.DataFrame:
    src_ts = (df_1h.index + pd.Timedelta(hours=1)).astype("datetime64[s]")
    tgt_ts = df_15m.index.astype("datetime64[s]")
    src_df = pd.DataFrame({"ts": src_ts, "v": df_1h["atr_14"].values}).sort_values("ts")
    tgt_df = pd.DataFrame({"ts": tgt_ts})
    merged = pd.merge_asof(tgt_df, src_df, on="ts", direction="backward")
    atr_1h = pd.Series(merged["v"].ffill().fillna(0.0).values, index=df_15m.index)
    out = df_15m.copy()
    out["atr_14"] = atr_1h
    return out


def _apply_1h_dir_filter(sig_15m: pd.DataFrame, sig_1h: pd.DataFrame) -> pd.DataFrame:
    comp = sig_1h["composite"].reindex(sig_15m.index, method="ffill").fillna(0.0)
    out  = sig_15m.copy()
    out.loc[(out["signal"] == 1)  & (comp < 0), "signal"] = 0
    out.loc[(out["signal"] == -1) & (comp > 0), "signal"] = 0
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward TP/SL optimiser
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_tpsl(
    df:       pd.DataFrame,   # price data (1H or 15M with 1H ATR)
    signals:  pd.DataFrame,   # pre-filtered signal matrix
    verbose:  bool = True,
    label:    str  = "",
) -> dict:
    """
    Walk-forward grid search of (atr_sl, atr_tp1).

    For each training window: exhaustive grid search maximising in-sample
    Calmar.  Best params applied to OOS window.  OOS equity pieces are
    stitched together to form a single continuous equity curve starting at
    INIT_CAP.

    Returns
    -------
    dict with keys:
        equity          pd.Series (stitched OOS)
        drawdown        pd.Series
        kpis            dict
        window_stats    list[dict]   per-window: best sl, tp1, IS calmar, OOS calmar
    """
    index   = df.index
    windows = _wf_windows(index)

    if verbose:
        print(f"  {'':2}Grid: SL={SL_GRID}  TP1={TP1_GRID}  |  {len(windows)} windows")

    window_stats: list = []
    oos_equity_pieces: list = []  # (index_slice, equity_array)

    running_capital = float(INIT_CAP)

    for i, (tr_s, tr_e, oo_s, oo_e) in enumerate(windows):
        tr_mask  = (index >= tr_s) & (index < tr_e)
        oos_mask = (index >= oo_s) & (index < oo_e)

        df_tr   = df[tr_mask];   sig_tr  = signals[tr_mask]
        df_oos  = df[oos_mask];  sig_oos = signals[oos_mask]

        if len(df_tr) < 50 or len(df_oos) < 4:
            continue

        # ── In-sample grid search ─────────────────────────────────────────
        best_calmar_is = -np.inf
        best_sl = 2.0; best_tp1 = 2.0

        grid_results = {}
        for atr_sl, atr_tp1 in itertools.product(SL_GRID, TP1_GRID):
            bt = run_backtest(df_tr, sig_tr,
                              atr_sl_override=atr_sl,
                              atr_tp1_override=atr_tp1)
            c = bt["kpis"]["calmar"]
            grid_results[(atr_sl, atr_tp1)] = c
            if c > best_calmar_is:
                best_calmar_is = c
                best_sl  = atr_sl
                best_tp1 = atr_tp1

        # ── OOS application ───────────────────────────────────────────────
        bt_oos = run_backtest(df_oos, sig_oos,
                              initial_capital=running_capital,
                              atr_sl_override=best_sl,
                              atr_tp1_override=best_tp1)

        oos_calmar = bt_oos["kpis"]["calmar"]
        oos_ret    = bt_oos["kpis"]["total_return"]
        oos_trades = bt_oos["kpis"]["n_trades"]

        # carry forward final capital
        final_cap = float(bt_oos["equity"].iloc[-1])
        oos_equity_pieces.append(bt_oos["equity"])
        running_capital = final_cap

        window_stats.append({
            "window":     i + 1,
            "oos_start":  str(oo_s.date()),
            "oos_end":    str(oo_e.date()),
            "best_sl":    best_sl,
            "best_tp1":   best_tp1,
            "is_calmar":  round(best_calmar_is, 3),
            "oos_calmar": round(oos_calmar, 3),
            "oos_ret":    round(oos_ret * 100, 2),
            "oos_trades": oos_trades,
        })

        if verbose:
            print(f"    Win {i+1:2d} [{oo_s.date()}→{oo_e.date()}]: "
                  f"best SL={best_sl:.1f}  TP1={best_tp1:.1f}  "
                  f"IS Calmar={best_calmar_is:+.3f}  "
                  f"OOS Ret={oos_ret*100:+.1f}%  trades={oos_trades}")

    if not oos_equity_pieces:
        empty = pd.Series([INIT_CAP], index=df.index[:1])
        return {"equity": empty, "drawdown": empty * 0, "kpis": {}, "window_stats": []}

    # Stitch OOS equity (already in dollar terms, chained)
    equity  = pd.concat(oos_equity_pieces)
    equity  = equity[~equity.index.duplicated(keep="last")]
    rmax    = equity.cummax()
    dd      = (equity - rmax) / rmax

    # Recompute KPIs on stitched series
    from src.strategy.engine import _compute_kpis
    # Reconstruct trades_df from window stats (simplified — use equity for KPIs)
    kpis = _compute_kpis(equity, dd, pd.DataFrame(), INIT_CAP)

    return {
        "equity":       equity,
        "drawdown":     dd,
        "kpis":         kpis,
        "window_stats": window_stats,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n══ Walk-Forward TP/SL Optimisation ════════════════════════════")

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("\n[1/7] Loading data …")
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
    df_15m_bt = _build_df15m_with_1h_atr(df_15m, df_1h)
    print(f"  1H: {len(df_1h):,} bars   15M: {len(df_15m):,} bars")

    # ── 2. Signals ────────────────────────────────────────────────────────────
    print("\n[2/7] Building signals …")
    raw_sig_1h  = build_signal_matrix(
        tf_data=tf_ind, oi_df=oi_df, funding=funding,
        premium_1h=None, df_15m=df_15m, df_1m=None,
    )
    raw_sig_15m = build_signal_matrix_15m(
        tf_data=tf_ind, oi_df=oi_df, funding=funding,
        premium_1h=None, crossing_only=True,
    )
    sig_1h       = apply_filters(raw_sig_1h,  SESSION_CFG)
    sig_15m      = apply_filters(raw_sig_15m, SESSION_CFG)
    sig_15m_filt = _apply_1h_dir_filter(sig_15m, sig_1h)

    # ── 3. A/A'. Baseline 1H ─────────────────────────────────────────────────
    print("\n[3/7] A. Baseline 1H (fixed TP/SL) …")
    bt_a = run_backtest(df_1h, sig_1h)
    row_a = kpi_row("A. Baseline 1H", bt_a)
    print(f"  → Return={row_a['total_return']:+.1f}%  DD={row_a['max_dd']:.1f}%  "
          f"Trades={row_a['n_trades']}  Calmar={row_a['calmar']:.3f}")

    print("       A'. + WF TP/SL optimisation …")
    wf_a = walk_forward_tpsl(df_1h, sig_1h, verbose=True, label="A'")
    bt_a_opt = {"equity": wf_a["equity"], "drawdown": wf_a["drawdown"], "kpis": wf_a["kpis"]}
    row_a_opt = kpi_row("A'. Baseline 1H + WF TP/SL", bt_a_opt,
                        extra={"window_stats": wf_a["window_stats"]})
    print(f"  → Return={row_a_opt['total_return']:+.1f}%  DD={row_a_opt['max_dd']:.1f}%  "
          f"Trades={row_a_opt['n_trades']}  Calmar={row_a_opt['calmar']:.3f}")

    # ── 4. B/B'. ML Gate 1H P≥0.50 ───────────────────────────────────────────
    print("\n[4/7] B. ML Gate 1H P≥0.50 (fixed TP/SL) …")
    feat_1h = build_feature_matrix(tf_ind, sig_1h, base_tf="1H")
    gate_1h = walk_forward_binary_gate(
        df_1h, sig_1h, feat_1h, gate_threshold=0.50,
        use_feat_sel=True, verbose=True,
    )
    bt_b = run_gated_backtest(df_1h, sig_1h, gate_1h)
    n_b  = int((gate_1h.gated_signal != 0).sum())
    n_1h = int((sig_1h["signal"] != 0).sum())
    row_b = kpi_row("B. ML Gate 1H P≥0.50", bt_b,
                    extra={"filter_rate": round((1 - n_b / n_1h) * 100, 1)})
    print(f"  → Return={row_b['total_return']:+.1f}%  DD={row_b['max_dd']:.1f}%  "
          f"Trades={row_b['n_trades']}  Calmar={row_b['calmar']:.3f}  "
          f"Filtered={row_b['filter_rate']:.0f}%")

    # Build gated signal dataframe for the optimiser
    sig_1h_gated = sig_1h.copy()
    sig_1h_gated["signal"] = gate_1h.gated_signal.reindex(sig_1h.index).fillna(0).astype(int)

    print("       B'. + WF TP/SL optimisation …")
    wf_b = walk_forward_tpsl(df_1h, sig_1h_gated, verbose=True, label="B'")
    bt_b_opt = {"equity": wf_b["equity"], "drawdown": wf_b["drawdown"], "kpis": wf_b["kpis"]}
    row_b_opt = kpi_row("B'. ML Gate 1H + WF TP/SL", bt_b_opt,
                        extra={"window_stats": wf_b["window_stats"]})
    print(f"  → Return={row_b_opt['total_return']:+.1f}%  DD={row_b_opt['max_dd']:.1f}%  "
          f"Trades={row_b_opt['n_trades']}  Calmar={row_b_opt['calmar']:.3f}")

    # ── 5. E/E'. Baseline 15M + 1H dir filter ─────────────────────────────────
    print("\n[5/7] E. Baseline 15M + 1H dir filter (fixed TP/SL) …")
    bt_e = run_backtest(df_15m_bt, sig_15m_filt)
    n_e  = int((sig_15m_filt["signal"] != 0).sum())
    n_15 = int((sig_15m["signal"] != 0).sum())
    row_e = kpi_row("E. Baseline 15M + dir filter", bt_e,
                    extra={"filter_rate": round((1 - n_e / n_15) * 100, 1)})
    print(f"  → Return={row_e['total_return']:+.1f}%  DD={row_e['max_dd']:.1f}%  "
          f"Trades={row_e['n_trades']}  Calmar={row_e['calmar']:.3f}")

    print("       E'. + WF TP/SL optimisation …")
    wf_e = walk_forward_tpsl(df_15m_bt, sig_15m_filt, verbose=True, label="E'")
    bt_e_opt = {"equity": wf_e["equity"], "drawdown": wf_e["drawdown"], "kpis": wf_e["kpis"]}
    row_e_opt = kpi_row("E'. Baseline 15M + dir filter + WF TP/SL", bt_e_opt,
                        extra={"window_stats": wf_e["window_stats"]})
    print(f"  → Return={row_e_opt['total_return']:+.1f}%  DD={row_e_opt['max_dd']:.1f}%  "
          f"Trades={row_e_opt['n_trades']}  Calmar={row_e_opt['calmar']:.3f}")

    results = [row_a, row_a_opt, row_b, row_b_opt, row_e, row_e_opt]

    # ── 6. Console summary ────────────────────────────────────────────────────
    print("\n── Results Summary ────────────────────────────────────────────")
    print(f"  {'Config':<42} {'Return%':>8} {'DD%':>7} {'Calmar':>8} {'Trades':>7}")
    print("  " + "─" * 74)
    for r in results:
        print(f"  {r['name']:<42} {r['total_return']:>+8.1f} {r['max_dd']:>7.1f} "
              f"{r['calmar']:>8.3f} {r['n_trades']:>7}")
    print()
    pairs = [("A'−A", "A'. Baseline 1H + WF TP/SL",             "A. Baseline 1H"),
             ("B'−B", "B'. ML Gate 1H + WF TP/SL",              "B. ML Gate 1H P≥0.50"),
             ("E'−E", "E'. Baseline 15M + dir filter + WF TP/SL","E. Baseline 15M + dir filter")]
    res = {r["name"]: r for r in results}
    for label, new_n, base_n in pairs:
        n = res[new_n]; b = res[base_n]
        dc = round(n["calmar"] - b["calmar"], 3)
        dr = round(n["total_return"] - b["total_return"], 1)
        print(f"  {label}: ΔReturn={dr:+.1f}%  ΔCalmar={dc:+.3f}")
    print()

    # ── 7. HTML ───────────────────────────────────────────────────────────────
    print("[7/7] Generating report …")
    html = _build_html(results, wf_a, wf_b, wf_e)
    out  = Path("reports/report_tpsl.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"  Saved → {out}")
    print("\n══ Done ══════════════════════════════════════════════════════════")


# ─────────────────────────────────────────────────────────────────────────────
# HTML builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_html(results: list, wf_a: dict, wf_b: dict, wf_e: dict) -> str:

    COLORS = {
        "A. Baseline 1H":                             "#6e7681",
        "A'. Baseline 1H + WF TP/SL":                "#c9d1d9",
        "B. ML Gate 1H P≥0.50":                       "#81c784",
        "B'. ML Gate 1H + WF TP/SL":                  "#4caf50",
        "E. Baseline 15M + dir filter":               "#4fc3f7",
        "E'. Baseline 15M + dir filter + WF TP/SL":   "#0288d1",
    }

    def _cc(v, good_high=True):
        cls = ("pos" if v > 0 else "neg") if good_high else ("neg" if v < 0 else "pos")
        return f'<td class="{cls}">{v}</td>'

    def _summary_rows():
        rows = ""
        groups = [
            ("1H Baseline", ["A. Baseline 1H", "A'. Baseline 1H + WF TP/SL"]),
            ("1H ML Gate",  ["B. ML Gate 1H P≥0.50", "B'. ML Gate 1H + WF TP/SL"]),
            ("15M Baseline + Dir Filter",
             ["E. Baseline 15M + dir filter", "E'. Baseline 15M + dir filter + WF TP/SL"]),
        ]
        res_by = {r["name"]: r for r in results}
        for grp, names in groups:
            rows += f'<tr><td colspan="8" style="background:#21262d;color:#8b949e;font-size:0.76rem;padding:5px 10px">{grp}</td></tr>'
            for name in names:
                r = res_by.get(name)
                if not r:
                    continue
                c = COLORS.get(name, "#aaa")
                rows += f"""
            <tr>
              <td><span class="dot" style="background:{c}"></span>{r['name']}</td>
              <td>{r['n_trades']}</td>
              {_cc(r['total_return'])}
              {_cc(r['max_dd'], False)}
              {_cc(r['calmar'])}
              {_cc(r['sharpe'])}
              {_cc(r['win_rate'])}
              {_cc(r['profit_factor'])}
            </tr>"""
        return rows

    def _impact_rows():
        res = {r["name"]: r for r in results}
        pairs = [
            ("WF TP/SL on Baseline 1H",        "A'. Baseline 1H + WF TP/SL",
             "A. Baseline 1H"),
            ("WF TP/SL on ML Gate 1H",          "B'. ML Gate 1H + WF TP/SL",
             "B. ML Gate 1H P≥0.50"),
            ("WF TP/SL on 15M dir filter",      "E'. Baseline 15M + dir filter + WF TP/SL",
             "E. Baseline 15M + dir filter"),
        ]
        rows = ""
        for lbl, new_n, base_n in pairs:
            n = res.get(new_n); b = res.get(base_n)
            if not n or not b:
                continue
            dr = round(n["total_return"] - b["total_return"], 1)
            dd = round(n["max_dd"]       - b["max_dd"],       1)
            dc = round(n["calmar"]       - b["calmar"],       3)
            ds = round(n["sharpe"]       - b["sharpe"],       3)
            dt = n["n_trades"]           - b["n_trades"]
            cr = "pos" if dr > 0 else "neg"
            cc = "pos" if dc > 0 else "neg"
            rows += f"""
            <tr>
              <td>{lbl}</td>
              <td class="{cr}">{dr:+.1f}%</td>
              <td class="{'neg' if dd < 0 else 'pos'}">{dd:+.1f}%</td>
              <td class="{cc}">{dc:+.3f}</td>
              <td class="{cc}">{ds:+.3f}</td>
              <td>{dt:+d}</td>
            </tr>"""
        return rows

    def _ws_table(ws: list) -> str:
        if not ws:
            return "<p style='color:#6e7681'>No data</p>"
        rows = ""
        for w in ws:
            c_oos = "pos" if w["oos_calmar"] > 0 else "neg"
            c_ret = "pos" if w["oos_ret"]    > 0 else "neg"
            rows += f"""
            <tr>
              <td>{w['oos_start']}→{w['oos_end']}</td>
              <td>{w['best_sl']:.1f}</td>
              <td>{w['best_tp1']:.1f}</td>
              <td>{w['best_tp1']*2:.1f}</td>
              <td>{w['best_tp1']*3:.1f}</td>
              <td>{w['is_calmar']:+.3f}</td>
              <td class="{c_ret}">{w['oos_ret']:+.1f}%</td>
              <td class="{c_oos}">{w['oos_calmar']:+.3f}</td>
              <td>{w['oos_trades']}</td>
            </tr>"""
        return f"""
        <table>
          <thead><tr>
            <th style="text-align:left">OOS window</th>
            <th>SL</th><th>TP1</th><th>TP2</th><th>TP3</th>
            <th>IS Calmar</th><th>OOS Ret%</th><th>OOS Calmar</th><th>Trades</th>
          </tr></thead>
          <tbody>{rows}</tbody>
        </table>"""

    data = {
        "results": [{
            "name":   r["name"],
            "equity": r["equity"],
            "dd":     r["drawdown"],
            "index":  r["index"],
            "color":  COLORS.get(r["name"], "#aaa"),
        } for r in results],
    }
    data_json = json.dumps(data, default=str)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Walk-Forward TP/SL Optimisation — BTCUSDT</title>
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
  .note {{ font-size: 0.78rem; color: #6e7681; margin-bottom: 10px; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .pill {{ display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.72rem;
           font-weight: 600; background: #21262d; margin: 2px; }}
</style>
</head>
<body>

<h1>Walk-Forward TP/SL Optimisation — BTCUSDT Perpetual Futures</h1>
<p class="subtitle">
  Grid: SL ∈ {{{', '.join(str(v) for v in SL_GRID)}}} × TP1 ∈ {{{', '.join(str(v) for v in TP1_GRID)}}} = {len(SL_GRID)*len(TP1_GRID)} combinations &nbsp;|&nbsp;
  TP2 = 2×TP1 &nbsp; TP3 = 3×TP1 &nbsp;|&nbsp;
  Objective: max in-sample Calmar &nbsp;|&nbsp;
  Walk-forward 6m train / 2m OOS / 23 windows
</p>

<script>const DATA = {data_json};</script>

<!-- ── Summary table ──────────────────────────────────────────────────────── -->
<h2>1. Performance Summary</h2>
<div class="card">
<table>
  <thead>
    <tr><th>Strategy</th><th>Trades</th><th>Return%</th><th>Max DD%</th>
    <th>Calmar</th><th>Sharpe</th><th>Win%</th><th>Profit Factor</th></tr>
  </thead>
  <tbody>{_summary_rows()}</tbody>
</table>
</div>

<!-- ── Impact table ───────────────────────────────────────────────────────── -->
<h2>2. Incremental Impact of WF TP/SL Optimisation</h2>
<div class="card">
<p class="note">Delta versus fixed TP/SL baseline.</p>
<table>
  <thead><tr><th>Applied to</th><th>ΔReturn%</th><th>ΔMax DD%</th>
  <th>ΔCalmar</th><th>ΔSharpe</th><th>ΔTrades</th></tr></thead>
  <tbody>{_impact_rows()}</tbody>
</table>
</div>

<!-- ── Equity curves ──────────────────────────────────────────────────────── -->
<h2>3. Equity Curves — 1H Strategies</h2>
<div class="card"><canvas id="eq_1h"></canvas></div>

<h2>4. Equity Curves — 15M Strategy</h2>
<div class="card"><canvas id="eq_15m"></canvas></div>

<!-- ── Per-window optimal params ─────────────────────────────────────────── -->
<h2>5. Per-Window Optimal Parameters</h2>
<p class="note" style="margin:8px 0 10px">
  Each row shows the in-sample optimal (SL, TP1) and the OOS out-of-sample realised performance using those params.
</p>

<h3 style="color:#c9d1d9;font-size:0.9rem;margin:14px 0 6px">A'. Baseline 1H</h3>
<div class="card">{_ws_table(wf_a['window_stats'])}</div>

<h3 style="color:#4caf50;font-size:0.9rem;margin:14px 0 6px">B'. ML Gate 1H</h3>
<div class="card">{_ws_table(wf_b['window_stats'])}</div>

<h3 style="color:#0288d1;font-size:0.9rem;margin:14px 0 6px">E'. Baseline 15M + dir filter</h3>
<div class="card">{_ws_table(wf_e['window_stats'])}</div>

<script>
function makeChart(id, nameFilter, yFmt) {{
  const ds = DATA.results.filter(r => nameFilter(r.name));
  const ref = ds[0];
  new Chart(document.getElementById(id).getContext('2d'), {{
    type: 'line',
    data: {{ labels: ref.index, datasets: ds.map(r => ({{
      label: r.name, data: r.equity, borderColor: r.color,
      borderWidth: r.name.includes("'") ? 2.5 : 1.5, pointRadius: 0, fill: false,
    }})) }},
    options: {{
      responsive: true, animation: false,
      plugins: {{ legend: {{ labels: {{ color: '#8b949e' }} }} }},
      scales: {{
        x: {{ ticks: {{ color: '#8b949e', maxTicksLimit: 12 }}, grid: {{ color: '#21262d' }} }},
        y: {{ ticks: {{ color: '#8b949e', callback: yFmt }}, grid: {{ color: '#21262d' }} }},
      }},
    }},
  }});
}}
const fmt$ = v => '$' + v.toLocaleString();
makeChart('eq_1h',  n => n.includes('1H'),  fmt$);
makeChart('eq_15m', n => n.includes('15M'), fmt$);
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
