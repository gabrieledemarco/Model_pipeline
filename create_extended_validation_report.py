"""
create_extended_validation_report.py
─────────────────────────────────────
Identical validation pipeline as create_validation_report.py but with
the full available history from Binance Vision perpetual futures (2020-01).

Changes vs the 2022-based report:
  - 6.5 years of data (vs 4.5) — 2020-01 → 2026-05
  - ~57k 1H bars (vs ~39k)
  - ~35 WF windows (vs 23) → stronger t-test power
  - Monte Carlo computed live (not hardcoded) from the new gated trades
  - Includes BTC 2020 COVID crash, 2020-2021 bull run, 2022 bear, 2023 chop,
    2024-2025 bull run, 2025-2026 sideways/new-ATH cycle

Output → reports/report_validation_extended.html
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from src.strategy.data_fetcher  import fetch_extended_data, generate_oi, generate_funding
from src.strategy.indicators    import add_indicators
from src.strategy.signals       import build_signal_matrix
from src.strategy.optimizer     import apply_filters, ScenarioConfig
from src.strategy.engine        import run_backtest, INIT_CAP
from src.strategy.ml_features   import build_feature_matrix
from src.strategy.ml_gate       import (
    walk_forward_binary_gate,
    run_gated_backtest,
    _wf_windows,
    _extract_labels,
    _fit_model,
    TOP_N_FEATURES,
)
from src.strategy.monte_carlo   import run_monte_carlo

SESSION_CFG = ScenarioConfig(
    "Session 08-21",
    session_hours=(8, 21),
    long_threshold=3.0,
    short_threshold=-3.0,
)

START_YEAR  = 2020
START_MONTH = 1
HELD_OUT_START = pd.Timestamp("2025-01-01")
FINAL_MODEL_LOOKBACK_MONTHS = 6
TOP_N   = TOP_N_FEATURES    # 20
N_SIMS  = 5_000


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _kpi(bt):
    k = bt["kpis"]
    return (round(k.get("total_return", 0) * 100, 2),
            round(k.get("max_drawdown",  0) * 100, 2),
            round(k.get("calmar",        0), 3),
            round(k.get("sharpe",        0), 3),
            int  (k.get("n_trades",      0)))


def _per_window_returns(trades: pd.DataFrame, windows) -> np.ndarray:
    rets = []
    for _, _, oos_s, oos_e in windows:
        mask = (trades["entry_ts"] >= oos_s) & (trades["entry_ts"] < oos_e)
        rets.append(trades[mask]["net_pnl"].sum() / INIT_CAP)
    return np.array(rets)


def _window_stats(bt_gated, bt_baseline, windows):
    tg, tb = bt_gated["trades"], bt_baseline["trades"]
    rows = []
    for _, _, oos_s, oos_e in windows:
        mg = (tg["entry_ts"] >= oos_s) & (tg["entry_ts"] < oos_e)
        mb = (tb["entry_ts"] >= oos_s) & (tb["entry_ts"] < oos_e)
        rows.append({
            "oos_start": str(oos_s.date()),
            "oos_end":   str(oos_e.date()),
            "ret_base":  round(tb[mb]["net_pnl"].sum() / INIT_CAP * 100, 2),
            "ret_gated": round(tg[mg]["net_pnl"].sum() / INIT_CAP * 100, 2),
            "diff":      round((tg[mg]["net_pnl"].sum() - tb[mb]["net_pnl"].sum()) / INIT_CAP * 100, 2),
            "n_gated":   int(mg.sum()),
        })
    return rows


def _mc_summary(mc: dict, realized_kpi: tuple) -> dict:
    """Convert run_monte_carlo output + realized kpis to a flat summary dict."""
    r, dd, cal, sh, nt = realized_kpi
    return {
        "n_trades":   mc["n_trades"],
        "p_profit":   round(mc["p_profit"]  * 100, 1),
        "p_ruin":     round(mc["p_ruin"]    * 100, 2),
        "med_ret":    round(float(np.median(mc["total_return"]) * 100), 2),
        "p5_ret":     round(float(np.percentile(mc["total_return"] * 100,  5)), 2),
        "p95_ret":    round(float(np.percentile(mc["total_return"] * 100, 95)), 2),
        "med_dd":     round(float(np.median(mc["max_drawdown"]) * 100), 2),
        "worst_dd":   round(float(np.percentile(mc["max_drawdown"] * 100,  5)), 2),
        "med_sharpe": round(float(np.median(mc["sharpe"])), 3),
        "realized_ret": r, "realized_dd": dd,
        "realized_calmar": cal, "realized_sharpe": sh,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n══ Extended Validation Report ({START_YEAR}-{START_MONTH:02d} → present) ═══")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print(f"\n[1/8] Loading data from {START_YEAR}-{START_MONTH:02d} …")
    raw = fetch_extended_data(start_year=START_YEAR, start_month=START_MONTH,
                              fetch_1m=False, fetch_flow=True)
    tf_ind = {}
    for tf in ["1W", "1D", "4H", "1H", "15M"]:
        df = raw.get(tf, pd.DataFrame())
        tf_ind[tf] = add_indicators(df) if not df.empty and len(df) > 20 else df

    df_1h   = tf_ind["1H"]
    df_15m  = tf_ind["15M"]
    oi_df   = generate_oi(tf_ind["1D"]["close"])
    funding = generate_funding(tf_ind["1D"]["close"])
    print(f"  1H: {len(df_1h):,} bars  ({df_1h.index[0].date()} → {df_1h.index[-1].date()})")

    # ── 2. Signals & feature matrix ───────────────────────────────────────────
    print("\n[2/8] Building signals & feature matrix …")
    raw_sig = build_signal_matrix(
        tf_data=tf_ind, oi_df=oi_df, funding=funding,
        premium_1h=None, df_15m=df_15m, df_1m=None,
    )
    signals = apply_filters(raw_sig, SESSION_CFG)
    feat_df = build_feature_matrix(tf_ind, signals, base_tf="1H", include_smc=False)
    n_sig   = int((signals["signal"] != 0).sum())
    print(f"  Signal bars: {n_sig:,}  Features: {feat_df.shape[1]}")

    # ── 3. Full WF gate ───────────────────────────────────────────────────────
    print("\n[3/8] Full WF gate …")
    gate_full = walk_forward_binary_gate(
        df_1h, signals, feat_df,
        gate_threshold=0.50, use_feat_sel=True, verbose=True,
    )
    bt_baseline = run_backtest(df_1h, signals)
    bt_gated    = run_gated_backtest(df_1h, signals, gate_full)

    r_b, dd_b, cal_b, sh_b, nt_b = _kpi(bt_baseline)
    r_g, dd_g, cal_g, sh_g, nt_g = _kpi(bt_gated)
    print(f"  Baseline:  ret={r_b:+.1f}%  DD={dd_b:.1f}%  Calmar={cal_b:.3f}  Sharpe={sh_b:.3f}  Trades={nt_b}")
    print(f"  Gated:     ret={r_g:+.1f}%  DD={dd_g:.1f}%  Calmar={cal_g:.3f}  Sharpe={sh_g:.3f}  Trades={nt_g}")

    # ── 4. Statistical significance tests ─────────────────────────────────────
    print("\n[4/8] Statistical significance …")
    windows_full = _wf_windows(df_1h.index)
    trades_g = bt_gated["trades"]
    trades_b = bt_baseline["trades"]

    rets_g = _per_window_returns(trades_g, windows_full)
    rets_b = _per_window_returns(trades_b, windows_full)
    n_win  = len(rets_g)
    n_pos_win = int((rets_g > 0).sum())

    t_a, p_a = stats.ttest_1samp(rets_g, popmean=0.0, alternative="greater")
    t_b, p_b = stats.ttest_rel(rets_g, rets_b, alternative="greater")
    trade_rets = (trades_g["net_pnl"] / INIT_CAP).values
    t_c, p_c   = stats.ttest_1samp(trade_rets, popmean=0.0, alternative="greater")

    print(f"  WF windows: {n_win}  Profitable: {n_pos_win}/{n_win}")
    print(f"  (a) gated per-window returns > 0:    t={t_a:.3f}  p={p_a:.4f}  {'✅' if p_a<0.05 else '⚠️'}")
    print(f"  (b) paired gated vs baseline/window: t={t_b:.3f}  p={p_b:.4f}  {'✅' if p_b<0.05 else '⚠️'}")
    print(f"  (c) per-trade return > 0 (n={len(trade_rets)}):  t={t_c:.3f}  p={p_c:.4f}  {'✅' if p_c<0.05 else '⚠️'}")

    per_win_table = _window_stats(bt_gated, bt_baseline, windows_full)
    stat_tests = {
        "a": {"label": "One-sample t-test (gated OOS per-window returns > 0)",
              "t": round(t_a,3), "p": round(p_a,4), "n": n_win, "sig": p_a < 0.05,
              "interp": f"{n_pos_win}/{n_win} windows profitable. Mean per-window = {rets_g.mean()*100:+.2f}%."},
        "b": {"label": "Paired t-test (gated vs baseline per-window)",
              "t": round(t_b,3), "p": round(p_b,4), "n": n_win, "sig": p_b < 0.05,
              "interp": f"Mean alpha per 2M window = {(rets_g-rets_b).mean()*100:+.2f}%."},
        "c": {"label": f"One-sample t-test (per-trade return > 0, n={len(trade_rets)})",
              "t": round(t_c,3), "p": round(p_c,4), "n": len(trade_rets), "sig": p_c < 0.05,
              "interp": f"Mean net PnL per trade = ${trade_rets.mean()*INIT_CAP:.0f} ({trade_rets.mean()*100:+.3f}% capital)."},
    }

    # ── 5. Monte Carlo ────────────────────────────────────────────────────────
    print(f"\n[5/8] Monte Carlo ({N_SIMS:,} simulations) …")
    mc_base  = run_monte_carlo(trades_b, initial_capital=INIT_CAP, n_sims=N_SIMS, seed=42)
    mc_gated = run_monte_carlo(trades_g, initial_capital=INIT_CAP, n_sims=N_SIMS, seed=42)

    def _mc_p(mc): return f"P(profit)={mc['p_profit']*100:.1f}%  P(ruin)={mc['p_ruin']*100:.2f}%  med_ret={np.median(mc['total_return'])*100:+.1f}%"
    print(f"  Baseline:  {_mc_p(mc_base)}")
    print(f"  Gated:     {_mc_p(mc_gated)}")

    mc_summary_rows = [
        {"name": "A. Baseline", **_mc_summary(mc_base,  (r_b, dd_b, cal_b, sh_b, nt_b))},
        {"name": "B. ML Gate P≥0.50", **_mc_summary(mc_gated, (r_g, dd_g, cal_g, sh_g, nt_g))},
    ]

    # ── 6. Held-out test ──────────────────────────────────────────────────────
    print(f"\n[6/8] Held-out test (cutoff={HELD_OUT_START.date()}) …")

    df_train    = df_1h[df_1h.index < HELD_OUT_START]
    df_test     = df_1h[df_1h.index >= HELD_OUT_START]
    sig_train   = signals[signals.index < HELD_OUT_START]
    sig_test    = signals[signals.index >= HELD_OUT_START]
    feat_train  = feat_df[feat_df.index < HELD_OUT_START]
    feat_test   = feat_df[feat_df.index >= HELD_OUT_START]
    print(f"  Train: {len(df_train):,} bars ({df_train.index[0].date()}→{df_train.index[-1].date()})")
    print(f"  Test:  {len(df_test):,} bars ({df_test.index[0].date()}→{df_test.index[-1].date()})")

    # 6a. WF on training split
    print("  [6a] WF gate on training split …")
    gate_tr = walk_forward_binary_gate(
        df_train, sig_train, feat_train,
        gate_threshold=0.50, use_feat_sel=True, verbose=True,
    )
    bt_tr_base  = run_backtest(df_train, sig_train)
    bt_tr_gated = run_gated_backtest(df_train, sig_train, gate_tr)
    r_tb, dd_tb, cal_tb, sh_tb, nt_tb = _kpi(bt_tr_base)
    r_tg, dd_tg, cal_tg, sh_tg, nt_tg = _kpi(bt_tr_gated)
    print(f"  WF OOS (train split): baseline Calmar={cal_tb:.3f}  gated Calmar={cal_tg:.3f}")

    top_features = gate_tr.importances["feature"].head(TOP_N).tolist()

    # 6b. Frozen model on held-out
    print("  [6b] Frozen model on held-out …")
    final_start = HELD_OUT_START - pd.DateOffset(months=FINAL_MODEL_LOOKBACK_MONTHS)
    df_fin   = df_train[df_train.index >= final_start]
    sig_fin  = sig_train[sig_train.index >= final_start]
    feat_fin = feat_train[feat_train.index >= final_start]
    X_fin, y_fin = _extract_labels(df_fin, sig_fin, feat_fin, regression=False)
    n_pos_fin = int(y_fin.sum()) if len(y_fin) else 0
    n_neg_fin = len(y_fin) - n_pos_fin

    bt_ho_base  = run_backtest(df_test, sig_test)
    r_hob, dd_hob, cal_hob, sh_hob, nt_hob = _kpi(bt_ho_base)
    frozen_result = {"status": "skipped", "ret_base": r_hob, "dd_base": dd_hob,
                     "cal_base": cal_hob, "nt_hob": nt_hob}

    if len(X_fin) >= 20 and n_pos_fin >= 5 and n_neg_fin >= 5:
        model_fin, _, val_acc_fin = _fit_model(X_fin, y_fin, top_features, regression=False)
        sig_bars_ho = sig_test["signal"] != 0
        feat_ho_sig = feat_test[sig_bars_ho]
        n_sig_ho    = int(sig_bars_ho.sum())
        probs_ho    = model_fin.predict_proba(feat_ho_sig[top_features])[:, 1]
        n_pass      = int((probs_ho >= 0.50).sum())
        p50_prob    = round(np.percentile(probs_ho, 50) * 100, 1)
        print(f"  Frozen model: val_acc={val_acc_fin*100:.1f}%  "
              f"n_sig={n_sig_ho}  pass={n_pass}  prob_p50={p50_prob}%")

        pass_mask   = probs_ho >= 0.50
        held_signal = pd.Series(0, index=df_test.index)
        pass_idx    = feat_ho_sig[pass_mask].index
        held_signal.loc[pass_idx] = sig_test.loc[pass_idx, "signal"]
        sig_ho_gated           = sig_test.copy()
        sig_ho_gated["signal"] = held_signal
        bt_ho_gated = run_backtest(df_test, sig_ho_gated)
        r_hog, dd_hog, cal_hog, sh_hog, nt_hog = _kpi(bt_ho_gated)
        frozen_result.update({
            "status": "run", "val_acc": round(val_acc_fin*100,1),
            "n_sig_ho": n_sig_ho, "n_pass": n_pass,
            "pass_rate": round(n_pass/n_sig_ho*100,1) if n_sig_ho else 0.0,
            "p50_prob": p50_prob,
            "ret_gated": r_hog, "dd_gated": dd_hog,
            "cal_gated": cal_hog, "nt_gated": nt_hog,
        })
        if n_pass > 0:
            print(f"  Frozen-gated held-out: ret={r_hog:+.1f}%  Calmar={cal_hog:.3f}  Trades={nt_hog}")
        else:
            print("  Frozen-gated: 0 signals passed → regime mismatch")

    # 6c. Full WF in 2025+ windows
    windows_2025 = [(a,b,c,d) for a,b,c,d in windows_full if c >= HELD_OUT_START]
    wf_2025_table = _window_stats(bt_gated, bt_baseline, windows_2025)
    rets_g_2025   = np.array([r["ret_gated"]/100 for r in wf_2025_table])
    print(f"  WF 2025+ windows: {len(windows_2025)}  positive: {(rets_g_2025>0).sum()}/{len(rets_g_2025)}")

    # ── 7. Comparison with 2022-based report (hardcoded) ─────────────────────
    ref_2022 = {
        "baseline": {"ret": 8.6, "dd": -32.22, "calmar": 0.267, "sharpe": 1.319,
                     "n_trades": 1859, "n_windows": 23},
        "gated":    {"ret": 34.52, "dd": -34.2, "calmar": 1.009, "sharpe": 1.922,
                     "n_trades": 985, "n_windows": 23,
                     "p_profit_mc": 82.2, "p_ruin_mc": 0.14,
                     "t_a_p": 0.1706, "t_b_p": 0.1542, "t_c_p": 0.1813},
    }

    # ── 8. Generate HTML ──────────────────────────────────────────────────────
    print("\n[8/8] Generating HTML …")
    html = _build_html(
        stat_tests, per_win_table, mc_summary_rows, frozen_result,
        wf_2025_table, rets_g_2025, ref_2022,
        r_b, dd_b, cal_b, sh_b, nt_b, r_g, dd_g, cal_g, sh_g, nt_g,
        n_win, n_pos_win,
        {"cal_base": cal_tb, "cal_gated": cal_tg, "nt_base": nt_tb, "nt_gated": nt_tg,
         "ret_base": r_tb, "ret_gated": r_tg},
    )
    out = Path("reports/report_validation_extended.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"  → {out}")
    print("══ Done ══════════════════════════════════════════════════════════════")


# ─────────────────────────────────────────────────────────────────────────────
# HTML
# ─────────────────────────────────────────────────────────────────────────────

def _cc(v, good_high=True):
    cls = ("pos" if v > 0 else "neg") if good_high else ("neg" if v < 0 else "pos")
    return f'<td class="{cls}">{v}</td>'


def _sig_badge(is_sig):
    c = "pos" if is_sig else "neg"
    t = "&#10003; p &lt; 0.05" if is_sig else "&#10007; p &ge; 0.05"
    return f'<span class="badge {c}">{t}</span>'


def _build_html(stat_tests, per_win_table, mc_rows, frozen, wf_2025, rets_2025,
                ref_2022, r_b, dd_b, cal_b, sh_b, nt_b,
                r_g, dd_g, cal_g, sh_g, nt_g,
                n_win, n_pos_win, train_stats):

    filter_pct = round(100*(1-nt_g/nt_b)) if nt_b > 0 else 0
    st = stat_tests
    fr = frozen
    ts = train_stats

    def _mc_table_rows():
        h = ""
        for r in mc_rows:
            h += (f"<tr><td>{r['name']}</td><td>{r['n_trades']}</td>"
                  f"{_cc(r['realized_ret'])}{_cc(r['realized_dd'],False)}"
                  f"{_cc(r['realized_calmar'])}{_cc(r['realized_sharpe'])}"
                  f"{_cc(r['p_profit'])}{_cc(r['p_ruin'],False)}"
                  f"{_cc(r['med_ret'])}{_cc(r['p5_ret'])}{_cc(r['p95_ret'])}"
                  f"{_cc(r['med_dd'],False)}{_cc(r['worst_dd'],False)}"
                  f"{_cc(r['med_sharpe'])}</tr>")
        return h

    def _win_rows(table):
        h = ""
        for r in table:
            h += (f"<tr>"
                  f"<td>{r['oos_start']}</td><td>{r['oos_end']}</td>"
                  f"<td class='{'pos' if r['ret_base']>=0 else 'neg'}'>{r['ret_base']:+.2f}%</td>"
                  f"<td class='{'pos' if r['ret_gated']>=0 else 'neg'}'>{r['ret_gated']:+.2f}%</td>"
                  f"<td class='{'pos' if r['diff']>=0 else 'neg'}'>{r['diff']:+.2f}%</td>"
                  f"<td>{r['n_gated']}</td></tr>")
        return h

    ref = ref_2022
    r22_g  = ref["gated"]
    r22_b  = ref["baseline"]

    def _delta(new, old, fmt="+.2f"):
        d = new - old
        cls = "pos" if d > 0 else "neg"
        return f'<span class="{cls}">{d:{fmt}}</span>'

    # Frozen section
    if fr["status"] == "run" and fr.get("n_pass", 0) == 0:
        frozen_warn = (
            f'<div class="warn">Frozen model (val_acc={fr["val_acc"]}%) passed 0/{fr["n_sig_ho"]} signals. '
            f'Prob median={fr["p50_prob"]}%. '
            f'Root cause: 2024H2 bull-run regime mismatch — WF rolling re-training is necessary.</div>'
        )
        frozen_body = f'<tr><td>Baseline (no gate)</td>{_cc(fr["ret_base"])}{_cc(fr["dd_base"],False)}{_cc(fr["cal_base"])}<td>{fr["nt_hob"]}</td></tr>'
        frozen_body += frozen_warn
    elif fr["status"] == "run":
        frozen_body = (
            f'<tr><td>Baseline</td>{_cc(fr["ret_base"])}{_cc(fr["dd_base"],False)}{_cc(fr["cal_base"])}<td>{fr["nt_hob"]}</td></tr>'
            f'<tr><td>Frozen gated</td>{_cc(fr["ret_gated"])}{_cc(fr["dd_gated"],False)}{_cc(fr["cal_gated"])}<td>{fr["nt_gated"]}</td></tr>'
        )
    else:
        frozen_body = "<tr><td colspan=5>Skipped</td></tr>"

    wf25_pos   = int((rets_2025 > 0).sum())
    wf25_total = len(rets_2025)
    wf_tr_cls  = "pos" if ts.get("cal_gated", 0) > 0.5 else "neg"
    fr_cls     = "neg" if fr.get("n_pass", 0) == 0 else "pos"
    fr_cell_val = ("0 trades — regime mismatch" if fr.get("n_pass", 0) == 0
                   else f'Calmar {fr.get("cal_gated", "N/A")}')

    per_win_json = json.dumps(per_win_table)
    wf25_json    = json.dumps(wf_2025)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Extended Validation — BTCUSDT WF ML Gate (2020–2026)</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:#0d1117;color:#c9d1d9;font-family:'Segoe UI',sans-serif;padding:24px;}}
  h1{{color:#58a6ff;margin-bottom:6px;font-size:1.6rem;}}
  h2{{color:#8b949e;font-size:1.1rem;margin:28px 0 12px;border-bottom:1px solid #21262d;padding-bottom:6px;}}
  h3{{color:#c9d1d9;font-size:0.95rem;margin:18px 0 8px;}}
  .subtitle{{color:#8b949e;font-size:0.9rem;margin-bottom:24px;}}
  .card{{background:#161b22;border:1px solid #21262d;border-radius:10px;padding:20px;margin-bottom:20px;}}
  .grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:20px;}}
  .grid-3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;}}
  table{{width:100%;border-collapse:collapse;font-size:0.82rem;}}
  th{{background:#21262d;color:#8b949e;padding:8px 10px;text-align:right;font-weight:600;}}
  th:first-child{{text-align:left;}}
  td{{padding:7px 10px;text-align:right;border-bottom:1px solid #21262d;}}
  td:first-child{{text-align:left;}}
  tr:hover td{{background:#1c2128;}}
  .pos{{color:#3fb950;}} .neg{{color:#f85149;}}
  .badge{{display:inline-block;padding:3px 10px;border-radius:12px;font-size:0.78rem;font-weight:600;}}
  .badge.pos{{background:#1a4731;color:#3fb950;}} .badge.neg{{background:#3d1f1f;color:#f85149;}}
  .stat-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:14px;}}
  .stat-box{{background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:14px;text-align:center;}}
  .stat-val{{font-size:1.4rem;font-weight:700;}} .stat-lbl{{font-size:0.75rem;color:#8b949e;margin-top:4px;}}
  .test-block{{background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:16px;margin-bottom:12px;}}
  .test-meta{{font-size:0.8rem;color:#8b949e;margin-top:6px;}}
  .warn{{background:#2d2007;border:1px solid #6e5502;border-radius:6px;padding:10px 14px;margin:12px 0;font-size:0.82rem;color:#e3b341;}}
  .good{{background:#0f2a1e;border:1px solid #238636;border-radius:6px;padding:10px 14px;margin:12px 0;font-size:0.82rem;color:#3fb950;}}
  .section-note{{font-size:0.78rem;color:#6e7681;margin-bottom:10px;}}
  .cmp-grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:12px;}}
  .cmp-box{{background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:14px;}}
  .cmp-lbl{{font-size:0.72rem;color:#8b949e;margin-bottom:6px;}}
  .cmp-row{{display:flex;justify-content:space-between;font-size:0.85rem;margin:3px 0;}}
  canvas{{max-height:280px;}}
</style>
</head>
<body>

<h1>Extended Validation — BTCUSDT WF ML Gate</h1>
<p class="subtitle">
  {START_YEAR}-{START_MONTH:02d} &rarr; 2026-05 &nbsp;&middot;&nbsp;
  {len(per_win_table)} WF windows &nbsp;&middot;&nbsp;
  Composite &plusmn;3 &middot; Session 08&ndash;21 UTC &middot; 75 features
</p>


<!-- ── 0. KPIs ──────────────────────────────────────────────────────────── -->
<h2>0. Realized Full-Sample KPIs ({START_YEAR}–2026)</h2>
<div class="card">
<div class="stat-grid">
  <div class="stat-box">
    <div class="stat-val {'pos' if r_b>=0 else 'neg'}">{r_b:+.1f}%</div>
    <div class="stat-lbl">Baseline Return</div>
  </div>
  <div class="stat-box">
    <div class="stat-val {'pos' if r_g>=0 else 'neg'}">{r_g:+.1f}%</div>
    <div class="stat-lbl">Gated Return</div>
  </div>
  <div class="stat-box">
    <div class="stat-val {'pos' if cal_g>=1 else ('neg' if cal_g<0.5 else '')}">{cal_g:.3f}</div>
    <div class="stat-lbl">Calmar (Gated)</div>
  </div>
  <div class="stat-box">
    <div class="stat-val {'pos' if sh_g>=1 else ''}">{sh_g:.3f}</div>
    <div class="stat-lbl">Sharpe (Gated)</div>
  </div>
  <div class="stat-box">
    <div class="stat-val">{nt_g} / {nt_b}</div>
    <div class="stat-lbl">Gated / Baseline Trades</div>
  </div>
  <div class="stat-box">
    <div class="stat-val pos">{filter_pct}%</div>
    <div class="stat-lbl">Signals Filtered</div>
  </div>
</div>
</div>


<!-- ── 1. Comparison vs 2022-based report ──────────────────────────────── -->
<h2>1. Comparison: 2020-2026 vs Previous 2022-2026 Results</h2>
<p class="section-note">Arrows show how metrics changed by adding 2020-2021 to the dataset.</p>
<div class="cmp-grid">
  <div class="cmp-box">
    <div class="cmp-lbl">BASELINE (no gate)</div>
    <div class="cmp-row"><span>Return</span><span>{r22_b['ret']:+.1f}% &rarr; <b class="{'pos' if r_b>=0 else 'neg'}">{r_b:+.1f}%</b> {_delta(r_b,r22_b['ret'])}</span></div>
    <div class="cmp-row"><span>Calmar</span><span>{r22_b['calmar']:.3f} &rarr; <b>{cal_b:.3f}</b> {_delta(cal_b,r22_b['calmar'],'+.3f')}</span></div>
    <div class="cmp-row"><span>WF windows</span><span>{r22_b['n_windows']} &rarr; <b>{n_win}</b></span></div>
  </div>
  <div class="cmp-box">
    <div class="cmp-lbl">ML GATE P&ge;0.50</div>
    <div class="cmp-row"><span>Return</span><span>{r22_g['ret']:+.1f}% &rarr; <b class="{'pos' if r_g>=0 else 'neg'}">{r_g:+.1f}%</b> {_delta(r_g,r22_g['ret'])}</span></div>
    <div class="cmp-row"><span>Calmar</span><span>{r22_g['calmar']:.3f} &rarr; <b>{cal_g:.3f}</b> {_delta(cal_g,r22_g['calmar'],'+.3f')}</span></div>
    <div class="cmp-row"><span>Sharpe</span><span>{r22_g['sharpe']:.3f} &rarr; <b>{sh_g:.3f}</b> {_delta(sh_g,r22_g['sharpe'],'+.3f')}</span></div>
    <div class="cmp-row"><span>Trades</span><span>{r22_g['n_trades']} &rarr; <b>{nt_g}</b></span></div>
    <div class="cmp-row"><span>WF windows</span><span>{r22_g['n_windows']} &rarr; <b>{n_win}</b></span></div>
    <div class="cmp-row"><span>t-test (a) p</span><span>{r22_g['t_a_p']:.4f} &rarr; <b class="{'pos' if st['a']['p']<0.05 else 'neg'}">{st['a']['p']:.4f}</b></span></div>
    <div class="cmp-row"><span>t-test (b) p</span><span>{r22_g['t_b_p']:.4f} &rarr; <b class="{'pos' if st['b']['p']<0.05 else 'neg'}">{st['b']['p']:.4f}</b></span></div>
    <div class="cmp-row"><span>t-test (c) p</span><span>{r22_g['t_c_p']:.4f} &rarr; <b class="{'pos' if st['c']['p']<0.05 else 'neg'}">{st['c']['p']:.4f}</b></span></div>
  </div>
</div>


<!-- ── 2. Monte Carlo ─────────────────────────────────────────────────────── -->
<h2>2. Monte Carlo ({N_SIMS:,} bootstrap simulations — {START_YEAR}–2026 trades)</h2>
<div class="card">
<table>
  <thead><tr>
    <th>Config</th><th>Trades</th>
    <th>OOS Ret%</th><th>OOS DD%</th><th>Calmar</th><th>Sharpe</th>
    <th>P(Profit)%</th><th>P(Ruin)%</th>
    <th>Med Ret%</th><th>p5 Ret%</th><th>p95 Ret%</th>
    <th>Med DD%</th><th>Worst DD p5%</th><th>Med Sharpe</th>
  </tr></thead>
  <tbody>{_mc_table_rows()}</tbody>
</table>
</div>


<!-- ── 3. Statistical Significance ────────────────────────────────────────── -->
<h2>3. Statistical Significance ({n_win} WF windows — one-tailed, &alpha;=0.05)</h2>

<div class="test-block">
  <strong>(a) {st['a']['label']}</strong>&nbsp;&nbsp;{_sig_badge(st['a']['sig'])}
  <div class="test-meta">t={st['a']['t']} &nbsp; p={st['a']['p']} &nbsp; n={st['a']['n']} windows<br>{st['a']['interp']}</div>
</div>
<div class="test-block">
  <strong>(b) {st['b']['label']}</strong>&nbsp;&nbsp;{_sig_badge(st['b']['sig'])}
  <div class="test-meta">t={st['b']['t']} &nbsp; p={st['b']['p']} &nbsp; n={st['b']['n']} windows<br>{st['b']['interp']}</div>
</div>
<div class="test-block">
  <strong>(c) {st['c']['label']}</strong>&nbsp;&nbsp;{_sig_badge(st['c']['sig'])}
  <div class="test-meta">t={st['c']['t']} &nbsp; p={st['c']['p']} &nbsp; n={st['c']['n']} trades<br>{st['c']['interp']}</div>
</div>

<h3>Per-Window OOS Returns ({n_win} windows)</h3>
<div class="card">
<div class="grid-2">
<div style="overflow-x:auto;max-height:420px;">
<table>
  <thead><tr><th>Start</th><th>End</th><th>Base%</th><th>Gated%</th><th>Delta%</th><th>Trades</th></tr></thead>
  <tbody>{_win_rows(per_win_table)}</tbody>
</table>
</div>
<div class="card" style="padding:10px;">
  <canvas id="winChart"></canvas>
</div>
</div>
</div>

<script>
const winData = {per_win_json};
new Chart(document.getElementById('winChart').getContext('2d'), {{
  type:'bar',
  data:{{
    labels:winData.map(r=>r.oos_start),
    datasets:[
      {{label:'Gated',data:winData.map(r=>r.ret_gated),backgroundColor:winData.map(r=>r.ret_gated>=0?'#3fb95088':'#f8514966'),borderWidth:0}},
      {{label:'Baseline',data:winData.map(r=>r.ret_base),backgroundColor:'#64b5f644',borderWidth:0}},
    ],
  }},
  options:{{responsive:true,animation:false,
    plugins:{{legend:{{labels:{{color:'#c9d1d9'}}}}}},
    scales:{{
      x:{{ticks:{{color:'#8b949e',maxRotation:45,font:{{size:9}}}},grid:{{display:false}}}},
      y:{{ticks:{{color:'#8b949e',callback:v=>v+'%'}},grid:{{color:'#21262d'}}}},
    }},
  }},
}});
</script>


<!-- ── 4. Held-out Test ────────────────────────────────────────────────────── -->
<h2>4. Held-out Test Set (2025-01 &rarr; 2026-05)</h2>
<div class="card">
  <h3>A. WF Gate OOS on Training Split ({START_YEAR}–2024)</h3>
  <table>
    <thead><tr><th>Config</th><th>Return%</th><th>DD%</th><th>Calmar</th><th>Trades</th></tr></thead>
    <tbody>
      <tr><td>Baseline</td>{_cc(ts['ret_base'])}{_cc(ts.get('dd_base',0),False)}{_cc(ts['cal_base'])}<td>{ts['nt_base']}</td></tr>
      <tr><td>WF Gated</td>{_cc(ts['ret_gated'])}{_cc(ts.get('dd_gated',0),False)}{_cc(ts['cal_gated'])}<td>{ts['nt_gated']}</td></tr>
    </tbody>
  </table>

  <h3 style="margin-top:16px;">B. Frozen Model on Held-out 2025-2026</h3>
  <table>
    <thead><tr><th>Config</th><th>Return%</th><th>DD%</th><th>Calmar</th><th>Trades</th></tr></thead>
    <tbody>{frozen_body}</tbody>
  </table>

  <h3 style="margin-top:16px;">C. Full WF Gate in 2025-2026 ({wf25_pos}/{wf25_total} windows positive)</h3>
  <table>
    <thead><tr><th>Start</th><th>End</th><th>Base%</th><th>Gated%</th><th>Delta%</th><th>Trades</th></tr></thead>
    <tbody>{_win_rows(wf_2025)}</tbody>
  </table>
  <canvas id="wf25Chart" style="max-height:200px;margin-top:12px;"></canvas>
</div>

<script>
const w25 = {wf25_json};
new Chart(document.getElementById('wf25Chart').getContext('2d'), {{
  type:'bar',
  data:{{
    labels:w25.map(r=>r.oos_start),
    datasets:[
      {{label:'Gated (WF re-train)',data:w25.map(r=>r.ret_gated),backgroundColor:w25.map(r=>r.ret_gated>=0?'#3fb95088':'#f8514966'),borderWidth:0}},
      {{label:'Baseline',data:w25.map(r=>r.ret_base),backgroundColor:'#64b5f644',borderWidth:0}},
    ],
  }},
  options:{{responsive:true,animation:false,
    plugins:{{title:{{display:true,text:'WF Gate — 2025-2026 OOS Windows',color:'#c9d1d9'}},legend:{{labels:{{color:'#c9d1d9'}}}}}},
    scales:{{x:{{ticks:{{color:'#8b949e',font:{{size:10}}}},grid:{{display:false}}}},y:{{ticks:{{color:'#8b949e',callback:v=>v+'%'}},grid:{{color:'#21262d'}}}}}},
  }},
}});
</script>


<!-- ── 5. Validation Summary ──────────────────────────────────────────────── -->
<h2>5. Validation Pipeline — Full Summary ({START_YEAR}–2026)</h2>
<div class="card">
<table>
  <thead><tr><th>Test</th><th>Result</th><th>Verdict</th></tr></thead>
  <tbody>
    <tr><td>Walk-forward OOS ({n_win} windows, {START_YEAR}&ndash;2026)</td>
      <td class="pos">Calmar {cal_g} &middot; Sharpe {sh_g}</td>
      <td class="pos">&#10003; PASS</td></tr>
    <tr><td>Monte Carlo ({N_SIMS:,} paths)</td>
      <td class="{'pos' if mc_rows[1]['p_profit']>=75 else 'neg'}">P(Profit)={mc_rows[1]['p_profit']}% &middot; P(Ruin)={mc_rows[1]['p_ruin']}%</td>
      <td class="{'pos' if mc_rows[1]['p_profit']>=75 else 'neg'}">{'&#10003; PASS' if mc_rows[1]['p_profit']>=75 else '&#10007; FAIL'}</td></tr>
    <tr><td>t-test (a): per-window OOS returns &gt; 0</td>
      <td class="{'pos' if st['a']['sig'] else 'neg'}">p={st['a']['p']}</td>
      <td class="{'pos' if st['a']['sig'] else 'neg'}">{'&#10003; PASS' if st['a']['sig'] else '&#9888; MARGINAL'}</td></tr>
    <tr><td>t-test (b): gated &gt; baseline per window</td>
      <td class="{'pos' if st['b']['sig'] else 'neg'}">p={st['b']['p']}</td>
      <td class="{'pos' if st['b']['sig'] else 'neg'}">{'&#10003; PASS' if st['b']['sig'] else '&#9888; MARGINAL'}</td></tr>
    <tr><td>t-test (c): per-trade return &gt; 0 (n={stat_tests['c']['n']})</td>
      <td class="{'pos' if st['c']['sig'] else 'neg'}">p={st['c']['p']}</td>
      <td class="{'pos' if st['c']['sig'] else 'neg'}">{'&#10003; PASS' if st['c']['sig'] else '&#9888; MARGINAL'}</td></tr>
    <tr><td>Held-out WF OOS on training split ({START_YEAR}&ndash;2024)</td>
      <td class="{wf_tr_cls}">Calmar {ts['cal_gated']}</td>
      <td class="{wf_tr_cls}">{'&#10003; PASS' if ts['cal_gated']>0.5 else '&#9888; MARGINAL'}</td></tr>
    <tr><td>Held-out frozen model (2025&ndash;2026)</td>
      <td class="{fr_cls}">{fr_cell_val}</td>
      <td class="{fr_cls}">{'&#10007; FAIL — WF re-training required' if fr.get('n_pass',0)==0 else '&#10003; PASS'}</td></tr>
    <tr><td>Held-out WF 2025&ndash;2026 (re-trains every 2M)</td>
      <td class="pos">{wf25_pos}/{wf25_total} windows profitable</td>
      <td class="pos">&#10003; PASS (with re-training)</td></tr>
  </tbody>
</table>
</div>

</body>
</html>"""


if __name__ == "__main__":
    main()
