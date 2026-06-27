"""
test_ny_orb_scan.py
===================
NY Opening Range Breakout su BTCUSDT 15M.

Setup:
  1. NY open = 14:30 UTC (09:30 AM ET)
  2. Prima candela 15M: barra 14:30–14:44 UTC
     Body range: body_hi = max(open, close),  body_lo = min(open, close)
  3. Breakout:  prima barra 15M successiva (14:45+) che chiude
                sopra body_hi → LONG
                sotto body_lo → SHORT
                (una sola direzione per giornata, la prima che scatta)
  4. Entry:    close della breakout bar (approssimazione next-bar open)
  5. SL:       body_lo - sl_buf × ATR_1H  (LONG)
               body_hi + sl_buf × ATR_1H  (SHORT)
  6. TP:       entry + tp_frac × body_range  (LONG)
               entry - tp_frac × body_range  (SHORT)
  7. Max hold: 32 barre 15M (~8h) oppure 20:00 UTC (fine sessione NY cash)

IC del segnale:
  Calcolato come correlazione di Spearman tra direzione segnale (+1/-1)
  e forward return cumulato sulle successive 16 barre 15M (= 4h).

Parametri scansionati:
  tp_frac  ∈ {0.5, 1.0, 1.5, 2.0, 3.0}
  sl_buf   ∈ {0.00, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00}
  rba_max  ∈ {999, 2.0, 1.5, 1.0}   (body_range / ATR_1H ≤ soglia)
"""
from __future__ import annotations

import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from src.strategy.data_fetcher import fetch_extended_data
from src.strategy.indicators   import add_indicators

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
START_YEAR   = 2020
NY_HOUR      = 14       # 14:30 UTC = 09:30 AM ET
NY_MIN       = 30
NY_END_HOUR  = 20       # 20:00 UTC ≈ fine NY cash session
MAX_HOLD     = 32       # barre 15M massime per trade (~8h)
IC_HORIZON   = 16       # barre 15M per calcolo IC (= 4h)
MIN_BODY_PTS = 10.0     # body minimo in USD (scarta doji)

TP_FRAC_GRID = [0.5, 1.0, 1.5, 2.0, 3.0]
SL_BUF_GRID  = [0.00, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00]
RBA_MAX_GRID = [999, 2.0, 1.5, 1.0]

SEP  = "─" * 82
SEP2 = "═" * 82


# ─────────────────────────────────────────────────────────────────────────────
# 1. Caricamento dati
# ─────────────────────────────────────────────────────────────────────────────
print(SEP2)
print("NY ORB Scan — First-Candle Body Breakout | BTCUSDT 15M | 2020-2026")
print(SEP2)
print("\n[1/4] Caricamento dati …")
raw   = fetch_extended_data(start_year=START_YEAR, start_month=1,
                             fetch_15m=True, fetch_1m=False, fetch_flow=False)
df_1h  = add_indicators(raw["1H"])
df_15m = add_indicators(raw["15M"])
print(f"  1H : {len(df_1h):,} bar  ({df_1h.index[0].date()} → {df_1h.index[-1].date()})")
print(f"  15M: {len(df_15m):,} bar")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Array numpy
# ─────────────────────────────────────────────────────────────────────────────
IDX    = df_15m.index
H_arr  = IDX.hour
M_arr  = IDX.minute
HI     = df_15m["high"].values
LO     = df_15m["low"].values
CL     = df_15m["close"].values
OP     = df_15m["open"].values
N      = len(df_15m)
yr_arr = np.array([t.year for t in IDX], dtype=int)

# ATR 1H dalla barra precedente completa (evita look-ahead)
print("  Allineamento ATR 1H → 15M (barra 1H precedente) …")
prev_1h    = IDX.floor("h") - pd.Timedelta("1h")
atr_1h_map = df_1h["atr_14"].clip(lower=1.0).to_dict()
fallback   = df_15m["atr_14"].clip(lower=1.0).values
ATR_1H     = np.array([atr_1h_map.get(t, np.nan) for t in prev_1h], dtype=float)
ATR_1H     = np.where(np.isnan(ATR_1H), fallback, ATR_1H)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Raccolta eventi NY ORB
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/4] Raccolta eventi NY ORB (breakout body prima candela) …")

events: list[dict] = []
seen_dates: set = set()

for i in range(N - MAX_HOLD - 2):
    # Prima candela NY = barra esattamente a 14:30 UTC
    if H_arr[i] != NY_HOUR or M_arr[i] != NY_MIN:
        continue
    d = IDX[i].normalize()
    if d in seen_dates:
        continue
    seen_dates.add(d)

    op_i    = OP[i]
    cl_i    = CL[i]
    body_hi = max(op_i, cl_i)
    body_lo = min(op_i, cl_i)
    rw      = body_hi - body_lo
    if rw < MIN_BODY_PTS:
        continue

    atr_i = ATR_1H[i]
    if np.isnan(atr_i) or atr_i <= 0:
        continue
    rba = rw / atr_i

    # Cerca breakout dalla barra successiva fino a NY_END o MAX_HOLD
    for j in range(i + 1, min(i + MAX_HOLD + 1, N - MAX_HOLD - 1)):
        if IDX[j].hour >= NY_END_HOUR:
            break

        direction = None
        if CL[j] > body_hi:
            direction = "long"
        elif CL[j] < body_lo:
            direction = "short"

        if direction is not None:
            ei = j          # entry bar (use close of breakout bar as entry px)
            events.append({
                "direction": direction,
                "brk_i":    j,
                "entry_i":  ei,
                "entry_px": CL[j],
                "body_hi":  body_hi,
                "body_lo":  body_lo,
                "rw":       rw,
                "atr_1h":   atr_i,
                "rba":      rba,
                "year":     yr_arr[i],
                "ny_bar":   i,
            })
            break  # una sola direzione per giornata

print(f"  Totale eventi: {len(events)}")
print(f"    Long:  {sum(1 for e in events if e['direction']=='long')}")
print(f"    Short: {sum(1 for e in events if e['direction']=='short')}")
if events:
    rbas = [e["rba"] for e in events]
    print(f"    rba   mean={np.mean(rbas):.2f}  p25={np.percentile(rbas,25):.2f}"
          f"  p50={np.percentile(rbas,50):.2f}  p75={np.percentile(rbas,75):.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# IC del segnale grezzo (prima del filtro rba)
# IC = Spearman(direction, forward_return_4h)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/4] Calcolo IC del segnale grezzo (4H forward return) …")
signals  = []
fwd_rets = []
for ev in events:
    ei   = ev["entry_i"]
    end  = min(ei + IC_HORIZON, N - 1)
    fwd  = (CL[end] - ev["entry_px"]) / ev["entry_px"] * 100.0
    sig  = 1.0 if ev["direction"] == "long" else -1.0
    signals.append(sig)
    fwd_rets.append(sig * fwd)   # signed return (positivo = segnale corretto)

if len(signals) >= 10:
    ic, ic_p = st.spearmanr(signals, fwd_rets)
    pos_frac  = sum(1 for x in fwd_rets if x > 0) / len(fwd_rets)
    print(f"  N segnali        : {len(signals)}")
    print(f"  IC (Spearman)    : {ic:.4f}  (p={ic_p:.4f})")
    print(f"  Hit rate (≥0%)   : {pos_frac:.1%}")
    print(f"  Mean signed ret  : {np.mean(fwd_rets):.3f}%")
else:
    print("  Dati insufficienti per IC.")


# ─────────────────────────────────────────────────────────────────────────────
# Pre-cache price paths (una volta sola per tutti i parametri)
# Entry = close di j; path inizia da j+1 (barra successiva all'entry)
# ─────────────────────────────────────────────────────────────────────────────
event_paths: list[tuple] = []
for ev in events:
    start = ev["entry_i"] + 1   # scan starts from the bar AFTER entry bar
    event_paths.append((
        HI[start : start + MAX_HOLD],
        LO[start : start + MAX_HOLD],
    ))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Grid scan TP × SL × rba_max
# ─────────────────────────────────────────────────────────────────────────────
n_combos = len(TP_FRAC_GRID) * len(SL_BUF_GRID) * len(RBA_MAX_GRID)
print(f"\n[4/4] Scanning {n_combos} combinazioni "
      f"({len(TP_FRAC_GRID)}×{len(SL_BUF_GRID)}×{len(RBA_MAX_GRID)}) …\n")

results = []

for tp_frac, sl_buf, rba_max in product(TP_FRAC_GRID, SL_BUF_GRID, RBA_MAX_GRID):
    wins = losses = n_valid = 0
    total_tp_dist = total_sl_dist = 0.0
    yearly: dict[int, dict] = {}

    for ev, (ph, pl) in zip(events, event_paths):
        if ev["rba"] > rba_max:
            continue

        entry  = ev["entry_px"]
        rw     = ev["rw"]
        atr_sw = ev["atr_1h"]

        if ev["direction"] == "long":
            tp_px = entry + tp_frac * rw
            sl_px = ev["body_lo"] - sl_buf * atr_sw
            if sl_px >= entry or tp_px <= entry:
                continue
            sl_dist = entry - sl_px
            tp_dist = tp_px - entry
            hit_tp = hit_sl = False
            for ph_bar, pl_bar in zip(ph, pl):
                if ph_bar >= tp_px and not hit_sl:
                    hit_tp = True; break
                if pl_bar <= sl_px:
                    hit_sl = True; break
        else:
            tp_px = entry - tp_frac * rw
            sl_px = ev["body_hi"] + sl_buf * atr_sw
            if sl_px <= entry or tp_px >= entry:
                continue
            sl_dist = sl_px - entry
            tp_dist = entry - tp_px
            hit_tp = hit_sl = False
            for ph_bar, pl_bar in zip(ph, pl):
                if pl_bar <= tp_px and not hit_sl:
                    hit_tp = True; break
                if ph_bar >= sl_px:
                    hit_sl = True; break

        if sl_dist <= 0 or tp_dist <= 0:
            continue

        n_valid += 1
        total_tp_dist += tp_dist / entry * 100.0
        total_sl_dist += sl_dist / entry * 100.0
        yr = ev["year"]
        if yr not in yearly:
            yearly[yr] = {"w": 0, "n": 0}
        yearly[yr]["n"] += 1

        if hit_tp:
            wins += 1
            yearly[yr]["w"] += 1
        elif hit_sl:
            losses += 1

    if n_valid < 10:
        continue

    wr     = wins / n_valid * 100
    rr     = (total_tp_dist / n_valid) / (total_sl_dist / n_valid) if total_sl_dist > 0 else 0
    be     = 1 / (1 + rr) * 100 if rr > 0 else 50.0
    avg_tp = total_tp_dist / n_valid
    avg_sl = total_sl_dist / n_valid
    exp_pnl = (wr / 100 * avg_tp) - ((1 - wr / 100) * avg_sl)
    margin  = wr - be
    p_val   = st.binomtest(int(round(wr / 100 * n_valid)), n_valid,
                           be / 100, alternative="greater").pvalue
    # Positive years fraction
    pos_yrs = sum(1 for v in yearly.values() if v["n"] > 0
                  and v["w"] / v["n"] > be / 100) / max(len(yearly), 1)

    results.append({
        "tp_frac": tp_frac,
        "sl_buf":  sl_buf,
        "rba_max": rba_max,
        "n":       n_valid,
        "wr":      round(wr, 2),
        "rr":      round(rr, 2),
        "be_wr":   round(be, 2),
        "margin":  round(margin, 2),
        "exp_pnl": round(exp_pnl, 4),
        "p_val":   round(p_val, 4),
        "pos_yrs": round(pos_yrs, 2),
        "yearly":  yearly,
    })

df_res = pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# Output — top combos (nessun filtro rba)
# ─────────────────────────────────────────────────────────────────────────────
def _print_top(label: str, subset: pd.DataFrame, n: int = 20) -> None:
    print(f"\n{SEP2}")
    print(label)
    print(SEP2)
    top = subset.sort_values("exp_pnl", ascending=False).head(n)
    print(f"{'tp_frac':>8} {'sl_buf':>7} {'rba_max':>8} {'N':>6} "
          f"{'WR%':>7} {'R:R':>6} {'BE%':>6} {'Margin':>8} "
          f"{'ExpPnL%':>9} {'p-val':>7} {'PosYrs':>7}")
    print(SEP)
    for _, r in top.iterrows():
        print(f"{r.tp_frac:>8.2f} {r.sl_buf:>7.2f} "
              f"{'∞' if r.rba_max > 100 else r.rba_max:>8} "
              f"{int(r.n):>6} {r.wr:>7.2f} {r.rr:>6.2f} {r.be_wr:>6.2f} "
              f"{r.margin:>+8.2f} {r.exp_pnl:>+9.4f} {r.p_val:>7.4f} "
              f"{r.pos_yrs:>7.0%}")
    print(SEP)


_print_top("TOP 20 — nessun filtro rba", df_res[df_res.rba_max > 100])
_print_top("TOP 20 — rba_max ≤ 2.0",    df_res[df_res.rba_max == 2.0])
_print_top("TOP 20 — rba_max ≤ 1.5",    df_res[df_res.rba_max == 1.5])
_print_top("TOP 20 — rba_max ≤ 1.0",    df_res[df_res.rba_max == 1.0])


# ─────────────────────────────────────────────────────────────────────────────
# Matrice WR%  per tp_frac × sl_buf  (rba_max = best o ∞)
# ─────────────────────────────────────────────────────────────────────────────
def _print_matrix(label: str, subset: pd.DataFrame,
                  col: str = "exp_pnl") -> None:
    print(f"\n{SEP2}")
    print(f"Matrice {col}  —  {label}")
    print(SEP2)
    pivot = subset.pivot_table(index="tp_frac", columns="sl_buf",
                               values=col, aggfunc="mean")
    print(pivot.to_string(float_format=lambda x: f"{x:+.4f}"))

for rba_val in [999, 1.5, 1.0]:
    sub = df_res[(df_res.rba_max == rba_val) | (df_res.rba_max > 100)]
    sub = df_res[df_res.rba_max == rba_val] if rba_val != 999 \
        else df_res[df_res.rba_max > 100]
    _print_matrix(f"rba_max={'∞' if rba_val > 100 else rba_val}", sub, "exp_pnl")
    _print_matrix(f"rba_max={'∞' if rba_val > 100 else rba_val}", sub, "wr")


# ─────────────────────────────────────────────────────────────────────────────
# Effetto sl_buf (a tp_frac migliore)
# ─────────────────────────────────────────────────────────────────────────────
best_tp = (df_res[df_res.rba_max > 100]
           .groupby("tp_frac")["exp_pnl"].mean()
           .idxmax())
print(f"\n{SEP2}")
print(f"Effetto sl_buf a tp_frac={best_tp}  (rba_max=∞)")
print(SEP2)
sub_tp = df_res[(df_res.rba_max > 100) & (df_res.tp_frac == best_tp)]
print(f"{'sl_buf':>8} {'N':>6} {'WR%':>7} {'R:R':>6} {'BE%':>6} "
      f"{'Margin':>8} {'ExpPnL%':>9} {'p-val':>7}")
print(SEP)
for _, r in sub_tp.sort_values("sl_buf").iterrows():
    print(f"{r.sl_buf:>8.2f} {int(r.n):>6} {r.wr:>7.2f} {r.rr:>6.2f} "
          f"{r.be_wr:>6.2f} {r.margin:>+8.2f} {r.exp_pnl:>+9.4f} {r.p_val:>7.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Best combo summary
# ─────────────────────────────────────────────────────────────────────────────
best = df_res.sort_values("exp_pnl", ascending=False).iloc[0]
print(f"\n{SEP2}")
print("BEST COMBO (massimo ExpPnL%)")
print(SEP2)
print(f"  tp_frac  = {best.tp_frac}")
print(f"  sl_buf   = {best.sl_buf}")
print(f"  rba_max  = {'∞' if best.rba_max > 100 else best.rba_max}")
print(f"  N        = {int(best.n)}")
print(f"  WR%      = {best.wr:.2f}%")
print(f"  R:R      = {best.rr:.2f}")
print(f"  BE WR%   = {best.be_wr:.2f}%")
print(f"  Margin   = {best.margin:+.2f}pp")
print(f"  ExpPnL%  = {best.exp_pnl:+.4f}%")
print(f"  p-value  = {best.p_val:.4f}")

print(f"\n  → usa questi parametri in create_ny_orb_report.py")
print(SEP2)
