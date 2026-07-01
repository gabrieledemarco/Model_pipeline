"""
create_orb_research_report.py
──────────────────────────────
Open Range Breakout — Ricerca Statistica sulle Sessioni di Mercato BTCUSDT.

Sessioni definite (UTC, non-overlapping):
  Asia   : 00:00 – 07:59  (ore 0-7)
  London : 08:00 – 15:59  (ore 8-15)
  New York: 16:00 – 23:59  (ore 16-23)

Open Range per sessione = primi 2 barre 1H (prime 2 ore).

Moduli di ricerca
─────────────────
1. Session Statistics          — range, return, volatility per sessione
2. Directional Correlation     — contingency table Asia→EU→NY + chi-squared
3. Open Range Breakout (ORB)   — success rate e profilo di payoff
4. Asia Range Spillover        — effetto ampiezza range asiatico su sessioni successive
5. Hour-of-Day Bias            — return medio per ora UTC (tutte le 24)
6. Sequential Patterns         — catene a 3 sessioni (8 combinazioni)
7. Conditional Probabilities   — tabella completa P(direzione | precedente)

Output → reports/report_orb_research.html
"""
from __future__ import annotations

import base64
import io
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

from src.strategy.data_fetcher import fetch_extended_data

# ─────────────────────────────────────────────────────────────────────────────
# Costanti
# ─────────────────────────────────────────────────────────────────────────────
SESSIONS = {
    "Asia":   (0,  8),   # [0, 8)
    "London": (8,  16),  # [8, 16)
    "NY":     (16, 24),  # [16, 24)
}
ORB_HOURS = 2   # durata Open Range in ore (barre 1H)

# Colori dark-theme
_BG   = "#0f1117"
_CARD = "#12151f"
_GRID = "#1e2130"
_TEXT = "#e0e0e0"
_C = {
    "Asia":   "#42a5f5",
    "London": "#66bb6a",
    "NY":     "#ef5350",
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers plotting
# ─────────────────────────────────────────────────────────────────────────────

def _style_ax(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    ax.set_facecolor(_CARD)
    ax.tick_params(colors=_TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(_GRID)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    if title:
        ax.set_title(title, color=_TEXT, fontsize=9, pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, color=_GRID, linewidth=0.5, alpha=0.7)


def _to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=_BG, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _img(b64: str, w: int = 700) -> str:
    return f'<img src="data:image/png;base64,{b64}" width="{w}" style="border-radius:6px;margin:8px 0">'


# ─────────────────────────────────────────────────────────────────────────────
# 1. Costruzione aggregati per sessione
# ─────────────────────────────────────────────────────────────────────────────

def build_session_df(df_1h: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Per ogni sessione, aggrega le barre 1H in un DataFrame giornaliero con:
    date, open, high, low, close, volume, return (%), range ($), range_pct (%)
    orb_high, orb_low  (massimo/minimo delle prime ORB_HOURS barre)
    direction: +1 (bullish) / -1 (bearish)
    """
    results: dict[str, pd.DataFrame] = {}

    for sess, (h_start, h_end) in SESSIONS.items():
        mask = (df_1h.index.hour >= h_start) & (df_1h.index.hour < h_end)
        df_s = df_1h[mask].copy()

        # Raggruppa per data UTC
        date_col = df_s.index.date
        df_s = df_s.copy()
        df_s["date"] = date_col

        def agg_session(g):
            g = g.sort_index()
            orb  = g.iloc[:ORB_HOURS]
            row = {
                "open":      g["open"].iloc[0],
                "high":      g["high"].max(),
                "low":       g["low"].min(),
                "close":     g["close"].iloc[-1],
                "volume":    g["volume"].sum(),
                "orb_high":  orb["high"].max(),
                "orb_low":   orb["low"].min(),
                "n_bars":    len(g),
            }
            row["ret_pct"]   = (row["close"] / row["open"] - 1.0) * 100
            row["range_abs"] = row["high"] - row["low"]
            row["range_pct"] = row["range_abs"] / row["open"] * 100
            row["orb_range"] = row["orb_high"] - row["orb_low"]
            row["direction"] = 1 if row["ret_pct"] >= 0 else -1
            return pd.Series(row)

        sess_daily = df_s.groupby("date").apply(agg_session).reset_index()
        sess_daily["date"] = pd.to_datetime(sess_daily["date"])
        # Keep only complete sessions (all 8 bars present)
        sess_daily = sess_daily[sess_daily["n_bars"] == (h_end - h_start)].copy()
        results[sess] = sess_daily.set_index("date")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. ORB — analisi breakout intra-sessione
# ─────────────────────────────────────────────────────────────────────────────

def orb_analysis(df_1h: pd.DataFrame) -> dict[str, dict]:
    """
    Per ogni sessione:
    - Definisce ORB = range delle prime ORB_HOURS barre
    - Classifica ogni giorno: ORB_LONG (prima rottura verso l'alto),
      ORB_SHORT (prima rottura verso il basso), NO_BREAK
    - Calcola: % di breakout up/down, % di False Breakout,
      return medio del resto della sessione dopo il breakout
    """
    results = {}

    for sess, (h_start, h_end) in SESSIONS.items():
        mask = (df_1h.index.hour >= h_start) & (df_1h.index.hour < h_end)
        df_s = df_1h[mask].copy()
        df_s["date"] = df_s.index.date

        records = []
        for date, g in df_s.groupby("date"):
            g = g.sort_index()
            if len(g) < (h_end - h_start):
                continue

            orb_bars = g.iloc[:ORB_HOURS]
            rem_bars = g.iloc[ORB_HOURS:]

            orb_h = orb_bars["high"].max()
            orb_l = orb_bars["low"].min()
            orb_r = orb_h - orb_l
            orb_open = orb_bars["open"].iloc[0]

            if rem_bars.empty or orb_r < 1.0:
                continue

            # Prima rottura nel resto della sessione
            first_break = "NONE"
            break_price = np.nan
            post_close  = g["close"].iloc[-1]

            for _, bar in rem_bars.iterrows():
                if bar["high"] > orb_h and first_break == "NONE":
                    first_break = "UP"
                    break_price = orb_h
                    break
                if bar["low"] < orb_l and first_break == "NONE":
                    first_break = "DOWN"
                    break_price = orb_l
                    break

            # Return dopo breakout: close sessione vs break price
            if first_break == "UP":
                post_ret = (post_close - orb_h) / orb_h * 100
                # False breakout: se dopo il breakout il close è sotto l'ORB low
                false_break = post_close < orb_l
            elif first_break == "DOWN":
                post_ret = (orb_l - post_close) / orb_l * 100   # positivo se continua
                false_break = post_close > orb_h
            else:
                post_ret   = 0.0
                false_break = False

            sess_ret = (g["close"].iloc[-1] / g["open"].iloc[0] - 1) * 100
            orb_ret  = (orb_bars["close"].iloc[-1] / orb_open - 1) * 100

            records.append({
                "date":        date,
                "orb_high":    orb_h,
                "orb_low":     orb_l,
                "orb_range":   orb_r,
                "orb_range_pct": orb_r / orb_open * 100,
                "first_break": first_break,
                "post_ret":    post_ret,
                "false_break": false_break,
                "sess_ret":    sess_ret,
                "orb_ret":     orb_ret,
            })

        df_orb = pd.DataFrame(records)
        total   = len(df_orb)
        n_up    = (df_orb["first_break"] == "UP").sum()
        n_dn    = (df_orb["first_break"] == "DOWN").sum()
        n_none  = (df_orb["first_break"] == "NONE").sum()

        orb_up   = df_orb[df_orb["first_break"] == "UP"]
        orb_dn   = df_orb[df_orb["first_break"] == "DOWN"]

        results[sess] = {
            "df":              df_orb,
            "total":           total,
            "pct_up":          n_up / total * 100,
            "pct_down":        n_dn / total * 100,
            "pct_none":        n_none / total * 100,
            "false_break_up":  (orb_up["false_break"].sum() / len(orb_up) * 100) if len(orb_up) > 0 else 0,
            "false_break_dn":  (orb_dn["false_break"].sum() / len(orb_dn) * 100) if len(orb_dn) > 0 else 0,
            "mean_post_ret_up": orb_up["post_ret"].mean() if len(orb_up) > 0 else 0,
            "mean_post_ret_dn": orb_dn["post_ret"].mean() if len(orb_dn) > 0 else 0,
            "med_orb_range_pct": df_orb["orb_range_pct"].median(),
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. Correlazione direzionale tra sessioni
# ─────────────────────────────────────────────────────────────────────────────

def directional_correlation(session_dfs: dict[str, pd.DataFrame]) -> dict:
    """
    Allinea Asia, London, NY sulla stessa data.
    Calcola contingency table e chi-squared per ogni coppia.
    """
    asia = session_dfs["Asia"][["direction", "ret_pct", "range_pct"]].rename(
        columns=lambda c: f"asia_{c}")
    lon  = session_dfs["London"][["direction", "ret_pct", "range_pct"]].rename(
        columns=lambda c: f"lon_{c}")
    ny   = session_dfs["NY"][["direction", "ret_pct", "range_pct"]].rename(
        columns=lambda c: f"ny_{c}")

    # NOTA: La sessione NY del giorno D è dopo London del giorno D.
    # Asia del giorno D precede London del giorno D, che precede NY del giorno D.
    combined = asia.join(lon, how="inner").join(ny, how="inner").dropna()

    def contingency(col_a, col_b, label_a, label_b):
        ct = pd.crosstab(combined[col_a], combined[col_b])
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        n = ct.values.sum()
        cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))

        # Conditional probabilities
        cond = {}
        for a_val, a_lbl in [(1, "Bull"), (-1, "Bear")]:
            sub = combined[combined[col_a] == a_val]
            cond[f"P({label_b} Bull | {label_a} {a_lbl})"] = (
                (sub[col_b] == 1).mean() * 100)
            cond[f"P({label_b} Bear | {label_a} {a_lbl})"] = (
                (sub[col_b] == -1).mean() * 100)

        return {
            "ct":        ct,
            "chi2":      chi2,
            "p":         p,
            "dof":       dof,
            "cramers_v": cramers_v,
            "cond":      cond,
            "n":         n,
        }

    return {
        "Asia→London": contingency("asia_direction", "lon_direction",  "Asia", "London"),
        "Asia→NY":     contingency("asia_direction", "ny_direction",   "Asia", "NY"),
        "London→NY":   contingency("lon_direction",  "ny_direction",   "London", "NY"),
        "combined":    combined,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Effetto ampiezza range asiatico
# ─────────────────────────────────────────────────────────────────────────────

def asia_range_effect(session_dfs: dict[str, pd.DataFrame]) -> dict:
    """
    Divide i giorni in quartili di range asiatico.
    Per ogni quartile: distribuzione dei return London e NY.
    """
    asia = session_dfs["Asia"][["range_pct", "direction"]].rename(
        columns=lambda c: f"asia_{c}")
    lon  = session_dfs["London"][["ret_pct", "direction"]].rename(
        columns=lambda c: f"lon_{c}")
    ny   = session_dfs["NY"][["ret_pct", "direction"]].rename(
        columns=lambda c: f"ny_{c}")

    comb = asia.join(lon, how="inner").join(ny, how="inner").dropna()

    q25, q50, q75 = comb["asia_range_pct"].quantile([0.25, 0.5, 0.75])
    comb["asia_range_q"] = pd.cut(
        comb["asia_range_pct"],
        bins=[-np.inf, q25, q50, q75, np.inf],
        labels=["Q1 Narrow", "Q2", "Q3", "Q4 Wide"],
    )

    summary = {}
    for q_label in ["Q1 Narrow", "Q2", "Q3", "Q4 Wide"]:
        sub = comb[comb["asia_range_q"] == q_label]
        summary[q_label] = {
            "n":          len(sub),
            "lon_bull_pct": (sub["lon_direction"] == 1).mean() * 100,
            "ny_bull_pct":  (sub["ny_direction"] == 1).mean() * 100,
            "lon_ret_mean": sub["lon_ret_pct"].mean(),
            "ny_ret_mean":  sub["ny_ret_pct"].mean(),
            "lon_ret_std":  sub["lon_ret_pct"].std(),
            "ny_ret_std":   sub["ny_ret_pct"].std(),
        }

    return {"comb": comb, "quartiles": summary, "q25": q25, "q50": q50, "q75": q75}


# ─────────────────────────────────────────────────────────────────────────────
# 5. Hour-of-Day bias
# ─────────────────────────────────────────────────────────────────────────────

def hour_of_day_bias(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Return medio, win rate e vol per ora UTC."""
    df = df_1h.copy()
    df["ret_pct"] = df["close"] / df["open"] * 100 - 100
    df["hour"]    = df.index.hour

    agg = df.groupby("hour").agg(
        mean_ret  = ("ret_pct", "mean"),
        med_ret   = ("ret_pct", "median"),
        std_ret   = ("ret_pct", "std"),
        win_rate  = ("ret_pct", lambda x: (x > 0).mean() * 100),
        n         = ("ret_pct", "count"),
    ).reset_index()

    # t-test vs 0
    agg["t_stat"] = agg["mean_ret"] / (agg["std_ret"] / np.sqrt(agg["n"]))
    agg["sig"]    = agg["t_stat"].abs() > 2.0   # ~5% sig livello

    return agg


# ─────────────────────────────────────────────────────────────────────────────
# 6. Catene a 3 sessioni
# ─────────────────────────────────────────────────────────────────────────────

def three_session_chains(session_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Mappa tutte le 8 combinazioni Asia/London/NY (Bull/Bear).
    Frequenza, rendimento medio della sessione NY, hit rate NY bull.
    """
    asia = session_dfs["Asia"][["direction"]].rename(columns={"direction": "asia_dir"})
    lon  = session_dfs["London"][["direction", "ret_pct"]].rename(
        columns={"direction": "lon_dir", "ret_pct": "lon_ret"})
    ny   = session_dfs["NY"][["direction", "ret_pct"]].rename(
        columns={"direction": "ny_dir", "ret_pct": "ny_ret"})

    comb = asia.join(lon, how="inner").join(ny, how="inner").dropna()

    def dir_label(v):
        return "↑" if v == 1 else "↓"

    records = []
    for a_dir in [1, -1]:
        for l_dir in [1, -1]:
            sub = comb[(comb["asia_dir"] == a_dir) & (comb["lon_dir"] == l_dir)]
            if len(sub) == 0:
                continue
            records.append({
                "Pattern":     f"Asia {dir_label(a_dir)} → London {dir_label(l_dir)}",
                "N":           len(sub),
                "Freq %":      len(sub) / len(comb) * 100,
                "P(NY↑) %":   (sub["ny_dir"] == 1).mean() * 100,
                "P(NY↓) %":   (sub["ny_dir"] == -1).mean() * 100,
                "NY mean ret": sub["ny_ret"].mean(),
                "NY med ret":  sub["ny_ret"].median(),
                "NY ret std":  sub["ny_ret"].std(),
                "Edge":        abs((sub["ny_dir"] == 1).mean() - 0.5) * 100,
            })

    df_chains = pd.DataFrame(records).sort_values("Freq %", ascending=False)
    return df_chains


# ─────────────────────────────────────────────────────────────────────────────
# Plotting functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_session_returns_dist(session_dfs: dict) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor(_BG)
    fig.suptitle("Distribuzione Return per Sessione (%)", color=_TEXT, fontsize=10)

    for ax, (sess, df) in zip(axes, session_dfs.items()):
        data = df["ret_pct"].dropna()
        data_clipped = data.clip(-5, 5)
        ax.hist(data_clipped, bins=60, color=_C[sess], alpha=0.8, edgecolor="none")
        ax.axvline(data.mean(), color="white", linewidth=1.5, linestyle="--",
                   label=f"Mean: {data.mean():.3f}%")
        ax.axvline(data.median(), color="#ffeb3b", linewidth=1.0, linestyle=":",
                   label=f"Median: {data.median():.3f}%")
        _style_ax(ax, sess, "Return %", "Freq")
        ax.legend(facecolor=_CARD, edgecolor=_GRID, labelcolor=_TEXT, fontsize=7)

    fig.tight_layout(pad=1.2)
    return _to_b64(fig)


def plot_session_range(session_dfs: dict) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor(_BG)
    fig.suptitle("Distribuzione Ampiezza Range per Sessione (% of open)", color=_TEXT, fontsize=10)

    for ax, (sess, df) in zip(axes, session_dfs.items()):
        data = df["range_pct"].dropna()
        ax.hist(data.clip(0, 6), bins=60, color=_C[sess], alpha=0.8, edgecolor="none")
        ax.axvline(data.median(), color="white", linewidth=1.5, linestyle="--",
                   label=f"Median: {data.median():.2f}%")
        ax.axvline(data.quantile(0.75), color="#ff9800", linewidth=1.0, linestyle=":",
                   label=f"p75: {data.quantile(0.75):.2f}%")
        _style_ax(ax, sess, "Range %", "Freq")
        ax.legend(facecolor=_CARD, edgecolor=_GRID, labelcolor=_TEXT, fontsize=7)

    fig.tight_layout(pad=1.2)
    return _to_b64(fig)


def plot_hour_of_day(hod: pd.DataFrame) -> str:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    fig.patch.set_facecolor(_BG)

    hours = hod["hour"].values
    mean_ret = hod["mean_ret"].values
    win_rate = hod["win_rate"].values
    sig      = hod["sig"].values

    # Colori per sessione
    bar_colors = []
    for h in hours:
        if h < 8:
            bar_colors.append(_C["Asia"])
        elif h < 16:
            bar_colors.append(_C["London"])
        else:
            bar_colors.append(_C["NY"])

    # Plot 1: mean return
    bars = ax1.bar(hours, mean_ret, color=bar_colors, alpha=0.85, width=0.8)
    # Evidenzia barre significative
    for i, (bar, s) in enumerate(zip(bars, sig)):
        if s:
            bar.set_edgecolor("white")
            bar.set_linewidth(1.5)
    ax1.axhline(0, color=_TEXT, linewidth=0.7)
    _style_ax(ax1, "Return Medio per Ora UTC (barra bordata = significativo α=5%)",
              "", "Mean Return %")

    # Plot 2: win rate
    ax2.bar(hours, win_rate - 50, color=bar_colors, alpha=0.85, width=0.8)
    ax2.axhline(0, color=_TEXT, linewidth=0.7)
    _style_ax(ax2, "Win Rate per Ora UTC (scarto da 50%)", "Ora UTC", "Win Rate − 50%")

    # Legenda sessioni
    patches = [
        mpatches.Patch(color=_C["Asia"],   label="Asia (00-07)"),
        mpatches.Patch(color=_C["London"], label="London (08-15)"),
        mpatches.Patch(color=_C["NY"],     label="NY (16-23)"),
    ]
    ax1.legend(handles=patches, facecolor=_CARD, edgecolor=_GRID,
               labelcolor=_TEXT, fontsize=8, loc="upper right")

    ax2.set_xticks(hours)
    fig.tight_layout(pad=1.2)
    return _to_b64(fig)


def plot_contingency_heatmap(corr_results: dict) -> str:
    pairs = ["Asia→London", "Asia→NY", "London→NY"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.patch.set_facecolor(_BG)
    fig.suptitle("Contingency Tables — Direzione Sessioni (% riga)", color=_TEXT, fontsize=10)

    labels_map = {1: "↑ Bull", -1: "↓ Bear"}

    for ax, pair in zip(axes, pairs):
        res = corr_results[pair]
        ct  = res["ct"]
        # Normalizza per riga (conditional probability)
        ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
        mat = ct_pct.values

        im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels([labels_map.get(c, str(c)) for c in ct_pct.columns],
                           color=_TEXT, fontsize=9)
        ax.set_yticklabels([labels_map.get(r, str(r)) for r in ct_pct.index],
                           color=_TEXT, fontsize=9)

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.1f}%", ha="center", va="center",
                        color="black" if 30 < mat[i, j] < 70 else _TEXT, fontsize=11, fontweight="bold")

        a_lbl, b_lbl = pair.split("→")
        ax.set_xlabel(f"→ {b_lbl}", color=_TEXT, fontsize=9)
        ax.set_ylabel(f"{a_lbl} →", color=_TEXT, fontsize=9)
        v = res["cramers_v"]
        p = res["p"]
        p_str = f"{p:.4f}" if p >= 0.0001 else "<0.0001"
        ax.set_title(f"{pair}\nCramér's V={v:.3f}  p={p_str}", color=_TEXT, fontsize=8)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout(pad=1.5)
    return _to_b64(fig)


def plot_orb_breakdown(orb_results: dict) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor(_BG)
    fig.suptitle(f"Open Range Breakout — Analisi per Sessione (ORB = prime {ORB_HOURS}h)",
                 color=_TEXT, fontsize=10)

    for ax, (sess, res) in zip(axes, orb_results.items()):
        cats = ["ORB Up\nBreak", "ORB Down\nBreak", "No Break"]
        vals = [res["pct_up"], res["pct_down"], res["pct_none"]]
        colors = ["#4caf50", "#ef5350", "#78909c"]
        bars = ax.bar(cats, vals, color=colors, alpha=0.85, width=0.6)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", va="bottom", color=_TEXT, fontsize=9)

        # Aggiunge linea post-ret
        ax2 = ax.twinx()
        pr = [res["mean_post_ret_up"], -res["mean_post_ret_dn"], 0]
        ax2.plot(cats[:2], pr[:2], "o", color="#ffeb3b", markersize=8, zorder=5)
        ax2.axhline(0, color=_TEXT, linewidth=0.5, linestyle=":")
        ax2.set_ylabel("Post-break return %", color="#ffeb3b", fontsize=7)
        ax2.tick_params(colors="#ffeb3b", labelsize=7)
        ax2.spines["right"].set_color(_GRID)

        fb_u = res["false_break_up"]
        fb_d = res["false_break_dn"]
        _style_ax(ax, f"{sess}  |  FB↑={fb_u:.0f}%  FB↓={fb_d:.0f}%",
                  "", "% giorni")

    fig.tight_layout(pad=1.2)
    return _to_b64(fig)


def plot_asia_range_effect(are: dict) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(_BG)
    fig.suptitle("Effetto Ampiezza Range Asiatico sulle Sessioni Successive",
                 color=_TEXT, fontsize=10)

    q_labels = list(are["quartiles"].keys())
    lon_bull = [are["quartiles"][q]["lon_bull_pct"] for q in q_labels]
    ny_bull  = [are["quartiles"][q]["ny_bull_pct"]  for q in q_labels]
    lon_ret  = [are["quartiles"][q]["lon_ret_mean"] for q in q_labels]
    ny_ret   = [are["quartiles"][q]["ny_ret_mean"]  for q in q_labels]
    ns       = [are["quartiles"][q]["n"]            for q in q_labels]

    x = np.arange(len(q_labels))
    w = 0.35

    # Plot 1: Bull probability
    ax1.bar(x - w/2, lon_bull, w, color=_C["London"], alpha=0.85, label="London Bull %")
    ax1.bar(x + w/2, ny_bull,  w, color=_C["NY"],     alpha=0.85, label="NY Bull %")
    ax1.axhline(50, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{q}\n(n={n})" for q, n in zip(q_labels, ns)], color=_TEXT, fontsize=8)
    for i, (lv, nv) in enumerate(zip(lon_bull, ny_bull)):
        ax1.text(i - w/2, lv + 0.5, f"{lv:.0f}%", ha="center", va="bottom",
                 color=_C["London"], fontsize=8)
        ax1.text(i + w/2, nv + 0.5, f"{nv:.0f}%", ha="center", va="bottom",
                 color=_C["NY"],     fontsize=8)
    _style_ax(ax1, "P(Sessione Bullish) per Quartile Range Asia",
              "Range Asia", "Bull %")
    ax1.legend(facecolor=_CARD, edgecolor=_GRID, labelcolor=_TEXT, fontsize=8)
    ax1.set_ylim(35, 65)

    # Plot 2: Mean return
    ax2.bar(x - w/2, lon_ret, w, color=_C["London"], alpha=0.85, label="London ret %")
    ax2.bar(x + w/2, ny_ret,  w, color=_C["NY"],     alpha=0.85, label="NY ret %")
    ax2.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(q_labels, color=_TEXT, fontsize=8)
    _style_ax(ax2, "Return Medio Sessione per Quartile Range Asia",
              "Range Asia", "Mean Return %")
    ax2.legend(facecolor=_CARD, edgecolor=_GRID, labelcolor=_TEXT, fontsize=8)

    fig.tight_layout(pad=1.2)
    return _to_b64(fig)


def plot_orb_post_ret_dist(orb_results: dict) -> str:
    """Distribuzione dei post-breakout return (up vs down) per sessione."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor(_BG)
    fig.suptitle("Distribuzione Post-Breakout Return per Sessione",
                 color=_TEXT, fontsize=10)

    for ax, (sess, res) in zip(axes, orb_results.items()):
        df_orb = res["df"]
        up = df_orb[df_orb["first_break"] == "UP"]["post_ret"].clip(-3, 3)
        dn = df_orb[df_orb["first_break"] == "DOWN"]["post_ret"].clip(-3, 3)

        if len(up) > 0:
            ax.hist(up, bins=40, color="#4caf50", alpha=0.7, edgecolor="none",
                    label=f"ORB↑ (n={len(up)}) μ={up.mean():.2f}%")
        if len(dn) > 0:
            ax.hist(dn, bins=40, color="#ef5350", alpha=0.7, edgecolor="none",
                    label=f"ORB↓ (n={len(dn)}) μ={dn.mean():.2f}%")
        ax.axvline(0, color="white", linewidth=1.0, linestyle="--")
        _style_ax(ax, sess, "Post-break Return %", "Freq")
        ax.legend(facecolor=_CARD, edgecolor=_GRID, labelcolor=_TEXT, fontsize=7)

    fig.tight_layout(pad=1.2)
    return _to_b64(fig)


def plot_rolling_hit_rate(corr_results: dict, session_dfs: dict) -> str:
    """Rolling 90-day hit rate: P(London Bull | Asia Bull) nel tempo."""
    combined = corr_results["combined"]
    window = 90

    # P(London Bull | Asia Bull): rolling
    asia_bull_mask = combined["asia_direction"] == 1
    lon_dir = combined["lon_direction"]

    roll_all    = lon_dir.rolling(window).mean() * 100
    roll_asia_b = lon_dir.where(asia_bull_mask).rolling(window).mean() * 100
    roll_asia_r = lon_dir.where(~asia_bull_mask).rolling(window).mean() * 100

    fig, ax = plt.subplots(figsize=(13, 4))
    fig.patch.set_facecolor(_BG)

    ax.plot(combined.index, roll_all,    color=_TEXT,        linewidth=1.0,
            label="P(London↑) unconditional", alpha=0.6)
    ax.plot(combined.index, roll_asia_b, color=_C["London"], linewidth=1.5,
            label="P(London↑ | Asia↑)")
    ax.plot(combined.index, roll_asia_r, color=_C["NY"],     linewidth=1.5,
            label="P(London↑ | Asia↓)")
    ax.axhline(50, color="white", linewidth=0.7, linestyle="--", alpha=0.4)
    ax.fill_between(combined.index, roll_asia_b, roll_asia_r,
                    alpha=0.1, color="#90caf9", label="spread")

    _style_ax(ax, f"Rolling {window}-day: P(London Bull) condizionale su direzione Asia",
              "Data", "P(London↑) %")
    ax.set_ylim(25, 75)
    ax.legend(facecolor=_CARD, edgecolor=_GRID, labelcolor=_TEXT, fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    fig.tight_layout(pad=1.2)
    return _to_b64(fig)


def plot_orb_cumulative(orb_results: dict) -> str:
    """Equity curve sintetica degli ORB: long su breakout up, short su breakout down."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.patch.set_facecolor(_BG)
    fig.suptitle("Equity Curve Sintetica ORB (post-breakout return cumulato, 1 USD/trade)",
                 color=_TEXT, fontsize=10)

    for ax, (sess, res) in zip(axes, orb_results.items()):
        df_orb = res["df"].copy()
        df_orb = df_orb.sort_values("date")

        # Long su ORB UP, short su ORB DOWN
        df_orb["trade_ret"] = 0.0
        df_orb.loc[df_orb["first_break"] == "UP",   "trade_ret"] = df_orb["post_ret"]
        df_orb.loc[df_orb["first_break"] == "DOWN",  "trade_ret"] = df_orb["post_ret"]
        df_orb = df_orb[df_orb["first_break"] != "NONE"]
        if df_orb.empty:
            continue

        eq = (1 + df_orb["trade_ret"] / 100).cumprod()
        ax.plot(df_orb["date"], eq, color=_C[sess], linewidth=1.2)
        ax.axhline(1.0, color=_TEXT, linewidth=0.7, linestyle="--", alpha=0.5)
        ax.fill_between(df_orb["date"], 1.0, eq, where=eq >= 1.0,
                        alpha=0.2, color="#4caf50")
        ax.fill_between(df_orb["date"], 1.0, eq, where=eq < 1.0,
                        alpha=0.2, color="#ef5350")
        final = eq.iloc[-1]
        _style_ax(ax, f"{sess}  (final={final:.2f}×)", "Data", "Equity")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))

    fig.tight_layout(pad=1.2)
    return _to_b64(fig)


# ─────────────────────────────────────────────────────────────────────────────
# HTML
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
body{font-family:system-ui,sans-serif;background:#0f1117;color:#e0e0e0;margin:0;padding:24px;max-width:1200px;margin:0 auto;padding:32px 24px}
h1{font-size:1.6rem;color:#fff;border-bottom:2px solid #2196f3;padding-bottom:8px;margin-bottom:4px}
h2{font-size:1.1rem;color:#90caf9;margin-top:36px;border-left:3px solid #2196f3;padding-left:10px}
h3{font-size:.92rem;color:#b0bec5;margin-top:18px}
p,li{font-size:.88rem;line-height:1.6;color:#cfd8dc}
p.meta{color:#546e7a;font-size:.78rem}
code{background:#1e2130;padding:2px 6px;border-radius:3px;font-size:.82rem;color:#80cbc4}
table{border-collapse:collapse;width:100%;margin-top:12px;font-size:.82rem}
th{background:#1e2130;color:#90caf9;padding:7px 12px;text-align:center;border-bottom:2px solid #2196f3}
td{padding:6px 12px;border-bottom:1px solid #1e2130;text-align:center}
td:first-child{text-align:left}
tr:hover td{background:#1a1e2e}
.pos{color:#4caf50;font-weight:600}
.neg{color:#f44336;font-weight:600}
.warn{color:#ff9800;font-weight:600}
.hl td{background:#1e2a1e!important;border-left:3px solid #4caf50}
.card{background:#12151f;border:1px solid #1e2130;border-radius:8px;padding:16px 20px;margin-top:16px}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px;margin-top:14px}
.kpi{background:#12151f;border:1px solid #1e2130;border-radius:8px;padding:14px;text-align:center}
.kpi .val{font-size:1.4rem;font-weight:700;color:#fff}
.kpi .lbl{font-size:.75rem;color:#78909c;margin-top:4px}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:.78rem;font-weight:600;margin-left:6px}
.badge-green{background:#1b5e20;color:#a5d6a7}
.badge-red{background:#b71c1c;color:#ef9a9a}
.badge-gray{background:#263238;color:#b0bec5}
.scroll{overflow-x:auto}
img{max-width:100%;display:block}
.finding{background:#0d2137;border-left:3px solid #42a5f5;padding:10px 14px;margin:8px 0;border-radius:0 6px 6px 0}
.finding strong{color:#90caf9}
"""


def _pnl_cls(v: float) -> str:
    return "pos" if v > 0 else ("neg" if v < 0 else "")


def _kpi(val: str, lbl: str) -> str:
    return f'<div class="kpi"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>'


def _finding(text: str) -> str:
    return f'<div class="finding">{text}</div>'


def build_html(
    session_dfs: dict,
    orb_results: dict,
    corr_results: dict,
    are: dict,
    hod: pd.DataFrame,
    chains: pd.DataFrame,
    imgs: dict,
    elapsed: float,
) -> str:

    # ── KPI per sessione
    kpi_blocks = ""
    for sess, df in session_dfs.items():
        mean_r = df["ret_pct"].mean()
        med_r  = df["ret_pct"].median()
        bull_p = (df["direction"] == 1).mean() * 100
        med_rng= df["range_pct"].median()
        kpi_blocks += f"""
        <h3 style="color:{_C[sess]}">{sess} Session</h3>
        <div class="kpi-grid">
          {_kpi(f'{mean_r:+.3f}%',  'Mean Return')}
          {_kpi(f'{med_r:+.3f}%',   'Median Return')}
          {_kpi(f'{bull_p:.1f}%',   'Bullish Rate')}
          {_kpi(f'{med_rng:.2f}%',  'Median Range')}
          {_kpi(f'{len(df):,}',     'Days')}
        </div>"""

    # ── Tabella correlazioni
    corr_rows = ""
    for pair, res in corr_results.items():
        if pair == "combined":
            continue
        sig = "✓ Significativo" if res["p"] < 0.05 else "✗ Non significativo"
        cls = "pos" if res["p"] < 0.05 else "neg"
        corr_rows += (
            f"<tr><td>{pair}</td>"
            f"<td>{res['n']:,}</td>"
            f"<td>{res['chi2']:.2f}</td>"
            f"<td>{res['dof']}</td>"
            f"<td class='{_pnl_cls(-res['p'])}'>{res['p']:.4f}</td>"
            f"<td>{res['cramers_v']:.4f}</td>"
            f"<td class='{cls}'>{sig}</td></tr>"
        )

    # ── Probabilità condizionali
    cond_rows = ""
    for pair, res in corr_results.items():
        if pair == "combined":
            continue
        for k, v in res["cond"].items():
            col = "pos" if v > 52 else ("neg" if v < 48 else "")
            cond_rows += f"<tr><td>{pair}</td><td>{k}</td><td class='{col}'>{v:.1f}%</td></tr>"

    # ── ORB summary
    orb_rows = ""
    for sess, res in orb_results.items():
        fb_avg = (res["false_break_up"] + res["false_break_dn"]) / 2
        orb_rows += (
            f"<tr><td style='color:{_C[sess]}'>{sess}</td>"
            f"<td>{res['total']}</td>"
            f"<td>{res['pct_up']:.1f}%</td>"
            f"<td>{res['pct_down']:.1f}%</td>"
            f"<td>{res['pct_none']:.1f}%</td>"
            f"<td>{res['med_orb_range_pct']:.2f}%</td>"
            f"<td class='pos'>{res['mean_post_ret_up']:.2f}%</td>"
            f"<td class='pos'>{res['mean_post_ret_dn']:.2f}%</td>"
            f"<td class='warn'>{fb_avg:.1f}%</td></tr>"
        )

    # ── Tabella Asia range quartile
    are_rows = ""
    for q_label, d in are["quartiles"].items():
        col_l = _pnl_cls(d["lon_bull_pct"] - 50)
        col_n = _pnl_cls(d["ny_bull_pct"] - 50)
        are_rows += (
            f"<tr><td>{q_label}</td>"
            f"<td>{d['n']}</td>"
            f"<td class='{col_l}'>{d['lon_bull_pct']:.1f}%</td>"
            f"<td class='{col_n}'>{d['ny_bull_pct']:.1f}%</td>"
            f"<td class='{_pnl_cls(d['lon_ret_mean'])}'>{d['lon_ret_mean']:+.3f}%</td>"
            f"<td class='{_pnl_cls(d['ny_ret_mean'])}'>{d['ny_ret_mean']:+.3f}%</td></tr>"
        )

    # ── Tabella catene 3 sessioni
    chain_rows = ""
    for _, row in chains.iterrows():
        edge = row["Edge"]
        p_ny_up = row["P(NY↑) %"]
        best_dir = "↑" if p_ny_up > 50 else "↓"
        cls = "pos" if edge > 3 else "warn"
        chain_rows += (
            f"<tr><td>{row['Pattern']}</td>"
            f"<td>{int(row['N'])}</td>"
            f"<td>{row['Freq %']:.1f}%</td>"
            f"<td class='pos'>{row['P(NY↑) %']:.1f}%</td>"
            f"<td class='neg'>{row['P(NY↓) %']:.1f}%</td>"
            f"<td class='{_pnl_cls(row['NY mean ret'])}'>{row['NY mean ret']:+.3f}%</td>"
            f"<td><b>{best_dir}</b></td>"
            f"<td class='{cls}'>{edge:.1f}pp</td></tr>"
        )

    # ── Hour-of-day table (top 5 long + top 5 short)
    hod_sorted_bull = hod.sort_values("mean_ret", ascending=False).head(6)
    hod_sorted_bear = hod.sort_values("mean_ret", ascending=True).head(6)

    def hod_row(r):
        sess_lbl = "Asia" if r["hour"] < 8 else ("London" if r["hour"] < 16 else "NY")
        sig_badge = '<span class="badge badge-green">sig</span>' if r["sig"] else ""
        cls = _pnl_cls(r["mean_ret"])
        return (f"<tr><td>{int(r['hour']):02d}:00</td>"
                f"<td style='color:{_C[sess_lbl]}'>{sess_lbl}</td>"
                f"<td class='{cls}'>{r['mean_ret']:+.4f}%{sig_badge}</td>"
                f"<td>{r['win_rate']:.1f}%</td>"
                f"<td>{r['std_ret']:.3f}%</td>"
                f"<td>{int(r['n'])}</td></tr>")

    hod_bull_rows = "".join(hod_row(r) for _, r in hod_sorted_bull.iterrows())
    hod_bear_rows = "".join(hod_row(r) for _, r in hod_sorted_bear.iterrows())

    # ── Findings chiave (pre-compute strings to avoid backslash in f-string)
    best_chain      = chains.iloc[0]
    best_edge_chain = chains.sort_values("Edge", ascending=False).iloc[0]
    corr_al_p = corr_results["Asia→London"]["p"]
    corr_al_v = corr_results["Asia→London"]["cramers_v"]
    corr_ln_p = corr_results["London→NY"]["p"]
    corr_ln_v = corr_results["London→NY"]["cramers_v"]

    asia_orb = orb_results["Asia"]
    lon_orb  = orb_results["London"]
    ny_orb   = orb_results["NY"]

    best_hod_bull = hod.loc[hod["mean_ret"].idxmax()]
    best_hod_bear = hod.loc[hod["mean_ret"].idxmin()]

    # Pre-built finding strings (no backslash in f-expression)
    sig_al   = "<b>statisticamente significativo</b>" if corr_al_p < 0.05 else "non significativo"
    sig_ln   = "anticipa" if corr_ln_p < 0.05 else "non anticipa significativamente"
    hod_b_sig = "sig" if best_hod_bull["sig"] else "non-sig"
    hod_r_sig = "sig" if best_hod_bear["sig"] else "non-sig"
    cramer_s  = "Cramér's"   # Cramér's — avoid \' in f-string

    f1 = _finding(
        f"<strong>Correlazione Asia→London:</strong> {cramer_s} V = {corr_al_v:.3f},"
        f" p = {corr_al_p:.4f}. La direzione asiatica ha un effetto {sig_al} sulla direzione londinese.")
    f2 = _finding(
        f"<strong>Correlazione London→NY:</strong> {cramer_s} V = {corr_ln_v:.3f},"
        f" p = {corr_ln_p:.4f}. La sessione londinese {sig_ln} la direzione NY.")
    f3 = _finding(
        f"<strong>ORB London:</strong> Breakout in "
        f"{lon_orb['pct_up']+lon_orb['pct_down']:.0f}% dei giorni. "
        f"Post-break mean return: ↑ {lon_orb['mean_post_ret_up']:+.2f}% "
        f"| ↓ {lon_orb['mean_post_ret_dn']:+.2f}%. "
        f"False breakout medio: {(lon_orb['false_break_up']+lon_orb['false_break_dn'])/2:.0f}%.")
    f4 = _finding(
        f"<strong>Pattern più frequente:</strong> {best_chain['Pattern']} "
        f"→ {best_chain['N']} giorni ({best_chain['Freq %']:.1f}%). "
        f"P(NY↑) = {best_chain['P(NY↑) %']:.1f}%.")
    f5 = _finding(
        f"<strong>Edge massimo 3-sessioni:</strong> {best_edge_chain['Pattern']} "
        f"→ P(NY↑) = {best_edge_chain['P(NY↑) %']:.1f}% "
        f"(edge {best_edge_chain['Edge']:.1f}pp vs random)")
    f6 = _finding(
        f"<strong>Ora UTC più bullish:</strong> {int(best_hod_bull['hour']):02d}:00 "
        f"(mean {best_hod_bull['mean_ret']:+.4f}%, WR {best_hod_bull['win_rate']:.1f}%, {hod_b_sig}).")
    f7 = _finding(
        f"<strong>Ora UTC più bearish:</strong> {int(best_hod_bear['hour']):02d}:00 "
        f"(mean {best_hod_bear['mean_ret']:+.4f}%, WR {best_hod_bear['win_rate']:.1f}%, {hod_r_sig}).")

    return f"""<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<title>ORB Research — BTCUSDT Sessions 2020-2026</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Open Range Breakout — Ricerca Statistica Sessioni BTCUSDT</h1>
<p class="meta">
  Periodo: 2020-01 → 2026-05 &middot; 2.343 giorni &middot; Timeframe: 1H &middot;
  Sessioni UTC: Asia 00-07 | London 08-15 | NY 16-23 &middot;
  Open Range = prime {ORB_HOURS} ore di sessione &middot; Generato in {elapsed:.0f}s
</p>

<!-- ═══ EXECUTIVE SUMMARY ═══════════════════════════════════════════════════ -->
<h2>Executive Summary — Findings Principali</h2>
<div class="card">
  {f1}
  {f2}
  {f3}
  {f4}
  {f5}
  {f6}
  {f7}
</div>

<!-- ═══ 1. SESSION STATISTICS ════════════════════════════════════════════════ -->
<h2>1. Statistiche per Sessione</h2>
<div class="card">
  {kpi_blocks}
</div>
{_img(imgs["sess_ret"])}
{_img(imgs["sess_range"])}

<!-- ═══ 2. CORRELAZIONE DIREZIONALE ══════════════════════════════════════════ -->
<h2>2. Correlazione Direzionale tra Sessioni</h2>
<p>Chi-squared test su contingency tables (Bull/Bear per coppia di sessioni).
Cramér's V misura la forza dell'associazione (0 = nessuna, 1 = perfetta).</p>
{_img(imgs["contingency"], 900)}

<div class="scroll">
<table>
<thead><tr><th>Coppia</th><th>N</th><th>χ²</th><th>DoF</th><th>p-value</th><th>Cramér's V</th><th>Significatività</th></tr></thead>
<tbody>{corr_rows}</tbody>
</table>
</div>

<h3>Probabilità Condizionali Complete</h3>
<div class="scroll">
<table>
<thead><tr><th>Coppia</th><th>Condizione</th><th>Probabilità</th></tr></thead>
<tbody>{cond_rows}</tbody>
</table>
</div>

{_img(imgs["rolling_hit"])}

<!-- ═══ 3. OPEN RANGE BREAKOUT ════════════════════════════════════════════════ -->
<h2>3. Open Range Breakout Analysis</h2>
<p>Open Range = high e low delle prime <code>{ORB_HOURS} ore</code> di ogni sessione.
<em>False Breakout</em> = breakout in direzione X ma close di sessione oltre il limite opposto.</p>

<div class="scroll">
<table>
<thead><tr>
  <th>Sessione</th><th>Giorni</th><th>ORB↑ %</th><th>ORB↓ %</th>
  <th>No Break %</th><th>ORB Range median</th>
  <th>Mean Post-ret ↑</th><th>Mean Post-ret ↓</th><th>False Break avg</th>
</tr></thead>
<tbody>{orb_rows}</tbody>
</table>
</div>

{_img(imgs["orb_break"])}
{_img(imgs["orb_post_ret"])}
{_img(imgs["orb_equity"])}

<!-- ═══ 4. EFFETTO RANGE ASIATICO ════════════════════════════════════════════ -->
<h2>4. Effetto Ampiezza Range Asiatico sulle Sessioni Successive</h2>
<p>I giorni sono divisi in quartili del range asiatico (% sull'open).
Q1 = giornata asiatica stretta, Q4 = giornata asiatica ampia.</p>
<p class="meta">Q25={are["q25"]:.2f}%  Q50={are["q50"]:.2f}%  Q75={are["q75"]:.2f}%</p>

<div class="scroll">
<table>
<thead><tr>
  <th>Quartile Asia Range</th><th>N giorni</th>
  <th>P(London↑)</th><th>P(NY↑)</th>
  <th>London mean ret</th><th>NY mean ret</th>
</tr></thead>
<tbody>{are_rows}</tbody>
</table>
</div>
{_img(imgs["asia_range"])}

<!-- ═══ 5. HOUR-OF-DAY BIAS ══════════════════════════════════════════════════ -->
<h2>5. Hour-of-Day Bias (UTC)</h2>
<p>Return medio e win rate per ogni ora UTC su 2020-2026.
Barre con bordo bianco = significative (|t| &gt; 2, α≈5%).</p>
{_img(imgs["hour_of_day"])}

<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
<div>
<h3>Top 6 Ore Bullish</h3>
<table>
<thead><tr><th>Ora UTC</th><th>Sessione</th><th>Mean Ret</th><th>WR</th><th>Std</th><th>N</th></tr></thead>
<tbody>{hod_bull_rows}</tbody>
</table>
</div>
<div>
<h3>Top 6 Ore Bearish</h3>
<table>
<thead><tr><th>Ora UTC</th><th>Sessione</th><th>Mean Ret</th><th>WR</th><th>Std</th><th>N</th></tr></thead>
<tbody>{hod_bear_rows}</tbody>
</table>
</div>
</div>

<!-- ═══ 6. CATENE A 3 SESSIONI ══════════════════════════════════════════════ -->
<h2>6. Pattern Sequenziali — Catene Asia→London→NY</h2>
<p>Analisi delle 4 combinazioni possibili Asia/London (Bull↑/Bear↓).
<em>Edge</em> = distanza da 50% nella previsione NY (più alto = più predittivo).</p>

<div class="scroll">
<table>
<thead><tr>
  <th>Pattern</th><th>N giorni</th><th>Freq</th>
  <th>P(NY↑)</th><th>P(NY↓)</th>
  <th>NY mean ret</th><th>Direzione NY</th><th>Edge</th>
</tr></thead>
<tbody>{chain_rows}</tbody>
</table>
</div>

<!-- ═══ 7. NOTE METODOLOGICHE ════════════════════════════════════════════════ -->
<h2>7. Note Metodologiche</h2>
<div class="card">
  <h3>Definizione Sessioni (UTC)</h3>
  <ul>
    <li><strong>Asia</strong>: 00:00–07:59 UTC — mercati asiatici, liquidità BTC principalmente dalla Corea, Giappone, HK</li>
    <li><strong>London</strong>: 08:00–15:59 UTC — apertura EU, overlap con Asia fino alle 09:00 circa</li>
    <li><strong>NY</strong>: 16:00–23:59 UTC — apertura Wall Street, massima liquidità del giorno</li>
  </ul>
  <h3>Open Range</h3>
  <p>ORB definita come il range delle prime {ORB_HOURS} barre 1H di ogni sessione.
  Un breakout è classificato come la prima violazione del high/low dell'ORB nel
  resto della sessione. Il <em>post-break return</em> misura il rendimento dal
  livello di breakout al close di sessione.</p>
  <h3>Limitazioni</h3>
  <ul>
    <li>Il mercato crypto è 24/7: la separazione in sessioni è convenzionale, non strutturale</li>
    <li>I return intra-sessione non includono il funding rate sui futures</li>
    <li>L'analisi è in-sample su 2020-2026 — la robustezza va verificata con WF</li>
    <li>La significatività statistica non implica tradabilità: occorre modellare slippage e fee</li>
  </ul>
</div>

</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("══ ORB Research Report ══\n")
    t0 = time.time()

    print("[1/7] Loading 1H data …")
    raw  = fetch_extended_data(start_year=2020, start_month=1,
                               fetch_flow=False, fetch_15m=False)
    df_1h = raw["1H"].copy()
    print(f"  {len(df_1h):,} bars  ({df_1h.index[0].date()} → {df_1h.index[-1].date()})")

    print("[2/7] Building session aggregates …")
    session_dfs = build_session_df(df_1h)
    for s, df in session_dfs.items():
        bull = (df["direction"] == 1).mean() * 100
        print(f"  {s}: {len(df)} days  bull={bull:.1f}%  "
              f"med_range={df['range_pct'].median():.2f}%")

    print("[3/7] ORB analysis …")
    orb_results = orb_analysis(df_1h)
    for s, r in orb_results.items():
        print(f"  {s}: ↑{r['pct_up']:.0f}%  ↓{r['pct_down']:.0f}%  "
              f"none={r['pct_none']:.0f}%  "
              f"post_ret: ↑{r['mean_post_ret_up']:+.2f}% ↓{r['mean_post_ret_dn']:+.2f}%")

    print("[4/7] Directional correlation …")
    corr_results = directional_correlation(session_dfs)
    for pair, res in corr_results.items():
        if pair == "combined":
            continue
        print(f"  {pair}: chi2={res['chi2']:.2f}  p={res['p']:.4f}  V={res['cramers_v']:.3f}")

    print("[5/7] Asia range effect …")
    are = asia_range_effect(session_dfs)

    print("[6/7] Hour-of-day bias …")
    hod = hour_of_day_bias(df_1h)

    print("[7/7] Three-session chains …")
    chains = three_session_chains(session_dfs)

    print("\nGenerating charts …")
    imgs = {
        "sess_ret":    plot_session_returns_dist(session_dfs),
        "sess_range":  plot_session_range(session_dfs),
        "hour_of_day": plot_hour_of_day(hod),
        "contingency": plot_contingency_heatmap(corr_results),
        "orb_break":   plot_orb_breakdown(orb_results),
        "orb_post_ret":plot_orb_post_ret_dist(orb_results),
        "orb_equity":  plot_orb_cumulative(orb_results),
        "asia_range":  plot_asia_range_effect(are),
        "rolling_hit": plot_rolling_hit_rate(corr_results, session_dfs),
    }

    elapsed = time.time() - t0
    html = build_html(session_dfs, orb_results, corr_results, are, hod, chains, imgs, elapsed)

    out = Path("reports/report_orb_research.html")
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"\n  → {out}  ({out.stat().st_size//1024}KB, {elapsed:.0f}s)")


if __name__ == "__main__":
    main()
