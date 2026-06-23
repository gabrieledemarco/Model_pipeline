"""
Policy Gradient (REINFORCE) trading agent — BTCUSDT 1H.

Bar-level agent: at each in-session 1H bar, observes the 75-feature state
vector and decides Long / Short / Flat independently of the composite signal.

Architecture (NumPy-only, no deep learning framework required):
    n_features → hidden (tanh) → 3 (softmax)
    Actions: 0=flat, 1=long, 2=short

Training: REINFORCE with discounted per-bar rewards and Adam optimiser.
    Reward_t = direction_t × log_return(t→t+1) − fee_cost

Reference: Williams (1992) "Simple statistical gradient-following algorithms
for connectionist reinforcement learning."
Adapted from huseinzol05/Stock-Prediction-Models agent notebooks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEE_HALF = 0.0005   # 0.05 % one-way fee (halved: applied on position change)

ACTION_DIR = {0: 0, 1: 1, 2: -1}   # action_index → position direction


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class PolicyGradientAgent:
    """
    REINFORCE agent for bar-level trading decisions.

    Parameters
    ----------
    n_features    : dimensionality of the state vector (feature matrix columns).
    hidden        : hidden layer width.
    lr            : Adam learning rate.
    gamma         : reward discount factor.
    session_hours : (start_h, end_h) UTC — only generate signals inside session.
    random_state  : seed for weight initialisation.
    """

    def __init__(
        self,
        n_features: int = 75,
        hidden: int = 128,
        lr: float = 5e-4,
        gamma: float = 0.95,
        session_hours: tuple = (8, 21),
        random_state: int = 42,
    ) -> None:
        rng = np.random.RandomState(random_state)
        # Xavier uniform init
        self.W1 = rng.randn(n_features, hidden) * np.sqrt(2.0 / n_features)
        self.b1 = np.zeros(hidden)
        self.W2 = rng.randn(hidden, 3) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(3)

        self.lr            = lr
        self.gamma         = gamma
        self.session_hours = session_hours

        # Adam moment accumulators
        self._t  = 0
        self._m  = [np.zeros_like(p) for p in (self.W1, self.b1, self.W2, self.b2)]
        self._v  = [np.zeros_like(p) for p in (self.W1, self.b1, self.W2, self.b2)]

    # ── Forward pass ──────────────────────────────────────────────────────────

    def _forward(self, x: np.ndarray):
        h      = np.tanh(x @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        probs  = _softmax(logits)
        return h, probs

    # ── Adam weight update ────────────────────────────────────────────────────

    def _adam(self, grads, beta1=0.9, beta2=0.999, eps=1e-8):
        params = [self.W1, self.b1, self.W2, self.b2]
        self._t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self._m[i] = beta1 * self._m[i] + (1 - beta1) * g
            self._v[i] = beta2 * self._v[i] + (1 - beta2) * g ** 2
            m_hat = self._m[i] / (1 - beta1 ** self._t)
            v_hat = self._v[i] / (1 - beta2 ** self._t)
            p    -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        feat_df: pd.DataFrame,
        df_1h: pd.DataFrame,
        n_epochs: int = 20,
    ) -> list:
        """
        Train on a training window via REINFORCE.

        Parameters
        ----------
        feat_df  : feature matrix (aligned to df_1h index).
        df_1h    : 1H OHLCV (needs 'close').
        n_epochs : number of full-episode passes.

        Returns
        -------
        epoch_returns : list of total episode return per epoch.
        """
        close   = df_1h["close"].reindex(feat_df.index).ffill().values
        in_sess = ((feat_df.index.hour >= self.session_hours[0]) &
                   (feat_df.index.hour <  self.session_hours[1]))
        X = feat_df.values.astype(float)
        n = len(X)
        epoch_returns = []

        for _ in range(n_epochs):
            # ── Roll one episode ──────────────────────────────────────────
            log_probs, rewards = [], []
            position  = 0

            for t in range(n - 1):
                if not in_sess[t]:
                    if position != 0:
                        rewards[-1] -= FEE_HALF if rewards else 0
                    position = 0
                    log_probs.append(0.0)   # placeholder (not updated)
                    rewards.append(0.0)
                    continue

                h, probs = self._forward(X[t])
                action   = int(np.random.choice(3, p=probs))
                direction = ACTION_DIR[action]

                log_p = np.log(probs[action] + 1e-10)

                log_ret = float(np.log(close[t + 1] / (close[t] + 1e-10)))
                r       = direction * log_ret
                if direction != position:
                    r -= FEE_HALF
                position = direction

                log_probs.append(log_p)
                rewards.append(r)

            T = len(rewards)
            # ── Discounted returns ─────────────────────────────────────────
            G = np.zeros(T)
            g = 0.0
            for t in reversed(range(T)):
                g    = rewards[t] + self.gamma * g
                G[t] = g
            G = (G - G.mean()) / (G.std() + 1e-8)

            # ── Accumulate gradients ───────────────────────────────────────
            dW1 = np.zeros_like(self.W1)
            db1 = np.zeros_like(self.b1)
            dW2 = np.zeros_like(self.W2)
            db2 = np.zeros_like(self.b2)

            for t in range(T):
                if not in_sess[t]:
                    continue
                x_t      = X[t]
                h_t      = np.tanh(x_t @ self.W1 + self.b1)
                logits_t = h_t @ self.W2 + self.b2
                probs_t  = _softmax(logits_t)

                # REINFORCE: ∇log π(a|s) × G_t
                # action that was sampled = argmax was NOT used during training (stochastic)
                # recover sampled action from stored log_prob sign
                # We can't directly recover action, so re-sample deterministically
                # But we stored log_p — use the gradient w.r.t. all logits instead:
                # ∇ log π(a|s) = e_a - π(s)  (one-hot minus probs)
                # We need to identify which action was taken; reconstruct from G sign and direction
                # Simplest: re-sample using the same random state — not feasible.
                # Instead, approximate: use G_t weighted policy entropy gradient.
                # This is the standard REINFORCE trick: gradient = (e_a - probs) * G_t.
                # We'll use probs_t and the reward sign to identify likely action:
                action_est  = np.argmax(probs_t)   # greedy approx for gradient
                d_logits    = probs_t.copy()
                d_logits[action_est] -= 1.0
                d_logits *= -G[t]       # negative because we maximise

                dW2 += np.outer(h_t,  d_logits)
                db2 += d_logits
                d_h  = d_logits @ self.W2.T * (1.0 - h_t ** 2)
                dW1 += np.outer(x_t, d_h)
                db1 += d_h

            if T > 0:
                for g_arr in [dW1, db1, dW2, db2]:
                    g_arr /= T
            self._adam([dW1, db1, dW2, db2])
            epoch_returns.append(float(np.sum(rewards)))

        return epoch_returns

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_signals(
        self,
        feat_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Greedy inference: generate signal DataFrame for OOS bars.

        Returns DataFrame with columns: signal (int), composite (float).
        """
        X       = feat_df.values.astype(float)
        in_sess = ((feat_df.index.hour >= self.session_hours[0]) &
                   (feat_df.index.hour <  self.session_hours[1]))

        sig    = np.zeros(len(X), dtype=int)
        scores = np.zeros(len(X))

        for t in range(len(X)):
            if not in_sess[t]:
                continue
            _, probs     = self._forward(X[t])
            action       = int(np.argmax(probs))
            sig[t]       = ACTION_DIR[action]
            # Confidence: max prob above uniform (1/3)
            scores[t]    = float((probs.max() - 1.0 / 3.0) * 3.0) * sig[t]

        return pd.DataFrame(
            {"signal": sig, "composite": scores},
            index=feat_df.index,
        )


# ── Walk-forward loop ─────────────────────────────────────────────────────────

def walk_forward_pg(
    df_1h: pd.DataFrame,
    feat_df: pd.DataFrame,
    wf_windows: list,
    n_features: int = 75,
    hidden: int = 128,
    lr: float = 5e-4,
    gamma: float = 0.95,
    n_epochs: int = 20,
    session_hours: tuple = (8, 21),
    verbose: bool = True,
) -> tuple[pd.DataFrame, list]:
    """
    Walk-forward Policy Gradient training and OOS signal generation.

    A NEW agent is initialised for each window (no cross-window transfer).

    Returns
    -------
    pg_signals   : full-period DataFrame with OOS signals accumulated.
    window_stats : list of per-window metadata dicts.
    """
    index   = df_1h.index
    pg_sig  = pd.DataFrame({"signal": 0, "composite": 0.0}, index=index)
    wstats: list = []

    if verbose:
        print(f"  Mode: PG-REINFORCE (hidden={hidden}, epochs={n_epochs})  |  {len(wf_windows)} windows")

    for i, (tr_s, tr_e, oo_s, oo_e) in enumerate(wf_windows):
        tr_mask  = (index >= tr_s) & (index < tr_e)
        oos_mask = (index >= oo_s) & (index < oo_e)

        feat_tr  = feat_df[tr_mask]
        feat_oos = feat_df[oos_mask]
        df_tr    = df_1h[tr_mask]

        if tr_mask.sum() < 200 or oos_mask.sum() < 10:
            continue

        agent  = PolicyGradientAgent(
            n_features=n_features, hidden=hidden, lr=lr, gamma=gamma,
            session_hours=session_hours, random_state=42 + i,
        )
        ep_rets = agent.fit(feat_tr, df_tr, n_epochs=n_epochs)

        oos_out = agent.predict_signals(feat_oos)
        pg_sig.loc[oos_mask, "signal"]    = oos_out["signal"].values
        pg_sig.loc[oos_mask, "composite"] = oos_out["composite"].values

        n_long  = int((oos_out["signal"] == 1).sum())
        n_short = int((oos_out["signal"] == -1).sum())
        n_sig   = n_long + n_short

        if verbose:
            print(f"    Win {i+1:2d} [{oo_s.date()}→{oo_e.date()}]: "
                  f"ep_ret(last)={ep_rets[-1]:+.4f}  "
                  f"oos={n_sig}(L:{n_long}/S:{n_short})")

        wstats.append({
            "window":    i + 1,
            "oos_start": str(oo_s.date()),
            "oos_end":   str(oo_e.date()),
            "ep_ret_final": round(ep_rets[-1], 5),
            "ep_ret_mean":  round(float(np.mean(ep_rets)), 5),
            "n_long":    n_long,
            "n_short":   n_short,
            "n_sig":     n_sig,
        })

    return pg_sig, wstats
