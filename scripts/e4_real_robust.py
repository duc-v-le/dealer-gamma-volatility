#!/usr/bin/env python3
"""
e4_real_robust.py — robustness for the real-data E4 lead-lag result
===================================================================
Addresses the two caveats from e4_leadlag_real.py on the real SPX panel:

  (1) Newey-West (HAC) inference. The H=5 overlapping future-vol windows make OLS
      residuals autocorrelated, inflating t-stats. Recompute the predictive
      coefficient's t-stat with Bartlett-kernel HAC SEs (lag L). The point estimate
      and sign are unchanged; the honest t-stat is the HAC one.

  (2) Episode overlay. Does dealer short-gamma concentration build AHEAD of known
      stress? Plot stationarized short-gamma intensity vs realized vol around
      Feb-2018 (volmageddon), Mar-2020 (COVID), Jan-2021 (meme/gamma), 2022 selloff.
      (The 2010 Flash Crash is intentionally NOT shown: it predates 0DTE and is a
      liquidity/HFT event, not a dealer-gamma one — including it would mislead.)

Run:  python3 e4_real_robust.py  →  figures/e4_real_episodes.png, data/e4_real_hac.csv
"""
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
DATA = ROOT / "data"; DATA.mkdir(exist_ok=True)
PANEL = ROOT / "data" / "real" / "spx_gamma_panel.csv"
NAVY, TEAL, RED, AMBER = "#1f3b63", "#2a9d8f", "#c1432e", "#e9a23b"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "axes.titleweight": "bold"})

def roll_z(s, w=60):
    s = pd.Series(s); return ((s - s.rolling(w).mean()) / s.rolling(w).std()).values

def predictive_hac(conc, vol, H=5, L=10):
    """Future H-day mean vol on conc_t controlling for vol_t; OLS + Newey-West(L) t-stats."""
    T = len(vol)
    Y = np.array([np.nanmean(vol[t+1:t+1+H]) for t in range(T - H - 1)])
    c, v = conc[:T-H-1], vol[:T-H-1]
    m = np.isfinite(Y) & np.isfinite(c) & np.isfinite(v)
    Y, c, v = Y[m], c[m], v[m]
    X = np.column_stack([np.ones_like(c), c, v])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ Y
    u = Y - X @ beta
    n, k = X.shape
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for l in range(1, L + 1):                       # Bartlett kernel
        w_l = 1.0 - l / (L + 1)
        G = Xu[l:].T @ Xu[:n - l]
        S += w_l * (G + G.T)
    V_hac = XtX_inv @ S @ XtX_inv
    se_hac = np.sqrt(np.diag(V_hac))
    se_ols = np.sqrt(np.diag((u @ u) / (n - k) * XtX_inv))
    return beta[1], beta[1] / se_ols[1], beta[1] / se_hac[1]

def main():
    if not PANEL.exists():
        raise SystemExit("Run wrds_download_spx_gamma.py first.")
    p = pd.read_csv(PANEL, parse_dates=["date"]).dropna(subset=["realized_vol"]).reset_index(drop=True)
    p["short_intensity"] = np.maximum(-p["net_gex_dollar_pct"], 0.0)
    si_z = roll_z(p["short_intensity"].values)
    vol_z = roll_z(p["realized_vol"].values)

    # (1) HAC inference
    print("E4 robustness (1): Newey-West HAC inference on the real lead-lag predictive test")
    rows = [("lag_L", "beta_conc->futureVol", "t_OLS", "t_HAC")]
    for L in (5, 10, 21):
        b, t_ols, t_hac = predictive_hac(si_z, vol_z, H=5, L=L)
        rows.append((L, round(b, 3), round(t_ols, 2), round(t_hac, 2)))
        print(f"  L={L:<3}  beta={b:+.3f}  t_OLS={t_ols:+.2f}  t_HAC={t_hac:+.2f}")
    (DATA / "e4_real_hac.csv").write_text("\n".join(",".join(str(c) for c in r) for r in rows))
    print("  wrote data/e4_real_hac.csv  (HAC t-stats are the honest ones; OLS overstated)")

    # (2) episode overlay
    p["si_z"], p["vol_z"] = si_z, vol_z
    episodes = [("2018-02-05", "Feb-2018 volmageddon"),
                ("2020-03-16", "Mar-2020 COVID crash"),
                ("2021-01-27", "Jan-2021 meme/gamma"),
                ("2022-06-13", "2022 selloff")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, (d, label) in zip(axes.ravel(), episodes):
        ev = pd.Timestamp(d)
        idx = (p["date"] - ev).abs().idxmin()
        lo, hi = max(0, idx - 60), min(len(p), idx + 25)
        seg = p.iloc[lo:hi]
        x = (seg["date"] - ev).dt.days
        ax.plot(x, seg["si_z"], color=RED, lw=1.6, label="dealer short-gamma (z)")
        ax.plot(x, seg["vol_z"], color=NAVY, lw=1.6, label="realized vol (z)")
        ax.axvline(0, color="grey", ls="--", lw=1)
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_title(label, fontsize=10); ax.set_xlabel("trading days from event")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("E4 — does dealer short-gamma concentration build AHEAD of stress? (real SPX)",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG / "e4_real_episodes.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("\n  wrote figures/e4_real_episodes.png")
    print("  (2010 Flash Crash intentionally omitted: pre-0DTE, liquidity/HFT not dealer-gamma.)")
    print("\n  Honest reading: the HAC t-stat is the inference to quote (OLS overstated by overlap).")
    print("  If short-gamma (red) rises before realized vol (navy) across episodes, it corroborates")
    print("  the +2-day lead; episodes where it spikes only WITH vol are honest exceptions to note.")

if __name__ == "__main__":
    main()
