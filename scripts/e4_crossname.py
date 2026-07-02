#!/usr/bin/env python3
"""
e4_crossname.py — E4 extensions: single-name generalization + VIX control
=========================================================================
(1) Cross-name: does dealer short-gamma concentration LEAD realized vol for the
    AI-trade single names (NVDA, TSLA, AAPL) as it does for SPX? Same stationarized
    lead-lag + Newey-West HAC test, run per name.
(2) VIX control (SPX): does the SPX lead survive controlling for the contemporaneous
    VIX *level*? (i.e., is short-gamma concentration more than a repackaging of the
    implied-vol level?) VIX pulled from OptionMetrics (secid 117801).

Inputs: data/real/{SPX,NVDA,TSLA,AAPL}_gamma_panel.csv (SPX = spx_gamma_panel.csv).
Run:    python3 e4_crossname.py  →  figures/e4_crossname.png, data/e4_crossname.csv
"""
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wrds_lib as w

ROOT = pathlib.Path(__file__).resolve().parent.parent
REAL = ROOT / "data" / "real"
FIG = ROOT / "figures"; DATA = ROOT / "data"
NAVY, TEAL, RED, AMBER = "#1f3b63", "#2a9d8f", "#c1432e", "#e9a23b"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "axes.titleweight": "bold"})

def roll_z(s, w=60):
    s = pd.Series(s); return ((s - s.rolling(w).mean()) / s.rolling(w).std()).values

def hac_t(X, y, L=10):
    """OLS beta with Newey-West (Bartlett, L) t-stats for every coefficient."""
    XtX_inv = np.linalg.inv(X.T @ X); beta = XtX_inv @ X.T @ y
    u = y - X @ beta; n, k = X.shape; Xu = X * u[:, None]
    S = Xu.T @ Xu
    for l in range(1, L + 1):
        wl = 1 - l / (L + 1); G = Xu[l:].T @ Xu[:n - l]; S += wl * (G + G.T)
    se = np.sqrt(np.diag(XtX_inv @ S @ XtX_inv))
    return beta, beta / se

def future_mean(vol, H=5):
    T = len(vol)
    return np.array([np.nanmean(vol[t+1:t+1+H]) for t in range(T - H - 1)]), T

def xcorr_peak(c, v, maxlag=15):
    c = (c - np.nanmean(c)) / np.nanstd(c); v = (v - np.nanmean(v)) / np.nanstd(v)
    best, bl = -9, 0
    for L in range(-maxlag, maxlag + 1):
        a, b = (c[:len(c)-L], v[L:]) if L >= 0 else (c[-L:], v[:len(v)+L])
        r = np.corrcoef(a, b)[0, 1]
        if r > best: best, bl = r, L
    return bl

def panel(ticker):
    f = REAL / ("spx_gamma_panel.csv" if ticker == "SPX" else f"{ticker}_gamma_panel.csv")
    p = pd.read_csv(f, parse_dates=["date"]).dropna(subset=["realized_vol"]).reset_index(drop=True)
    p["short_intensity"] = np.maximum(-p["net_gex_dollar_pct"], 0.0)
    return p

def leadlag(ticker, H=5, L=10):
    p = panel(ticker)
    siz, vz = roll_z(p["short_intensity"].values), roll_z(p["realized_vol"].values)
    Yf, T = future_mean(vz, H); c, v = siz[:T-H-1], vz[:T-H-1]
    m = np.isfinite(Yf) & np.isfinite(c) & np.isfinite(v)
    X = np.column_stack([np.ones(m.sum()), c[m], v[m]])
    beta, t = hac_t(X, Yf[m], L)
    peak = xcorr_peak(siz[np.isfinite(siz) & np.isfinite(vz)], vz[np.isfinite(siz) & np.isfinite(vz)])
    return dict(n=len(p), short_share=float((p["net_gamma_oi"] < 0).mean()),
                beta=float(beta[1]), t_hac=float(t[1]), xcorr_peak=peak)

def vix_control():
    """SPX: add contemporaneous VIX level as a control; does the lead survive?"""
    vix = pd.concat([w.query(f"SELECT date, close AS vix FROM optionm.secprd{y} WHERE secid=117801 ORDER BY date;")
                     for y in range(2016, 2025)]).reset_index(drop=True)
    vix["date"] = pd.to_datetime(vix["date"])
    p = panel("SPX").merge(vix, on="date", how="inner")
    siz, vz, xz = roll_z(p["short_intensity"].values), roll_z(p["realized_vol"].values), roll_z(p["vix"].values.astype(float))
    Yf, T = future_mean(vz, 5)
    c, v, x = siz[:T-6], vz[:T-6], xz[:T-6]
    m = np.isfinite(Yf) & np.isfinite(c) & np.isfinite(v) & np.isfinite(x)
    out = {}
    for name, cols in [("no VIX control", [c]), ("with VIX control", [c, x])]:
        X = np.column_stack([np.ones(m.sum())] + [a[m] for a in ([c, v] if name == "no VIX control" else [c, v, x])])
        beta, t = hac_t(X, Yf[m], 10)
        out[name] = (float(beta[1]), float(t[1]))
    return out

def main():
    print("E4 cross-name + VIX control")
    tickers = ["SPX", "NVDA", "TSLA", "AAPL"]
    rows = [("ticker", "n_days", "dealer_short_share", "beta_conc->futVol", "t_HAC", "xcorr_peak_lag")]
    res = {}
    for tk in tickers:
        try:
            r = leadlag(tk); res[tk] = r
            rows.append((tk, r["n"], round(r["short_share"], 3), round(r["beta"], 3),
                         round(r["t_hac"], 2), r["xcorr_peak"]))
            print(f"  {tk:5} t_HAC={r['t_hac']:+6.2f}  beta={r['beta']:+.3f}  xcorr_peak={r['xcorr_peak']:+d}  "
                  f"short_share={r['short_share']:.0%}")
        except FileNotFoundError:
            print(f"  {tk}: panel missing (run wrds_download_gamma.py {tk})")
    (DATA / "e4_crossname.csv").write_text("\n".join(",".join(str(c) for c in r) for r in rows))

    print("\n  VIX control (SPX): does the lead survive controlling for the VIX level?")
    vc = vix_control()
    for k, (b, t) in vc.items():
        print(f"    {k:18}  beta_conc={b:+.3f}  t_HAC={t:+.2f}")

    # figure: HAC t by name
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    names = [r[0] for r in rows[1:]]; ts = [r[4] for r in rows[1:]]
    ax.bar(names, ts, color=[NAVY, RED, AMBER, TEAL][:len(names)])
    ax.axhline(2, color="grey", ls="--", lw=1); ax.text(0, 2.3, "t=2", color="grey", fontsize=8)
    ax.set_ylabel("Newey-West t: short-gamma → future vol")
    ax.set_title("E4 — short-gamma concentration leads vol across SPX and AI single names")
    fig.tight_layout(); fig.savefig(FIG / "e4_crossname.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("\n  wrote figures/e4_crossname.png, data/e4_crossname.csv")
    print("  Honest reading: consistent positive HAC t across names ⇒ the lead generalizes; a")
    print("  surviving (shrunken) coeff under VIX control ⇒ concentration adds info beyond the vol level.")

if __name__ == "__main__":
    main()
