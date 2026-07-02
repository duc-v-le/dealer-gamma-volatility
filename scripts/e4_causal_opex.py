#!/usr/bin/env python3
"""
e4_causal_opex.py — getting past "subsumed by VIX": an expiration-calendar IV
============================================================================
The cross-name test showed dealer short-gamma concentration LEADS vol but is
largely SUBSUMED BY VIX as a predictor. To argue the gamma channel is CAUSAL
(not just correlated with the vol environment) requires variation in dealer gamma
that is orthogonal to VIX. The monthly **options-expiration (OPEX) calendar**
provides it: on the 3rd Friday a large block of gamma mechanically rolls off, so
dealer gamma swings on a fixed schedule that is plausibly exogenous to that day's
volatility innovation.

Design (all on the real SPX panel + VIX):
  • First stage — does the calendar move gamma?  gross_gamma_z ~ days_to_OPEX + VIX_z.
  • Reduced form — does the calendar move FUTURE vol beyond VIX?  futureVol ~ days_to_OPEX + VIX_z.
  • 2SLS — calendar-instrumented gamma's effect on future vol, net of VIX.
  • Event study — gamma & vol in event-time around OPEX.

The exclusion restriction (OPEX affects vol only through dealer
gamma) is not airtight — OPEX also brings hedging-unwind and volume effects — so the
estimates are best read as suggestive causal evidence, not airtight identification.

Run:  python3 e4_causal_opex.py  →  figures/e4_opex_eventstudy.png, data/e4_causal_opex.csv
"""
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wrds_lib as w

ROOT = pathlib.Path(__file__).resolve().parent.parent
REAL = ROOT / "data" / "real"; FIG = ROOT / "figures"; DATA = ROOT / "data"
NAVY, TEAL, RED, AMBER = "#1f3b63", "#2a9d8f", "#c1432e", "#e9a23b"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "axes.titleweight": "bold"})

def roll_z(s, w=60):
    s = pd.Series(s); return ((s - s.rolling(w).mean()) / s.rolling(w).std()).values

def third_friday(y, m):
    d = pd.Timestamp(y, m, 1)
    first_fri = d + pd.Timedelta(days=(4 - d.weekday()) % 7)
    return first_fri + pd.Timedelta(days=14)

def hac_t(X, y, L=10):
    XtX_inv = np.linalg.inv(X.T @ X); beta = XtX_inv @ X.T @ y
    u = y - X @ beta; n, k = X.shape; Xu = X * u[:, None]; S = Xu.T @ Xu
    for l in range(1, L + 1):
        S += (1 - l/(L+1)) * (Xu[l:].T @ Xu[:n-l] + (Xu[l:].T @ Xu[:n-l]).T)
    se = np.sqrt(np.diag(XtX_inv @ S @ XtX_inv))
    return beta, beta / se

def main():
    p = pd.read_csv(REAL / "spx_gamma_panel.csv", parse_dates=["date"]).dropna(subset=["realized_vol"]).reset_index(drop=True)
    vix = pd.concat([w.query(f"SELECT date, close AS vix FROM optionm.secprd{y} WHERE secid=117801 ORDER BY date;")
                     for y in range(2016, 2025)])
    vix["date"] = pd.to_datetime(vix["date"]); vix["vix"] = vix["vix"].astype(float)
    p = p.merge(vix, on="date", how="inner")

    # OPEX calendar → days to next monthly OPEX for each trading day
    opex = sorted({third_friday(d.year, d.month) for d in p["date"]} |
                  {third_friday((d + pd.offsets.MonthBegin(1)).year, (d + pd.offsets.MonthBegin(1)).month) for d in p["date"]})
    opex = np.array([np.datetime64(x) for x in opex])
    dts = p["date"].values.astype("datetime64[D]")
    d2o = np.array([ (opex[opex >= t].min() - t).astype("timedelta64[D]").astype(int) if (opex >= t).any() else np.nan
                     for t in dts ], dtype=float)
    p["days_to_opex"] = d2o

    gz, vz, xz = roll_z(p["gross_gamma_oi"].values), roll_z(p["realized_vol"].values), roll_z(p["vix"].values)
    Z = roll_z(p["days_to_opex"].values)        # instrument (calendar position), standardized
    # future 5d vol
    H = 5; T = len(vz)
    Yf = np.array([np.nanmean(vz[t+1:t+1+H]) for t in range(T)])
    m = np.isfinite(gz) & np.isfinite(vz) & np.isfinite(xz) & np.isfinite(Z) & np.isfinite(Yf)
    gz, vz, xz, Z, Yf = gz[m], vz[m], xz[m], Z[m], Yf[m]
    n = m.sum()

    # First stage: gamma ~ Z + VIX
    X1 = np.column_stack([np.ones(n), Z, xz]); b1, t1 = hac_t(X1, gz)
    Ghat = X1 @ b1
    F1 = t1[1] ** 2
    # Reduced form: future vol ~ Z + VIX
    Xr = np.column_stack([np.ones(n), Z, xz]); br, tr = hac_t(Xr, Yf)
    # 2SLS: future vol ~ Ghat + VIX  (calendar-driven gamma, net of VIX)
    X2 = np.column_stack([np.ones(n), Ghat, xz]); b2, t2 = hac_t(X2, Yf)
    # Benchmark OLS (gamma not instrumented), with VIX control
    Xo = np.column_stack([np.ones(n), gz, xz]); bo, to = hac_t(Xo, Yf)

    print("E4 causal (OPEX-calendar IV for dealer gamma; SPX, controlling for VIX)")
    print(f"  first stage  gamma ~ days_to_OPEX:   coef={b1[1]:+.3f}  t={t1[1]:+.2f}  (F≈{F1:.1f})")
    print(f"  reduced form futVol ~ days_to_OPEX:  coef={br[1]:+.3f}  t={tr[1]:+.2f}")
    print(f"  2SLS         futVol ~ gamma_hat:     coef={b2[1]:+.3f}  t={t2[1]:+.2f}   <-- causal-flavored")
    print(f"  (cf. OLS gamma w/ VIX control:       coef={bo[1]:+.3f}  t={to[1]:+.2f})")

    # event study around OPEX
    idx_opex = np.array([np.argmin(np.abs(dts - o)) for o in opex if (dts.min() <= o <= dts.max())])
    win = range(-6, 9)
    g_es = {k: [] for k in win}; v_es = {k: [] for k in win}
    gz_full, vz_full = roll_z(p["gross_gamma_oi"].values), roll_z(p["realized_vol"].values)
    for i in idx_opex:
        for k in win:
            j = i + k
            if 0 <= j < len(p):
                g_es[k].append(gz_full[j]); v_es[k].append(vz_full[j])
    gm = [np.nanmean(g_es[k]) for k in win]; vm = [np.nanmean(v_es[k]) for k in win]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(list(win), gm, "o-", color=NAVY, label="gross gamma (z)")
    ax.plot(list(win), vm, "s-", color=RED, label="realized vol (z)")
    ax.axvline(0, color="grey", ls="--", lw=1); ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlabel("trading days from monthly OPEX (3rd Friday)"); ax.set_ylabel("event-time mean (z)")
    ax.set_title("E4 — OPEX gamma roll-off: gamma drops, vol picks up after expiration")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "e4_opex_eventstudy.png", bbox_inches="tight", facecolor="white"); plt.close(fig)

    rows = [("spec", "coef", "t_HAC"),
            ("first_stage_gamma~Z", round(b1[1], 4), round(t1[1], 2)),
            ("reduced_form_vol~Z", round(br[1], 4), round(tr[1], 2)),
            ("2sls_vol~gammahat", round(b2[1], 4), round(t2[1], 2)),
            ("ols_vol~gamma|VIX", round(bo[1], 4), round(to[1], 2)),
            ("first_stage_F", round(F1, 1), "")]
    (DATA / "e4_causal_opex.csv").write_text("\n".join(",".join(str(c) for c in r) for r in rows))
    print("\n  wrote figures/e4_opex_eventstudy.png, data/e4_causal_opex.csv")
    print("  Honest reading: if first stage is strong (F≫10) AND reduced form / 2SLS is")
    print("  significant beyond VIX, the calendar-driven (exogenous) part of gamma moves vol")
    print("  → causal-flavored support for the mechanism, rescuing it from the 'just VIX'")
    print("  critique. A weak/insignificant 2SLS = a genuine null (can't separate from VIX even")
    print("  with the IV). Exclusion restriction (OPEX→vol only via gamma) is not airtight.")

if __name__ == "__main__":
    main()
