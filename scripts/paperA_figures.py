#!/usr/bin/env python3
"""
paperA_figures.py — Paper A figures F1–F4 (saved to figures/, never shown inline).
F1 first-stage scatter (roll-off -> net gamma, rolling-z, SPX)
F2 IV beta across forward-vol horizons (1/5/10/21d) with 95% CI
F3 aggregate index vol by dealer-gamma regime (short vs long half)
F4 sign-convention decomposition (net/call/put/gross betas)
Run: python3 paperA_figures.py
"""
import pathlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA, REAL, FIG = ROOT / "data", ROOT / "data" / "real", ROOT / "figures"
FIG.mkdir(exist_ok=True)
BLUE, RED, GREY, INK = "#2166AC", "#B2182B", "#696868", "#1a1a1a"  # refined ColorBrewer-style palette
LIGHT_BLUE, LIGHT_RED, LIGHT_GREY = "#6397B5", "#D35F55", "#A9A8A8"  # lighter shades for the fit line and annotations
BOLD_BLUE, BOLD_RED, BOLD_GREY = "#1407CE", "#F11414", "#313030"   # fresher palette for the dots
plt.rcParams.update({
    "font.family": "serif", "font.size": 12, "axes.titlesize": 12.5, "axes.labelsize": 12,
    "axes.labelweight": "bold",                                   # bold axis labels
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "axes.linewidth": 1.4, "axes.edgecolor": INK,                 # bold axes
    "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    "xtick.major.size": 5, "ytick.major.size": 5, "xtick.color": INK, "ytick.color": INK,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.6, "lines.linewidth": 2.2,
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
})
FW = (8.6, 5.0)  # uniform full-width figure size
def rz(s, w=60): return (s - s.rolling(w).mean()) / s.rolling(w).std()

# ---------- F1: first-stage scatter ----------
def f1():
    d = pd.read_csv(REAL / "spx_rolloff.csv")
    x = rz(d["scheduled_rolloff_net"]); y = rz(d["net_all"])
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    b1, b0 = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    fig, ax = plt.subplots(figsize=FW)
    # dots split by dealer-gamma sign (blue=long, red=short); black fit line (mirrors Adams et al.)
    yv = np.asarray(y); xv = np.asarray(x); lng = yv >= 0
    ax.scatter(xv[lng],  yv[lng],  s=13, alpha=0.45, color=BLUE, edgecolors="none", label="dealer long gamma")
    ax.scatter(xv[~lng], yv[~lng], s=13, alpha=0.50, color=RED,  edgecolors="none", label="dealer short gamma")
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, b0 + b1 * xs, color=INK, lw=2.4, label=f"first stage: slope = {b1:.2f}")
    ax.set_xlabel("Scheduled gamma roll-off  $Z_t$  (rolling-z)")
    ax.set_ylabel("Dealer net gamma  $\\tilde G_t$  (rolling-z)")
    ax.set_title("F1 — First stage: the roll-off strongly shifts dealer gamma (SPX)")
    # report the PROPER controlled first-stage F from the 2SLS (Table 2), not a univariate proxy
    ax.text(0.04, 0.94, f"slope={b1:.2f},  $R^2$={r**2:.2f},  $n=${len(x):,}\n"
            f"controlled first-stage $F\\approx$1,443 (Table 2)", transform=ax.transAxes,
            va="top", fontsize=9, bbox=dict(boxstyle="round", fc="white", ec=GREY))

    # Create the legend normally
    leg = ax.legend(loc="lower right", frameon=False)
    
    # Safely modify only the scatter markers in the legend
    for handle in leg.legend_handles:
        handle.set_alpha(0.8)  # Makes all colors solid/bold
        if hasattr(handle, "set_sizes"):
            handle.set_sizes([40])  # Makes the red/blue circles bigger

    fig.tight_layout(); fig.savefig(FIG / "paperA_F1_firststage.png", dpi=150); plt.close(fig)
    return f"F1 (slope={b1:.2f}, R2={r**2:.2f}, n={len(x):,})"

# ---------- F2: IV beta across horizons ----------
def f2():
    d = pd.read_csv(DATA / "e4_horizon.csv")
    se = (d["beta"].abs() / d["t"].abs()); ci = 1.96 * se
    fig, ax = plt.subplots(figsize=FW)
    ax.errorbar(d["horizon_days"], d["beta"], yerr=ci, fmt="o-", color=GREY, 
                markerfacecolor=BOLD_BLUE, markeredgecolor=BOLD_BLUE, ecolor=BOLD_BLUE,
                capsize=4, lw=1.2, ms=7, label="IV $\\hat\\beta$ (net gamma)")
    ax.axhline(0, color=GREY, lw=1, ls="--")
    for _, r in d.iterrows():
        ax.annotate(f"t={r['t']:.0f}", (r["horizon_days"], r["beta"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8, color=GREY)
    ax.set_xticks(d["horizon_days"])
    ax.set_xlabel("Forward realized-volatility horizon (trading days)")
    ax.set_ylabel("IV coefficient  $\\hat\\beta$  (standardized)")
    ax.set_title("F2 — Causal effect is robust across horizons (1–21 days)")
    ax.legend(loc="lower left", frameon=False); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(FIG / "paperA_F2_horizons.png", dpi=150); plt.close(fig)
    return "F2 (horizons 1/5/10/21)"

# ---------- F3: aggregate vol by gamma regime ----------
def f3():
    s = pd.read_csv(DATA / "e4_systemic_aggregate.csv").set_index("metric")["value"]
    short = float(s.loc["avg ann index vol: low-gamma (more short) half"])
    long_ = float(s.loc["avg ann index vol: high-gamma (more long) half"])
    shr = float(s.loc["share of days aggregate dealers NET SHORT gamma"]) * 100
    fig, ax = plt.subplots(figsize=FW)
    bars = ax.bar(["Dealers more SHORT\ngamma (half of days)", "Dealers more LONG\ngamma (half of days)"],
                  [short, long_], color=[LIGHT_RED, LIGHT_BLUE], width=0.38, edgecolor="#222222", linewidth=1.0)
    for b, v in zip(bars, [short, long_]):
        ax.text(b.get_x() + b.get_width()/2, v + 0.3, f"{v:.1f}%", ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Average annualized S&P 500 index volatility")
    ax.set_title("F3 — Short-gamma states ≈ double the index volatility", fontsize=11)
    ax.text(0.5, 0.92, f"Dealers net short on only {shr:.1f}% of days", transform=ax.transAxes,
            ha="center", fontsize=10, color=GREY)
    ax.set_ylim(0, max(short, long_) * 1.2); ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(FIG / "paperA_F3_regime.png", dpi=150); plt.close(fig)
    return f"F3 (short {short:.1f}% vs long {long_:.1f}%)"

# ---------- F4: sign-convention decomposition ----------
def f4():
    d = pd.read_csv(DATA / "e4_sign_robust.csv")
    lab = {"NET (call−put) [baseline]": "NET\n(call−put)", "CALL gamma only": "CALL\nonly",
           "PUT gamma only": "PUT\nonly", "GROSS (call+put)": "GROSS\n(call+put)"}
    d["lab"] = d["convention"].map(lab)
    colors = [LIGHT_BLUE if b < 0 else LIGHT_RED for b in d["beta"]]
    fig, ax = plt.subplots(figsize=FW)
    bars = ax.bar(d["lab"], d["beta"], color=colors, width=0.62, edgecolor="#222222", linewidth=0.9)
    ax.axhline(0, color="black", lw=1)
    lo, hi = d["beta"].min(), d["beta"].max()
    ax.set_ylim(lo * 1.45, hi * 2.4)
    for b, beta, t in zip(bars, d["beta"], d["t"]):
        # negative bars: label inside, just below the zero line; positive bar: just above its top
        y = (-0.0010) if beta < 0 else (beta + 0.0010)
        va = "top" if beta < 0 else "bottom"
        ax.text(b.get_x()+b.get_width()/2, y, f"{beta:+.3f}\n(t={t:+.1f})",
                ha="center", va=va, fontsize=9)
    ax.set_ylabel("Panel-IV coefficient  $\\hat\\beta$")
    ax.set_title("F4 — Sign convention validated: call dampens (−), put amplifies (+)", fontsize=10.5)
    ax.text(0.015, 0.97, "blue = vol-dampening (β<0)\nred = vol-amplifying (β>0)",
            transform=ax.transAxes, va="top", fontsize=8, color=GREY)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(FIG / "paperA_F4_sign.png", dpi=150); plt.close(fig)
    return "F4 (call<0, put>0)"

# ---------- F5: SPX net dealer gamma time series ----------
def f5():
    d = pd.read_csv(REAL / "spx_rolloff.csv", parse_dates=["date"])
    g = (pd.to_numeric(d["net_all"], errors="coerce") * pd.to_numeric(d["spot"], errors="coerce")**2 * 0.01 / 1e9).values
    x = d["date"].values
    fig, ax = plt.subplots(figsize=FW)
    # line/area form (blue = long gamma, red = short gamma)
    ax.fill_between(x, g, 0, where=(g >= 0), color=BLUE, alpha=0.55, label="dealers long gamma (dampening)")
    ax.fill_between(x, g, 0, where=(g < 0),  color=RED,  alpha=0.6,  label="dealers short gamma (amplifying)")
    ax.axhline(0, color=INK, lw=1.1)
    ax.set_ylabel("SPX dealer net gamma\n($bn per 1% move)")
    ax.set_xlabel("")
    ax.set_title("F5 — Dealer net gamma over time (SPX): the variation the instrument exploits", fontsize=10.5)
    ax.legend(loc="upper left", frameon=False, fontsize=9); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(FIG / "paperA_F5_gamma_timeseries.png", dpi=150); plt.close(fig)
    return "F5 (SPX gamma time series)"

# ---------- F6: distribution of firm-day dealer net gamma ----------
def f6():
    p = pd.read_parquet(REAL / "panel_sp500_cp.parquet", columns=["call_g", "put_g", "spot"])
    g = (p["call_g"] - p["put_g"]) * p["spot"]**2 * 0.01 / 1e9
    g = g.replace([np.inf, -np.inf], np.nan).dropna()
    short = float((g < 0).mean()) * 100
    lo, hi = -0.03, 0.06                       # bounded view of the bulk (no clip pile-up)
    edges = np.linspace(lo, hi, 121)
    inrange = float(((g >= lo) & (g <= hi)).mean()) * 100
    fig, ax = plt.subplots(figsize=FW)
    ax.hist(g[g >= 0], bins=edges, color=BLUE, alpha=0.85, label="long gamma", edgecolor="white", linewidth=0.5)
    ax.hist(g[g < 0],  bins=edges, color=RED,  alpha=0.9,  label="short gamma", edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", lw=1)
    ax.set_yscale("log")                        # huge near-zero mass + short tail both visible
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Firm-day dealer net gamma ($bn per 1% move)")
    ax.set_ylabel("Firm-days (log scale)")
    ax.set_title("F6 — Dealers are net short gamma on 26% of firm-days (cross-sectional variation)", fontsize=10)
    ax.text(0.02, 0.95, f"short-gamma mass = {short:.0f}% of firm-days\n({inrange:.0f}% of firm-days shown in range)",
            transform=ax.transAxes, va="top", fontsize=8.5, color=RED,
            bbox=dict(boxstyle="round", fc="white", ec=GREY))
    ax.legend(loc="upper right", frameon=False); ax.grid(alpha=0.2, axis="y")
    fig.tight_layout(); fig.savefig(FIG / "paperA_F6_gamma_distribution.png", dpi=150); plt.close(fig)
    return f"F6 (short mass {short:.0f}%)"

# ---------- F7: OPEX event study (visualizes the roll-off identification) ----------
def _third_fridays():
    out = []
    for y in range(2016, 2025):
        for m in range(1, 13):
            c = pd.Timestamp(y, m, 1)
            out.append((c + pd.Timedelta(days=(4 - c.weekday()) % 7) + pd.Timedelta(days=14)).normalize())
    return sorted(out)

def f7():
    d = pd.read_csv(REAL / "spx_rolloff.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    d["d0"] = d["date"].dt.normalize()
    # trading-day index of each monthly OPEX (last trading day on/before the third Friday)
    opex_idx = sorted({d.index[d["d0"] <= o][-1] for o in _third_fridays() if (d["d0"] <= o).any()})
    oi = np.array(opex_idx)
    off = np.array([i - oi[np.argmin(np.abs(oi - i))] for i in range(len(d))])
    d["evt"] = off
    netg = (pd.to_numeric(d["net_all"], errors="coerce")
            * pd.to_numeric(d["spot"], errors="coerce")**2 * 0.01 / 1e9)
    rv = pd.to_numeric(d["realized_vol"], errors="coerce") * np.sqrt(252) * 100
    g = pd.DataFrame({"evt": d["evt"], "g": netg, "rv": rv})
    g = g[g["evt"].abs() <= 10]
    mg = g.groupby("evt")["g"].mean(); mrv = g.groupby("evt")["rv"].mean()
    fig, ax = plt.subplots(figsize=FW)
    ax.axvline(0, color=GREY, lw=1.2, ls="--")
    l1 = ax.plot(mg.index, mg.values, "o-", color=GREY, markerfacecolor=BLUE, markeredgecolor=BOLD_BLUE, lw=1, ms=5, label="dealer net gamma ($bn per 1%)")
    ax.annotate("gamma rolls off\nat expiration", xy=(0, mg.loc[0]), xytext=(-6.5, mg.loc[0] + 0.4),
                fontsize=9.5, color=GREY, ha="center",
                arrowprops=dict(arrowstyle="->", color=LIGHT_GREY, connectionstyle="arc3,rad=-0.2"))
    ax.set_xlabel("Event time: trading days relative to monthly expiration (0 = OPEX)")
    ax.set_ylabel("Mean dealer net gamma ($bn per 1%)", color=BLUE)
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True); ax2.spines["top"].set_visible(False); ax2.grid(False)
    l2 = ax2.plot(mrv.index, mrv.values, "s-", color=GREY, markerfacecolor=RED, markeredgecolor=BOLD_RED, lw=1, ms=5, label="realized volatility (ann. %)")
    ax2.set_ylabel("Mean realized volatility (ann. %)", color=RED)
    ax.set_title("F7 — Event study around monthly expiration: dealer gamma and volatility")
    ax.legend(l1 + l2, [x.get_label() for x in l1 + l2], loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "paperA_F7_opex_eventstudy.png", dpi=150); plt.close(fig)
    return f"F7 (OPEX event study, net gamma, n_evt={g['evt'].nunique()})"

# ---------- F8: lead-lag cross-correlation (motivates the VIX-subsumption story) ----------
def f8():
    d = pd.read_csv(REAL / "spx_rolloff.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    short = np.maximum(-(pd.to_numeric(d["net_all"], errors="coerce")
                         * pd.to_numeric(d["spot"], errors="coerce")**2 * 0.01), 0.0)
    cs = rz(short); v = rz(pd.to_numeric(d["realized_vol"], errors="coerce"))
    lags = list(range(-10, 11))
    cc = [cs.corr(v.shift(-k)) for k in lags]     # corr(short-gamma_t, vol_{t+k})
    kbest = lags[int(np.nanargmax(cc))]
    fig, ax = plt.subplots(figsize=FW)
    colors = [RED if L > 0 else GREY for L in lags]
    ax.bar(lags, cc, color=colors, width=0.8, edgecolor="#222222", linewidth=0.8)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Lag $k$ (trading days): correlation of dealer short-gamma$_t$ with volatility$_{t+k}$")
    ax.set_ylabel("Cross-correlation")
    ax.set_title("F8 — Dealer short-gamma leads realized volatility (peak at $k>0$)", fontsize=10.5)
    ax.text(0.98, 0.95, f"peak lead at k = {kbest} days\n(red = future volatility)", transform=ax.transAxes,
            ha="right", va="top", fontsize=8.5, color=RED, bbox=dict(boxstyle="round", fc="white", ec=GREY))
    ax.grid(alpha=0.2, axis="y"); fig.tight_layout()
    fig.savefig(FIG / "paperA_F8_leadlag.png", dpi=150); plt.close(fig)
    return f"F8 (lead-lag, peak k={kbest})"

if __name__ == "__main__":
    for fn in (f1, f2, f3, f4, f5, f6, f7, f8):
        print("  saved", fn())
    print("  -> figures/paperA_F1..F8 (PNG, 150 dpi)")
