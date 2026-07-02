#!/usr/bin/env python3
"""
e4_panel_mechanism.py — non-linearity / mechanism slice (convexity near stress)
===================================================================================
The gamma-feedback model is CONVEX: g = 1/(1-βf) accelerates as the system nears the
instability threshold (βf→1). Empirical prediction: the causal effect of dealer gamma on
volatility should be STRONGER in stress regimes (high VIX, closer to the cliff) than in
calm regimes — a signature of the specific feedback mechanism, not a linear correlation.

Test: re-estimate the full-S&P-500 panel IV (time-FE, date-clustered, roll-off instrument)
WITHIN VIX terciles. VIX is a date-level state (not the endogenous gamma), so conditioning
on it does not induce the usual conditioning-on-endogenous bias. If |β| rises low→high VIX,
that is the convex-feedback signature linking the empirics to the simulation model (E1-E5b).

Run:  python3 e4_panel_mechanism.py [panel=panel_sp500.parquet]  →  data/e4_panel_mechanism.csv
"""
import sys
import pathlib
import numpy as np
import pandas as pd
import wrds_lib as w
from linearmodels.iv import IV2SLS

ROOT = pathlib.Path(__file__).resolve().parent.parent
REAL = ROOT / "data" / "real"; DATA = ROOT / "data"

def rz(s, win=60):
    return (s - s.rolling(win).mean()) / s.rolling(win).std()

def main():
    src = REAL / (sys.argv[1] if len(sys.argv) > 1 else "panel_sp500.parquet")
    p = pd.read_parquet(src) if src.suffix == ".parquet" else pd.read_csv(src, parse_dates=["date"])
    p["date"] = pd.to_datetime(p["date"])
    vix = pd.concat([w.query(f"SELECT date, close AS vix FROM optionm.secprd{y} WHERE secid=117801 ORDER BY date;")
                     for y in range(2016, 2025)])
    vix["date"] = pd.to_datetime(vix["date"]); vix["vix"] = vix["vix"].astype(float)
    p = p.merge(vix, on="date", how="left").sort_values(["secid", "date"]).reset_index(drop=True)

    g = p.groupby("secid", group_keys=False)
    p["c"]  = g["net_gex"].apply(rz)
    p["Z"]  = g["scheduled_rolloff_net"].apply(rz)
    p["fv"] = g["realized_vol"].apply(lambda s: s.shift(-5))
    p["Y"]  = g.apply(lambda x: rz(x["fv"])).reset_index(level=0, drop=True)
    p["vixz"] = rz(p["vix"])

    # VIX terciles by date (date-level state)
    dvix = p.groupby("date")["vix"].first().dropna()
    q1, q2 = dvix.quantile([1/3, 2/3])
    def regime(v): return "calm (low VIX)" if v <= q1 else ("stress (high VIX)" if v >= q2 else "mid VIX")
    p["regime"] = p["vix"].map(regime)

    def panel_iv(df, label):
        df = df.dropna(subset=["Y", "c", "Z", "vixz"]).copy()
        # time-FE within the subsample
        for v in ("Y", "c", "Z", "vixz"):
            df[v + "_d"] = df[v] - df.groupby("date")[v].transform("mean")
        df["const"] = 1.0
        df = df.dropna(subset=["Y_d", "c_d", "Z_d", "vixz_d"])
        m = IV2SLS(df["Y_d"], df[["const", "vixz_d"]], df["c_d"], df["Z_d"]).fit(
            cov_type="clustered", clusters=df["date"].astype("category").cat.codes)
        F = m.first_stage.diagnostics.loc["c_d", "f.stat"]
        return (label, len(df), round(m.params["c_d"], 4), round(m.tstats["c_d"], 2), round(F, 0))

    rows = [("regime", "firm_days", "beta(net_gamma)", "t", "firstF")]
    rows.append(panel_iv(p, "ALL (pooled)"))
    for r in ("calm (low VIX)", "mid VIX", "stress (high VIX)"):
        rows.append(panel_iv(p[p["regime"] == r], r))

    print("mechanism: is the gamma→vol effect stronger near stress (convex feedback)?\n")
    print(f"  {'regime':<22}{'firm-days':>11}{'beta':>9}{'t':>7}{'F':>9}")
    for r in rows[1:]:
        print(f"  {r[0]:<22}{r[1]:>11,}{r[2]:>9}{r[3]:>7}{r[4]:>9}")
    (DATA / "e4_panel_mechanism.csv").write_text("\n".join(",".join(str(c) for c in r) for r in rows))
    print("\n  wrote data/e4_panel_mechanism.csv")
    print("  Prediction (convex feedback g=1/(1-βf)): |β| rises calm → stress. If so, the empirics")
    print("  carry the simulation model's signature (effect accelerates near the instability cliff),")
    print("  not just a linear correlation. Flat |β| across regimes = honest non-confirmation.")

if __name__ == "__main__":
    main()
