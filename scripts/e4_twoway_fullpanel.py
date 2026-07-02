#!/usr/bin/env python3
"""
e4_twoway_fullpanel.py — additional robustness, with BOTH clusterings reported side by side.
  - Panel + heterogeneity (All / high- / low-gamma-exposure), baseline outcome.
  - Full S&P 500 panel for the additional outcomes (vol / tail / variance).
Each spec reports beta, t under date-clustering AND t under two-way (firm+date) clustering, plus
the first-stage F. Panel uses cross-sectional date-demeaning (time FE), which absorbs the VIX, so
no WRDS series is needed. Out: data/e4_twoway_fullpanel.csv
"""
import pathlib
import numpy as np, pandas as pd
from linearmodels.iv import IV2SLS

REAL = pathlib.Path(__file__).resolve().parent.parent / "data" / "real"
DATA = pathlib.Path(__file__).resolve().parent.parent / "data"

def rz(s, win=60): return (s - s.rolling(win).mean()) / s.rolling(win).std()

def run(dfin, ycol, label):
    df = dfin[["date", "secid", ycol, "c", "Z"]].dropna().copy()
    for v in (ycol, "c", "Z"):
        df[v + "_d"] = df[v] - df.groupby("date")[v].transform("mean")
    df = df.dropna(subset=[ycol + "_d", "c_d", "Z_d"]); df["const"] = 1.0
    dcodes = df["date"].astype("category").cat.codes
    scodes = df["secid"].astype("category").cat.codes
    md = IV2SLS(df[ycol+"_d"], df[["const"]], df["c_d"], df["Z_d"]).fit(
        cov_type="clustered", clusters=dcodes.values)
    mt = IV2SLS(df[ycol+"_d"], df[["const"]], df["c_d"], df["Z_d"]).fit(
        cov_type="clustered", clusters=np.column_stack([scodes, dcodes]))
    F = float(md.first_stage.diagnostics.loc["c_d", "f.stat"])
    return (label, len(df), round(float(md.params["c_d"]), 4),
            round(float(md.tstats["c_d"]), 2), round(float(mt.tstats["c_d"]), 2), round(F, 0))

def main():
    p = pd.read_parquet(REAL / "panel_sp500_cp.parquet")
    p["date"] = pd.to_datetime(p["date"]); p = p.sort_values(["secid", "date"]).reset_index(drop=True)
    for c in ("call_g", "put_g", "spot", "ret", "realized_vol", "roll_call", "roll_put"):
        p[c] = pd.to_numeric(p[c], errors="coerce")
    sp2 = p["spot"] ** 2 * 0.01
    p["net"] = (p["call_g"] - p["put_g"]) * sp2; p["roll"] = (p["roll_call"] - p["roll_put"])
    p["c"] = p.groupby("secid")["net"].transform(rz)
    p["Z"] = p.groupby("secid")["roll"].transform(rz)
    p["Yb"]   = p.groupby("secid")["realized_vol"].transform(lambda s: rz(s.shift(-5)))
    p["Yvol"] = p.groupby("secid")["ret"].transform(lambda r: rz(r.rolling(5).std().shift(-5)))
    p["Ytail"]= p.groupby("secid")["ret"].transform(lambda r: rz(r.abs().rolling(5).max().shift(-5)))
    p["Yvar"] = p.groupby("secid")["ret"].transform(lambda r: rz((r**2).rolling(5).sum().shift(-5)))
    # gamma-exposure split (per-name mean |net dollar gamma|), median split across names
    expo = p.groupby("secid")["net"].apply(lambda s: s.abs().mean())
    hi = set(expo[expo >= expo.median()].index)
    p["half"] = np.where(p["secid"].isin(hi), "high", "low")

    rows = [("spec", "firm_days", "beta", "t_date", "t_twoway", "F")]
    print("=== Panel IV + heterogeneity (baseline outcome) ===")
    for r in (run(p, "Yb", "All (710 names)"),
              run(p[p.half == "high"], "Yb", "High gamma-exposure half"),
              run(p[p.half == "low"],  "Yb", "Low gamma-exposure half")):
        rows.append(r); print("  ", r)
    print("=== Full-panel additional outcomes ===")
    for yc, lab in (("Yvol", "Realized vol (5d)"), ("Ytail", "Tail max|ret| (5d)"),
                    ("Yvar", "Realized variance (5d)")):
        r = run(p, yc, lab); rows.append(r); print("  ", r)
    (DATA / "e4_twoway_fullpanel.csv").write_text("\n".join(",".join(str(c) for c in r) for r in rows))
    print("\n  wrote data/e4_twoway_fullpanel.csv")

if __name__ == "__main__":
    main()
