#!/usr/bin/env python3
"""
e4_iv2sls.py — PROPER 2SLS (linearmodels) for the gamma→vol causal claim
=============================================================================
Sharpens the hand-rolled IV with correct 2SLS inference and a richer instrument
set, on the expiry-bucketed SPX panel (wrds_download_spx_expiry.py).

Endogenous:  total dealer short-gamma intensity (rolling-z).
Instruments: (Z1) OPEX-calendar position (days-to-next-3rd-Friday), (Z2) near-expiry
             gamma share gamma_le7/gross_all — the mechanically-rolling component.
Control:     contemporaneous VIX level (rolling-z).
Outcome:     future 5-day realized vol (rolling-z).

Reports: OLS-with-VIX benchmark; just-identified IV (Z1) with kernel(HAC) SEs and the
first-stage F; over-identified IV (Z1,Z2) with the Wooldridge robust over-id (J) test.

The exclusion restriction (calendar/near-expiry gamma affect vol
only through dealer gamma) is not airtight; this specification tightens inference but does
not by itself make the identification airtight.

Run:  python3 e4_iv2sls.py  →  data/e4_iv2sls.csv
"""
import pathlib
import numpy as np
import pandas as pd
import wrds_lib as w
from linearmodels.iv import IV2SLS

ROOT = pathlib.Path(__file__).resolve().parent.parent
REAL = ROOT / "data" / "real"; DATA = ROOT / "data"

def roll_z(s, win=60):
    s = pd.Series(s); return ((s - s.rolling(win).mean()) / s.rolling(win).std())

def third_friday(y, m):
    d = pd.Timestamp(y, m, 1)
    return d + pd.Timedelta(days=(4 - d.weekday()) % 7) + pd.Timedelta(days=14)

def main():
    p = pd.read_csv(REAL / "spx_gamma_expiry.csv", parse_dates=["date"]).dropna(subset=["realized_vol"]).reset_index(drop=True)
    vix = pd.concat([w.query(f"SELECT date, close AS vix FROM optionm.secprd{y} WHERE secid=117801 ORDER BY date;")
                     for y in range(2016, 2025)])
    vix["date"] = pd.to_datetime(vix["date"]); vix["vix"] = vix["vix"].astype(float)
    p = p.merge(vix, on="date", how="inner")

    # OPEX calendar: days to next 3rd Friday
    opex = sorted({third_friday(d.year, d.month) for d in p["date"]} |
                  {third_friday((d + pd.offsets.MonthBegin(1)).year, (d + pd.offsets.MonthBegin(1)).month) for d in p["date"]})
    opex = np.array([np.datetime64(x) for x in opex])
    dts = p["date"].values.astype("datetime64[D]")
    p["days_to_opex"] = [int((opex[opex >= t].min() - t) / np.timedelta64(1, "D")) if (opex >= t).any() else np.nan for t in dts]

    # variables (stationarized)
    p["short_intensity"] = np.maximum(-(p["net_all"] * p["spot"]**2 * 0.01), 0.0)
    df = pd.DataFrame({
        "c":   roll_z(p["short_intensity"]),   # POLICY-relevant: dealer short-gamma intensity
        "cg":  roll_z(p["gross_all"]),         # gross gamma (what the hand-rolled IV instrumented — reconciliation)
        "vix": roll_z(p["vix"]),
        "Z1":  roll_z(p["days_to_opex"]),
        "Z2":  roll_z(p["near_share"]),
    })
    df["Y"] = roll_z(p["realized_vol"]).shift(-1).rolling(5).mean().shift(-(5-1))  # future 5d mean vol (z)
    df["const"] = 1.0
    df = df.dropna().reset_index(drop=True)
    n = len(df)

    out = []
    # (0) OLS with VIX control (benchmark: the "subsumed by VIX" result)
    ols = IV2SLS(df["Y"], df[["const", "c", "vix"]], None, None).fit(cov_type="kernel", kernel="bartlett")
    out.append(("OLS  Y~c+vix", round(ols.params["c"], 4), round(ols.tstats["c"], 2), "", ""))

    # (1) just-identified IV with the OPEX calendar (Z1)
    iv1 = IV2SLS(df["Y"], df[["const", "vix"]], df["c"], df["Z1"]).fit(cov_type="kernel", kernel="bartlett")
    F1 = iv1.first_stage.diagnostics.loc["c", "f.stat"]
    out.append(("IV(Z1=calendar) Y~c|vix", round(iv1.params["c"], 4), round(iv1.tstats["c"], 2), f"F={F1:.1f}", ""))

    # (1b) reconciliation: IV with GROSS gamma endogenous (what the hand-rolled IV used) — strong F, but NOT the policy variable
    ivg = IV2SLS(df["Y"], df[["const", "vix"]], df["cg"], df["Z1"]).fit(cov_type="kernel", kernel="bartlett")
    Fg = ivg.first_stage.diagnostics.loc["cg", "f.stat"]
    out.append(("IV(Z1) GROSS gamma [recon]", round(ivg.params["cg"], 4), round(ivg.tstats["cg"], 2), f"F={Fg:.1f}", "not policy var"))

    # (2) over-identified IV with calendar + near-expiry share (Z1,Z2)
    iv2 = IV2SLS(df["Y"], df[["const", "vix"]], df["c"], df[["Z1", "Z2"]]).fit(cov_type="kernel", kernel="bartlett")
    F2 = iv2.first_stage.diagnostics.loc["c", "f.stat"]
    try:
        j = iv2.wooldridge_overid; overid = f"J p={j.pval:.2f}"
    except Exception:
        overid = "overid n/a"
    out.append(("IV(Z1,Z2) Y~c|vix", round(iv2.params["c"], 4), round(iv2.tstats["c"], 2), f"F={F2:.1f}", overid))

    print(f"proper 2SLS (linearmodels, kernel/Bartlett HAC), n={n}\n")
    print(f"  {'spec':<26}{'beta_c':>9}{'t':>7}{'first_stage':>14}{'overid':>12}")
    for r in out:
        print(f"  {r[0]:<26}{r[1]:>9}{r[2]:>7}{str(r[3]):>14}{str(r[4]):>12}")
    rows = [("spec", "beta_c", "t", "first_stage", "overid")] + out
    (DATA / "e4_iv2sls.csv").write_text("\n".join(",".join(str(c) for c in r) for r in rows))
    print("\n  wrote data/e4_iv2sls.csv")
    print("\n  Reading: compare the IV beta_c/t to the OLS benchmark (subsumed by VIX, t≈1.5-1.7).")
    print("  A strong first stage (F≫10) + a significant, correctly-signed IV beta_c ⇒ the")
    print("  exogenous part of dealer gamma moves vol beyond VIX. A non-rejected over-id (J p>0.1)")
    print("  supports instrument validity. Borderline/insignificant ⇒ honest: can't cleanly")
    print("  separate gamma's causal effect from VIX even with proper IV.")

if __name__ == "__main__":
    main()
