#!/usr/bin/env python3
"""
e4_panel_iv.py — PANEL roll-off IV (50 liquid names) + liquidity heterogeneity
==================================================================================
Pooled 2SLS over the firm×day panel (wrds_download_panel.py). Firm normalization via
rolling-z within name; TIME fixed effects via date-demeaning (absorbs common market
moves); SEs clustered by date. Endogenous = net dealer gamma; instrument = scheduled
roll-off; control = VIX; outcome = future 5d realized vol.

Plus HETEROGENEITY: split names by option liquidity (rank) — is the gamma→vol effect
stronger where options are more liquid / dealer gamma is more central? (mechanism evidence.)

Run:  python3 e4_panel_iv.py  →  data/e4_panel_iv.csv
"""
import pathlib
import numpy as np
import pandas as pd
import wrds_lib as w
from linearmodels.iv import IV2SLS

ROOT = pathlib.Path(__file__).resolve().parent.parent
REAL = ROOT / "data" / "real"; DATA = ROOT / "data"

def rz(s, win=60):
    return (s - s.rolling(win).mean()) / s.rolling(win).std()

def _read(path):
    path = pathlib.Path(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, parse_dates=["date"])

def main():
    import sys
    src = REAL / (sys.argv[1] if len(sys.argv) > 1 else "panel_gamma.csv")
    p = _read(src); p["date"] = pd.to_datetime(p["date"])
    print(f"panel: {src.name}")
    vix = pd.concat([w.query(f"SELECT date, close AS vix FROM optionm.secprd{y} WHERE secid=117801 ORDER BY date;")
                     for y in range(2016, 2025)])
    vix["date"] = pd.to_datetime(vix["date"]); vix["vix"] = vix["vix"].astype(float)
    p = p.merge(vix, on="date", how="left").sort_values(["secid", "date"]).reset_index(drop=True)

    g = p.groupby("secid", group_keys=False)
    p["c"]  = g["net_gex"].apply(rz)
    p["Z"]  = g["scheduled_rolloff_net"].apply(rz)
    p["fv"] = g["realized_vol"].apply(lambda s: s.shift(-5))      # future 5d realized vol
    p["Y"]  = g.apply(lambda x: rz(x["fv"])).reset_index(level=0, drop=True)
    p["vixz"] = rz(p["vix"])

    # absorb TIME fixed effects: demean each var by its cross-sectional date mean
    for v in ("Y", "c", "Z", "vixz"):
        p[v + "_d"] = p[v] - p.groupby("date")[v].transform("mean")

    def panel_iv(df, label):
        df = df.dropna(subset=["Y_d", "c_d", "Z_d", "vixz_d"]).copy()
        df["const"] = 1.0
        m = IV2SLS(df["Y_d"], df[["const", "vixz_d"]], df["c_d"], df["Z_d"]).fit(
            cov_type="clustered", clusters=df["date"].astype("category").cat.codes)
        F = m.first_stage.diagnostics.loc["c_d", "f.stat"]
        return (label, len(df), round(m.params["c_d"], 4), round(m.tstats["c_d"], 2), round(F, 0))

    rows = [("sample", "firm_days", "beta(net_gamma)", "t_clustered", "firstF")]
    # name-level gamma-exposure proxy for the heterogeneity split (if no liq_rank present)
    if "liq_rank" not in p.columns:
        sz = p.groupby("secid")["net_gex"].apply(lambda s: s.abs().mean())
        p["liq_rank"] = p["secid"].map(sz.rank(ascending=False))   # 1 = largest dealer gamma
    n = p["secid"].nunique(); med = p["liq_rank"].median()
    rows.append(panel_iv(p, f"ALL ({n} names, pooled time-FE)"))
    rows.append(panel_iv(p[p["liq_rank"] <= med], "HIGH gamma-exposure half"))
    rows.append(panel_iv(p[p["liq_rank"] > med],  "LOW gamma-exposure half"))

    print(f"PANEL roll-off IV (pooled 2SLS, time-FE, date-clustered) — {src.name}\n")
    print(f"  {'sample':<34}{'firm-days':>10}{'beta':>9}{'t':>7}{'F':>8}")
    for r in rows[1:]:
        print(f"  {r[0]:<34}{r[1]:>10,}{r[2]:>9}{r[3]:>7}{r[4]:>8}")
    (DATA / "e4_panel_iv.csv").write_text("\n".join(",".join(str(c) for c in r) for r in rows))
    print("\n  wrote data/e4_panel_iv.csv")
    print("  Reading: a significant negative pooled β ⇒ the causal gamma→vol effect holds in a")
    print("  liquid-options PANEL with time fixed effects (not just SPX + 3 names). A larger")
    print("  |β| for HIGH-liquidity names ⇒ heterogeneity consistent with the dealer-gamma")
    print("  mechanism (stronger where options/dealer hedging matter more).")

if __name__ == "__main__":
    main()
