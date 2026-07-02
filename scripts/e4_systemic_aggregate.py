#!/usr/bin/env python3
"""
e4_systemic_aggregate.py — aggregate (systemic) decomposition of the gamma channel
======================================================================================
The firm-level IV identifies dealer gamma's causal effect on each name's vol.
This aggregates to a MARKET-LEVEL systemic statement: how much of S&P 500 index volatility
(and tail-day frequency) is attributable to AGGREGATE dealer short-gamma?

Build market dealer gamma = Σ over S&P 500 names of net dealer gamma (from panel_sp500),
and aggregate roll-off Σ scheduled_rolloff_net (the market-level instrument). Outcome =
future S&P 500 index realized vol. Time-series IV (roll-off-instrumented, VIX-controlled),
plus a simple decomposition: index vol on aggregate-short- vs long-gamma days, and the
implied contribution of the channel.

Run:  python3 e4_systemic_aggregate.py  →  data/e4_systemic_aggregate.csv
"""
import pathlib
import numpy as np
import pandas as pd
import wrds_lib as w
from linearmodels.iv import IV2SLS

ROOT = pathlib.Path(__file__).resolve().parent.parent
REAL = ROOT / "data" / "real"; DATA = ROOT / "data"

def rz(s, win=60):
    s = pd.Series(s); return ((s - s.rolling(win).mean()) / s.rolling(win).std())

def main():
    # aggregate market dealer gamma + roll-off across the S&P 500 panel
    p = pd.read_parquet(REAL / "panel_sp500.parquet")
    p["date"] = pd.to_datetime(p["date"])
    agg = p.groupby("date").agg(mkt_net=("net_gex", "sum"),
                                mkt_roll=("scheduled_rolloff_net", "sum")).reset_index()
    # market index (SPX) returns + realized vol as the systemic outcome
    spx = pd.read_csv(REAL / "spx_gamma_panel.csv", parse_dates=["date"])[["date", "ret", "realized_vol"]]
    vix = pd.concat([w.query(f"SELECT date, close AS vix FROM optionm.secprd{y} WHERE secid=117801 ORDER BY date;")
                     for y in range(2016, 2025)])
    vix["date"] = pd.to_datetime(vix["date"]); vix["vix"] = vix["vix"].astype(float)
    m = agg.merge(spx, on="date").merge(vix, on="date").sort_values("date").reset_index(drop=True)

    m["fut_vol"] = m["realized_vol"].shift(-5)                 # future 5d index vol
    df = pd.DataFrame({"c": rz(m["mkt_net"]), "Z": rz(m["mkt_roll"]), "vix": rz(m["vix"]),
                       "Y": rz(m["fut_vol"])})
    df["const"] = 1.0
    d = df.dropna().reset_index(drop=True)
    iv = IV2SLS(d["Y"], d[["const", "vix"]], d["c"], d["Z"]).fit(cov_type="kernel", kernel="bartlett")
    F = iv.first_stage.diagnostics.loc["c", "f.stat"]
    beta, t = iv.params["c"], iv.tstats["c"]

    # magnitude in annualized index-vol points
    ann_sd = np.nanstd(m["fut_vol"]) * np.sqrt(252) * 100
    mag = abs(beta) * ann_sd

    # decomposition: index vol on aggregate-short vs long gamma days
    m["ann_vol"] = m["realized_vol"] * np.sqrt(252) * 100
    short = m[m["mkt_net"] < m["mkt_net"].median()]["ann_vol"].mean()   # more short / less long
    long_ = m[m["mkt_net"] >= m["mkt_net"].median()]["ann_vol"].mean()
    p_short_days = (m["mkt_net"] < 0).mean()

    rows = [("metric", "value"),
            ("aggregate IV beta (mkt gamma->index vol)", round(float(beta), 4)),
            ("t (HAC)", round(float(t), 2)),
            ("first-stage F", round(float(F), 0)),
            ("magnitude (ann index-vol pts per 1-SD mkt gamma)", round(float(mag), 2)),
            ("avg ann index vol: low-gamma (more short) half", round(float(short), 1)),
            ("avg ann index vol: high-gamma (more long) half", round(float(long_), 1)),
            ("vol gap (short - long) ann pts", round(float(short - long_), 1)),
            ("share of days aggregate dealers NET SHORT gamma", round(float(p_short_days), 3))]
    print("aggregate systemic decomposition (market dealer gamma -> S&P 500 index vol)\n")
    for r in rows[1:]:
        print(f"  {r[0]:<52}{r[1]}")
    (DATA / "e4_systemic_aggregate.csv").write_text("\n".join(",".join(str(c) for c in r) for r in rows))
    print("\n  wrote data/e4_systemic_aggregate.csv")
    print("  Reading: a significant aggregate IV beta + a vol gap between short- and long-gamma")
    print("  days quantifies the SYSTEMIC footprint — how much index volatility the dealer-gamma")
    print("  channel accounts for at the market level (the paper's financial-stability hook).")

if __name__ == "__main__":
    main()
