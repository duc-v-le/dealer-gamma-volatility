#!/usr/bin/env python3
"""
wrds_rank_universe.py — rank the liquid-options universe (auditable, reusable)
============================================================================
Saves the universe-selection step that wrds_download_panel.py uses inline, so the
sample is documented and reproducible (data-appendix material). Ranks optionable
secids by total option volume in a reference year, maps to tickers, flags VIX, and
writes a ranked CSV.

Run:  python3 wrds_rank_universe.py [N=80] [year=2023]
Out:  data/real/universe_ranked.csv   (rank, secid, ticker, option_volume, is_vix)
"""
import sys
import pathlib
import pandas as pd
import wrds_lib as w

OUT = pathlib.Path(__file__).resolve().parent.parent / "data" / "real"; OUT.mkdir(parents=True, exist_ok=True)

def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    year = int(sys.argv[2]) if len(sys.argv) > 2 else 2023
    rank = w.query(f"""SELECT secid, SUM(volume) AS option_volume
                       FROM optionm.opprcd{year} WHERE volume IS NOT NULL
                       GROUP BY secid ORDER BY option_volume DESC LIMIT {N};""")
    nm = w.query("SELECT secid, ticker FROM optionm.secnmd GROUP BY secid, ticker")
    rank = rank.merge(nm, on="secid", how="left").drop_duplicates("secid").reset_index(drop=True)
    rank.insert(0, "rank", range(1, len(rank) + 1))
    rank["is_vix"] = (rank["secid"] == 117801)
    rank["option_volume"] = pd.to_numeric(rank["option_volume"], errors="coerce")
    out = OUT / "universe_ranked.csv"
    rank.to_csv(out, index=False)
    print(f"Ranked top {N} optionable names by {year} option volume -> {out}\n")
    print(rank[["rank", "ticker", "option_volume", "is_vix"]].head(40).to_string(index=False))
    print(f"\n(The panel in wrds_download_panel.py drops VIX and takes the top N of this list.)")

if __name__ == "__main__":
    main()
