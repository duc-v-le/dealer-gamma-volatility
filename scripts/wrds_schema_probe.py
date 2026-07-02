#!/usr/bin/env python3
"""
wrds_schema_probe.py — document the OptionMetrics schema discovery (one-time inspection)
=======================================================================================
Saves the exploratory queries used to discover the OptionMetrics tables/columns/secids
that the download scripts rely on. Pure inspection (no analysis); makes the data
provenance fully script-backed. The findings are also summarized in data/real/DATA_GUIDE.md.

Run:  python3 wrds_schema_probe.py
"""
import wrds_lib as w

def show(sql, label, params=None):
    print(f"\n### {label}")
    try:
        print(w.query(sql, params).to_string(index=False))
    except Exception as e:
        print("  (query failed:", e, ")")

def main():
    show("""SELECT schema_name FROM information_schema.schemata
            WHERE schema_name ILIKE '%option%' ORDER BY 1;""", "option schemas")
    show("""SELECT MIN(table_name) AS first, MAX(table_name) AS last
            FROM information_schema.tables
            WHERE table_schema='optionm' AND table_name ~ '^opprcd[0-9]{4}$';""", "opprcd year range")
    show("""SELECT column_name, data_type FROM information_schema.columns
            WHERE table_schema='optionm' AND table_name='opprcd2022' ORDER BY ordinal_position;""",
         "opprcd (option price+greeks) columns")
    show("""SELECT column_name FROM information_schema.columns
            WHERE table_schema='optionm' AND table_name='secprd2022'
            AND column_name IN ('secid','date','close','return','open','high','low','volume') ORDER BY 1;""",
         "secprd (underlying price) key columns")
    show("SELECT secid, ticker, issuer FROM optionm.secnmd WHERE ticker IN ('SPX','VIX') GROUP BY secid, ticker, issuer;",
         "key index secids (SPX=108105, VIX=117801)")

if __name__ == "__main__":
    main()
