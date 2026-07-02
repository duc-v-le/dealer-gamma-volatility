#!/usr/bin/env python3
"""
wrds_lib.py — minimal WRDS (PostgreSQL) connection helper.
Credentials: the PASSWORD is never stored here or passed in code — libpq/psycopg2
reads it from ~/.pgpass. The username is also read from ~/.pgpass (field 4), or
the WRDS_USER env var. So nothing secret lives in the repo.

    import wrds_lib as w
    df = w.query("SELECT 1 AS x")
"""
import os
import psycopg2
import pandas as pd

HOST, PORT, DB = "wrds-pgdata.wharton.upenn.edu", 9737, "wrds"


def _user():
    pgpass = os.path.expanduser("~/.pgpass")
    if os.path.exists(pgpass):
        for line in open(pgpass):
            p = line.strip().split(":")
            if len(p) >= 5 and (p[0] == HOST or p[0] == "*"):
                return p[3]
    if os.environ.get("WRDS_USER"):
        return os.environ["WRDS_USER"]
    raise SystemExit("No WRDS username found in ~/.pgpass or $WRDS_USER")


def connect():
    return psycopg2.connect(host=HOST, port=PORT, dbname=DB, user=_user(),
                            sslmode="require", connect_timeout=30)


def query(sql, params=None) -> pd.DataFrame:
    with connect() as c, c.cursor() as cur:
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)


if __name__ == "__main__":
    print("WRDS user:", _user())
    print(query("SELECT 1 AS ok"))
