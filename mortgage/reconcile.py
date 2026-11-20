"""Check independent SQLite policy aggregates against the in-memory evaluation."""
from contextlib import closing
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

KEYS = ["model", "calibration_target", "period"]
MONTHLY_COUNTS = ["loans", "events", "loans_retained", "events_retained", "events_flagged"]
MONTHLY_RATES = ["observed_rate", "retention", "retained_event_rate", "capture_rate"]
PORTFOLIO_COUNTS = ["loans_retained", "retained_balance"]
PORTFOLIO_RATES = ["retention", "retained_event_rate", "capture_rate"]


def compare(sql_results, expected, counts, rates):
    merged = sql_results.merge(expected, on=KEYS, how="outer", validate="one_to_one",
                               indicator=True, suffixes=("_sql", "_python"))
    if not len(merged) or not merged["_merge"].eq("both").all():
        raise ValueError("SQL and Python policy groups differ")

    for field in [*counts, *rates]:
        sql_values, python_values = (merged[field + suffix] for suffix in ("_sql", "_python"))
        if field in counts:
            match = sql_values.eq(python_values)
        else:
            match = np.isclose(sql_values, python_values, atol=1e-12, rtol=0, equal_nan=True)
        if not np.all(match):
            sample = merged.loc[~match, KEYS].iloc[0].to_dict()
            raise ValueError(f"SQL and Python disagree on {field} for {sample}")


def reconcile(database, scores, thresholds, monthly, policies):
    # Read the private source database without persisting loan scores or thresholds.
    uri = Path(database).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        db.execute("""
            CREATE TEMP TABLE audit_scores (
                loan_id TEXT, model TEXT, probability REAL, PRIMARY KEY (model, loan_id)
            )
        """)
        db.execute("""
            CREATE TEMP TABLE audit_thresholds (
                model TEXT, calibration_target REAL, threshold REAL,
                PRIMARY KEY (model, calibration_target)
            )
        """)
        db.executemany("INSERT INTO audit_scores VALUES (?, ?, ?)",
                       scores[["loan_id", "model", "probability"]].itertuples(index=False, name=None))
        db.executemany("INSERT INTO audit_thresholds VALUES (?, ?, ?)", thresholds)
        sql = Path(__file__).with_name("policy_reconciliation.sql").read_text()
        results = pd.read_sql_query(sql, db)

    monthly_expected = monthly.rename(columns={"first_payment_month": "period"})
    portfolio_expected = policies.rename(columns={
        "loans": "loans_retained", "share_events_flagged": "capture_rate",
    }).assign(period="Portfolio")
    monthly_sql = results[results.period != "Portfolio"]
    portfolio_sql = results[results.period == "Portfolio"]
    compare(monthly_sql, monthly_expected, MONTHLY_COUNTS, MONTHLY_RATES)
    compare(portfolio_sql, portfolio_expected, PORTFOLIO_COUNTS, PORTFOLIO_RATES)
    return results
