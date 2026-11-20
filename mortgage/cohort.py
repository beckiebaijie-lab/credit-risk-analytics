"""Build a fixed-horizon Freddie Mac cohort from authorized local sample archives."""
from contextlib import closing
from pathlib import Path
import argparse
import sqlite3
from zipfile import ZipFile

import pandas as pd

# Zero-based positions verified against the July 2026 official file headers.
ORIG = {0: "fico", 1: "first_payment", 7: "occupancy", 8: "cltv", 9: "dti",
        10: "original_upb", 11: "ltv", 17: "property_type", 19: "loan_id",
        20: "purpose", 21: "term"}
PERF = {0: "loan_id", 1: "report_month", 3: "delinquency", 8: "zero_balance"}
PERFORMANCE_CUTOFF = "2026-06"
SPLITS = {
    "train": ("2021-01", "2022-06"),
    "calibrate": ("2023-07", "2023-12"),
    "test": ("2025-01", "2025-06"),
}


def read_fields(archive, name, fields, **kwargs):
    with archive.open(name) as source:
        yield from pd.read_csv(source, sep="|", header=None, usecols=list(fields),
                               dtype=str, keep_default_na=False, chunksize=250_000,
                               **kwargs)


def month_index(values):
    dates = pd.to_datetime(values, format="%Y%m", errors="coerce")
    return (12 * dates.dt.year + dates.dt.month - 1).astype("Int64")


def prepare(paths, db):
    for path in paths:
        with ZipFile(path) as archive:
            for prefix, fields, table in [
                ("sample_orig_", ORIG, "origination"),
                ("sample_perf_", PERF, "performance"),
            ]:
                names = [name for name in archive.namelist()
                         if Path(name).name.startswith(prefix) and name.endswith(".txt")]
                if len(names) != 1:
                    raise ValueError(f"{path}: expected one {prefix} file")
                for chunk in read_fields(archive, names[0], fields):
                    chunk = chunk.rename(columns=fields)
                    if table == "origination":
                        chunk["first_index"] = month_index(chunk.first_payment)
                        month = pd.to_datetime(chunk.first_payment, format="%Y%m", errors="coerce")
                        chunk["split"] = None
                        for split, (start, end) in SPLITS.items():
                            chunk.loc[month.dt.strftime("%Y-%m").between(start, end), "split"] = split
                        for col, minimum, maximum in [
                            ("fico", 300, 850), ("cltv", 1, 998), ("dti", 1, 65),
                            ("ltv", 1, 998), ("original_upb", 1, float("inf")), ("term", 1, 600),
                        ]:
                            numeric = pd.to_numeric(chunk[col], errors="coerce")
                            chunk[col] = numeric.where(numeric.between(minimum, maximum))
                    else:
                        chunk["report_index"] = month_index(chunk.report_month)
                        chunk["delinquency"] = pd.to_numeric(chunk.delinquency, errors="coerce")
                    chunk.to_sql(table, db, if_exists="append", index=False)
    db.execute("CREATE INDEX perf_loan_month ON performance(loan_id, report_index)")
    cutoff = int(month_index(pd.Series([PERFORMANCE_CUTOFF.replace("-", "")])).iloc[0])
    sql = Path(__file__).with_name("cohort.sql").read_text()
    db.executescript(sql.replace("{{PERFORMANCE_CUTOFF}}", str(cutoff)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--database", type=Path, required=True,
                        help="New database path under an ignored private data folder")
    args = parser.parse_args()
    args.database.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects prior runs from accidental overwrite.
    args.database.touch(exist_ok=False)
    with closing(sqlite3.connect(args.database)) as db:
        prepare(args.archives, db)
        print(pd.read_sql_query("""
            SELECT split, outcome_status, count(*) AS loans
            FROM mortgage_cohort GROUP BY split, outcome_status ORDER BY 1, 2
        """, db).to_string(index=False))


if __name__ == "__main__":
    main()
