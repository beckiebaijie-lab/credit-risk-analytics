"""Train on older LendingClub vintages and export aggregate evaluation tables."""
import argparse
import hashlib
import sqlite3
from contextlib import closing
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dashboard import render

ROOT = Path(__file__).resolve().parent
MEMBER = "data/lending_club/data/df_all.csv"
CATEGORICAL = ["purpose", "home_ownership"]
RAW_NUMERIC = [
    "loan_amnt", "annual_inc", "dti", "fico_approx", "credit_history_age",
    "open_acc", "total_acc", "revol_bal", "revol_util", "inq_last_6mths",
]
NUMERIC = [
    "log_loan_amnt", "log_annual_inc", "loan_to_income", "term_months",
    "dti", "fico_approx", "credit_history_age", "open_acc", "total_acc",
    "log_revol_bal", "revol_util", "inq_last_6mths",
]
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 1.00]


def quality_check(raw):
    staged = raw.copy()
    # Explicit English month map avoids relying on the machine's locale.
    months = dict(zip("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(),
                      (f"{n:02}" for n in range(1, 13))))
    month = staged["issue_d"].str.extract(r"^([A-Za-z]{3})-(\d{4})$")
    staged["issue_month"] = month[1] + "-" + month[0].map(months)
    for col in RAW_NUMERIC:
        staged[col] = pd.to_numeric(staged[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    with closing(sqlite3.connect(":memory:")) as db:
        staged.to_sql("staged", db, index=False)
        db.executescript((ROOT / "sql/quality.sql").read_text())
        quality = pd.read_sql_query(
            "SELECT quality_status, count(*) AS loans FROM checked GROUP BY quality_status", db
        )
        cohort = pd.read_sql_query("SELECT * FROM cohort ORDER BY issue_month, id", db)
    missing = pd.DataFrame({"field": [*RAW_NUMERIC, *CATEGORICAL],
                            "missing": cohort[[*RAW_NUMERIC, *CATEGORICAL]].isna().sum().values,
                            "eligible_loans": len(cohort)})
    return cohort, quality, missing


def features(frame):
    x = frame[["term", *RAW_NUMERIC, *CATEGORICAL]].copy()
    income = x["annual_inc"].where(x["annual_inc"] > 0)
    x["log_annual_inc"] = np.log1p(income)
    x["log_loan_amnt"] = np.log1p(x["loan_amnt"])
    x["loan_to_income"] = x["loan_amnt"] / income
    x["log_revol_bal"] = np.log1p(x["revol_bal"].where(x["revol_bal"] >= 0))
    x["term_months"] = x["term"].str.strip().map({"36 months": 36, "60 months": 60})
    return x[NUMERIC + CATEGORICAL].replace([np.inf, -np.inf], np.nan)


def vintage_split(frame):
    year = frame.issue_month.str[:4]
    return frame[year < "2011"], frame[year == "2011"], frame[year == "2012"]


def fit_models(train, calibration, test):
    x_train, x_cal, x_test = map(features, (train, calibration, test))
    for name, model in [
        ("Logistic Regression", LogisticRegression(max_iter=500)),
        ("Random Forest", RandomForestClassifier(
            n_estimators=160, min_samples_leaf=40, max_depth=12, random_state=42, n_jobs=1
        )),
    ]:
        preprocess = ColumnTransformer([
            ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
            ("category", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                       OneHotEncoder(handle_unknown="ignore")), CATEGORICAL),
        ])
        estimator = make_pipeline(preprocess, model).fit(x_train, train.charged_off)

        def log_odds(x):
            p = np.clip(estimator.predict_proba(x)[:, 1], 1e-6, 1 - 1e-6)
            return np.log(p / (1 - p)).reshape(-1, 1)

        sigmoid = LogisticRegression(C=1e6).fit(log_odds(x_cal), calibration.charged_off)
        yield name, estimator.predict_proba(x_test)[:, 1], sigmoid.predict_proba(log_odds(x_test))[:, 1]


def calibration_table(scored):
    bins = np.minimum((scored.pd * 10).astype(int), 9)
    return scored.groupby(bins.rename("bin")).agg(
        loans=("charged_off", "size"), mean_pd=("pd", "mean"),
        actual_rate=("charged_off", "mean"),
    ).reset_index()


def risk_tiers(scored):
    groups = pd.cut(scored.pd, [-np.inf, .10, .20, np.inf],
                    labels=["Low", "Medium", "High"], right=False)
    return scored.groupby(groups.rename("tier"), observed=False).agg(
        loans=("charged_off", "size"), chargeoffs=("charged_off", "sum"),
        actual_rate=("charged_off", "mean"), mean_pd=("pd", "mean"),
        loan_amount=("loan_amnt", "sum"),
    ).reset_index()


def policy(scored, threshold):
    kept = scored[scored.pd < threshold] if threshold < 1 else scored
    n = len(kept)
    events = int(kept.charged_off.sum())
    total_events = int(scored.charged_off.sum())
    return dict(threshold=threshold, retained=n, retention=n / len(scored),
                retained_amount=kept.loan_amnt.sum(), retained_chargeoffs=events,
                retained_chargeoff_rate=events / n if n else np.nan,
                flagged_chargeoff_share=1 - events / total_events if total_events else np.nan)


def run(archive, report):
    report.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as z, z.open(MEMBER) as handle:
        raw = pd.read_csv(handle, usecols=[
            "id", "loan_status", "issue_d", "term", *RAW_NUMERIC, *CATEGORICAL
        ], dtype={"id": str})
    cohort, quality, missing = quality_check(raw)
    train, cal, test = vintage_split(cohort)
    if any(frame.charged_off.nunique() != 2 for frame in (train, cal, test)):
        raise ValueError("Each vintage split must contain both outcomes")
    yearly = cohort.assign(year=cohort.issue_month.str[:4]).groupby("year").charged_off.agg(
        loans="size", chargeoffs="sum", actual_rate="mean").reset_index()
    factors = pd.concat([
        test.assign(segment=pd.cut(test.fico_approx, [0, 680, 720, np.inf],
                                   labels=["<680", "680–719", "720+"], right=False), factor="FICO"),
        test.assign(segment=test.term.str.strip(), factor="Term"),
    ]).groupby(["factor", "segment"], observed=True).charged_off.agg(
        loans="size", chargeoffs="sum", actual_rate="mean").reset_index()
    scores, calibration, tiers, scenarios = [], [], [], []
    y = test.charged_off
    for model, raw_pd, calibrated_pd in fit_models(train, cal, test):
        for version, probabilities in [("Raw", raw_pd), ("Calibrated", calibrated_pd)]:
            scored = test[["charged_off", "loan_amnt"]].assign(pd=probabilities)
            prediction = probabilities >= .20
            table = calibration_table(scored)
            ece = ((table.mean_pd - table.actual_rate).abs() * table.loans).sum() / len(test)
            scores.append(dict(model=model, version=version, roc_auc=roc_auc_score(y, probabilities),
                               precision=precision_score(y, prediction, zero_division=0),
                               recall=recall_score(y, prediction), brier=brier_score_loss(y, probabilities),
                               ece=ece, mean_pd=probabilities.mean(), actual_rate=y.mean()))
            calibration.append(table.assign(model=model, version=version))
        tiers.append(risk_tiers(scored).assign(model=model))
        scenarios.extend(dict(model=model, **policy(scored, t)) for t in THRESHOLDS)
    tables = dict(quality=quality, missing=missing, vintages=yearly, risk_factors=factors,
                  metrics=pd.DataFrame(scores), calibration=pd.concat(calibration),
                  risk_tiers=pd.concat(tiers), policy=pd.DataFrame(scenarios))
    for name, table in tables.items():
        table.to_csv(report / f"{name}.csv", index=False, float_format="%.8f")
    render(tables, report / "dashboard.html", (len(train), len(cal), len(test)))
    with archive.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    print(f"Archive SHA-256: {digest}")
    print(quality.to_string(index=False))
    print(tables["metrics"].round(4).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data/lending_club.zip")
    parser.add_argument("--report", type=Path, default=ROOT / "report")
    args = parser.parse_args()
    run(args.archive, args.report)
