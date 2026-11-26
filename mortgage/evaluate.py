"""Fit mortgage models and export aggregate 2025 holdout results."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if __package__:
    from .reconcile import reconcile
    from .stability import monthly_policy
    from .uncertainty import comparison
else:
    from reconcile import reconcile
    from stability import monthly_policy
    from uncertainty import comparison

NUMERIC = ["fico", "cltv", "dti", "log_upb", "ltv", "term"]
CATEGORY = ["occupancy", "property_type", "purpose"]
TARGET_RETENTIONS = [0.70, 0.75, 0.80, 0.90, 1.00]


def load_cohort(database):
    with closing(sqlite3.connect(database)) as db:
        frame = pd.read_sql_query("SELECT * FROM mortgage_cohort ORDER BY first_payment, loan_id", db)
    quality = frame.groupby(["split", "outcome_status"]).size().rename("loans").reset_index()
    frame = frame[frame.outcome_status.isin([
        "event", "prepaid_without_event", "complete_without_event",
    ])].copy()
    frame["event"] = frame.outcome_status.eq("event").astype(int)
    frame["log_upb"] = np.log(frame.original_upb)
    frame[CATEGORY] = frame[CATEGORY].replace({"": np.nan, "9": np.nan, "99": np.nan})
    return frame, quality


def predict(estimator, frame):
    return estimator.predict_proba(frame)[:, 1]


def log_odds(predictions):
    p = np.clip(predictions, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def policy_scores(frame, probability, threshold):
    retained = frame[probability < threshold]
    return dict(
        loans=len(retained), retention=len(retained) / len(frame),
        retained_balance=retained.original_upb.sum(),
        retained_event_rate=retained.event.mean(),
        share_events_flagged=1 - retained.event.sum() / frame.event.sum(),
    )


def review_capacity(events, risk_scores):
    """Evaluate exact holdout review budgets with deterministic score ties."""
    labels = np.asarray(events)
    ranked = labels[np.argsort(-risk_scores, kind="stable")]
    total_events = int(labels.sum())
    for fraction in (.10, .20):
        reviewed = int(len(labels) * fraction)
        captured = int(ranked[:reviewed].sum())
        actual_share = reviewed / len(labels)
        capture = captured / total_events
        yield dict(review_target=fraction, reviewed=reviewed, reviewed_share=actual_share,
                   events_flagged=captured, total_events=total_events,
                   capture_rate=capture, lift=capture / actual_share if reviewed else np.nan)


def evaluate(database, output):
    frame, quality = load_cohort(database)
    train, cal, test = (frame[frame.split == name] for name in ("train", "calibrate", "test"))
    if any(part.event.nunique() != 2 for part in (train, cal, test)):
        raise ValueError("Each split must contain observed events and non-events")
    x_train, x_cal, x_test = (part[NUMERIC + CATEGORY] for part in (train, cal, test))
    splits = frame.groupby("split").event.agg(loans="size", events="sum", event_rate="mean")
    splits = splits.reindex(["train", "calibrate", "test"]).reset_index()
    metrics, bins, policies, risk, stability = [], [], [], [], []
    holdout_predictions, audit_scores, audit_thresholds, capacity = [], [], [], []
    for name, classifier in [
        ("Logistic Regression", LogisticRegression(max_iter=500)),
        ("Random Forest", RandomForestClassifier(
            n_estimators=160, max_depth=12, min_samples_leaf=40, random_state=42, n_jobs=1)),
    ]:
        preprocess = ColumnTransformer([
            ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
            ("category", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                       OneHotEncoder(handle_unknown="ignore")), CATEGORY),
        ])
        model = make_pipeline(preprocess, classifier).fit(x_train, train.event)
        raw_cal, raw_test = (predict(model, x) for x in (x_cal, x_test))
        sigmoid = LogisticRegression(C=1e6).fit(log_odds(raw_cal), cal.event)
        calibrated_cal, calibrated_test = (predict(sigmoid, log_odds(p)) for p in (raw_cal, raw_test))
        for version, p_cal, p_test in [
            ("Raw", raw_cal, raw_test), ("Calibrated", calibrated_cal, calibrated_test),
        ]:
            threshold = np.quantile(p_cal, .75)
            flagged = p_test >= threshold
            # Fixed bins concentrate resolution on the low probabilities typical of mortgages.
            bin_id = pd.cut(p_test, [-.001, .0025, .005, .01, .02, .05, 1],
                            labels=["<0.25%", "0.25–0.50%", "0.50–1%", "1–2%", "2–5%", "5%+"],
                            right=False)
            calibration = pd.DataFrame({"bin": bin_id, "prediction": p_test, "event": test.event})
            calibration = calibration.groupby("bin", observed=True).agg(
                loans=("event", "size"), mean_pd=("prediction", "mean"), observed=("event", "mean")
            ).reset_index()
            ece = ((calibration.mean_pd - calibration.observed).abs() * calibration.loans).sum() / len(test)
            metrics.append(dict(model=name, version=version, roc_auc=roc_auc_score(test.event, p_test),
                                average_precision=average_precision_score(test.event, p_test),
                                precision=precision_score(test.event, flagged, zero_division=0),
                                recall=recall_score(test.event, flagged), threshold=threshold,
                                brier=brier_score_loss(test.event, p_test), ece=ece,
                                mean_pd=p_test.mean(), actual_rate=test.event.mean()))
            bins.append(calibration.assign(model=name, version=version))
        capacity.extend(dict(model=name, **row) for row in review_capacity(test.event, calibrated_test))
        holdout_predictions.append((calibrated_test, np.quantile(calibrated_cal, .75)))
        audit_scores.append(test[["loan_id"]].assign(model=name, probability=calibrated_test))
        for target in TARGET_RETENTIONS:
            threshold = float(np.quantile(calibrated_cal, target)) if target < 1 else 1.01
            audit_thresholds.append((name, target, threshold))
            policies.append(dict(model=name, calibration_target=target,
                                 threshold=threshold if target < 1 else np.nan,
                                 **policy_scores(test, calibrated_test, threshold)))
            stability.append(monthly_policy(test, calibrated_test, threshold).assign(
                model=name, calibration_target=target,
                threshold=threshold if target < 1 else np.nan,
            ))
        lower, upper = np.quantile(calibrated_cal, [.50, .75])
        tier = np.select([calibrated_test < lower, calibrated_test < upper],
                         ["Low", "Medium"], default="High")
        risk.append(test.assign(tier=tier, prediction=calibrated_test).groupby("tier").agg(
            loans=("event", "size"), events=("event", "sum"),
            observed=("event", "mean"), mean_pd=("prediction", "mean"),
        ).reset_index().assign(model=name))
    factors = []
    for factor, values, levels, labels in [
        ("FICO", test.fico, [-np.inf, 680, 740, np.inf], ["Below 680", "680–739", "740 or above"]),
        ("LTV", test.ltv, [-np.inf, 70, 80, np.inf], ["Below 70%", "70%–79%", "80% or above"]),
    ]:
        segment = pd.cut(values, levels, labels=labels, right=False)
        factors.append(test.assign(segment=segment).groupby("segment", observed=True).event.agg(
            loans="size", events="sum", observed="mean").reset_index().assign(factor=factor))
    # Lower FICO ranks first. Keep loans without a FICO in the comparison.
    fico_risk = -test.fico.fillna(np.inf).to_numpy()
    capacity.extend(dict(model="FICO-only", **row)
                    for row in review_capacity(test.event, fico_risk))

    (lr_scores, lr_cutoff), (rf_scores, rf_cutoff) = holdout_predictions
    intervals = comparison(test.event, lr_scores, rf_scores, lr_cutoff, rf_cutoff)
    tables = dict(cohort_quality=quality, splits=splits, metrics=pd.DataFrame(metrics),
                  calibration=pd.concat(bins), policy=pd.DataFrame(policies),
                  risk_tiers=pd.concat(risk), risk_factors=pd.concat(factors),
                  monthly_stability=pd.concat(stability, ignore_index=True),
                  model_uncertainty=intervals, review_capacity=pd.DataFrame(capacity))
    tables["policy_reconciliation"] = reconcile(
        database, pd.concat(audit_scores, ignore_index=True), audit_thresholds,
        tables["monthly_stability"], tables["policy"],
    )
    print(f"SQLite reconciliation passed for {len(tables['policy_reconciliation'])} policy groups")
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False, float_format="%.8f")
    payload = json.dumps({name: json.loads(table.to_json(orient="records"))
                          for name, table in tables.items()
                          if name != "policy_reconciliation"}).replace("<", "\\u003c")
    template = Path(__file__).with_name("dashboard_template.html").read_text()
    (output / "dashboard.html").write_text(template.replace("{{DATA}}", payload), encoding="utf-8")
    print(splits.to_string(index=False))
    print(tables["metrics"].round(5).to_string(index=False))
    print(tables["policy"].round(5).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path(__file__).parent / "report")
    args = parser.parse_args()
    evaluate(args.database, args.report)
