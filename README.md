# LendingClub Credit Risk Analytics

Estimate charge-off risk using information available at origination, then use the scores to segment risk and simulate approval policies on a historical loan portfolio.

Download and open the [interactive dashboard](report/dashboard.html) in a browser to switch models, compare calibration, and explore approval thresholds offline. See the [Power BI import guide](POWER_BI.md) to recreate the report in Power BI.

## 1. Source, cohort, and outcome

The project uses the pinned [teaching release v2026.09](https://jaredsmurray.github.io/intro-stats/data.html) of the historical LendingClub accepted-loans data. The `df_all.csv` extract covers funded 36- and 60-month loans originated in 2008–2012 that reached a terminal status. See the [data dictionary](https://jaredsmurray.github.io/intro-stats/appendix_data.html#a.6-lendingclub-accepted-loans) and the [upstream LendingClub archive](https://www.kaggle.com/datasets/wordsforthewise/lending-club). This repository contains only aggregate outputs.

`Charged Off` is labeled 1 and `Fully Paid` is labeled 0. The observation window runs from origination through the terminal outcome recorded in the snapshot and varies by loan. The upstream archive documents origination coverage through 2018Q4. The exact performance cutoff and charge-off dates are unavailable, so a fixed 12-month default outcome cannot be constructed. The source excludes declined and unresolved loans, which can introduce selection and resolution bias.

## 2. SQL data quality

[pipeline.py](pipeline.py) reads the required columns, normalizes issue months and numeric fields, and stages the data in SQLite. [quality.sql](sql/quality.sql) uses `COUNT(*) OVER (PARTITION BY id)` to find duplicates and checks identifiers, dates, loan amounts, terms, and outcome status. All records with a duplicate ID are excluded. Each excluded row is counted under its first failed rule. All 92,902 input records passed these checks. See the [quality audit](report/quality.csv) and [missing-value counts](report/missing.csv).

## 3. Origination features

`features()` uses an explicit allowlist with FICO midpoint, debt-to-income ratio, credit history, accounts, inquiries, home ownership, loan purpose, and contract term. It also derives loan amount divided by annual income, log income, and log revolving balance. The model pipelines fit imputation, scaling, and one-hot encoding on the training cohort. Identifiers, terminal status, last payment date, cumulative payments, recoveries, and outstanding principal are excluded from predictors. The allowlist also excludes LendingClub grade, subgrade, interest rate, and funded amount to reduce reliance on the platform's prior underwriting and pricing decisions.

## 4. Time-based evaluation

| Stage | Origination years | Loans |
|---|---|---:|
| Model and preprocessing fit | 2008–2010 | 17,814 |
| Sigmoid probability calibration | 2011 | 21,721 |
| Holdout evaluation and policy simulation | 2012 | 53,367 |

`fit_models()` fits Logistic Regression and Random Forest. A sigmoid calibrator is fit on each model's 2011 predicted log-odds and observed outcomes. Both models are evaluated on the 2012 holdout. This is a retrospective origination-year split. Some training or calibration outcomes may have become available after the test period began, and the data do not identify when each outcome became available.

| Calibrated results, 2012 holdout | ROC-AUC | Precision at 20% | Recall at 20% | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6781 | 29.35% | 40.36% | 0.1280 | 1.06% |
| Random Forest | 0.6705 | 27.38% | 45.07% | 0.1287 | 0.80% |

ROC-AUC measures ranking. Precision is the share of flagged loans that charged off. Recall is the share of all charge-offs flagged. Brier is the mean squared probability error. Expected calibration error (ECE) is the loan-count-weighted absolute difference between predicted and observed rates across fixed 10-percentage-point bins. After calibration, Brier fell from 0.1281 to 0.1280 for Logistic Regression and from 0.1299 to 0.1287 for Random Forest. Logistic Regression predicted 15.32% charge-offs on average, compared with 16.20% observed. See the [model metrics](report/metrics.csv) and [calibration bins](report/calibration.csv).

## 5. Dashboard walkthrough

1. Expand the SQL audit and review the vintage counts. The 2012 holdout includes 8,644 charge-offs and a 16.20% observed charge-off rate.
2. Switch between Logistic Regression and Random Forest, then toggle raw and calibrated probabilities to compare Brier scores and calibration bins.
3. Select Logistic Regression. The low (below 10%), medium (10% to below 20%), and high (20% or more) tiers show observed charge-off rates of 7.04%, 15.56%, and 29.35%, respectively. For discussion, the tiers could correspond to standard review, manual review, and enhanced review.
4. Switch the policy from “Retain all” to “Probability below 20%.” The simulation retains 41,479 loans (77.72%) and $503.9 million in original loan amounts. The retained portfolio's observed charge-off rate is 12.43%, and 40.36% of charge-offs are flagged. At 15%, retention falls to 58.99% and the retained portfolio's charge-off rate falls to 10.23%. These thresholds illustrate trade-offs on the funded portfolio. Live underwriting would require timestamped outcomes, evaluation of declined applicants, economics, and fairness validation.
5. Review the risk factors. The observed charge-off rate was 22.29% for FICO below 680 and 10.29% for FICO of 720 or above. It was 13.58% for 36-month loans and 27.70% for 60-month loans. These are descriptive associations.

## 6. Reproduce the results

The project was run with Python 3.14. The downloader verifies the SHA-256 of the pinned source archive. The raw data directory is ignored by Git.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python download.py
python -m unittest -v
python pipeline.py
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`. Outputs are written to `report/`. Use `python pipeline.py --archive PATH --report PATH` to specify other paths.
