# Freddie Mac Mortgage Credit Risk

This analysis compares Logistic Regression and Random Forest for predicting 12-month serious delinquency on Freddie Mac mortgages. It uses a January–June 2025 first-payment holdout and evaluates how much of the historical portfolio each model would retain when thresholds are set using an earlier cohort.

## Source and reproduction

Sign in to [Freddie Mac Clarity](https://capitalmarkets.freddiemac.com/clarity), open **Data Download → CRT & Historical Data → SFLLD Data**, and download the Standard Dataset sample archives for 2020 through 2025 from the same release. This analysis uses the release with performance through June 2026. The 2020 and 2024 files capture earlier archive vintages whose first scheduled payments can fall in the subsequent year. See the [dataset and terms](https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset).

From the repository root, use Python 3.14 and the pinned root requirements. Keep the source archives and SQLite database in the ignored `data/` folder.

```sh
python -m pip install -r requirements.txt
python -m unittest -v test_pipeline mortgage.test_cohort mortgage.test_stability mortgage.test_uncertainty mortgage.test_reconcile
python mortgage/cohort.py data/sample_2020.zip data/sample_2021.zip data/sample_2022.zip data/sample_2023.zip data/sample_2024.zip data/sample_2025.zip --database data/mortgage.sqlite
python mortgage/evaluate.py --database data/mortgage.sqlite
```

Open `mortgage/report/dashboard.html` in a browser. The generated HTML and eleven aggregate CSVs are saved locally. The database path must be new for each cohort build. To generate a second report from an existing database, pass `--report PATH` to the evaluator.

## Cohort and leakage controls

| Stage | First scheduled payment month |
|---|---|
| Fit models and preprocessing | January 2021–June 2022 |
| Fit calibration and select policy thresholds | July–December 2023 |
| Evaluate locked models and policies | January–June 2025 |

The outcome window includes the first scheduled payment month and the following 11 months. An event is any known monthly delinquency status of 3 or higher within that window. A negative requires either 12 known months below 90 days past due or an observed voluntary payoff with complete preceding months below 90 days past due. Duplicate performance months and histories insufficient to identify an outcome are excluded. The outcome measures serious delinquency rather than terminal charge-off.

The final training observation window closes before calibration starts. The final calibration observation window closes before the 2025 holdout starts. This is a retrospective analysis using the updated release. Period end dates do not establish when historical records were first published or revised.

SQL uses loan identifiers to connect origination and servicing records. The models receive origination FICO, CLTV, DTI, LTV, log original balance, original term, occupancy, property type, and purpose. Monthly servicing fields, loan identifiers, and future delinquency information are excluded from model inputs. Imputation, scaling, and one-hot encoding are fit on the training cohort.

## Evaluation and decision walkthrough

1. Open the cohort coverage audit. Compare included events and negatives with histories that could not be classified.
2. Toggle the model and probability version. Compare ROC-AUC, average precision, calibration bins, and Brier. Evaluate average precision alongside the observed event rate because serious delinquency is rare.
3. Set the calibration retention target to 75%. Each model's threshold is derived from its own earlier calibration scores and stays fixed on the 2025 holdout. Realized holdout retention can differ from the calibration target and between models. Precision and recall use this operating point for the selected probability version.
4. Compare the share of serious delinquencies flagged and the balance retained. Risk tiers use score cutoffs determined by the 50th and 75th percentiles of calibrated scores on the calibration cohort. Historical mortgages were already acquired or guaranteed. This simulation does not evaluate declined borrowers, policy causality, or production fairness.

5. Review **Monthly policy stability** for the selected model and calibration target. Each row groups the holdout by first scheduled payment month and uses the same locked threshold. Compare origination cohort size, observed event rate, retention, and the share of observed events flagged. Read event counts alongside rates because a small number of events can move the monthly estimates substantially.

6. Open **Model comparison uncertainty** to see differences between Logistic Regression and Random Forest. The analysis resamples the same 2025 loans for both models in 1,000 draws, separately within observed events and non-events. It reports percentile intervals for the model gaps, conditional on the fixed scores, thresholds, and observed class counts. These intervals do not include model refitting uncertainty or account for geographic or other clustering.

7. In **Equal review capacity**, rank each model’s calibrated holdout scores and review the top 10% or 20%, rounded down to whole loans. Ties keep the existing first-payment-month and loan-ID order. Capture rate divides flagged events by all observed events. Lift divides capture rate by the actual review share. This comparison measures rank performance at equal holdout capacity.

## SQL policy reconciliation

[policy_reconciliation.sql](policy_reconciliation.sql) joins the eligible holdout cohort to temporary model scores and locked thresholds in SQLite. It independently applies the strict retention rule, aggregates counts by first payment month and for the full portfolio, and calculates rates from those counts. Python compares the results by model, threshold scenario, and period. It requires exact agreement on counts and portfolio balances, handles undefined rates as missing values, and stops report generation when a check fails. Loan scores and thresholds are kept in connection-local staging tables.

## Power BI setup

On Windows, use **Home → Get data → Text/CSV** to import `mortgage/report/*.csv` with UTF-8 and comma separators. Use the following tables independently or create a model dimension for a shared slicer.

| CSV | Suggested view |
|---|---|
| `cohort_quality.csv`, `splits.csv` | Cohort size, observed events, and outcome coverage |
| `metrics.csv` | Model comparison with model and probability version filters |
| `calibration.csv` | Mean predicted versus observed rate by probability bin |
| `policy.csv` | Realized retention and flagged events by calibration target |
| `policy_reconciliation.csv` | SQLite audit results by model, scenario, and period |
| `model_uncertainty.csv` | Paired model differences with 95% percentile bounds |
| `review_capacity.csv` | Events captured and lift at equal 10% and 20% review budgets |
| `monthly_stability.csv` | Monthly retention and event capture with model and calibration target filters |
| `risk_tiers.csv`, `risk_factors.csv` | Observed event rates with loan and event counts |

Format rates and probability thresholds as percentages and retained balances as USD. Filter to one model and probability version before charting individual calibration bins. Policy rows describe overlapping scenarios, so select one model and target when reading retained balances and counts.

For monthly stability, use `first_payment_month` on the axis and show event counts in tooltips. When rolling up months, divide summed counts to get the combined rate.
