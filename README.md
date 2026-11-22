# Credit Risk Analytics

Two Python and SQL case studies comparing credit-risk models and simulating how fixed thresholds affect loan retention.

| Case study | Test cohort | Logistic regression | Random forest |
|---|---|---:|---:|
| Freddie Mac · 12-month serious delinquency | 23,873 loans with first payments in Jan–Jun 2025 | 0.802 ROC-AUC | 0.748 ROC-AUC |
| LendingClub · Charge-off | 53,367 loans originated in 2012 | 0.678 ROC-AUC | 0.671 ROC-AUC |

- **Freddie Mac.** With thresholds set on the calibration cohort for 75% target retention, logistic regression retained 78.14% of test loans and flagged 56 of 91 serious delinquencies, a 2.82× lift over the holdout event rate. See the [mortgage study](mortgage/README.md) for data access, replication, monthly stability, and SQLite reconciliation.
- **LendingClub.** At a 20% predicted charge-off threshold, logistic regression retained 77.72% of test loans. The retained cohort's observed charge-off rate was 12.43%, compared with 16.20% across the holdout. [Download the interactive dashboard](report/dashboard.html) or use the [Power BI import guide](POWER_BI.md).

Both simulations use historical funded loans.

## Run the LendingClub study

Use Python 3.14 in a virtual environment.

```sh
python -m pip install -r requirements.txt
python download.py
python -m unittest -v
python pipeline.py
```

The pipeline writes aggregate results to `report/`. Freddie Mac has separate [data and run instructions](mortgage/README.md).
