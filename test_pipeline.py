import unittest

import numpy as np
import pandas as pd

from pipeline import (
    CATEGORICAL, NUMERIC, RAW_NUMERIC, calibration_table, features, policy,
    quality_check, risk_tiers, vintage_split,
)


class PipelineTests(unittest.TestCase):
    def test_sql_reconciles_invalid_records_and_defines_outcome(self):
        row = dict.fromkeys(RAW_NUMERIC, 1) | dict.fromkeys(CATEGORICAL, "other")
        row.update(id="one", loan_status="Charged Off", issue_d="Jan-2011", term="36 months")
        raw = pd.DataFrame([
            row, row | {"id": "two", "loan_status": "Fully Paid"},
            row | {"id": "dupe"}, row | {"id": "dupe"},
            row | {"id": "bad_month", "issue_d": "Abc-2011"},
            row | {"id": "pending", "loan_status": "Current"},
            row | {"id": "zero", "loan_amnt": 0},
            row | {"id": "bad_term", "term": "99 months"},
            row | {"id": None},
        ])
        cohort, audit, _ = quality_check(raw)
        self.assertEqual(audit.loans.sum(), len(raw))
        self.assertEqual(dict(zip(cohort.id, cohort.charged_off)), {"one": 1, "two": 0})
        self.assertEqual(dict(zip(audit.quality_status, audit.loans)), {
            "duplicate_id": 2, "eligible": 2, "invalid_issue_month": 1,
            "unknown_outcome": 1, "invalid_loan_amount": 1, "invalid_term": 1, "missing_id": 1,
        })

    def test_features_ignore_post_origination_fields(self):
        row = dict.fromkeys(RAW_NUMERIC, 10) | dict.fromkeys(CATEGORICAL, "other")
        row.update(term="36 months", annual_inc=0, loan_status="Charged Off",
                   last_pymnt_d="Jan-2019", recoveries=123, int_rate=99)
        x = features(pd.DataFrame([row]))
        self.assertEqual(x.columns.tolist(), NUMERIC + CATEGORICAL)
        self.assertTrue(np.isnan(x.loan_to_income.iloc[0]))
        self.assertEqual(x.term_months.iloc[0], 36)

    def test_year_boundaries(self):
        frame = pd.DataFrame({"issue_month": ["2008-01", "2010-12", "2011-01", "2012-01"]})
        self.assertEqual([x.index.to_list() for x in vintage_split(frame)], [[0, 1], [2], [3]])

    def test_threshold_and_risk_tier_boundaries(self):
        scored = pd.DataFrame({"pd": [.09, .10, .20, 1.0],
                               "charged_off": [0, 1, 1, 1], "loan_amnt": [100, 200, 300, 400]})
        self.assertEqual(risk_tiers(scored).loans.tolist(), [1, 1, 2])
        self.assertEqual(calibration_table(scored).loans.sum(), len(scored))
        self.assertEqual(calibration_table(scored).bin.max(), 9)
        at_threshold = policy(scored, .20)
        self.assertEqual(at_threshold["retained"], 2)
        self.assertEqual(at_threshold["retained_amount"], 300)
        self.assertAlmostEqual(at_threshold["flagged_chargeoff_share"], 2 / 3)
        self.assertEqual(policy(scored, 1)["retained"], 4)


if __name__ == "__main__":
    unittest.main()
