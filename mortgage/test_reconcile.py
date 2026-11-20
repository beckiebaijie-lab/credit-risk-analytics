from contextlib import closing
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from mortgage.reconcile import reconcile


class ReconciliationTest(unittest.TestCase):
    def test_sql_boundary_and_mismatch(self):
        with TemporaryDirectory() as directory:
            database = Path(directory) / "fixture.sqlite"
            with closing(sqlite3.connect(database)) as db:
                pd.DataFrame({
                    "loan_id": ["a", "b"], "first_payment": ["202501", "202502"],
                    "original_upb": [100, 200], "split": ["test", "test"],
                    "outcome_status": ["complete_without_event", "event"],
                }).to_sql("mortgage_cohort", db, index=False)
            scores = pd.DataFrame({
                "loan_id": ["a", "b"], "model": ["LR", "LR"], "probability": [.05, .1],
            })
            monthly = pd.DataFrame({
                "model": ["LR", "LR"], "calibration_target": [.75, .75],
                "first_payment_month": ["2025-01", "2025-02"], "loans": [1, 1],
                "events": [0, 1], "loans_retained": [1, 0], "events_retained": [0, 0],
                "events_flagged": [0, 1], "observed_rate": [0., 1.], "retention": [1., 0.],
                "retained_event_rate": [0., np.nan], "capture_rate": [np.nan, 1.],
            })
            policies = pd.DataFrame({
                "model": ["LR"], "calibration_target": [.75], "loans": [1],
                "retained_balance": [100], "retention": [.5],
                "retained_event_rate": [0.], "share_events_flagged": [1.],
            })
            result = reconcile(database, scores, [("LR", .75, .1)], monthly, policies)
            self.assertEqual(result.events_flagged.tolist(), [0, 1, 1])
            monthly.loc[1, "events_flagged"] = 0
            with self.assertRaisesRegex(ValueError, "events_flagged"):
                reconcile(database, scores, [("LR", .75, .1)], monthly, policies)
