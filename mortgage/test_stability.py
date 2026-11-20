import unittest

import numpy as np
import pandas as pd

from mortgage.stability import monthly_policy


class MonthlyPolicyTest(unittest.TestCase):
    def test_fixed_threshold_and_undefined_rates(self):
        loans = pd.DataFrame({
            "first_payment": ["202502", "202501", "202501", "202503"],
            "event": [0, 1, 0, 1],
        }, index=[6, 3, 9, 12])
        monthly = monthly_policy(loans, np.array([.1, .2, .1, .3]), .2)
        self.assertEqual(monthly.first_payment_month.tolist(), ["2025-01", "2025-02", "2025-03"])
        self.assertEqual(monthly.loans_retained.tolist(), [1, 1, 0])
        self.assertEqual(monthly.events_flagged.tolist(), [1, 0, 1])
        self.assertEqual(monthly.loc[0, "capture_rate"], 1)
        self.assertTrue(pd.isna(monthly.loc[1, "capture_rate"]))
        self.assertTrue(pd.isna(monthly.loc[2, "retained_event_rate"]))
