"""Regression coverage for the 12-month outcome construction."""
from contextlib import closing
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile

from mortgage.cohort import prepare


def pipe_row(width, fields):
    values = [""] * width
    for position, value in fields.items():
        values[position] = str(value)
    return "|".join(values)


class CohortTest(unittest.TestCase):
    def test_observed_window_payoff_missing_and_duplicate(self):
        loans = ["event", "full", "payoff", "missing", "duplicate"]
        orig = [
            pipe_row(31, {0: 720, 1: 202506, 9: 35, 10: 300000, 19: loan})
            for loan in loans
        ]
        perf = [
            pipe_row(35, {0: loan, 1: month, 3: status, 8: code})
            for loan, month, status, code in [
                ("event", 202506, 3, ""),
                ("payoff", 202506, 0, "01"),
                ("duplicate", 202506, 0, ""),
                ("duplicate", 202506, 3, ""),
                ("full", 202606, 3, ""),  # Month 13 must not set the outcome.
            ]
        ]
        perf.extend(
            pipe_row(35, {0: "full", 1: month, 3: 0})
            for month in [*range(202506, 202513), *range(202601, 202606)]
        )
        with TemporaryDirectory() as folder, closing(sqlite3.connect(":memory:")) as db:
            archive = Path(folder) / "sample_2025.zip"
            with ZipFile(archive, "w") as z:
                z.writestr("sample_orig_2025.txt", "\n".join(orig))
                z.writestr("sample_perf_2025.txt", "\n".join(perf))
            prepare([archive], db)
            outcomes = dict(db.execute("SELECT loan_id, outcome_status FROM mortgage_cohort"))
        self.assertEqual(outcomes, {
            "event": "event", "full": "complete_without_event",
            "payoff": "prepaid_without_event", "missing": "insufficient_history",
            "duplicate": "duplicate_performance_month",
        })
