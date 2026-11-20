import unittest

import numpy as np

from mortgage.uncertainty import comparison


class BootstrapComparisonTest(unittest.TestCase):
    def test_identical_models_have_exactly_zero_gaps(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([.1, .6, .5, .9])
        result = comparison(labels, scores, scores, .5, .5, draws=10)
        self.assertTrue((result[["estimate", "lower_95", "upper_95"]] == 0).all().all())

    def test_known_difference_and_reproducible_intervals(self):
        labels = [0, 0, 1, 1]
        first = [.1, .2, .8, .9]
        second = [.8, .9, .1, .2]
        result = comparison(labels, first, second, .5, .5, draws=10, seed=9)
        self.assertEqual(result.estimate.tolist(), [1., 1., 0.])
        self.assertTrue(result.equals(comparison(labels, first, second, .5, .5, draws=10, seed=9)))
