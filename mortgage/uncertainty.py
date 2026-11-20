"""Paired bootstrap intervals for fixed, already-fitted holdout scores."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def comparison(labels, first_scores, second_scores, first_cutoff, second_cutoff,
               draws=1000, seed=2025):
    """Compare models on identical resamples, preserving the observed class counts."""
    labels = np.asarray(labels)
    scores = np.column_stack((first_scores, second_scores))
    flagged = scores >= [first_cutoff, second_cutoff]
    events = np.flatnonzero(labels == 1)
    non_events = np.flatnonzero(labels == 0)
    if not len(events) or not len(non_events):
        raise ValueError("Both outcome classes are required")
    if draws < 1:
        raise ValueError("draws must be positive")

    def differences(indices, event_indices):
        outcomes = labels[indices]
        sample = scores[indices]
        auc_gap = roc_auc_score(outcomes, sample[:, 0]) - roc_auc_score(outcomes, sample[:, 1])
        capture = flagged[event_indices].mean(axis=0)
        retention = 1 - flagged[indices].mean(axis=0)
        return auc_gap, capture[0] - capture[1], retention[0] - retention[1]

    observed = differences(np.arange(len(labels)), events)
    rng = np.random.default_rng(seed)
    estimates = np.empty((draws, 3))
    for draw in range(draws):
        sampled_events = rng.choice(events, size=len(events), replace=True)
        sampled_non_events = rng.choice(non_events, size=len(non_events), replace=True)
        indices = np.concatenate((sampled_events, sampled_non_events))
        estimates[draw] = differences(indices, sampled_events)

    lower, upper = np.quantile(estimates, [.025, .975], axis=0)
    return pd.DataFrame({
        "metric": ["ROC-AUC gap", "Event capture rate gap", "Loan retention rate gap"],
        "estimate": observed,
        "lower_95": lower,
        "upper_95": upper,
        "bootstrap_draws": draws,
    })
