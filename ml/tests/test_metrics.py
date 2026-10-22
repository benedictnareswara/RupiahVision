import numpy as np
import pytest

from rupiahvision.training import metrics


def test_softmax_rows_sum_to_one_and_temperature_softens():
    logits = np.array([[2.0, 0.0, -1.0], [0.0, 0.0, 0.0]])
    probs = metrics.softmax(logits)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    assert metrics.softmax(logits, 3.0)[0, 0] < probs[0, 0]


def test_fit_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    true_logits = rng.normal(size=(4000, 5)) * 1.5
    labels = np.array([rng.choice(5, p=p) for p in metrics.softmax(true_logits)])
    # A model that is 3x overconfident should be calibrated by T close to 3.
    assert metrics.fit_temperature(true_logits * 3, labels) == pytest.approx(3.0, rel=0.1)


def probs_for(preds, confidences, k=4):
    probs = np.full((len(preds), k), 0.0)
    for i, (p, c) in enumerate(zip(preds, confidences, strict=True)):
        probs[i] = (1 - c) / (k - 1)
        probs[i, p] = c
    return probs


def test_selective_metrics():
    # labels: two notes (class 1, 2) and one unknown (0)
    labels = np.array([1, 2, 0, 1])
    probs = probs_for([1, 3, 2, 1], [0.9, 0.95, 0.6, 0.4])
    m = metrics.selective_metrics(probs, labels, threshold=0.5)
    # announced: 0 (right), 1 (wrong), 2 (unknown announced as 2: wrong); 3 is below threshold
    assert m["announcements"] == 3
    assert m["wrong_announcements"] == 2
    assert m["coverage"] == pytest.approx(2 / 3)
    assert m["risk"] == pytest.approx(2 / 3)
    assert m["false_announcement_rate_on_unknown"] == 1.0


def test_unknown_predictions_are_never_announced():
    probs = probs_for([0, 0], [0.99, 0.99])
    assert metrics.selective_metrics(probs, np.array([1, 0]), 0.0)["announcements"] == 0


def test_choose_threshold_is_lowest_meeting_target():
    labels = np.array([1, 1, 2, 2])
    probs = probs_for([1, 1, 3, 2], [0.95, 0.9, 0.7, 0.6])  # the 0.7 one is wrong
    t = metrics.choose_threshold(probs, labels, target_risk=0.0)
    assert t == pytest.approx(0.9)
    assert metrics.selective_metrics(probs, labels, t)["risk"] == 0.0
    assert metrics.choose_threshold(probs, labels, target_risk=0.5) == pytest.approx(0.6)


def test_choose_threshold_never_announces_if_target_unreachable():
    probs = probs_for([2], [0.99])
    assert metrics.choose_threshold(probs, np.array([1]), target_risk=0.0) > 1.0


def test_confusion_matrix():
    m = metrics.confusion_matrix(np.array([0, 1, 1, 2]), np.array([0, 1, 2, 2]), 3)
    assert m.tolist() == [[1, 0, 0], [0, 1, 1], [0, 0, 1]]


def test_cluster_bootstrap_is_wider_than_naive_for_correlated_groups():
    # 10 groups of 50 identical outcomes: really only 10 independent observations.
    values = np.repeat(np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1], dtype=float), 50)
    groups = np.repeat(np.arange(10), 50)
    low, high = metrics.cluster_bootstrap_ci(values, groups)
    naive_low, naive_high = metrics.cluster_bootstrap_ci(values, np.arange(len(values)))
    assert low <= values.mean() <= high
    assert (high - low) > 2 * (naive_high - naive_low)


def test_wilson_ci_matches_known_values():
    low, high = metrics.wilson_ci(5, 5)
    assert high == 1.0 and low == pytest.approx(0.566, abs=1e-3)
    low, high = metrics.wilson_ci(5, 10)
    assert (low, high) == (pytest.approx(0.237, abs=1e-3), pytest.approx(0.763, abs=1e-3))


def test_proportion_ci_does_not_collapse_when_all_correct():
    correct = np.ones(40)
    groups = np.repeat(np.arange(5), 8)
    low, high = metrics.proportion_ci(correct, groups)
    assert high == 1.0 and low < 0.6  # 5 groups can't prove near-perfect accuracy
    mixed = np.array([1.0] * 30 + [0.0] * 10)
    boot = metrics.cluster_bootstrap_ci(mixed, groups)
    wilson = metrics.wilson_ci(0.75 * 5, 5)
    assert metrics.proportion_ci(mixed, groups) == (
        min(boot[0], wilson[0]),
        max(boot[1], wilson[1]),
    )
