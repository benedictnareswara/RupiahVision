"""Evaluation metrics, calibration and the "unknown" threshold (pure numpy).

Terms used here and in the README:
  announce   the app speaks a denomination: prediction isn't "unknown" and its calibrated
             probability is at least the threshold.
  coverage   share of banknote images that get announced.
  risk       share of announcements that are wrong (wrong denomination, or a note
             announced when there was none). This is the error a user actually hears.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

UNKNOWN_INDEX = 0


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    probs = softmax(logits, temperature)
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, None)).mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperature scaling: the single T that makes softmax(logits / T) best match reality.

    Networks are usually overconfident; dividing logits by T > 1 softens them without
    changing which class wins. T is found by golden-section search on log T, minimising
    negative log-likelihood on the validation set.
    """
    lo, hi = np.log(0.05), np.log(20.0)
    ratio = (np.sqrt(5) - 1) / 2

    def f(log_t: float) -> float:
        return nll(logits, labels, float(np.exp(log_t)))

    a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    fa, fb = f(a), f(b)
    for _ in range(60):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - ratio * (hi - lo)
            fa = f(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + ratio * (hi - lo)
            fb = f(b)
    return float(np.exp((lo + hi) / 2))


def decide(probs: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """(predictions, announced mask) for calibrated probabilities."""
    preds = probs.argmax(axis=1)
    announced = (preds != UNKNOWN_INDEX) & (probs.max(axis=1) >= threshold)
    return preds, announced


def selective_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    preds, announced = decide(probs, threshold)
    is_note = labels != UNKNOWN_INDEX
    wrong = announced & (preds != labels)
    return {
        "threshold": float(threshold),
        "coverage": float(announced[is_note].mean()) if is_note.any() else 0.0,
        "risk": float(wrong.sum() / announced.sum()) if announced.any() else 0.0,
        "announcements": int(announced.sum()),
        "wrong_announcements": int(wrong.sum()),
        "false_announcement_rate_on_unknown": (
            float(announced[~is_note].mean()) if (~is_note).any() else 0.0
        ),
    }


def choose_threshold(probs: np.ndarray, labels: np.ndarray, target_risk: float) -> float:
    """Lowest threshold whose risk on these images is at most target_risk.

    Lower thresholds announce more, so the lowest acceptable one gives the most coverage.
    Returns a threshold above 1 (never announce) if no threshold reaches the target.
    """
    preds, _ = decide(probs, 0.0)
    confidence = probs.max(axis=1)
    candidates = np.unique(confidence[preds != UNKNOWN_INDEX])
    for threshold in candidates:  # ascending
        if selective_metrics(probs, labels, threshold)["risk"] <= target_risk:
            return float(threshold)
    return 1.0 + 1e-6


def risk_coverage_curve(probs: np.ndarray, labels: np.ndarray, points: int = 101) -> dict:
    thresholds = np.linspace(0.0, 1.0, points)
    rows = [selective_metrics(probs, labels, t) for t in thresholds]
    return {
        "threshold": thresholds.tolist(),
        "coverage": [r["coverage"] for r in rows],
        "risk": [r["risk"] for r in rows],
    }


def confusion_matrix(labels: np.ndarray, preds: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (labels, preds), 1)
    return matrix


def cluster_bootstrap_ci(
    values: np.ndarray,
    groups: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float]:
    """Confidence interval for `statistic(values)`, resampling whole groups.

    Images from one photo group are near-copies of each other, so resampling single images
    would pretend we have more independent evidence than we do and give intervals that are
    too narrow. Resampling groups (with replacement) respects that.
    """
    if len(values) == 0:
        return (float("nan"), float("nan"))
    unique, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(unique))]
    rng = np.random.default_rng(seed)
    stats = np.empty(n_resamples)
    for i in range(n_resamples):
        chosen = rng.integers(0, len(unique), size=len(unique))
        stats[i] = statistic(values[np.concatenate([members[g] for g in chosen])])
    alpha = (1 - level) / 2
    return float(np.quantile(stats, alpha)), float(np.quantile(stats, 1 - alpha))


def wilson_ci(successes: float, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a proportion; sensible even at 0% or 100%."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def proportion_ci(correct: np.ndarray, groups: np.ndarray, seed: int = 0) -> tuple[float, float]:
    """95% CI for an accuracy, treating photo groups as the independent observations.

    Takes the wider of two intervals:
      - the group bootstrap (cluster_bootstrap_ci), and
      - a Wilson interval with n = number of groups.
    The bootstrap alone is too narrow with only a handful of groups, and it collapses to a
    single point when every image is right (every resample gives 100%), which would claim
    certainty. Wilson over groups stays honest there: 5 all-correct groups give about
    57%-100%, not 100%-100%.
    """
    n_groups = len(np.unique(groups))
    if n_groups == 0:
        return (float("nan"), float("nan"))
    boot = cluster_bootstrap_ci(correct, groups, seed=seed)
    wilson = wilson_ci(float(correct.mean()) * n_groups, n_groups)
    return min(boot[0], wilson[0]), max(boot[1], wilson[1])
