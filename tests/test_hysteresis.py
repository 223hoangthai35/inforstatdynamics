"""
Unit tests for HysteresisGMMWrapper — Schmitt-trigger filter over GMM posteriors.

Three regression cases:
  1. Single-day noise must NOT flip a held regime label.
  2. A hard-margin penetration (>= delta_hard) MUST flip immediately.
  3. Sustained drift (margin in [delta_soft, delta_hard)) MUST flip after
     exactly t_persist consecutive bars — and not before.

Run:
    python -m pytest tests/test_hysteresis.py -v
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from skills.ds_skill import HysteresisGMMWrapper


class _StubGMM:
    """
    Minimal GMM stub mirroring sklearn's predict_proba semantics:
    returns one pre-canned posterior row per input row.
    Single-row calls advance an internal index (for step()); multi-row calls
    return the next len(X) rows en bloc (for transform()).
    """

    def __init__(self, posteriors):
        self.posteriors = np.asarray(posteriors, dtype=np.float64)
        self._i = 0

    def predict_proba(self, X):
        X = np.atleast_2d(X)
        n = X.shape[0]
        if n == 1:
            p = self.posteriors[self._i:self._i + 1]
            self._i += 1
            return p
        return self.posteriors[:n]


class _StubClassifier:
    """Wraps the stub GMM with the attributes HysteresisGMMWrapper expects."""

    def __init__(self, posteriors, label_map=None):
        self.gmm = _StubGMM(posteriors)
        # Identity mapping by default — semantic label == cluster index
        self._cluster_to_regime = label_map or {0: 0, 1: 1, 2: 2}


def test_single_day_noise_does_not_flip():
    """A one-bar excursion with sub-soft margin must not move the held label."""
    # Bar 0: clear cluster-0 dominance -> hold = 0
    # Bar 1: cluster-1 leads but margin (0.45 - 0.40 = 0.05) < delta_soft (0.20)
    # Bar 2: cluster-0 reasserts
    posteriors = [
        [0.80, 0.10, 0.10],
        [0.40, 0.45, 0.15],
        [0.75, 0.15, 0.10],
    ]
    clf = _StubClassifier(posteriors)
    wrap = HysteresisGMMWrapper(clf, delta_hard=0.40, delta_soft=0.20, t_persist=3)
    labels = [wrap.step(np.array([0.0, 0.0])) for _ in posteriors]
    assert labels == [0, 0, 0], f"Single-bar noise leaked: got {labels}"


def test_hard_margin_flips_immediately():
    """A single bar with margin >= delta_hard must flip the held label at once."""
    posteriors = [
        [0.80, 0.10, 0.10],   # adopt 0
        [0.10, 0.85, 0.05],   # margin 0.85 - 0.10 = 0.75 >= 0.40 -> flip to 1
    ]
    clf = _StubClassifier(posteriors)
    wrap = HysteresisGMMWrapper(clf, delta_hard=0.40, delta_soft=0.20, t_persist=3)
    labels = [wrap.step(np.array([0.0, 0.0])) for _ in posteriors]
    assert labels == [0, 1], f"Hard penetration failed to flip: got {labels}"


def test_sustained_drift_flips_after_t_persist():
    """
    Soft-margin (in [0.20, 0.40)) must require exactly t_persist consecutive
    bars on the same candidate before flipping; not earlier, not later.
    """
    # All "drift" bars use margin 0.30 -> soft (>= 0.20, < 0.40)
    drift = [0.30, 0.60, 0.10]   # held=0 -> margin = 0.60 - 0.30 = 0.30
    posteriors = [
        [0.80, 0.10, 0.10],   # bar 0 — adopt held = 0
        drift,                 # bar 1 — pending(1) = 1
        drift,                 # bar 2 — pending(1) = 2 (still holding 0)
        drift,                 # bar 3 — pending(1) reaches t_persist=3 -> flip
        [0.10, 0.65, 0.25],   # bar 4 — confirm new held = 1 sticks
    ]
    clf = _StubClassifier(posteriors)
    wrap = HysteresisGMMWrapper(clf, delta_hard=0.40, delta_soft=0.20, t_persist=3)
    labels = [wrap.step(np.array([0.0, 0.0])) for _ in posteriors]
    assert labels == [0, 0, 0, 1, 1], (
        f"Sustained-drift flip timing wrong: got {labels} "
        f"(expected flip on bar 3, the third consecutive soft-margin bar)"
    )


def test_pending_resets_when_candidate_changes():
    """If the soft-margin candidate switches, the persistence counter resets."""
    posteriors = [
        [0.80, 0.10, 0.10],   # bar 0 - hold = 0
        [0.30, 0.60, 0.10],   # bar 1 - pending=1, count=1 (margin 0.30)
        [0.30, 0.10, 0.60],   # bar 2 - candidate switched to 2: count resets to 1
        [0.30, 0.60, 0.10],   # bar 3 - candidate switched back to 1: count=1
    ]
    clf = _StubClassifier(posteriors)
    wrap = HysteresisGMMWrapper(clf, delta_hard=0.40, delta_soft=0.20, t_persist=3)
    labels = [wrap.step(np.array([0.0, 0.0])) for _ in posteriors]
    assert labels == [0, 0, 0, 0], f"Candidate-switch must not flip: got {labels}"


def test_transform_batch_matches_stepwise():
    """transform() must produce identical labels to repeated step() calls."""
    posteriors = [
        [0.85, 0.10, 0.05],
        [0.20, 0.70, 0.10],   # hard flip 0 -> 1 (margin 0.50)
        [0.10, 0.80, 0.10],
        [0.45, 0.40, 0.15],   # noise
    ]
    clf_a = _StubClassifier(posteriors)
    clf_b = _StubClassifier(posteriors)
    wrap_step = HysteresisGMMWrapper(clf_a, delta_hard=0.40, delta_soft=0.20, t_persist=3)
    wrap_batch = HysteresisGMMWrapper(clf_b, delta_hard=0.40, delta_soft=0.20, t_persist=3)
    stepwise = np.array([wrap_step.step(np.array([0.0, 0.0])) for _ in posteriors])
    batch = wrap_batch.transform(np.zeros((len(posteriors), 2)))
    assert np.array_equal(stepwise, batch), (
        f"step() vs transform() diverge: {stepwise} vs {batch}"
    )


if __name__ == "__main__":
    tests = [
        test_single_day_noise_does_not_flip,
        test_hard_margin_flips_immediately,
        test_sustained_drift_flips_after_t_persist,
        test_pending_resets_when_candidate_changes,
        test_transform_batch_matches_stepwise,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed.")
