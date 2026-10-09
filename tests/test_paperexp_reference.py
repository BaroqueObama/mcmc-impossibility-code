"""Reference intervals, coverage classification and coverage summaries."""

import numpy as np
import pytest
from scipy import stats

from mcmc.paperexp import reference as ref


def test_clopper_pearson_known_values():
    lo, hi = ref.clopper_pearson(0, 10, 0.05)
    assert lo == 0.0 and abs(hi - (1 - 0.025 ** (1 / 10))) < 1e-12
    lo, hi = ref.clopper_pearson(950, 1000, 0.05)
    ci = stats.binomtest(950, 1000).proportion_ci(0.95, method="exact")
    assert abs(lo - ci.low) < 1e-10 and abs(hi - ci.high) < 1e-10


def test_truth_intervals_and_classification():
    L, U = ref.truth_intervals(np.array([0.10, 0.02]), np.array([0.03, 0.03]), 0.01)
    assert np.allclose(L, [0.06, 0.0]) and np.allclose(U, [0.14, 0.06])
    B = np.array([[0.20, 0.07], [0.05, 0.07], [0.10, 0.07], [0.20, 0.03]])
    codes = ref.classify(B, L, U)
    assert codes.tolist() == [[1, 1], [-1, 1], [0, 1], [1, 0]]
    s = ref.coverage_summary(codes, beta_ref=0.01)
    assert (s["covered_all"], s["any_violation"], s["unresolved"]) == (1, 1, 2)
    assert s["coverage_if_unresolved_fail"] == 0.25 and s["coverage_if_unresolved_succeed"] == 0.75
    assert s["beta_ref"] == 0.01


def test_analytic_classification_and_input_checks():
    codes = ref.classify_analytic(np.array([[0.2, 0.1], [0.05, 0.3]]), np.array([0.1, 0.1]))
    assert codes.tolist() == [[1, 1], [-1, 1]]
    with pytest.raises(ValueError):
        ref.classify(np.ones((2, 2)), np.array([0.2, 0.1]), np.array([0.1, 0.1]))
    with pytest.raises(ValueError):
        ref.classify_analytic(np.array([[np.nan, 1.0]]), np.array([0.1, 0.1]))


def test_reference_radii():
    b_t, b_pi = ref.reference_radii(1.0, 4096, None, 0.01, 0.01, 8)
    assert b_pi == 0.0 and b_t > 1 / 64
