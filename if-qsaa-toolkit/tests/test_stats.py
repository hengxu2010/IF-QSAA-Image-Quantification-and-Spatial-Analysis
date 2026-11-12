"""Tests of if_qsaa.stats on synthetic data with known ground truth.

Every sample is generated here, seeded where it is random. Results are checked against
hand-computed values, against an independent brute-force enumeration, or against the
corresponding scipy routine where one exists.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps


from if_qsaa import stats as qs


# --------------------------------------------------------------------------- Wilcoxon

def test_wilcoxon_exact_all_differences_positive():
    # With five pairs all in the same direction the smaller rank sum is 0, and only one of
    # the 2^5 sign patterns reaches it: P = 2 x 1 / 32.
    x = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    res = qs.wilcoxon_signed_rank_exact(x, y)
    assert res.statistic == 0.0
    assert res.W_plus == 15.0
    assert res.P == pytest.approx(2.0 / 32.0)
    assert (res.n, res.n_pos, res.n_zero) == (5, 5, 0)
    assert res.ties_present is False
    assert res.method == "exact"


def test_wilcoxon_exact_matches_scipy_without_ties():
    rng = np.random.default_rng(41)
    x = rng.normal(1.0, 1.0, 9)
    y = rng.normal(0.0, 1.0, 9)
    for alternative in ("two-sided", "greater", "less"):
        own = qs.wilcoxon_signed_rank_exact(x, y, alternative=alternative)
        reference = sps.wilcoxon(x, y, alternative=alternative, method="exact")
        assert own.P == pytest.approx(float(reference.pvalue), rel=1e-12)
        assert own.statistic == pytest.approx(float(reference.statistic))


def test_wilcoxon_exact_matches_scipy_with_ties_and_zeros():
    d = np.array([1.0, 2.0, 2.0, 3.0, 0.0, -1.0, 4.0, 4.0])
    own = qs.wilcoxon_signed_rank_exact(d)
    reference = sps.wilcoxon(d, method="exact")
    assert own.ties_present is True
    assert own.n_zero == 1
    assert own.n == 7
    assert own.statistic == pytest.approx(float(reference.statistic))
    assert own.P == pytest.approx(float(reference.pvalue), rel=1e-12)


def test_wilcoxon_exact_refuse_policies():
    with pytest.raises(ValueError):
        qs.wilcoxon_signed_rank_exact(np.array([1.0, 0.0, 2.0]), zero_method="refuse")
    with pytest.raises(ValueError):
        qs.wilcoxon_signed_rank_exact(np.array([1.0, -1.0, 2.0]), ties="refuse")
    with pytest.raises(ValueError):
        qs.wilcoxon_signed_rank_exact(np.zeros(4))
    with pytest.raises(ValueError):
        qs.wilcoxon_signed_rank_exact(np.array([1.0, 2.0]), np.array([1.0]))
    with pytest.raises(ValueError):
        qs.wilcoxon_signed_rank_exact(np.array([1.0, 2.0]), alternative="sideways")
    with pytest.raises(ValueError):
        qs.wilcoxon_signed_rank_exact(np.arange(1.0, 10.0), max_n=5)


def test_wilcoxon_exact_single_pair():
    res = qs.wilcoxon_signed_rank_exact(np.array([3.0]), np.array([1.0]))
    assert (res.n, res.n_pos) == (1, 1)
    assert res.P == pytest.approx(1.0)
    assert qs.wilcoxon_signed_rank_exact(
        np.array([3.0]), np.array([1.0]), alternative="greater"
    ).P == pytest.approx(0.5)


def test_wilcoxon_normal_matches_scipy():
    rng = np.random.default_rng(42)
    x = rng.normal(0.4, 1.0, 40)
    for alternative in ("two-sided", "greater", "less"):
        own = qs.wilcoxon_signed_rank_normal(x, alternative=alternative)
        reference = sps.wilcoxon(x, alternative=alternative, method="approx")
        assert own.P == pytest.approx(float(reference.pvalue))
        assert own.statistic == pytest.approx(float(reference.statistic))
    corrected = qs.wilcoxon_signed_rank_normal(x, correction=True)
    assert corrected.P == pytest.approx(
        float(sps.wilcoxon(x, method="approx", correction=True).pvalue)
    )
    assert corrected.P > qs.wilcoxon_signed_rank_normal(x).P


def test_wilcoxon_dispatcher_picks_the_method_by_sample_size():
    rng = np.random.default_rng(43)
    small = rng.normal(0.8, 1.0, 12)
    paired_fields = rng.normal(0.3, 1.0, 75)
    large = rng.normal(0.3, 1.0, 120)
    assert qs.wilcoxon_signed_rank(small).method == "exact"
    assert qs.wilcoxon_signed_rank(paired_fields).method == "exact"
    assert qs.wilcoxon_signed_rank(large).method == "normal"
    assert qs.wilcoxon_signed_rank(paired_fields, max_exact_n=50).method == "normal"
    assert qs.wilcoxon_signed_rank(paired_fields, method="exact", max_exact_n=50).method == "exact"
    with pytest.raises(ValueError, match="not computed beyond"):
        qs.wilcoxon_signed_rank(large, method="exact")
    assert qs.wilcoxon_signed_rank(small, max_exact_n=5).method == "normal"
    with pytest.raises(ValueError):
        qs.wilcoxon_signed_rank(small, method="nonesuch")


def test_wilcoxon_exact_and_normal_agree_near_the_usual_cut_offs():
    # The approximation is good where P values are read off and gets worse in the far
    # tail, so the two methods are only compared in the range that matters.
    rng = np.random.default_rng(44)
    x = rng.normal(0.15, 1.0, 30)
    exact = qs.wilcoxon_signed_rank_exact(x).P
    approximate = qs.wilcoxon_signed_rank_normal(x).P
    assert 0.01 < exact < 0.1
    assert exact == pytest.approx(approximate, rel=0.05)


def test_wilcoxon_exact_no_crossing_of_the_direction():
    # Every difference in the same direction is the most extreme arrangement there is, so
    # the one-sided P value is the smallest the enumeration can give: 1 / 2^n.
    x = np.arange(1.0, 9.0)
    res = qs.wilcoxon_signed_rank_exact(x, alternative="greater")
    assert res.P == pytest.approx(1.0 / 2 ** 8)
    assert res.W_minus == 0.0


def test_signed_rank_counts_total_is_exactly_two_to_the_n():
    # The counts span every sign pattern, so they must sum to 2 ** n with no rounding and
    # no wrap-around. Past n = 62 that total no longer fits a 64-bit word, which is where a
    # fixed-width accumulator starts returning a plausible wrong answer.
    for n in (8, 40, 62, 63, 75, 90):
        ranks = np.arange(1.0, n + 1.0)
        counts, scale = qs._signed_rank_counts(ranks)
        assert scale == 1
        assert int(counts.sum()) == 2 ** n


def test_wilcoxon_exact_stays_a_probability_above_sixty_two_pairs():
    # Exact Wilcoxon at these sample sizes is what the paired same-section comparisons need,
    # and it is reachable only above the old ceiling. A wrapped accumulator showed up here as
    # a P value outside [0, 1] or one far from the normal approximation, not as an error.
    rng = np.random.default_rng(7)
    for n in (63, 65, 70, 75):
        d = rng.normal(0.25, 1.0, n)
        exact = qs.wilcoxon_signed_rank_exact(d).P
        approximate = qs.wilcoxon_signed_rank_normal(d).P
        assert 0.0 <= exact <= 1.0
        assert exact == pytest.approx(approximate, rel=0.1, abs=0.01)


def test_wilcoxon_exact_refuses_beyond_max_n_instead_of_guessing():
    d = np.random.default_rng(3).normal(0.2, 1.0, 40)
    with pytest.raises(ValueError, match="not computed beyond"):
        qs.wilcoxon_signed_rank_exact(d, max_n=20)


# --------------------------------------------------------------------------- rank tests

def test_kruskal_matches_scipy():
    rng = np.random.default_rng(51)
    groups = [rng.normal(m, 1.0, n) for m, n in ((0.0, 7), (1.0, 6), (3.0, 8))]
    res = qs.kruskal_wallis(*groups)
    reference = sps.kruskal(*groups)
    assert res.statistic == pytest.approx(float(reference.statistic))
    assert res.P == pytest.approx(float(reference.pvalue))
    assert res.df == 2
    assert res.group_n == (7, 6, 8)
    assert res.n == 21


def test_kruskal_identical_groups_give_no_evidence():
    g = np.array([1.0, 2.0, 3.0, 4.0])
    res = qs.kruskal_wallis(g, g + 0.0, g + 0.0)
    assert res.statistic == pytest.approx(0.0)
    assert res.P == pytest.approx(1.0)
    with pytest.raises(ValueError):
        qs.kruskal_wallis(g)


def friedman_statistic_bruteforce(ranks):
    n, k = ranks.shape
    rank_sums = ranks.sum(axis=0)
    return 12.0 / (n * k * (k + 1)) * float((rank_sums ** 2).sum()) - 3.0 * n * (k + 1)


def test_friedman_statistic_matches_scipy():
    rng = np.random.default_rng(61)
    data = rng.normal(size=(7, 4)) + np.array([0.0, 0.5, 1.0, 2.0])
    res = qs.friedman_test(data)
    reference = sps.friedmanchisquare(*data.T)
    assert res.statistic == pytest.approx(float(reference.statistic))
    assert res.df == 3
    assert (res.n, res.k) == (7, 4)


def test_friedman_exact_matches_complete_enumeration():
    data = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0], [1.0, 3.0, 2.0]])
    res = qs.friedman_test(data, method="exact")
    ranks = np.vstack([sps.rankdata(row) for row in data])
    observed = friedman_statistic_bruteforce(ranks)
    permutations = list(itertools.permutations([1.0, 2.0, 3.0]))
    count = 0
    total = 0
    for arrangement in itertools.product(permutations, repeat=3):
        total += 1
        if friedman_statistic_bruteforce(np.array(arrangement)) >= observed - 1e-9:
            count += 1
    assert total == 6 ** 3
    assert res.P == pytest.approx(count / total)
    assert res.method == "exact"


def test_friedman_exact_with_one_consistent_ordering():
    # Four blocks ranked the same way: only the k! column relabelings of that ordering
    # reach the largest statistic, so the exact P value is k! / (k!)^n.
    data = np.tile(np.array([1.0, 2.0, 3.0]), (4, 1))
    res = qs.friedman_test(data, method="exact")
    assert res.P == pytest.approx(math.factorial(3) / math.factorial(3) ** 4)


def test_friedman_falls_back_when_blocks_have_ties():
    data = np.array([[1.0, 1.0, 2.0], [1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
    auto = qs.friedman_test(data)
    assert auto.method == "asymptotic"
    assert auto.P == pytest.approx(float(sps.friedmanchisquare(*data.T).pvalue))
    with pytest.raises(ValueError):
        qs.friedman_test(data, method="exact")


def test_friedman_permutation_is_reproducible_and_bounded():
    rng = np.random.default_rng(62)
    data = rng.normal(size=(6, 3)) + np.array([0.0, 1.0, 2.0])
    first = qs.friedman_test(data, method="permutation", n_resamples=200, seed=7)
    again = qs.friedman_test(data, method="permutation", n_resamples=200, seed=7)
    other = qs.friedman_test(data, method="permutation", n_resamples=200, seed=8)
    assert first.P == again.P
    assert first.P >= 1.0 / 201.0
    assert first.n_resamples == 200
    assert other.P >= 1.0 / 201.0


def test_friedman_rejects_bad_input():
    with pytest.raises(ValueError):
        qs.friedman_test(np.ones((4, 2)))
    with pytest.raises(ValueError):
        qs.friedman_test(np.ones((1, 3)))
    with pytest.raises(ValueError):
        qs.friedman_test(np.ones(5))
    with pytest.raises(ValueError):
        qs.friedman_test(np.array([[1.0, 2.0, 3.0], [1.0, 2.0, np.nan]]))
    with pytest.raises(ValueError):
        qs.friedman_test(np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]]), method="nonesuch")


def test_friedman_accepts_a_dataframe():
    frame = pd.DataFrame({"c1": [1.0, 2.0, 1.0], "c2": [2.0, 1.0, 3.0], "c3": [3.0, 3.0, 2.0]})
    assert qs.friedman_test(frame, method="exact").k == 3


# --------------------------------------------------------------------------- resampling

def test_bootstrap_ci_brackets_a_known_mean_and_is_reproducible():
    rng = np.random.default_rng(71)
    sample = rng.normal(10.0, 1.0, 200)
    first = qs.bootstrap_ci(sample, np.mean, seed=5, n_resamples=2000)
    again = qs.bootstrap_ci(sample, np.mean, seed=5, n_resamples=2000)
    assert (first.low, first.high) == (again.low, again.high)
    assert first.statistic == pytest.approx(float(sample.mean()))
    assert first.low < 10.0 < first.high
    assert first.n == 200
    assert first.n_resamples == 2000
    # The percentile interval of a mean is close to the normal-theory interval.
    half_width = 1.96 * float(sample.std(ddof=1)) / math.sqrt(sample.size)
    assert first.high - first.low == pytest.approx(2 * half_width, rel=0.15)


def test_bootstrap_ci_narrows_with_more_observations():
    rng = np.random.default_rng(72)
    small = qs.bootstrap_ci(rng.normal(0.0, 1.0, 20), np.mean, seed=1, n_resamples=1000)
    large = qs.bootstrap_ci(rng.normal(0.0, 1.0, 400), np.mean, seed=1, n_resamples=1000)
    assert large.high - large.low < small.high - small.low


def test_bootstrap_ci_of_a_constant_sample_is_degenerate():
    res = qs.bootstrap_ci(np.full(10, 3.5), np.mean, seed=2, n_resamples=100)
    assert res.low == pytest.approx(3.5)
    assert res.high == pytest.approx(3.5)


def test_bootstrap_ci_resamples_dataframe_rows():
    rng = np.random.default_rng(73)
    frame = pd.DataFrame({"area": rng.normal(5.0, 1.0, 120), "group": ["a", "b"] * 60})

    def ratio(block):
        return float(block.loc[block.group == "a", "area"].sum()
                     / block.loc[block.group == "b", "area"].sum())

    res = qs.bootstrap_ci(frame, ratio, seed=3, n_resamples=300)
    assert res.low < res.statistic < res.high
    assert res.n == 120


def test_bootstrap_ci_rejects_bad_input():
    with pytest.raises(ValueError):
        qs.bootstrap_ci(np.arange(5.0), np.mean, confidence_level=1.0)
    with pytest.raises(ValueError):
        qs.bootstrap_ci(np.arange(5.0), np.mean, n_resamples=0)
    with pytest.raises(ValueError):
        qs.bootstrap_ci(np.array([]), np.mean)


# --------------------------------------------------------------------------- randomisation

def test_empirical_p_is_hand_computable_and_two_sided():
    null = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # centre 3
    high = qs.empirical_p_two_sided(5.0, null)
    low = qs.empirical_p_two_sided(1.0, null)
    assert high.n == 5
    assert high.n_extreme == 2
    assert high.P == pytest.approx(3.0 / 6.0)
    assert low.P == pytest.approx(high.P)
    assert high.centre == pytest.approx(3.0)
    assert high.null_mean == pytest.approx(3.0)


def test_empirical_p_floor_is_one_over_n_plus_one():
    rng = np.random.default_rng(81)
    null = rng.normal(0.0, 1.0, 999)
    res = qs.empirical_p_two_sided(50.0, null)
    assert res.n_extreme == 0
    assert res.P == pytest.approx(1.0 / 1000.0)
    assert qs.empirical_p_two_sided(float(null.mean()), null).P == pytest.approx(1.0)


def test_empirical_p_accepts_a_median_centre_or_a_number():
    null = np.array([0.0, 0.0, 0.0, 10.0])
    assert qs.empirical_p_two_sided(0.0, null, centre="median").P == pytest.approx(5.0 / 5.0)
    fixed = qs.empirical_p_two_sided(12.0, null, centre=0.0)
    assert fixed.centre == pytest.approx(0.0)
    assert fixed.P == pytest.approx(1.0 / 5.0)


def test_empirical_p_rejects_bad_input():
    with pytest.raises(ValueError):
        qs.empirical_p_two_sided(1.0, [])
    with pytest.raises(ValueError):
        qs.empirical_p_two_sided(np.nan, [1.0, 2.0])
    with pytest.raises(ValueError):
        qs.empirical_p_two_sided(1.0, [1.0, np.nan])
    with pytest.raises(ValueError):
        qs.empirical_p_two_sided(1.0, [1.0, 2.0], centre="mode")


# --------------------------------------------------------------------------- end to end

def test_tests_do_not_touch_the_file_system(tmp_path):
    # The module is pure: a run writes nothing, so a fresh directory stays empty.
    rng = np.random.default_rng(91)
    groups = [rng.normal(m, 1.0, 5) for m in (0.0, 1.0, 2.0)]
    qs.wilcoxon_signed_rank(groups[0], groups[1])
    qs.kruskal_wallis(*groups)
    qs.friedman_test(np.column_stack(groups))
    qs.bootstrap_ci(groups[0], np.mean, seed=1, n_resamples=50)
    qs.empirical_p_two_sided(0.5, rng.normal(0.0, 1.0, 200))
    assert list(tmp_path.iterdir()) == []


def test_wilcoxon_auto_follows_a_raised_exact_limit():
    # Asking 'auto' for exactness up to 150 pairs must not trip the exact test's own
    # default limit of 100.
    d = np.random.default_rng(29).normal(0.2, 1.0, 120)
    res = qs.wilcoxon_signed_rank(d, max_exact_n=150)
    assert res.method == "exact"
    assert res.P == pytest.approx(qs.wilcoxon_signed_rank_exact(d, max_n=150).P)
