"""Statistical procedures of the same-section analyses: signed-rank, rank-sum and
rank-block tests, bootstrap intervals and randomisation P values.

Every test works on plain arrays of measurements and returns a small dataclass that
carries the statistic, the degrees of freedom where the test has any, the P value and the
sample size. Nothing here formats a number, chooses a comparison or reads a file.

Values are dimensionless or carry the unit of the measurement that is passed in; no
function in this module converts units.

Rules the procedures follow
---------------------------
* Paired comparisons, such as observed against translation-control pairing across fields
  or animals, or band coverage against a whole-field baseline: exact Wilcoxon signed-rank
  test, with the normal approximation as the fallback for large samples.
* Several independent groups: Kruskal-Wallis; several paired conditions: Friedman.
* Uncertainty over objects: percentile bootstrap confidence intervals.
* Randomisation tests: two-sided empirical P values with the ``(r + 1) / (n + 1)``
  convention, without adjustment for multiplicity.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ._util import default_rng

__all__ = [
    "WilcoxonResult",
    "KruskalResult",
    "FriedmanResult",
    "BootstrapCIResult",
    "EmpiricalPResult",
    "wilcoxon_signed_rank_exact",
    "wilcoxon_signed_rank_normal",
    "wilcoxon_signed_rank",
    "kruskal_wallis",
    "friedman_test",
    "bootstrap_ci",
    "empirical_p_two_sided",
]


# --------------------------------------------------------------------------- results


@dataclass(frozen=True)
class WilcoxonResult:
    """Wilcoxon signed-rank test for paired measurements.

    Attributes
    ----------
    statistic : float
        Rank sum the P value is based on: the smaller of the two signed-rank sums for a
        two-sided test, the positive-rank sum for a one-sided test.
    W_plus, W_minus : float
        Sums of the ranks of the positive and of the negative differences.
    P : float
        P value for the requested alternative.
    n : int
        Number of pairs that entered the test (zero differences excluded under the
        ``'wilcox'`` zero handling).
    n_zero : int
        Number of pairs with a zero difference.
    n_pos : int
        Number of pairs with a positive difference among those tested.
    ties_present : bool
        True when tied absolute differences were found and mid-ranks were used.
    method : str
        ``'exact'`` for the complete signed-rank distribution, ``'normal'`` for the
        normal approximation.
    alternative : str
        ``'two-sided'``, ``'greater'`` or ``'less'``.
    """

    statistic: float
    W_plus: float
    W_minus: float
    P: float
    n: int
    n_zero: int
    n_pos: int
    ties_present: bool
    method: str
    alternative: str


@dataclass(frozen=True)
class KruskalResult:
    """Kruskal-Wallis test over several independent groups."""

    statistic: float
    df: int
    P: float
    group_n: Tuple[int, ...]
    n: int


@dataclass(frozen=True)
class FriedmanResult:
    """Friedman test over several paired conditions.

    Attributes
    ----------
    statistic : float
        Friedman chi-square statistic, with the correction for tied ranks within a block.
    df : int
        Number of conditions minus one.
    P : float
        P value of the requested method.
    n : int
        Number of blocks (paired units).
    k : int
        Number of conditions.
    method : str
        ``'exact'`` (complete enumeration of the within-block rank permutations),
        ``'asymptotic'`` (chi-square approximation) or ``'permutation'`` (seeded
        randomisation with the ``(r + 1) / (n + 1)`` convention).
    n_resamples : int or None
        Number of randomisations for ``'permutation'``.
    """

    statistic: float
    df: int
    P: float
    n: int
    k: int
    method: str
    n_resamples: Optional[int] = None


@dataclass(frozen=True)
class BootstrapCIResult:
    """Percentile bootstrap confidence interval of an arbitrary statistic.

    Attributes
    ----------
    statistic : float
        The statistic evaluated on the full sample, in its own unit.
    low, high : float
        Percentile confidence limits, in the unit of the statistic.
    confidence_level : float
        Nominal coverage as a fraction.
    n_resamples : int
        Number of bootstrap resamples drawn.
    n : int
        Number of observations resampled with replacement.
    """

    statistic: float
    low: float
    high: float
    confidence_level: float
    n_resamples: int
    n: int


@dataclass(frozen=True)
class EmpiricalPResult:
    """Two-sided empirical P value from a randomisation distribution.

    Attributes
    ----------
    statistic : float
        Observed value, in its own unit.
    P : float
        ``(r + 1) / (n + 1)``, with ``r`` the number of randomisations at least as far
        from the centre of the randomisation distribution as the observed value.
    n : int
        Number of randomisations.
    n_extreme : int
        The count ``r``.
    centre : float
        Centre of the randomisation distribution the deviations are measured from.
    null_mean : float
        Mean of the randomisation distribution, in the unit of the statistic.
    """

    statistic: float
    P: float
    n: int
    n_extreme: int
    centre: float
    null_mean: float


# --------------------------------------------------------------------------- helpers

def _groups_as_arrays(groups: Sequence[Any], name: str = "group") -> List[np.ndarray]:
    """Validate a sequence of groups and return them as 1-D float arrays."""
    if len(groups) == 0:
        raise ValueError(f"at least one {name} is required")
    out = []
    for i, g in enumerate(groups):
        arr = np.asarray(g, dtype=float).ravel()
        if arr.size == 0:
            raise ValueError(f"{name} {i} is empty")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} {i} contains non-finite values")
        out.append(arr)
    return out


# --------------------------------------------------------------------------- signed rank

def _signed_rank_setup(
    x: Any,
    y: Optional[Any],
    zero_method: str,
    ties: str,
    tie_decimals: int,
) -> Tuple[np.ndarray, np.ndarray, int, bool]:
    """Differences, their ranks, the number of zeros and whether ties were found.

    Zero differences are either refused or dropped; tied absolute differences are either
    refused or given mid-ranks. Ties are detected on the absolute differences rounded to
    ``tie_decimals`` decimals, and the ranks are taken from the same rounded values.
    """
    if zero_method not in ("wilcox", "refuse"):
        raise ValueError("zero_method must be 'wilcox' or 'refuse'")
    if ties not in ("midrank", "refuse"):
        raise ValueError("ties must be 'midrank' or 'refuse'")
    d = np.asarray(x, dtype=float).ravel()
    if y is not None:
        other = np.asarray(y, dtype=float).ravel()
        if other.shape != d.shape:
            raise ValueError("x and y must have the same length")
        d = d - other
    if d.size == 0:
        raise ValueError("at least one pair is required")
    if not np.isfinite(d).all():
        raise ValueError("the differences contain non-finite values")

    n_zero = int((d == 0).sum())
    if n_zero and zero_method == "refuse":
        raise ValueError("zero differences are not allowed with zero_method='refuse'")
    if n_zero:
        d = d[d != 0]
    if d.size == 0:
        raise ValueError("every difference is zero: the test is not defined")

    magnitude = np.round(np.abs(d), tie_decimals)
    ties_present = bool(np.unique(magnitude).size != magnitude.size)
    if ties_present and ties == "refuse":
        raise ValueError("tied absolute differences are not allowed with ties='refuse'")
    ranks = stats.rankdata(magnitude)
    return d, ranks, n_zero, ties_present


def _signed_rank_counts(ranks: np.ndarray) -> Tuple[np.ndarray, int]:
    """Counts of the positive-rank sum over all sign patterns, and the scale of the sums.

    The null distribution of the positive-rank sum is built by convolution: starting from
    a single arrangement with sum 0, every rank either joins the sum or does not. Mid-ranks
    are half-integers, so the ranks are scaled by two to keep the sums whole; the returned
    scale says by how much.

    The counts are Python integers, not machine words. The total over the whole support is
    ``2 ** n``, which passes 2 ** 63 at n = 63, so a fixed-width accumulator would wrap and
    answer a plausible but wrong tail count -- or a negative one -- instead of failing. With
    arbitrary-precision integers the enumeration stays exact at any n the caller allows, and
    the cost is the length of the support rather than the number of sign patterns.
    """
    scale = 1 if np.array_equal(ranks, np.round(ranks)) else 2
    integer_ranks = [int(r) for r in np.round(np.asarray(ranks, dtype=float) * scale)]
    counts = [0] * (sum(integer_ranks) + 1)
    counts[0] = 1
    filled = 0
    for r in integer_ranks:
        for index in range(filled, -1, -1):
            value = counts[index]
            if value:
                counts[index + r] += value
        filled += r
    return np.asarray(counts, dtype=object), scale


def wilcoxon_signed_rank_exact(
    x: Any,
    y: Optional[Any] = None,
    *,
    alternative: str = "two-sided",
    zero_method: str = "wilcox",
    ties: str = "midrank",
    tie_decimals: int = 12,
    max_n: int = 100,
) -> WilcoxonResult:
    """Exact Wilcoxon signed-rank test for paired measurements.

    The differences ``d = x - y`` are ranked by absolute value. The null distribution of
    the positive-rank sum over all ``2^n`` sign patterns is built by convolution, and the
    P value is the proportion of sign patterns at least as extreme as the observed rank
    sum; for the two-sided alternative it is twice the lower tail, capped at 1. With
    mid-ranks the enumeration is the exact randomisation distribution given the observed
    pattern of tied absolute differences.

    Parameters
    ----------
    x : array_like
        Paired measurements, or the differences themselves when ``y`` is None, in the unit
        of the measurement.
    y : array_like, optional
        The second member of each pair, in the same unit and order as ``x``.
    alternative : {'two-sided', 'greater', 'less'}, optional
        ``'greater'`` tests whether ``x`` exceeds ``y``. Default ``'two-sided'``.
    zero_method : {'wilcox', 'refuse'}, optional
        ``'wilcox'`` drops pairs with a zero difference and reduces n; ``'refuse'`` raises
        on them. Default ``'wilcox'``.
    ties : {'midrank', 'refuse'}, optional
        ``'midrank'`` gives tied absolute differences their average rank; ``'refuse'``
        raises on them. Default ``'midrank'``.
    tie_decimals : int, optional
        Decimals the absolute differences are rounded to before ties are detected and
        ranked. Default 12.
    max_n : int, optional
        Largest number of pairs the enumeration is allowed to run on. The limit is the
        running time, not the arithmetic: the counts are exact integers at any n, and the
        work grows with the length of the rank-sum support, about ``n ** 2 / 2`` additions
        of large integers per pair. Default 100, which runs in well under a second; raise
        it if the design needs more pairs and the wait is acceptable.

    Returns
    -------
    WilcoxonResult
        Rank sums, the P value, the number of pairs tested, the number of zero and of
        positive differences and the tie flag.
    """
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("alternative must be 'two-sided', 'greater' or 'less'")
    d, ranks, n_zero, ties_present = _signed_rank_setup(x, y, zero_method, ties, tie_decimals)
    n = int(d.size)
    if n > max_n:
        raise ValueError(
            f"the exact distribution over 2^n sign patterns is not computed beyond n = {max_n}; "
            "raise max_n to enumerate more pairs, or use the normal approximation"
        )
    w_plus = float(ranks[d > 0].sum())
    w_minus = float(ranks[d < 0].sum())
    counts, scale = _signed_rank_counts(ranks)
    total = int(2 ** n)

    def _lower(bound: float) -> int:
        """Number of sign patterns whose positive-rank sum is at most ``bound``."""
        index = int(math.floor(bound * scale + 1e-9))
        index = min(max(index, -1), counts.size - 1)
        return 0 if index < 0 else int(counts[: index + 1].sum())

    if alternative == "two-sided":
        statistic = min(w_plus, w_minus)
        # The doubling stays in integers: the tail count can exceed what a float holds
        # exactly, and only the ratio has to be representable.
        p = min(1.0, 2 * _lower(statistic) / total)
    elif alternative == "less":
        statistic = w_plus
        p = _lower(w_plus) / total
    else:
        statistic = w_plus
        p = (total - _lower(w_plus - 1.0 / scale)) / total
    return WilcoxonResult(
        statistic=float(statistic),
        W_plus=w_plus,
        W_minus=w_minus,
        P=float(p),
        n=n,
        n_zero=n_zero,
        n_pos=int((d > 0).sum()),
        ties_present=ties_present,
        method="exact",
        alternative=alternative,
    )


def wilcoxon_signed_rank_normal(
    x: Any,
    y: Optional[Any] = None,
    *,
    alternative: str = "two-sided",
    zero_method: str = "wilcox",
    ties: str = "midrank",
    tie_decimals: int = 12,
    correction: bool = False,
) -> WilcoxonResult:
    """Wilcoxon signed-rank test by the normal approximation.

    The positive-rank sum is standardised with mean ``n (n + 1) / 4`` and variance
    ``[n (n + 1) (2 n + 1) - sum(t^3 - t) / 2] / 24``, where the sum runs over the sizes
    ``t`` of the groups of tied absolute differences.

    Parameters
    ----------
    x, y : array_like
        As in :func:`wilcoxon_signed_rank_exact`.
    alternative : {'two-sided', 'greater', 'less'}, optional
        Default ``'two-sided'``.
    zero_method, ties, tie_decimals : optional
        As in :func:`wilcoxon_signed_rank_exact`.
    correction : bool, optional
        Apply the continuity correction of half a rank unit towards the mean.
        Default False.

    Returns
    -------
    WilcoxonResult
        Rank sums, the P value and the sample sizes; ``method`` is ``'normal'``.
    """
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("alternative must be 'two-sided', 'greater' or 'less'")
    d, ranks, n_zero, ties_present = _signed_rank_setup(x, y, zero_method, ties, tie_decimals)
    n = int(d.size)
    w_plus = float(ranks[d > 0].sum())
    w_minus = float(ranks[d < 0].sum())
    mean = n * (n + 1) / 4.0
    variance = n * (n + 1) * (2 * n + 1) / 24.0
    _, tie_sizes = np.unique(np.round(ranks, tie_decimals), return_counts=True)
    variance -= 0.5 * float((tie_sizes * (tie_sizes * tie_sizes - 1)).sum()) / 24.0
    se = math.sqrt(variance) if variance > 0 else 0.0

    if alternative == "two-sided":
        statistic = min(w_plus, w_minus)
    else:
        statistic = w_plus
    if se == 0.0:
        p = 1.0
    else:
        delta = statistic - mean
        if correction:
            # Half a rank unit towards the mean of the null distribution.
            if alternative == "two-sided":
                delta -= 0.5 * np.sign(delta)
            elif alternative == "greater":
                delta -= 0.5
            else:
                delta += 0.5
        z = delta / se
        if alternative == "two-sided":
            p = float(2.0 * stats.norm.sf(abs(z)))
        elif alternative == "greater":
            p = float(stats.norm.sf(z))
        else:
            p = float(stats.norm.cdf(z))
        p = min(1.0, p)
    return WilcoxonResult(
        statistic=float(statistic),
        W_plus=w_plus,
        W_minus=w_minus,
        P=float(p),
        n=n,
        n_zero=n_zero,
        n_pos=int((d > 0).sum()),
        ties_present=ties_present,
        method="normal",
        alternative=alternative,
    )


def wilcoxon_signed_rank(
    x: Any,
    y: Optional[Any] = None,
    *,
    alternative: str = "two-sided",
    method: str = "auto",
    max_exact_n: int = 100,
    **kwargs: Any,
) -> WilcoxonResult:
    """Wilcoxon signed-rank test, exact for small samples and approximate for large ones.

    Parameters
    ----------
    x, y : array_like
        As in :func:`wilcoxon_signed_rank_exact`.
    alternative : {'two-sided', 'greater', 'less'}, optional
        Default ``'two-sided'``.
    method : {'auto', 'exact', 'normal'}, optional
        ``'auto'`` uses the exact distribution while the number of tested pairs is at most
        ``max_exact_n`` and the normal approximation above it. Default ``'auto'``.
    max_exact_n : int, optional
        Largest number of pairs ``'auto'`` runs the exact test on. Default 100, which
        covers paired comparisons across fields of a section series in well under a
        second; above it the normal approximation is used. Raising it also raises the
        exact test's own ``max_n`` to match.
    **kwargs
        Passed on: ``zero_method``, ``ties``, ``tie_decimals`` and, for the normal
        approximation, ``correction``.

    Returns
    -------
    WilcoxonResult
        The result of whichever of the two tests was used; ``method`` says which.
    """
    if method not in ("auto", "exact", "normal"):
        raise ValueError("method must be 'auto', 'exact' or 'normal'")
    if method == "exact":
        return wilcoxon_signed_rank_exact(x, y, alternative=alternative, **kwargs)
    if method == "normal":
        return wilcoxon_signed_rank_normal(x, y, alternative=alternative, **kwargs)
    probe = _signed_rank_setup(
        x,
        y,
        kwargs.get("zero_method", "wilcox"),
        kwargs.get("ties", "midrank"),
        kwargs.get("tie_decimals", 12),
    )
    if probe[0].size <= max_exact_n:
        exact_kwargs = {k: v for k, v in kwargs.items() if k != "correction"}
        # The exact test refuses beyond its own max_n; a caller who raises max_exact_n
        # has asked for exactness up to that size, so the limit follows it.
        exact_kwargs["max_n"] = max(int(exact_kwargs.get("max_n", 100)), int(max_exact_n))
        return wilcoxon_signed_rank_exact(x, y, alternative=alternative, **exact_kwargs)
    return wilcoxon_signed_rank_normal(x, y, alternative=alternative, **kwargs)


# --------------------------------------------------------------------------- rank tests

def kruskal_wallis(*groups: Any) -> KruskalResult:
    """Kruskal-Wallis test over several independent groups.

    Wraps ``scipy.stats.kruskal``: the rank-based H statistic with the correction for
    tied values, referred to a chi-square distribution with ``k - 1`` degrees of freedom.

    Parameters
    ----------
    *groups : array_like
        One array of measurements per group, in the unit of the measurement. Groups may
        differ in size.

    Returns
    -------
    KruskalResult
        H, the degrees of freedom, the P value, the group sizes and the total n.
    """
    arrays = _groups_as_arrays(groups)
    if len(arrays) < 2:
        raise ValueError("the Kruskal-Wallis test needs at least two groups")
    res = stats.kruskal(*arrays)
    sizes = tuple(int(g.size) for g in arrays)
    return KruskalResult(
        statistic=float(res.statistic),
        df=len(arrays) - 1,
        P=float(res.pvalue),
        group_n=sizes,
        n=int(sum(sizes)),
    )


def _friedman_statistic(ranks: np.ndarray) -> float:
    """Friedman chi-square from a block-by-condition matrix of within-block ranks."""
    n, k = ranks.shape
    rank_sums = ranks.sum(axis=0)
    statistic = 12.0 / (n * k * (k + 1)) * float((rank_sums ** 2).sum()) - 3.0 * n * (k + 1)
    # Correction for ties within blocks.
    tie_term = 0.0
    for row in ranks:
        _, counts = np.unique(row, return_counts=True)
        tie_term += float((counts ** 3 - counts).sum())
    correction = 1.0 - tie_term / (n * (k ** 3 - k))
    return statistic / correction if correction > 0 else float("nan")


def _friedman_exact_p(ranks: np.ndarray, max_states: int) -> float:
    """Exact P value by complete enumeration of the within-block rank permutations.

    With no ties the statistic is a strictly increasing function of the sum of squares of
    the condition rank sums, so the distribution is obtained by a dynamic program over
    those sums. The statistic is symmetric in the conditions, so each state is stored in
    sorted order.
    """
    n, k = ranks.shape
    observed = float((ranks.sum(axis=0) ** 2).sum())
    permutations = [tuple(p) for p in itertools.permutations(range(1, k + 1))]
    states: Dict[Tuple[int, ...], int] = {tuple([0] * k): 1}
    for _ in range(n):
        nxt: Dict[Tuple[int, ...], int] = {}
        for state, count in states.items():
            for perm in permutations:
                key = tuple(sorted(s + p for s, p in zip(state, perm)))
                nxt[key] = nxt.get(key, 0) + count
        states = nxt
        if len(states) > max_states:
            raise ValueError(
                "the exact enumeration exceeds max_states; use method='asymptotic' "
                "or method='permutation'"
            )
    total = len(permutations) ** n
    extreme = sum(
        count for state, count in states.items()
        if sum(s * s for s in state) >= observed - 1e-9
    )
    return extreme / total


def friedman_test(
    data: Any,
    *,
    method: str = "auto",
    n_resamples: int = 10000,
    seed: Any = None,
    max_states: int = 200000,
) -> FriedmanResult:
    """Friedman test over several paired conditions.

    Each block (paired unit) is ranked across the conditions; the condition rank sums give
    the Friedman chi-square statistic, corrected for ties within a block. The P value is
    either exact (complete enumeration of the within-block rank permutations), asymptotic
    (chi-square with ``k - 1`` degrees of freedom, from ``scipy.stats.friedmanchisquare``)
    or from a seeded randomisation with the ``(r + 1) / (n + 1)`` convention.

    Parameters
    ----------
    data : array_like, shape (n_blocks, n_conditions)
        Measurements, one row per paired unit and one column per condition, in the unit of
        the measurement. A pandas DataFrame is accepted; its columns are the conditions.
    method : {'auto', 'exact', 'asymptotic', 'permutation'}, optional
        ``'auto'`` enumerates exactly when no block has tied values and the enumeration
        stays within ``max_states``, and falls back to the chi-square approximation
        otherwise. Default ``'auto'``.
    n_resamples : int, optional
        Number of randomisations for ``method='permutation'``. Default 10000.
    seed : int or numpy.random.Generator, optional
        Seed or generator for ``method='permutation'``; pass an int for a reproducible run.
    max_states : int, optional
        Guard on the size of the exact enumeration. Default 200000.

    Returns
    -------
    FriedmanResult
        The statistic, the degrees of freedom, the P value, the number of blocks and of
        conditions, and which method produced the P value.
    """
    if method not in ("auto", "exact", "asymptotic", "permutation"):
        raise ValueError("method must be 'auto', 'exact', 'asymptotic' or 'permutation'")
    values = np.asarray(data.values if isinstance(data, pd.DataFrame) else data, dtype=float)
    if values.ndim != 2:
        raise ValueError("data must be a 2-D block-by-condition table")
    n, k = values.shape
    if k < 3:
        raise ValueError("the Friedman test needs at least three conditions")
    if n < 2:
        raise ValueError("the Friedman test needs at least two blocks")
    if not np.isfinite(values).all():
        raise ValueError("data contains non-finite values")

    ranks = np.vstack([stats.rankdata(row) for row in values])
    statistic = _friedman_statistic(ranks)
    has_ties = any(np.unique(row).size != k for row in ranks)

    used = method
    if method == "auto":
        used = "asymptotic" if has_ties else "exact"
    if used == "exact":
        if has_ties:
            raise ValueError("the exact enumeration is not defined with ties within a block")
        try:
            p = _friedman_exact_p(ranks, max_states)
        except ValueError:
            if method == "exact":
                raise
            used = "asymptotic"
            p = float(stats.friedmanchisquare(*values.T).pvalue)
    elif used == "permutation":
        rng = default_rng(seed)
        at_least = 0
        for _ in range(int(n_resamples)):
            drawn = np.vstack([rng.permutation(row) for row in ranks])
            if _friedman_statistic(drawn) >= statistic - 1e-9:
                at_least += 1
        p = (at_least + 1) / (int(n_resamples) + 1)
    else:
        p = float(stats.friedmanchisquare(*values.T).pvalue)
    return FriedmanResult(
        statistic=float(statistic),
        df=k - 1,
        P=float(min(1.0, p)),
        n=int(n),
        k=int(k),
        method=used,
        n_resamples=int(n_resamples) if used == "permutation" else None,
    )


# --------------------------------------------------------------------------- resampling

def bootstrap_ci(
    data: Any,
    statistic: Callable[[Any], float],
    *,
    confidence_level: float = 0.95,
    n_resamples: int = 10000,
    seed: Any = None,
) -> BootstrapCIResult:
    """Percentile bootstrap confidence interval of an arbitrary statistic.

    Observations are resampled with replacement along the first axis, the statistic is
    evaluated on each resample, and the confidence limits are the percentiles of those
    values that leave equal tails outside the interval (linear interpolation between
    order statistics).

    Parameters
    ----------
    data : array_like or pandas.DataFrame
        Observations, resampled along the first axis. Rows may be scalars, vectors or
        table rows; the unit is that of the measurement.
    statistic : callable
        Called with one resample of ``data`` (the same type as ``data``) and returning one
        number, in whatever unit the statistic has.
    confidence_level : float, optional
        Nominal coverage as a fraction. Default 0.95, i.e. the 2.5th and 97.5th
        percentiles.
    n_resamples : int, optional
        Number of bootstrap resamples. Default 10000.
    seed : int or numpy.random.Generator, optional
        Seed or generator; pass an int for a reproducible interval.

    Returns
    -------
    BootstrapCIResult
        The statistic on the full sample, the two limits, the coverage, the number of
        resamples and n.
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    if int(n_resamples) < 1:
        raise ValueError("n_resamples must be positive")
    is_frame = isinstance(data, (pd.DataFrame, pd.Series))
    sample = data if is_frame else np.asarray(data)
    n = int(len(sample))
    if n == 0:
        raise ValueError("data is empty")
    rng = default_rng(seed)
    take = (lambda idx: sample.iloc[idx]) if is_frame else (lambda idx: sample[idx])
    draws = np.empty(int(n_resamples), dtype=float)
    for i in range(int(n_resamples)):
        draws[i] = float(statistic(take(rng.integers(0, n, size=n))))
    tail = 100.0 * (1.0 - confidence_level) / 2.0
    low, high = np.percentile(draws, [tail, 100.0 - tail])
    return BootstrapCIResult(
        statistic=float(statistic(sample)),
        low=float(low),
        high=float(high),
        confidence_level=float(confidence_level),
        n_resamples=int(n_resamples),
        n=n,
    )


def empirical_p_two_sided(
    observed: float,
    null_distribution: Any,
    *,
    centre: Any = "mean",
) -> EmpiricalPResult:
    """Two-sided empirical P value from a randomisation distribution.

    The observed value and every randomisation are measured as absolute deviations from
    the centre of the randomisation distribution. With ``r`` randomisations that deviate at
    least as far as the observed value and ``n`` randomisations in all, the P value is
    ``(r + 1) / (n + 1)``: the observed arrangement counts as one of the possible ones, so
    the P value can never be zero. No adjustment for multiplicity is applied.

    Parameters
    ----------
    observed : float
        Observed value of the statistic, in its own unit.
    null_distribution : array_like
        Values of the same statistic over the randomisations, in the same unit.
    centre : {'mean', 'median'} or float, optional
        Centre the deviations are measured from. Default ``'mean'``, the mean of the
        randomisation distribution.

    Returns
    -------
    EmpiricalPResult
        The observed value, the P value, the number of randomisations, the count ``r``,
        the centre used and the mean of the randomisation distribution.
    """
    draws = np.asarray(null_distribution, dtype=float).ravel()
    if draws.size == 0:
        raise ValueError("the randomisation distribution is empty")
    if not np.isfinite(draws).all():
        raise ValueError("the randomisation distribution contains non-finite values")
    observed = float(observed)
    if not np.isfinite(observed):
        raise ValueError("observed must be finite")
    if isinstance(centre, str):
        if centre == "mean":
            c = float(draws.mean())
        elif centre == "median":
            c = float(np.median(draws))
        else:
            raise ValueError("centre must be 'mean', 'median' or a number")
    else:
        c = float(centre)
    n = int(draws.size)
    r = int((np.abs(draws - c) >= abs(observed - c) - 1e-12).sum())
    return EmpiricalPResult(
        statistic=observed,
        P=min(1.0, (r + 1) / (n + 1)),
        n=n,
        n_extreme=r,
        centre=c,
        null_mean=float(draws.mean()),
    )
