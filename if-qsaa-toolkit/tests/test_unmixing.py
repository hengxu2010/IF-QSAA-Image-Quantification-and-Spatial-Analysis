"""Tests for if_qsaa.unmixing on synthetic images with known cross-talk."""

from __future__ import annotations


import numpy as np
import pytest


from if_qsaa import unmixing as um
from if_qsaa._util import default_rng, label_mask

# Ground-truth coupling of the synthetic field.
ALPHA = 0.18  # marker into readout
BETA = 0.12   # readout into marker
GAMMA = 0.25  # nuclear into readout
DELTA = 0.30  # readout into nuclear


# ------------------------------------------------------------------ synthetic images

def _blob(shape, centre, radius_px, amplitude):
    """Disk with a linear radial falloff, so intensity spans a range inside it."""
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    distance = np.sqrt((rows - centre[0]) ** 2 + (cols - centre[1]) ** 2)
    out = np.zeros(shape, dtype=float)
    inside = distance <= radius_px
    out[inside] = amplitude * (1.0 - 0.5 * distance[inside] / radius_px)
    return out


def _flat_disk(shape, centre, radius_px, amplitude):
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    inside = (rows - centre[0]) ** 2 + (cols - centre[1]) ** 2 <= radius_px ** 2
    return np.where(inside, float(amplitude), 0.0)


def _synthetic_field(seed=7, noise=1.0, alpha=ALPHA, beta=BETA, gamma=GAMMA):
    """Three channels built from disjoint structures with known coupling.

    Nuclei, marker structures and readout objects sit on a grid and never overlap, so
    each coefficient is identifiable on its own set of structures.
    """
    rng = default_rng(seed)
    size = 460
    shape = (size, size)
    nuclear_true = np.zeros(shape)
    marker_true = np.zeros(shape)
    readout_true = np.zeros(shape)
    for i in range(9):
        for j in range(9):
            centre = (30 + 50 * i, 30 + 50 * j)
            kind = (i + j) % 3
            if kind == 0:
                nuclear_true += _blob(shape, centre, 8, 120.0)
            elif kind == 1:
                marker_true += _blob(shape, centre, 8, 120.0)
            else:
                readout_true += _blob(shape, centre, 6, 150.0)

    nuclear_corr = nuclear_true + rng.normal(0.0, noise, shape)
    readout_res = (
        readout_true + alpha * marker_true + gamma * nuclear_corr
        + rng.normal(0.0, noise, shape)
    )
    marker_res = marker_true + beta * readout_true + rng.normal(0.0, noise, shape)
    tissue = np.ones(shape, dtype=bool)
    return dict(
        readout_res=readout_res,
        marker_res=marker_res,
        nuclear_corr=nuclear_corr,
        readout_true=readout_true,
        marker_true=marker_true,
        nuclear_true=nuclear_true,
        tissue=tissue,
        usable=tissue.copy(),
        nuclei=nuclear_true > 50.0,
        marker_mask=marker_true > 50.0,
        readout_mask=readout_true > 50.0,
    )


def _estimate(field, **kwargs):
    options = dict(
        tissue=field["tissue"],
        usable=field["usable"],
        nuclei=field["nuclei"],
        marker_thr=20.0,
        beta_min_level=90.0,
        min_source_range=8.0,
        max_px=100,
    )
    options.update(kwargs)
    return um.estimate_crosstalk(
        field["readout_res"], field["marker_res"], field["nuclear_corr"], **options
    )


# ------------------------------------------------------- fixed-coefficient subtraction

def test_native_full_scale_per_dtype():
    assert um.native_full_scale(np.zeros((2, 2), np.uint8)) == 255.0
    assert um.native_full_scale(np.zeros((2, 2), np.uint16)) == 65535.0
    assert um.native_full_scale(np.zeros((2, 2), bool)) == 1.0
    assert um.native_full_scale(np.full((2, 2), 0.5)) == 1.0
    assert um.native_full_scale(np.array([[0.0, 7.0]])) == 7.0


def test_subtract_crosstalk_recovers_a_known_component():
    rng = default_rng(0)
    source = rng.uniform(0.0, 200.0, (24, 24))
    own = np.full((24, 24), 10.0)
    observed = own + 0.35 * source
    corrected = um.subtract_crosstalk(observed, source, 0.35)
    assert np.allclose(corrected, own)


def test_subtract_crosstalk_clipping_and_clipped_fraction():
    source = np.full((4, 4), 100.0)
    observed = np.full((4, 4), 10.0)
    observed[0, 0] = 60.0
    clipped = um.subtract_crosstalk(observed, source, 0.5)
    assert clipped.min() == 0.0 and clipped[0, 0] == 10.0
    unclipped = um.subtract_crosstalk(observed, source, 0.5, clip_negative=False)
    assert unclipped[1, 1] == pytest.approx(-40.0)
    assert um.clipped_fraction(observed, source, 0.5) == pytest.approx(15.0 / 16.0)
    assert um.clipped_fraction(observed, source, 0.0) == 0.0


def test_subtract_crosstalk_across_bit_depths():
    source = np.full((3, 3), 255.0)
    observed = np.full((3, 3), 65535.0)
    corrected = um.subtract_crosstalk(
        observed, source, 0.5, source_full_scale=255.0, observed_full_scale=65535.0
    )
    assert np.allclose(corrected, 65535.0 * 0.5)
    with pytest.raises(ValueError):
        um.subtract_crosstalk(observed, source, 0.5, source_full_scale=255.0)


def test_source_background_percentile_removes_a_constant_pedestal():
    source = np.full((10, 10), 20.0)
    source[0, 0] = 120.0
    component = um.crosstalk_component(source, 1.0, source_background_percentile=10.0)
    assert component[5, 5] == pytest.approx(0.0)
    assert component[0, 0] == pytest.approx(100.0)


def test_source_background_in_grey_levels_removes_the_same_pedestal():
    source = np.full((10, 10), 20.0)
    source[0, 0] = 120.0
    in_levels = um.crosstalk_component(source, 1.0, source_background=20.0)
    as_percentile = um.crosstalk_component(source, 1.0, source_background_percentile=10.0)
    assert np.allclose(in_levels, as_percentile)

    # Without the term the pedestal is scaled by the coefficient and subtracted as a
    # constant offset from the whole observed channel; with it the background is kept.
    observed = np.full((10, 10), 50.0)
    observed[0, 0] = 60.0
    pedestal = um.subtract_crosstalk(observed, source, 0.5, clip_negative=False)
    kept = um.subtract_crosstalk(
        observed, source, 0.5, clip_negative=False, source_background=20.0
    )
    assert pedestal[5, 5] == pytest.approx(50.0 - 0.5 * 20.0)
    assert kept[5, 5] == pytest.approx(50.0)
    assert kept[0, 0] == pytest.approx(60.0 - 0.5 * 100.0)

    assert um.clipped_fraction(observed, source, 0.5, source_background=20.0) == 0.0
    corrected, components = um.unmix_fixed(
        {"source": source, "other": observed}, "source", {"other": 0.5},
        source_background=20.0,
    )
    assert np.allclose(components["other"], 0.5 * np.maximum(source - 20.0, 0.0))
    assert corrected["other"][5, 5] == pytest.approx(50.0)


def test_source_background_terms_are_exclusive_and_checked():
    source = np.full((4, 4), 20.0)
    with pytest.raises(ValueError):
        um.crosstalk_component(
            source, 1.0, source_background=5.0, source_background_percentile=10.0
        )
    with pytest.raises(ValueError):
        um.crosstalk_component(source, 1.0, source_background=-1.0)
    with pytest.raises(ValueError):
        um.crosstalk_component(source, 1.0, source_background=float("nan"))


def test_unmix_fixed_corrects_only_the_listed_channels():
    rng = default_rng(1)
    source = rng.uniform(0.0, 100.0, (12, 12))
    channels = {
        "source": source,
        "first": 5.0 + 0.4 * source,
        "second": 2.0 + 0.1 * source,
        "untouched": np.ones((12, 12)),
    }
    corrected, components = um.unmix_fixed(
        channels, "source", {"first": 0.4, "second": 0.1}
    )
    assert np.allclose(corrected["first"], 5.0)
    assert np.allclose(corrected["second"], 2.0)
    assert np.allclose(corrected["untouched"], 1.0)
    assert np.allclose(corrected["source"], source)
    assert set(components) == {"first", "second"}
    assert np.allclose(components["first"], 0.4 * source)


def test_unmix_fixed_rejects_a_self_correction_and_a_missing_image():
    channels = {"source": np.ones((4, 4)), "other": np.ones((4, 4))}
    with pytest.raises(ValueError):
        um.unmix_fixed(channels, "source", {"source": 0.2})
    with pytest.raises(KeyError):
        um.unmix_fixed(channels, "source", {"absent": 0.2})
    with pytest.raises(KeyError):
        um.unmix_fixed(channels, "absent", {"other": 0.2})


# ------------------------------------------------------------------------ calibration

def _calibration_pair(seed=3, coefficient=0.25, intercept=3.0, noise=0.5):
    rng = default_rng(seed)
    source = rng.uniform(0.0, 200.0, (80, 80))
    observed = intercept + coefficient * source + rng.normal(0.0, noise, source.shape)
    return source, observed


def test_least_squares_calibration_recovers_slope_and_intercept():
    source, observed = _calibration_pair()
    out = um.calibrate_coupling_coefficient(observed, source, source_min_level=10.0)
    assert out["coefficient"] == pytest.approx(0.25, abs=0.01)
    assert out["intercept"] == pytest.approx(3.0, abs=0.1)
    assert out["n_pixels"] == out["n_selected"] == int((source > 10.0).sum())
    assert out["slope_ci"] is None


def test_through_origin_and_median_ratio_force_a_zero_intercept():
    rng = default_rng(4)
    source = rng.uniform(20.0, 200.0, (40, 40))
    observed = 0.25 * source
    for estimator in ("least_squares_through_origin", "median_ratio"):
        out = um.calibrate_coupling_coefficient(
            observed, source, source_min_level=10.0, estimator=estimator
        )
        assert out["coefficient"] == pytest.approx(0.25, abs=1e-9)
        assert out["intercept"] == 0.0


def test_robust_calibration_survives_pixels_with_own_signal():
    source, observed = _calibration_pair(noise=0.0, intercept=0.0)
    contaminated = observed.copy()
    # Own signal sits on the brightest structures, so it is correlated with the source
    # and pulls a least-squares slope upwards.
    own = source > 180.0
    contaminated[own] += 100.0

    robust = um.calibrate_coupling_coefficient(
        contaminated, source, source_min_level=10.0, estimator="theil_sen",
        max_pixels=2000, seed=0,
    )
    least_squares = um.calibrate_coupling_coefficient(
        contaminated, source, source_min_level=10.0
    )
    assert robust["coefficient"] == pytest.approx(0.25, abs=0.01)
    assert robust["slope_ci"] is not None
    assert robust["n_pixels"] == 2000
    assert abs(least_squares["coefficient"] - 0.25) > 0.05

    # On a co-stained section the own-signal pixels can be excluded by a level instead.
    gated = um.calibrate_coupling_coefficient(
        contaminated, source, source_min_level=10.0, observed_max_level=60.0
    )
    assert gated["coefficient"] == pytest.approx(0.25, abs=0.01)
    assert gated["n_selected"] < robust["n_selected"]


def test_calibration_excludes_saturated_pixels():
    source = np.array([[10.0, 20.0, 200.0], [30.0, 40.0, 200.0]])
    observed = 0.5 * source
    out = um.calibrate_coupling_coefficient(
        observed, source, source_min_level=5.0, saturation_level=200.0,
        estimator="median_ratio",
    )
    assert out["n_selected"] == 4
    assert out["coefficient"] == pytest.approx(0.5)


def test_calibration_subsampling_is_reproducible():
    source, observed = _calibration_pair(seed=6)
    first = um.calibrate_coupling_coefficient(
        observed, source, source_min_level=10.0, max_pixels=500, seed=11
    )
    second = um.calibrate_coupling_coefficient(
        observed, source, source_min_level=10.0, max_pixels=500, seed=11
    )
    assert first["coefficient"] == second["coefficient"]
    assert first["n_pixels"] == 500


def test_calibration_edge_cases():
    source = np.zeros((4, 4))
    observed = np.zeros((4, 4))
    with pytest.raises(ValueError):
        um.calibrate_coupling_coefficient(observed, source, source_min_level=1.0)
    with pytest.raises(ValueError):
        um.calibrate_coupling_coefficient(
            observed, source, source_min_level=1.0, estimator="nonsense"
        )


def test_calibrate_coupling_over_several_detectors():
    rng = default_rng(8)
    source = rng.uniform(0.0, 200.0, (30, 30))
    channels = {
        "source": source,
        "first": 0.4 * source,
        "second": 0.05 * source,
    }
    out = um.calibrate_coupling(
        channels, "source", source_min_level=10.0, estimator="median_ratio"
    )
    assert set(out) == {"first", "second"}
    assert out["first"]["coefficient"] == pytest.approx(0.4)
    assert out["second"]["coefficient"] == pytest.approx(0.05)

    per_detector = um.calibrate_coupling(
        channels, "source", source_min_level=10.0, detectors=["second"],
        observed_max_level={"second": 5.0},
    )
    assert set(per_detector) == {"second"}
    assert per_detector["second"]["n_selected"] < out["second"]["n_selected"]


# ------------------------------------------------ per-field estimation and unmixing

def test_unsaturated_mask_is_strict_and_honours_a_region():
    first = np.array([[10.0, 200.0], [10.0, 10.0]])
    second = np.array([[10.0, 10.0], [200.0, 10.0]])
    mask = um.unsaturated_mask([first, second], 200.0)
    assert mask.tolist() == [[True, False], [False, True]]
    region = np.array([[True, True], [True, False]])
    assert um.unsaturated_mask([first, second], 200.0, within=region).sum() == 1
    with pytest.raises(ValueError):
        um.unsaturated_mask([], 200.0)


def test_subtract_bleed_is_exact_and_optionally_clips():
    source = np.array([[0.0, 100.0]])
    observed = np.array([[5.0, 20.0]])
    assert um.subtract_bleed(observed, source, 0.3).tolist() == [[5.0, -10.0]]
    assert um.subtract_bleed(observed, source, 0.3, clip_negative=True).tolist() == [[5.0, 0.0]]


def test_readout_to_nuclear_bleed_recovers_a_known_coefficient():
    shape = (120, 120)
    rng = default_rng(2)
    readout = np.zeros(shape)
    for i in range(4):
        for j in range(4):
            readout += _flat_disk(shape, (15 + 30 * i, 15 + 30 * j), 6, 200.0)
    own_nuclei = _flat_disk(shape, (60, 5), 3, 150.0)
    nuclear = DELTA * readout + own_nuclei + rng.normal(0.0, 1.0, shape)
    readout = readout + rng.normal(0.0, 1.0, shape)
    usable = np.ones(shape, dtype=bool)

    out = um.estimate_readout_to_nuclear_bleed(
        nuclear, readout, usable=usable, readout_thr=50.0, readout_sigma=1.0,
        min_level=70.0,
    )
    assert out["delta"] == pytest.approx(DELTA, abs=0.02)
    assert out["n_pixels"] > 200
    assert out["selection_level"] == pytest.approx(70.0)

    corrected = um.subtract_bleed(nuclear, readout, out["delta"])
    assert abs(float(corrected[readout > 100.0].mean())) < 3.0

    too_few = um.estimate_readout_to_nuclear_bleed(
        nuclear, readout, usable=usable, readout_thr=50.0, readout_sigma=1.0,
        min_level=70.0, min_pixels=10 ** 7,
    )
    assert too_few["delta"] == 0.0


def test_component_slopes_recovers_per_structure_slopes():
    shape = (40, 40)
    source = np.zeros(shape)
    observed = np.zeros(shape)
    labels = np.zeros(shape, dtype=int)
    ramp = np.linspace(0.0, 100.0, 10)
    labels[5:15, 5:15] = 1
    source[5:15, 5:15] = ramp[None, :]
    observed[5:15, 5:15] = 0.2 * ramp[None, :]
    labels[25:35, 25:35] = 2
    source[25:35, 25:35] = ramp[None, :]
    observed[25:35, 25:35] = 0.6 * ramp[None, :]
    usable = np.ones(shape, dtype=bool)

    slopes = um.component_slopes(
        labels, source, observed, usable, 20, default_rng(0), min_source_range=5.0
    )
    assert slopes == pytest.approx([0.2, 0.6])


def test_component_slopes_skips_small_and_flat_structures():
    shape = (30, 30)
    labels = np.zeros(shape, dtype=int)
    source = np.zeros(shape)
    observed = np.zeros(shape)
    usable = np.ones(shape, dtype=bool)
    # too few pixels
    labels[2:4, 2:4] = 1
    source[2:4, 2:4] = [[0.0, 50.0], [0.0, 50.0]]
    observed[2:4, 2:4] = [[0.0, 25.0], [0.0, 25.0]]
    # large enough but with a constant source
    labels[10:20, 10:20] = 2
    source[10:20, 10:20] = 40.0
    observed[10:20, 10:20] = 20.0

    slopes = um.component_slopes(
        labels, source, observed, usable, 20, default_rng(0), min_source_range=5.0
    )
    assert slopes.size == 0
    empty = um.component_slopes(
        np.zeros(shape, dtype=int), source, observed, usable, 20, default_rng(0),
        min_source_range=5.0,
    )
    assert empty.shape == (0,)


def test_component_slopes_takes_a_plain_seed_or_a_generator():
    shape = (60, 60)
    rng = default_rng(0)
    labels = np.zeros(shape, dtype=int)
    labels[5:45, 5:45] = 1
    source = rng.uniform(0.0, 100.0, shape)
    observed = 0.3 * source + rng.normal(0.0, 1.0, shape)
    usable = np.ones(shape, dtype=bool)
    options = dict(min_source_range=5.0, max_px=200)

    from_seed = um.component_slopes(labels, source, observed, usable, 30, 42, **options)
    again = um.component_slopes(labels, source, observed, usable, 30, 42, **options)
    from_generator = um.component_slopes(
        labels, source, observed, usable, 30, default_rng(42), **options
    )
    assert from_seed == pytest.approx(again)
    assert from_seed == pytest.approx(from_generator)
    assert from_seed[0] == pytest.approx(0.3, abs=0.02)

    # The default seed makes an unseeded call reproducible too.
    assert um.component_slopes(
        labels, source, observed, usable, 30, **options
    ) == pytest.approx(
        um.component_slopes(labels, source, observed, usable, 30, 0, **options)
    )


def test_component_slopes_subsampling_is_reproducible():
    shape = (60, 60)
    rng = default_rng(0)
    labels = np.zeros(shape, dtype=int)
    labels[5:45, 5:45] = 1
    source = rng.uniform(0.0, 100.0, shape)
    observed = 0.3 * source + rng.normal(0.0, 1.0, shape)
    usable = np.ones(shape, dtype=bool)

    first = um.component_slopes(
        labels, source, observed, usable, 30, default_rng(42),
        min_source_range=5.0, max_px=200,
    )
    second = um.component_slopes(
        labels, source, observed, usable, 30, default_rng(42),
        min_source_range=5.0, max_px=200,
    )
    assert first.size == 1
    assert first == pytest.approx(second)
    assert first[0] == pytest.approx(0.3, abs=0.02)


def test_estimate_crosstalk_recovers_the_three_coefficients():
    field = _synthetic_field()
    out = _estimate(field)
    assert out["gamma"] == pytest.approx(GAMMA, abs=0.01)
    assert out["alpha"] == pytest.approx(ALPHA, abs=0.02)
    assert out["beta"] == pytest.approx(BETA, abs=0.03)
    assert out["n_nuclear_structures"] == 27
    assert out["n_marker_structures"] == 27
    assert out["n_beta_pixels"] > 200
    assert out["beta_select_level"] == pytest.approx(90.0)
    assert len(out["gamma_slope_quartiles"]) == 2
    assert len(out["alpha_slope_quartiles"]) == 2


def test_estimate_crosstalk_is_reproducible_and_switchable():
    field = _synthetic_field()
    first = _estimate(field, seed=5)
    second = _estimate(field, seed=5)
    assert first == second

    without_nuclear = _estimate(field, estimate_nuclear=False)
    assert without_nuclear["gamma"] == 0.0

    off = _estimate(field, enabled=False)
    assert (off["alpha"], off["beta"], off["gamma"]) == (0.0, 0.0, 0.0)
    assert off["n_beta_pixels"] == 0


def test_estimate_crosstalk_returns_zeros_without_crosstalk_or_structures():
    clean = _synthetic_field(alpha=0.0, beta=0.0, gamma=0.0)
    out = _estimate(clean)
    assert abs(out["alpha"]) < 0.02
    assert abs(out["beta"]) < 0.02
    assert abs(out["gamma"]) < 0.02

    field = _synthetic_field()
    bare = _estimate(
        field,
        nuclei=np.zeros_like(field["nuclei"]),
        marker_thr=10 ** 6,
    )
    assert bare["gamma"] == 0.0 and bare["alpha"] == 0.0
    assert bare["n_nuclear_structures"] == 0 and bare["n_marker_structures"] == 0
    assert bare["gamma_slope_quartiles"] is None


def test_estimate_crosstalk_handles_a_two_channel_field_without_placeholders():
    field = _synthetic_field()
    options = dict(
        usable=field["usable"], marker_thr=20.0, beta_min_level=90.0,
        min_source_range=8.0, max_px=100, seed=3,
    )
    two_channel = um.estimate_crosstalk(
        field["readout_res"], field["marker_res"], **options
    )
    with_placeholders = um.estimate_crosstalk(
        field["readout_res"], field["marker_res"], np.zeros_like(field["readout_res"]),
        tissue=field["tissue"], nuclei=np.zeros_like(field["nuclei"]),
        estimate_nuclear=False, **options,
    )
    assert two_channel == with_placeholders
    assert two_channel["gamma"] == 0.0
    assert two_channel["n_nuclear_structures"] == 0
    assert two_channel["gamma_slope_quartiles"] is None
    assert two_channel["alpha"] == pytest.approx(ALPHA, abs=0.02)
    assert two_channel["beta"] == pytest.approx(BETA, abs=0.03)

    # tissue=None is the whole field.
    assert um.estimate_crosstalk(
        field["readout_res"], field["marker_res"], field["nuclear_corr"],
        nuclei=field["nuclei"], **options
    ) == _estimate(field, seed=3)


def test_estimate_crosstalk_requires_the_nuclear_channel_and_mask_together():
    field = _synthetic_field()
    options = dict(
        usable=field["usable"], marker_thr=20.0, beta_min_level=90.0,
        min_source_range=8.0, max_px=100,
    )
    with pytest.raises(ValueError):
        um.estimate_crosstalk(
            field["readout_res"], field["marker_res"], field["nuclear_corr"], **options
        )
    with pytest.raises(ValueError):
        um.estimate_crosstalk(
            field["readout_res"], field["marker_res"], nuclei=field["nuclei"], **options
        )


def test_unmix_two_channel_accepts_no_nuclear_channel():
    shape = (16, 16)
    rng = default_rng(1)
    readout_true = rng.uniform(0.0, 100.0, shape)
    marker_true = rng.uniform(0.0, 100.0, shape)
    readout_obs = readout_true + ALPHA * marker_true
    marker_obs = marker_true + BETA * readout_true

    readout_u, marker_u = um.unmix_two_channel(readout_obs, marker_obs, None, ALPHA, BETA)
    assert np.allclose(readout_u, readout_true)
    assert np.allclose(marker_u, marker_true)
    with_zeros = um.unmix_two_channel(
        readout_obs, marker_obs, np.zeros(shape), ALPHA, BETA
    )
    assert np.allclose(readout_u, with_zeros[0]) and np.allclose(marker_u, with_zeros[1])

    # A nuclear coefficient with no nuclear channel to apply it to is an error, not a
    # silently dropped term.
    with pytest.raises(ValueError):
        um.unmix_two_channel(readout_obs, marker_obs, None, ALPHA, BETA, GAMMA)


def test_estimate_crosstalk_accepts_a_caller_supplied_noise_estimator():
    field = _synthetic_field()
    calls = []

    def estimator(values):
        calls.append(values.size)
        return float(np.median(values)), float(values.std())

    out = _estimate(field, noise_estimator=estimator)
    assert calls and out["beta"] == pytest.approx(BETA, abs=0.03)


def test_unmix_two_channel_inverts_the_mixing_exactly():
    shape = (16, 16)
    rng = default_rng(0)
    readout_true = rng.uniform(0.0, 100.0, shape)
    marker_true = rng.uniform(0.0, 100.0, shape)
    nuclear = rng.uniform(0.0, 100.0, shape)
    readout_obs = readout_true + ALPHA * marker_true + GAMMA * nuclear
    marker_obs = marker_true + BETA * readout_true

    readout_u, marker_u = um.unmix_two_channel(
        readout_obs, marker_obs, nuclear, ALPHA, BETA, GAMMA
    )
    assert np.allclose(readout_u, readout_true)
    assert np.allclose(marker_u, marker_true)

    same_readout, same_marker = um.unmix_two_channel(
        readout_obs, marker_obs, nuclear, 0.0, 0.0, 0.0
    )
    assert np.allclose(same_readout, readout_obs)
    assert np.allclose(same_marker, marker_obs)

    with pytest.raises(ValueError):
        um.unmix_two_channel(readout_obs, marker_obs, nuclear, 1.0, 1.0)


def test_estimated_coefficients_remove_the_crosstalk_from_the_field():
    field = _synthetic_field()
    out = _estimate(field)
    readout_u, marker_u = um.unmix_two_channel(
        field["readout_res"], field["marker_res"], field["nuclear_corr"],
        out["alpha"], out["beta"], out["gamma"],
    )
    on_marker = field["marker_mask"]
    before = float(field["readout_res"][on_marker].mean())
    after = float(readout_u[on_marker].mean())
    assert before > 10.0
    assert abs(after) < 3.0

    on_readout = field["readout_mask"]
    assert float(field["marker_res"][on_readout].mean()) > 8.0
    assert abs(float(marker_u[on_readout].mean())) < 3.0


# ------------------------------------------------- residual cross-talk object filters

def _shadow_case():
    shape = (40, 40)
    marker = np.zeros(shape, dtype=bool)
    marker[20, 10:20] = True          # ten pixels, eight of them covered below
    marker[5, 30:35] = True           # far from any object
    objects = np.zeros(shape, dtype=bool)
    objects[21, 11:19] = True
    labels, _ = label_mask(objects, 8)
    return marker, labels


def test_remove_marker_shadows_uses_an_inclusive_coverage_rule():
    marker, labels = _shadow_case()
    removed, n_removed = um.remove_marker_shadows(marker, labels, 1.0, cover=0.80)
    assert n_removed == 1
    assert not removed[20, 10:20].any()
    assert removed[5, 30:35].all()

    kept, n_kept = um.remove_marker_shadows(marker, labels, 1.0, cover=0.81)
    assert n_kept == 0
    assert kept[20, 10:20].all()


def test_remove_marker_shadows_can_be_disabled_and_handles_empty_input():
    marker, labels = _shadow_case()
    same, n = um.remove_marker_shadows(marker, labels, 1.0, cover=1.0)
    assert n == 0 and np.array_equal(same, marker)

    no_objects, n_none = um.remove_marker_shadows(
        marker, np.zeros_like(labels), 1.0
    )
    assert n_none == 0 and np.array_equal(no_objects, marker)

    empty_marker, n_empty = um.remove_marker_shadows(
        np.zeros_like(marker), labels, 1.0
    )
    assert n_empty == 0 and not empty_marker.any()


def test_footprint_beta_correction_is_off_by_default():
    shape = (30, 30)
    marker_res = np.zeros(shape)
    readout = np.zeros(shape)
    marker_u = np.full(shape, 7.0)
    labels, _ = label_mask(_flat_disk(shape, (15, 15), 4, 1.0) > 0, 8)
    out, beta_used = um.footprint_beta_correction(
        marker_res, readout, marker_u, labels, 1.0,
        usable=np.ones(shape, bool), readout_thr=10.0, marker_thr=5.0,
        alpha=0.1, beta=0.2, min_level=20.0,
    )
    assert beta_used == 0.2
    assert np.array_equal(out, marker_u)


def test_footprint_beta_correction_raises_the_coefficient_inside_footprints():
    shape = (80, 80)
    readout = _flat_disk(shape, (40, 40), 8, 200.0)
    marker_res = 0.30 * readout
    marker_u = np.zeros(shape)
    marker_u[3:8, 3:8] = 100.0            # marker territory far from the object
    labels, _ = label_mask(readout > 0, 8)

    out, beta_used = um.footprint_beta_correction(
        marker_res, readout, marker_u, labels, 1.0,
        usable=np.ones(shape, bool), readout_thr=50.0, marker_thr=20.0,
        alpha=0.10, beta=0.10, min_level=20.0, quantile=0.90,
    )
    assert beta_used == pytest.approx(0.30)
    inside = _flat_disk(shape, (40, 40), 6, 1.0) > 0
    assert np.allclose(out[inside], 0.0)
    assert out[5, 5] == pytest.approx(100.0)


def test_surround_ring_excludes_neighbours_and_falls_back_when_too_small():
    shape = (30, 30)
    obj = _flat_disk(shape, (15, 15), 2, 1.0) > 0
    neighbour = _flat_disk(shape, (15, 21), 2, 1.0) > 0
    ring = um.surround_ring(obj, 1.0, occupied=obj | neighbour, ring_um=(1.0, 3.0))
    assert ring.any()
    assert not (ring & (obj | neighbour)).any()

    # A ring smaller than min_ring_px falls back to the full outer dilation.
    crowded = um.surround_ring(
        obj, 1.0, occupied=np.ones(shape, bool), ring_um=(1.0, 3.0)
    )
    assert crowded.sum() > 0
    assert not (crowded & obj).any()


def test_ring_contrast_measures_each_channel_against_the_surround():
    shape = (30, 30)
    obj = _flat_disk(shape, (15, 15), 2, 1.0) > 0
    own = np.where(obj, 100.0, 10.0)
    flat = np.full(shape, 5.0)
    out = um.ring_contrast(obj, {"own": own, "flat": flat}, 1.0)
    assert out["own"] == pytest.approx(90.0)
    assert out["flat"] == pytest.approx(0.0)


def test_contrast_ratio_and_gate_tie_rules():
    assert um.contrast_ratio(10.0, 5.0, min_contrast=1.0) == pytest.approx(2.0)
    assert um.contrast_ratio(10.0, 1.0, min_contrast=1.0) == float("inf")

    tie = um.spectral_gate(2.0, 2.0, 2.0, min_contrast=1.0)
    assert tie["ratio_nuclear"] == pytest.approx(1.0)
    assert tie["gate_pass"] is True

    strict = um.spectral_gate(2.0, 2.0, 2.0, min_contrast=1.0, gate_ratio=1.5)
    assert strict["gate_nuclear"] is False and strict["gate_pass"] is False

    marker_only = um.spectral_gate(
        10.0, 1.0, 20.0, min_contrast=1.0, marker_gate_ratio=1.0
    )
    assert marker_only["gate_marker"] is False
    assert marker_only["gate_pass"] is False
    assert um.spectral_gate(
        10.0, 1.0, 20.0, min_contrast=1.0, use_marker_gate=False
    )["gate_pass"] is True


def test_object_contrast_table_flags_a_copy_in_another_channel():
    shape = (60, 60)
    own = np.zeros(shape)
    nuclear = np.zeros(shape)
    marker = np.zeros(shape)
    genuine = _flat_disk(shape, (15, 15), 2, 1.0) > 0
    copy = _flat_disk(shape, (15, 45), 2, 1.0) > 0
    border = _flat_disk(shape, (0, 30), 2, 1.0) > 0
    own[genuine] = 100.0
    own[copy] = 50.0
    nuclear[copy] = 50.0
    own[border] = 80.0
    labels, n = label_mask(genuine | copy | border, 8)
    assert n == 3

    table = um.object_contrast_table(
        labels, own, nuclear, marker, 0.5, min_contrast=1.0
    )
    assert list(table.columns)[:3] == ["label", "area_px", "area_um2"]
    assert len(table) == 3
    rows = table.set_index("label")
    genuine_label = int(labels[15, 15])
    copy_label = int(labels[15, 45])
    border_label = int(labels[0, 30])

    assert rows.loc[genuine_label, "contrast_own"] == pytest.approx(100.0)
    assert rows.loc[genuine_label, "ratio_nuclear"] == pytest.approx(999.0)
    assert bool(rows.loc[genuine_label, "gate_pass"])

    assert rows.loc[copy_label, "ratio_nuclear"] == pytest.approx(1.0)
    assert bool(rows.loc[copy_label, "gate_pass"])

    assert rows.loc[border_label, "n_ring_px"] > 0
    assert rows.loc[border_label, "area_um2"] == pytest.approx(
        rows.loc[border_label, "area_px"] * 0.25
    )

    strict = um.object_contrast_table(
        labels, own, nuclear, marker, 0.5, min_contrast=1.0, gate_ratio=1.5
    ).set_index("label")
    assert not bool(strict.loc[copy_label, "gate_pass"])
    assert bool(strict.loc[genuine_label, "gate_pass"])


def test_object_contrast_table_on_an_empty_label_image():
    shape = (10, 10)
    zeros = np.zeros(shape)
    table = um.object_contrast_table(
        np.zeros(shape, dtype=int), zeros, zeros, zeros, 1.0, min_contrast=1.0
    )
    assert len(table) == 0
    assert "gate_pass" in table.columns


# ------------------------------------------------------------- robustness switch

def test_unmixing_variants_are_fresh_mappings():
    first = um.unmixing_variants()
    first["without_crosstalk_correction"]["unmix"] = True
    assert um.unmixing_variants()["without_crosstalk_correction"]["unmix"] is False
    assert um.unmixing_off_overrides()["shadow_cover"] == 1.0


def test_run_variants_applies_overrides_without_touching_the_base():
    base = {"unmix": True, "shadow_cover": 0.80, "gate_ratio": 1.0}
    seen = []

    def analyse(params):
        seen.append(dict(params))
        return {"enrichment": 2.0 if params["unmix"] else 1.5}

    table = um.run_variants(analyse, base, {"off": um.unmixing_off_overrides()})
    assert list(table["setting"]) == ["baseline", "off"]
    assert list(table["enrichment"]) == [2.0, 1.5]
    assert base == {"unmix": True, "shadow_cover": 0.80, "gate_ratio": 1.0}
    assert seen[1]["shadow_cover"] == 1.0
    assert seen[0]["shadow_cover"] == 0.80

    without_baseline = um.run_variants(
        analyse, base, {"off": um.unmixing_off_overrides()}, include_baseline=False
    )
    assert list(without_baseline["setting"]) == ["off"]


def test_compare_unmixing_gives_one_row_per_setting():
    base = {"unmix": True, "shadow_cover": 0.80, "use_marker_gate": True}

    def analyse(params):
        return {"n": 10 if params["unmix"] else 12}

    table = um.compare_unmixing(analyse, base)
    assert list(table["setting"]) == ["with_unmixing", "without_unmixing"]
    assert list(table["n"]) == [10, 12]


def test_compare_unmixing_rejects_overrides_the_base_does_not_hold():
    def analyse(params):
        return {"n": 10 if params["unmix"] else 12}

    # A misspelled key would override nothing, so both rows would be identical.
    misspelled = {"unmix": True, "shadow_covr": 0.80, "use_marker_gate": True}
    with pytest.raises(KeyError) as raised:
        um.compare_unmixing(analyse, misspelled)
    assert "shadow_cover" in str(raised.value)

    # The check can be switched off, and then the old behaviour is unchanged.
    relaxed = um.compare_unmixing(analyse, misspelled, validate_overrides=False)
    assert list(relaxed["n"]) == [10, 12]

    # An analysis with only one of the three steps names its own overrides.
    own = um.compare_unmixing(
        analyse, {"unmix": True}, off_overrides={"unmix": False}
    )
    assert list(own["n"]) == [10, 12]
    with pytest.raises(KeyError):
        um.compare_unmixing(analyse, {"unmix": True})


def test_run_variants_defaults_to_the_crosstalk_grid():
    calls = []

    def analyse(params):
        calls.append(params.get("alpha_quantile"))
        return {"ok": True}

    table = um.run_variants(analyse, {"alpha_quantile": 0.5})
    assert len(table) == len(um.unmixing_variants()) + 1
    assert 0.25 in calls and 0.75 in calls


# ------------------------------------------------------------------- nuclei mask

def _disc(shape, cy, cx, radius_px):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= radius_px * radius_px


def test_nuclei_mask_keeps_only_components_inside_the_size_range():
    rng = default_rng(5)
    plane = rng.normal(5.0, 2.0, (120, 120))
    planted = {"large": _disc(plane.shape, 70, 70, 6), "medium": _disc(plane.shape, 30, 30, 3),
               "speck": _disc(plane.shape, 100, 20, 1)}
    for mask in planted.values():
        plane[mask] += 120.0
    pixel_size_um = 0.5                                  # 0.25 um2 per pixel

    gated = um.nuclei_mask(plane, pixel_size_um, area_um2=(2.0, 40.0))
    labels, n = label_mask(gated, connectivity=8)
    assert n == 2                                        # the speck is below the range
    areas = np.bincount(labels.ravel())[1:] * pixel_size_um ** 2
    assert sorted(np.round(areas, 2)) == [6.25, 28.25]
    assert gated[70, 70] and gated[30, 30] and not gated[100, 20]

    # With the gate wide open all three survive.
    _, n_open = label_mask(um.nuclei_mask(plane, pixel_size_um, area_um2=(0.0, 1e6)), connectivity=8)
    assert n_open == 3

    # An upper end below the smallest component leaves nothing.
    assert not um.nuclei_mask(plane, pixel_size_um, area_um2=(0.0, 0.5)).any()


def test_nuclei_mask_respects_the_measured_area_and_an_explicit_threshold():
    plane = np.zeros((60, 60))
    plane[_disc(plane.shape, 15, 15, 4)] = 100.0
    plane[_disc(plane.shape, 45, 45, 4)] = 100.0
    tissue = np.zeros(plane.shape, bool)
    tissue[:30, :30] = True

    inside = um.nuclei_mask(plane, 1.0, area_um2=(5.0, 200.0), tissue=tissue, threshold=50.0)
    assert inside[15, 15] and not inside[45, 45]
    assert inside.sum() == _disc(plane.shape, 15, 15, 4).sum()

    both = um.nuclei_mask(plane, 1.0, area_um2=(5.0, 200.0), threshold=50.0)
    assert both[15, 15] and both[45, 45]


def test_nuclei_mask_refuses_an_empty_area_and_a_reversed_range():
    plane = np.zeros((20, 20))
    with pytest.raises(ValueError, match="no pixel"):
        um.nuclei_mask(plane, 1.0, area_um2=(1.0, 2.0), tissue=np.zeros(plane.shape, bool))
    with pytest.raises(ValueError, match="smallest"):
        um.nuclei_mask(plane, 1.0, area_um2=(5.0, 1.0))


def test_unmixing_switches_use_the_contact_parameter_names():
    """Every override must name a parameter the contact analysis reads.

    An override on a name the analysis does not know is either rejected as unknown or,
    worse, ignored, so the variant would rerun the baseline under another label.
    """
    from if_qsaa import contact

    known = set(contact.default_contact_parameters()) | set(contact.required_contact_parameters())
    assert set(um.unmixing_off_overrides()) <= known
    for name, override in um.unmixing_variants().items():
        assert set(override) <= known, name
