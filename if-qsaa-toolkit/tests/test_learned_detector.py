"""Tests for the learned detector, on synthetic images with known ground truth."""

from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest


from if_qsaa import learned_detector as ld
from if_qsaa._util import default_rng, disk


FIELD_SHAPE = (200, 200)
BRIGHT_SPOTS = 10
DIM_SPOTS = 55


# --------------------------------------------------------------------------- helpers

def _gaussian_spot(shape, centre_y, centre_x, amplitude, sigma_px):
    """Add-ready Gaussian spot of the given amplitude in grey levels."""
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    squared = (yy - centre_y) ** 2 + (xx - centre_x) ** 2
    return amplitude * np.exp(-squared / (2.0 * sigma_px**2))


def _synthetic_field(seed, shape=FIELD_SHAPE, n_bright=BRIGHT_SPOTS, n_dim=DIM_SPOTS):
    """Synthetic field: a few bright compact objects among many dim distractors.

    Returns the image in grey levels and the centres of the bright objects, which are the
    ground truth of the detection tests.
    """
    rng = default_rng(seed)
    image = rng.normal(500.0, 20.0, shape)
    cells = [(row, col) for row in range(1, 10) for col in range(1, 10)]
    picks = rng.permutation(len(cells))[: n_bright + n_dim]
    centres = []
    for order, index in enumerate(picks):
        row, col = cells[index]
        centre_y = row * 20 + int(rng.integers(-3, 4))
        centre_x = col * 20 + int(rng.integers(-3, 4))
        if order < n_bright:
            image += _gaussian_spot(shape, centre_y, centre_x, 2600.0, 1.5)
            centres.append((centre_y, centre_x))
        else:
            image += _gaussian_spot(shape, centre_y, centre_x, 330.0, 2.0)
    return image.astype(np.float32), centres


def _labels_from_truth(candidate_set, centres, tolerance_px=2.5):
    """Label a candidate 1 when its seed sits on one of the known object centres."""
    labels = np.zeros(len(candidate_set.candidates), dtype=int)
    for index, candidate in enumerate(candidate_set.candidates):
        for centre_y, centre_x in centres:
            distance = (candidate.seed_y - centre_y) ** 2 + (candidate.seed_x - centre_x) ** 2
            if distance <= tolerance_px**2:
                labels[index] = 1
                break
    return labels


def _fit_on_fields(seeds=(11, 12, 13)):
    """Fit a model on synthetic fields and pick its threshold at the maximum F1 score.

    Fitting and scoring are given the same tables: the raw features of each field, which
    both of them normalise per field.
    """
    tables, labels = [], []
    for seed in seeds:
        image, centres = _synthetic_field(seed)
        candidate_set = ld.detect_candidates(image)
        tables.append(candidate_set.features)
        labels.append(_labels_from_truth(candidate_set, centres))
    target = np.concatenate(labels)
    model = ld.fit_learned_model(tables, target, l2_penalty=1.0)
    scores = np.concatenate(
        [ld.score_candidates(table, model, has_reference=False) for table in tables]
    )
    threshold = ld.choose_threshold_max_f1(scores, target)
    return model.with_threshold(threshold), tables, target, scores


def _simple_model(coef, intercept=0.0, threshold=0.0):
    """Model with a unit standardisation, for testing the scoring path itself."""
    return ld.LearnedModel(
        feature_names=ld.FEATURE_NAMES,
        log_features=ld.LOG_FEATURES,
        clip=12.0,
        mean=np.zeros(len(ld.FEATURE_NAMES)),
        scale=np.ones(len(ld.FEATURE_NAMES)),
        coef=np.asarray(coef, dtype=float),
        intercept=intercept,
        threshold=threshold,
    )


def _candidate(ys, xs, seed_y=None, seed_x=None):
    ys = np.asarray(ys, dtype=np.int64)
    xs = np.asarray(xs, dtype=np.int64)
    return ld.Candidate(
        ys=ys,
        xs=xs,
        seed_y=int(ys[0] if seed_y is None else seed_y),
        seed_x=int(xs[0] if seed_x is None else seed_x),
    )


# --------------------------------------------------------------------------- input planes

def test_to_float_plane_scales_integer_and_float_inputs():
    assert ld.to_float_plane(np.full((64, 64), 200, np.uint8)).max() == pytest.approx(
        200 * 65535 / 255
    )
    assert ld.to_float_plane(np.full((64, 64), 7000, np.uint16)).max() == pytest.approx(7000.0)
    # A float image with a maximum of at most 1.5 is read as normalised.
    assert ld.to_float_plane(np.full((64, 64), 0.5, np.float32)).max() == pytest.approx(
        0.5 * 65535
    )
    assert ld.to_float_plane(np.full((64, 64), 7000.0)).max() == pytest.approx(7000.0)


def test_to_float_plane_reduces_channels_and_replaces_nonfinite():
    stack = np.zeros((64, 64, 4), np.uint16)
    stack[..., 0] = 10
    stack[..., 1] = 70
    stack[..., 2] = 30
    stack[..., 3] = 65535  # a fourth channel is ignored
    assert ld.to_float_plane(stack).max() == pytest.approx(70.0)

    broken = np.full((64, 64), np.nan)
    broken[0, 0] = np.inf
    broken[0, 1] = -np.inf
    plane = ld.to_float_plane(broken)
    assert plane[0, 0] == pytest.approx(65535.0)
    assert plane[0, 1] == pytest.approx(0.0)
    assert np.isfinite(plane).all()


def test_to_float_plane_rejects_small_image():
    with pytest.raises(ValueError):
        ld.to_float_plane(np.zeros((32, 128)))


def test_subtract_reference_removes_reference_above_its_offset():
    raw = np.full((8, 8), 500.0, np.float32)
    reference = np.full((8, 8), 100.0, np.float32)
    reference[3, 3] = 400.0
    corrected = ld.subtract_reference(raw, reference, offset_percentile=10.0)
    # The reference offset (its 10th percentile) is 100, so only the excess is removed.
    assert corrected[0, 0] == pytest.approx(500.0)
    assert corrected[3, 3] == pytest.approx(200.0)
    # Half the weight removes half the excess, and the result never goes below zero.
    half = ld.subtract_reference(raw, reference, coefficient=0.5)
    assert half[3, 3] == pytest.approx(350.0)
    assert ld.subtract_reference(raw, 10.0 * reference)[3, 3] == pytest.approx(0.0)


def test_subtract_reference_without_reference_returns_input():
    raw = np.arange(64, dtype=np.float32).reshape(8, 8)
    assert np.array_equal(ld.subtract_reference(raw, None), raw)


# --------------------------------------------------------------- candidate generation

def test_dog_response_peaks_on_objects_and_respects_noise_floor():
    image = np.full((128, 128), 500.0, np.float32)
    image += _gaussian_spot(image.shape, 40, 60, 2600.0, 1.5)
    response, noise = ld.dog_response(image, noise_floor=32.0)
    # A noiseless image would give a zero noise estimate; the floor takes over.
    assert noise == pytest.approx(32.0)
    peak = np.unravel_index(int(np.argmax(response)), response.shape)
    assert peak == (40, 60)
    assert response[40, 60] > 3.5
    assert abs(response[5, 5]) < 1e-3


def test_dog_response_without_a_floor_returns_no_contrast_on_a_flat_plane():
    # No floor is the default: a plane whose own residual vanishes carries no contrast,
    # so the response is zero everywhere instead of a division by zero.
    image = np.full((64, 64), 500.0, np.float32)
    response, noise = ld.dog_response(image)
    assert noise == 0.0
    assert not response.any()


def test_dog_response_is_scale_free_without_a_floor():
    # Without a floor the response is in units of the plane's own noise, so doubling the
    # grey levels of an image leaves it unchanged; a floor in grey levels does not.
    rng = default_rng(11)
    image = rng.normal(500.0, 50.0, (96, 96)).astype(np.float32)
    image += _gaussian_spot(image.shape, 30, 40, 900.0, 1.6)
    plain, noise_one = ld.dog_response(image)
    doubled, noise_two = ld.dog_response(2.0 * image)
    assert noise_two == pytest.approx(2.0 * noise_one, rel=1e-5)
    assert np.allclose(plain, doubled, atol=1e-4)


def test_dog_response_noise_tracks_the_image_noise():
    rng = default_rng(3)
    image = rng.normal(500.0, 200.0, (160, 160)).astype(np.float32)
    _, noise = ld.dog_response(image, noise_floor=1.0)
    # White noise of this width leaves a residual well above the floor.
    assert 100.0 < noise < 300.0


def test_propose_seeds_finds_isolated_maxima():
    response = np.zeros((40, 40), np.float32)
    response[10, 12] = 6.0
    response[30, 7] = 4.0
    response[20, 20] = 2.0  # below the seed threshold
    seeds = ld.propose_seeds(response, seed_threshold=3.5)
    assert {tuple(seed) for seed in seeds} == {(10, 12), (30, 7)}


def test_propose_seeds_keeps_one_pixel_per_plateau():
    response = np.zeros((30, 30), np.float32)
    response[5:7, 5:7] = 8.0  # a flat plateau of four equal pixels
    response[20, 25] = 5.0
    seeds = ld.propose_seeds(response, seed_threshold=3.5)
    assert len(seeds) == 2
    # The plateau keeps its smallest row and then its smallest column.
    assert tuple(seeds[0]) == (5, 5)
    assert tuple(seeds[1]) == (20, 25)


def test_propose_seeds_empty_without_any_maximum():
    seeds = ld.propose_seeds(np.zeros((20, 20), np.float32), seed_threshold=3.5)
    assert seeds.shape == (0, 2)
    assert seeds.dtype.kind == "i"


def test_grow_candidates_recovers_a_known_support():
    response = np.zeros((40, 40), np.float32)
    support = [(20, 20), (20, 21), (21, 20), (21, 21), (19, 20), (22, 21)]
    for y, x in support:
        response[y, x] = 5.0
    response[21, 22] = 5.0  # 8-connected through a corner, so it joins
    response[30, 30] = 5.0  # a separate region, so it does not
    response[20, 20] = 9.0  # the seed
    candidates = ld.grow_candidates(response, np.array([[20, 20]]))
    assert len(candidates) == 1
    found = set(zip(candidates[0].ys.tolist(), candidates[0].xs.tolist()))
    assert found == set(support) | {(21, 22)}
    assert candidates[0].area_px == len(support) + 1


def test_grow_candidates_threshold_follows_the_peak():
    response = np.zeros((40, 40), np.float32)
    response[10, 10] = 40.0  # peak, so the growth threshold is 0.18 * 40 = 7.2
    response[10, 11] = 8.0
    response[10, 12] = 6.0
    candidates = ld.grow_candidates(response, np.array([[10, 10]]), low_fraction=0.18)
    pixels = set(zip(candidates[0].ys.tolist(), candidates[0].xs.tolist()))
    assert pixels == {(10, 10), (10, 11)}


def test_grow_candidates_clips_at_the_image_border():
    response = np.zeros((40, 40), np.float32)
    response[0, 0:3] = 5.0
    response[0, 0] = 9.0
    candidates = ld.grow_candidates(response, np.array([[0, 0]]))
    assert candidates[0].ys.min() >= 0
    assert candidates[0].xs.min() >= 0
    assert set(zip(candidates[0].ys.tolist(), candidates[0].xs.tolist())) == {
        (0, 0),
        (0, 1),
        (0, 2),
    }


def test_grow_candidates_without_seeds_returns_empty_list():
    assert ld.grow_candidates(np.zeros((20, 20), np.float32), np.zeros((0, 2))) == []


# --------------------------------------------------------------------------- features

def test_local_noise_map_recovers_two_noise_levels():
    rng = default_rng(5)
    image = np.full((200, 200), 500.0)
    image[:, :100] += rng.normal(0.0, 5.0, (200, 100))
    image[:, 100:] += rng.normal(0.0, 20.0, (200, 100))
    noise = ld.local_noise_map(image.astype(np.float32), tile_px=50)
    quiet = float(np.median(noise[:, 10:80]))
    loud = float(np.median(noise[:, 120:190]))
    assert quiet == pytest.approx(5.0, rel=0.2)
    assert loud == pytest.approx(20.0, rel=0.2)


def test_local_noise_map_is_strictly_positive_on_a_constant_image():
    noise = ld.local_noise_map(np.full((120, 120), 700.0, np.float32), tile_px=50)
    assert np.all(noise > 0)
    assert noise.max() == pytest.approx(1e-3)


def test_candidate_features_match_a_hand_computation():
    corrected = np.full((60, 60), 100.0, np.float32)
    corrected[20:23, 20:23] = 500.0
    response = np.zeros((60, 60), np.float32)
    response[20:23, 20:23] = 7.0
    response[21, 21] = 9.0
    noise = np.full((60, 60), 10.0, np.float32)
    ys, xs = np.nonzero(corrected > 100.0)
    features = ld.candidate_features([_candidate(ys, xs, 21, 21)], corrected, response, noise)
    row = features.iloc[0]
    # (mean inside - median of the ring) / mean local noise, with a uniform ring at 100.
    assert row["ringnear"] == pytest.approx(40.0, rel=1e-4)
    assert row["zmean"] == pytest.approx((8 * 7.0 + 9.0) / 9.0, rel=1e-5)
    assert row["zmax"] == pytest.approx(9.0)
    assert row["noiserel"] == pytest.approx(1.0)
    assert row["reftone"] == pytest.approx(0.0)
    assert row["aspect"] == pytest.approx(1.0, rel=1e-5)


def test_candidate_features_zero_ring_contrast_when_the_ring_is_too_small():
    corrected = np.full((60, 60), 100.0, np.float32)
    corrected[20:23, 20:23] = 500.0
    response = np.zeros((60, 60), np.float32)
    response[20:23, 20:23] = 7.0
    noise = np.full((60, 60), 10.0, np.float32)
    ys, xs = np.nonzero(corrected > 100.0)
    features = ld.candidate_features(
        [_candidate(ys, xs, 21, 21)], corrected, response, noise, min_ring_px=10_000
    )
    assert features.iloc[0]["ringnear"] == pytest.approx(0.0)


def test_candidate_features_aspect_separates_a_line_from_a_blob():
    corrected = np.full((60, 60), 100.0, np.float32)
    response = np.full((60, 60), 5.0, np.float32)
    noise = np.full((60, 60), 10.0, np.float32)
    line = _candidate(np.full(9, 30), np.arange(20, 29))
    blob = _candidate(*np.nonzero(np.pad(disk(2).astype(bool), ((30, 20), (30, 20)))))
    pair = _candidate([10, 10], [10, 11])
    features = ld.candidate_features([line, blob, pair], corrected, response, noise)
    assert features.iloc[0]["aspect"] > 1000.0
    assert features.iloc[1]["aspect"] == pytest.approx(1.0, abs=0.05)
    # Two pixels carry no shape information, so the ratio is one by definition.
    assert features.iloc[2]["aspect"] == pytest.approx(1.0)


def test_candidate_features_empty_candidate_list():
    features = ld.candidate_features(
        [], np.zeros((20, 20)), np.zeros((20, 20)), np.ones((20, 20))
    )
    assert list(features.columns) == list(ld.FEATURE_NAMES)
    assert features.empty


def test_detect_candidates_uses_the_reference_channel():
    image, centres = _synthetic_field(31)
    reference = np.full(FIELD_SHAPE, 300.0, np.float32)
    centre_y, centre_x = centres[0]
    reference += _gaussian_spot(FIELD_SHAPE, centre_y, centre_x, 4000.0, 1.5)

    without = ld.detect_candidates(image)
    with_reference = ld.detect_candidates(image, reference)
    assert with_reference.has_reference and not without.has_reference
    assert (without.features["reftone"] == 0).all()
    assert with_reference.features["reftone"].max() > 1.0
    # The response is built from the target channel, so the candidates do not move.
    assert len(with_reference.candidates) == len(without.candidates)
    # Subtracting the reference lowers the ring contrast where the reference is bright.
    seeds = [(c.seed_y, c.seed_x) for c in with_reference.candidates]
    index = seeds.index(
        min(seeds, key=lambda s: (s[0] - centre_y) ** 2 + (s[1] - centre_x) ** 2)
    )
    assert (
        with_reference.features["ringnear"].iloc[index]
        < without.features["ringnear"].iloc[index]
    )


# --------------------------------------------------------------- normalisation, scoring

def test_normalize_per_image_matches_median_and_lower_half_mad():
    values = np.arange(1.0, 11.0)
    table = pd.DataFrame(
        {name: values if name == "noiserel" else np.zeros(10) for name in ld.FEATURE_NAMES}
    )
    normalized = ld.normalize_per_image(table, log_features=())
    median = 5.5
    scale = 1.4826 * 2.5  # median of (median - lower half) = 2.5
    expected = (values - median) / scale
    assert normalized["noiserel"].to_numpy() == pytest.approx(expected, rel=1e-5)
    # A constant column has a zero scale, which the lower bound turns into a zero score.
    assert normalized["aspect"].to_numpy() == pytest.approx(np.zeros(10))


def test_normalize_per_image_log_transforms_only_the_named_features():
    rng = default_rng(17)
    raw = rng.gamma(2.0, 3.0, (40, len(ld.FEATURE_NAMES)))
    table = pd.DataFrame(raw, columns=list(ld.FEATURE_NAMES))
    normalized = ld.normalize_per_image(table, log_features=("zmax",))

    logged = table.copy()
    logged["zmax"] = np.log1p(np.maximum(logged["zmax"], 0.0))
    expected = ld.normalize_per_image(logged, log_features=())
    assert normalized.to_numpy() == pytest.approx(expected.to_numpy(), rel=1e-4, abs=1e-5)
    assert normalized["zmax"].to_numpy() != pytest.approx(
        ld.normalize_per_image(table, log_features=())["zmax"].to_numpy()
    )


def test_normalize_per_image_clips_and_keeps_a_unit_scale_for_a_short_lower_half():
    table = pd.DataFrame(
        np.zeros((6, len(ld.FEATURE_NAMES))), columns=list(ld.FEATURE_NAMES)
    )
    table["zmean"] = [0.0, 1.0, 2.0, 3.0, 4.0, 500.0]
    normalized = ld.normalize_per_image(table, log_features=(), clip=4.0, min_lower=5)
    # Three values lie at or below the median, fewer than min_lower, so the scale is one.
    assert normalized["zmean"].to_numpy()[:3] == pytest.approx([-2.5, -1.5, -0.5])
    # Everything far above the median is bounded by the clip.
    assert normalized["zmean"].to_numpy()[-1] == pytest.approx(4.0)


def test_normalize_per_image_empty_table():
    empty = pd.DataFrame(
        np.zeros((0, len(ld.FEATURE_NAMES))), columns=list(ld.FEATURE_NAMES)
    )
    normalized = ld.normalize_per_image(empty)
    assert normalized.empty
    assert ld.NORMALIZATION_KEY in normalized.attrs


def test_normalize_per_image_records_the_settings_it_used():
    table = pd.DataFrame(
        np.arange(60.0).reshape(10, len(ld.FEATURE_NAMES)), columns=list(ld.FEATURE_NAMES)
    )
    normalized = ld.normalize_per_image(table, log_features=("zmax",), clip=4.0)
    record = normalized.attrs[ld.NORMALIZATION_KEY]
    assert record["rule"] == ld.NORMALIZATION
    assert record["log_features"] == ("zmax",)
    assert record["clip"] == pytest.approx(4.0)
    assert record["min_lower"] == 5
    # The raw table carries no such record, which is how fitting tells the two apart.
    assert ld.NORMALIZATION_KEY not in table.attrs
    # Concatenating the fields of one fit keeps it, since every field used the same rule.
    joined = pd.concat([normalized, normalized], ignore_index=True)
    assert joined.attrs[ld.NORMALIZATION_KEY] == record


def test_score_candidates_refuses_an_already_normalised_table():
    rng = default_rng(29)
    table = pd.DataFrame(
        rng.gamma(2.0, 3.0, (20, len(ld.FEATURE_NAMES))), columns=list(ld.FEATURE_NAMES)
    )
    model = _simple_model(np.ones(len(ld.FEATURE_NAMES)))
    with pytest.raises(ValueError, match="normalised"):
        ld.score_candidates(ld.normalize_per_image(table), model)
    # The raw table is what it wants, and the normalised one can still be scored by hand.
    assert ld.score_candidates(table, model).shape == (20,)


def test_score_candidates_without_reference_drops_the_reference_feature():
    rng = default_rng(23)
    table = pd.DataFrame(
        rng.normal(size=(30, len(ld.FEATURE_NAMES))), columns=list(ld.FEATURE_NAMES)
    )
    coef = np.array([0.5, 0.25, 0.1, -0.2, 3.0, -0.4])
    model = _simple_model(coef, intercept=0.75)

    with_reference = ld.score_candidates(table, model, has_reference=True)
    without = ld.score_candidates(table, model, has_reference=False)

    normalized = ld.normalize_per_image(table, log_features=model.log_features)
    standardized = ld.standardize_features(normalized, model)
    assert with_reference == pytest.approx(
        ld.logistic_score(standardized, coef, 0.75), rel=1e-5
    )
    blind = coef.copy()
    blind[model.reference_index] = 0.0
    assert without == pytest.approx(ld.logistic_score(standardized, blind, 0.75), rel=1e-5)


def test_logistic_score_rejects_a_wrong_coefficient_count():
    with pytest.raises(ValueError):
        ld.logistic_score(np.zeros((4, 6)), np.zeros(5), 0.0)


# --------------------------------------------------------------------------- masks

def test_paint_candidates_unions_the_selected_supports():
    candidates = [
        _candidate([1, 1], [1, 2]),
        _candidate([5, 5], [5, 6]),
        _candidate([9], [9]),
    ]
    mask, count = ld.paint_candidates((12, 12), candidates, [True, False, True])
    assert count == 2
    assert mask.sum() == 3
    assert mask[1, 1] and mask[1, 2] and mask[9, 9]
    assert not mask[5, 5]


def test_paint_candidates_checks_the_flag_count():
    with pytest.raises(ValueError):
        ld.paint_candidates((5, 5), [_candidate([1], [1])], [True, True])


# --------------------------------------------------------------------------- fitting

def test_fit_learned_model_recovers_a_known_rule():
    rng = default_rng(41)
    table = pd.DataFrame(
        rng.normal(size=(400, len(ld.FEATURE_NAMES))), columns=list(ld.FEATURE_NAMES)
    )
    # The label depends on two features only; the other four carry no information.
    labels = (table["zmean"] + table["zmax"] > 1.0).to_numpy().astype(int)
    # The table is taken as prepared: this checks the standardisation and the fit itself.
    model = ld.fit_learned_model(table, labels, l2_penalty=1.0, already_normalized=True)

    assert model.mean == pytest.approx(table.to_numpy().mean(axis=0))
    assert model.scale == pytest.approx(table.to_numpy().std(axis=0))

    weights = dict(zip(model.feature_names, model.coef))
    assert weights["zmean"] > 1.0 and weights["zmax"] > 1.0
    for name in ("ringnear", "noiserel", "reftone", "aspect"):
        assert abs(weights[name]) < min(weights["zmean"], weights["zmax"]) / 2.0

    scores = ld.logistic_score(
        ld.standardize_features(table, model), model.coef, model.intercept
    )
    accuracy = np.mean((scores >= 0.0) == (labels == 1))
    assert accuracy > 0.95


def test_fit_learned_model_l2_penalty_shrinks_the_coefficients():
    rng = default_rng(43)
    table = pd.DataFrame(
        rng.normal(size=(200, len(ld.FEATURE_NAMES))), columns=list(ld.FEATURE_NAMES)
    )
    labels = (table["zmax"] > 0.0).to_numpy().astype(int)
    loose = ld.fit_learned_model(table, labels, l2_penalty=0.1, already_normalized=True)
    tight = ld.fit_learned_model(table, labels, l2_penalty=100.0, already_normalized=True)
    assert np.linalg.norm(tight.coef) < np.linalg.norm(loose.coef)


def test_fit_learned_model_honours_sample_weights():
    table = pd.DataFrame(
        np.zeros((20, len(ld.FEATURE_NAMES))), columns=list(ld.FEATURE_NAMES)
    )
    table["zmax"] = np.concatenate([np.full(10, -1.0), np.full(10, 1.0)])
    labels = np.concatenate([np.zeros(10, int), np.ones(10, int)])
    weights = np.concatenate([np.full(10, 1.0), np.full(10, 20.0)])
    balanced = ld.fit_learned_model(table, labels, l2_penalty=1.0, already_normalized=True)
    weighted = ld.fit_learned_model(
        table, labels, l2_penalty=1.0, sample_weight=weights, already_normalized=True
    )
    # Weighting the positives up moves the decision towards accepting them.
    assert weighted.intercept > balanced.intercept


def test_fit_learned_model_rejects_bad_input():
    table = pd.DataFrame(
        np.zeros((4, len(ld.FEATURE_NAMES))), columns=list(ld.FEATURE_NAMES)
    )
    with pytest.raises(ValueError):
        ld.fit_learned_model(table, np.zeros(3, int))
    with pytest.raises(ValueError):
        ld.fit_learned_model(table, np.zeros(4, int))  # one class only
    with pytest.raises(ValueError):
        ld.fit_learned_model(table, np.array([0, 1, 2, 1]))  # not binary
    with pytest.raises(ValueError):
        ld.fit_learned_model(table.drop(columns=["zmax"]), np.array([0, 1, 0, 1]))
    broken = table.copy()
    broken.loc[0, "zmean"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        ld.fit_learned_model(broken, np.array([0, 1, 0, 1]))


def _two_raw_fields(seed=71, sizes=(40, 30)):
    """Two raw feature tables with different scales, one per image."""
    rng = default_rng(seed)
    tables = []
    for position, count in enumerate(sizes):
        raw = rng.gamma(2.0, 3.0 + 4.0 * position, (count, len(ld.FEATURE_NAMES)))
        tables.append(pd.DataFrame(raw, columns=list(ld.FEATURE_NAMES)))
    return tables


def test_fit_learned_model_normalises_every_raw_table_per_image():
    tables = _two_raw_fields()
    labels = np.concatenate(
        [(table["zmean"] > table["zmean"].median()).to_numpy().astype(int) for table in tables]
    )
    fitted = ld.fit_learned_model(tables, labels, l2_penalty=1.0)

    # The same model comes out of the hand-prepared route: normalise each image, then
    # concatenate in the same order.
    by_hand = pd.concat(
        [ld.normalize_per_image(table) for table in tables], ignore_index=True
    )
    expected = ld.fit_learned_model(
        by_hand, labels, l2_penalty=1.0, already_normalized=True
    )
    assert fitted.mean == pytest.approx(expected.mean)
    assert fitted.scale == pytest.approx(expected.scale)
    assert fitted.coef == pytest.approx(expected.coef)
    assert fitted.intercept == pytest.approx(expected.intercept)
    # A single raw table is one image.
    single = ld.fit_learned_model(tables[0], labels[: len(tables[0])], l2_penalty=1.0)
    assert single.coef.size == len(ld.FEATURE_NAMES)


def test_fit_learned_model_sees_what_score_candidates_sees():
    tables = _two_raw_fields()
    labels = np.concatenate(
        [(table["zmax"] > table["zmax"].median()).to_numpy().astype(int) for table in tables]
    )
    model = ld.fit_learned_model(tables, labels, l2_penalty=1.0)
    # Scoring a raw table reproduces the standardised features the fit was given.
    for table in tables:
        standardized = ld.standardize_features(
            ld.normalize_per_image(table, log_features=model.log_features, clip=model.clip),
            model,
        )
        assert ld.score_candidates(table, model) == pytest.approx(
            ld.logistic_score(standardized, model.coef, model.intercept), rel=1e-5
        )


def test_fit_learned_model_refuses_a_normalised_table_on_the_raw_route():
    table = _two_raw_fields(sizes=(40,))[0]
    labels = (table["zmean"] > table["zmean"].median()).to_numpy().astype(int)
    normalized = ld.normalize_per_image(table)
    with pytest.raises(ValueError, match="already_normalized"):
        ld.fit_learned_model(normalized, labels, l2_penalty=1.0)


def test_fit_learned_model_refuses_a_normalisation_it_would_not_record():
    table = _two_raw_fields(sizes=(40,))[0]
    labels = (table["zmean"] > table["zmean"].median()).to_numpy().astype(int)
    # Normalised with another clip than the model would carry.
    with pytest.raises(ValueError, match="clip"):
        ld.fit_learned_model(
            ld.normalize_per_image(table, clip=4.0),
            labels,
            already_normalized=True,
            clip=12.0,
        )
    # Normalised with another set of log-transformed features.
    with pytest.raises(ValueError, match="log_features"):
        ld.fit_learned_model(
            ld.normalize_per_image(table, log_features=()),
            labels,
            already_normalized=True,
        )
    # The settings that were used are accepted.
    model = ld.fit_learned_model(
        ld.normalize_per_image(table, clip=4.0, log_features=("zmax",)),
        labels,
        already_normalized=True,
        clip=4.0,
        log_features=("zmax",),
    )
    assert model.clip == pytest.approx(4.0)
    assert model.log_features == ("zmax",)


def test_fit_learned_model_refuses_raw_features_on_the_normalised_route():
    table = _two_raw_fields(sizes=(40,))[0]
    table.loc[0, "zmax"] = 400.0  # a raw response, far outside any clipped score
    labels = (table["zmean"] > table["zmean"].median()).to_numpy().astype(int)
    with pytest.raises(ValueError, match="outside"):
        ld.fit_learned_model(table, labels, already_normalized=True)


def test_fit_learned_model_label_count_names_the_table_sizes():
    tables = _two_raw_fields()
    with pytest.raises(ValueError, match="40, 30"):
        ld.fit_learned_model(tables, np.zeros(60, int))


def test_choose_threshold_max_f1_finds_a_perfect_split():
    scores = np.array([-3.0, -1.0, 0.5, 2.0, 4.0])
    labels = np.array([0, 0, 1, 1, 1])
    assert ld.choose_threshold_max_f1(scores, labels) == pytest.approx(0.5)


def test_choose_threshold_max_f1_breaks_ties_towards_the_larger_cut():
    scores = np.array([0.0, 1.0, 2.0, 3.0])
    labels = np.array([1, 0, 0, 1])
    # Accepting everything and accepting only the top candidate give the same F1.
    assert ld.choose_threshold_max_f1(scores, labels) == pytest.approx(3.0)


# --------------------------------------------------------------------------- model files

def test_model_json_round_trip(tmp_path):
    rng = default_rng(53)
    table = pd.DataFrame(
        rng.normal(size=(80, len(ld.FEATURE_NAMES))), columns=list(ld.FEATURE_NAMES)
    )
    labels = (table["zmean"] > 0).to_numpy().astype(int)
    model = ld.fit_learned_model(
        table, labels, l2_penalty=1.0, already_normalized=True
    ).with_threshold(1.25)

    path = ld.save_model(model, tmp_path / "model.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {
        "feats",
        "log_feats",
        "clip",
        "mean",
        "scale",
        "coef",
        "intercept",
        "threshold",
        "norm",
    }
    assert payload["feats"] == list(ld.FEATURE_NAMES)
    assert payload["norm"] == ld.NORMALIZATION

    loaded = ld.load_model(path)
    assert loaded.feature_names == model.feature_names
    assert loaded.log_features == model.log_features
    assert loaded.mean == pytest.approx(model.mean)
    assert loaded.scale == pytest.approx(model.scale)
    assert loaded.coef == pytest.approx(model.coef)
    assert loaded.intercept == pytest.approx(model.intercept)
    assert loaded.threshold == pytest.approx(1.25)


def test_load_model_rejects_a_foreign_normalisation(tmp_path):
    model = _simple_model(np.ones(6))
    payload = model.to_dict()
    payload["norm"] = "whole_range_mad"
    path = tmp_path / "foreign.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        ld.load_model(path)


def test_model_from_dict_reports_missing_keys(tmp_path):
    payload = _simple_model(np.ones(6)).to_dict()
    del payload["coef"]
    with pytest.raises(ValueError):
        ld.LearnedModel.from_dict(payload)


def test_model_checks_vector_lengths():
    with pytest.raises(ValueError):
        ld.LearnedModel(
            feature_names=ld.FEATURE_NAMES,
            log_features=ld.LOG_FEATURES,
            clip=12.0,
            mean=np.zeros(5),
            scale=np.ones(6),
            coef=np.ones(6),
            intercept=0.0,
            threshold=0.0,
        )


# --------------------------------------------------------------------------- end to end

def test_segment_learned_recovers_the_synthetic_objects():
    model, _, _, _ = _fit_on_fields()
    for seed in (21, 23):
        image, centres = _synthetic_field(seed)
        result = ld.segment_learned(image, model)
        labels = _labels_from_truth(result.candidate_set, centres)
        accepted = result.scores >= result.threshold
        true_positive = int(np.count_nonzero(accepted & (labels == 1)))
        false_positive = int(np.count_nonzero(accepted & (labels == 0)))
        assert true_positive == len(centres)
        assert false_positive <= 3
        # Every known object centre is inside the painted mask.
        for centre_y, centre_x in centres:
            assert result.main_mask[centre_y, centre_x]
        assert result.n_candidates > result.n_accepted
        assert not (result.main_mask & result.review_mask).any()


def test_segment_learned_review_band_sits_below_the_threshold():
    model, _, _, _ = _fit_on_fields()
    image, _ = _synthetic_field(21)
    base = ld.segment_learned(image, model)
    # A threshold above every score accepts nothing, but a wide band still flags the
    # best candidates for review.
    unreachable = float(base.scores.max()) + 1.0
    strict = ld.segment_learned(
        image, model, threshold=unreachable, review_margin=2.0
    )
    assert strict.n_accepted == 0
    assert strict.n_review > 0
    assert strict.review_mask.any()
    assert not strict.main_mask.any()
    assert strict.n_review < strict.n_candidates


def test_segment_learned_is_reproducible():
    model, _, _, _ = _fit_on_fields()
    first_image, _ = _synthetic_field(21)
    second_image, _ = _synthetic_field(21)
    assert np.array_equal(first_image, second_image)
    first = ld.segment_learned(first_image, model)
    second = ld.segment_learned(second_image, model)
    assert np.array_equal(first.main_mask, second.main_mask)
    assert np.array_equal(first.review_mask, second.review_mask)
    assert first.scores == pytest.approx(second.scores)


def test_segment_learned_on_a_flat_image_finds_nothing():
    model = _simple_model(np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), threshold=0.5)
    flat = np.full((80, 80), 500.0, np.float32)
    result = ld.segment_learned(flat, model)
    assert result.n_candidates == 0
    assert result.n_accepted == 0
    assert not result.main_mask.any()
    assert result.scores.shape == (0,)


# --------------------------------------------------------------------------- measurement

def _shape_mask():
    """Mask with a square, a digital disk, a line, an annulus and a border object."""
    mask = np.zeros((60, 60), bool)
    mask[5:10, 5:10] = True
    element = disk(5).astype(bool)
    mask[20 : 20 + element.shape[0], 20 : 20 + element.shape[1]] = element
    mask[40:41, 5:25] = True
    annulus = np.zeros((11, 11), bool)
    annulus[1:10, 1:10] = True
    annulus[4:7, 4:7] = False
    mask[45:56, 40:51] = annulus
    mask[0:3, 55:60] = True
    return mask


def test_measure_objects_on_known_shapes():
    mask = _shape_mask()
    plane = np.full((60, 60), 100.0)
    plane[5:10, 5:10] = 900.0
    response = np.full((60, 60), 2.0)
    response[5:10, 5:10] = 7.0
    table = ld.measure_objects(mask, plane, response, 0.5, response_reference=3.5)
    assert len(table) == 5

    square = table[(table["centroid_y_px"] == 7.0) & (table["centroid_x_px"] == 7.0)].iloc[0]
    assert square["area_px"] == 25
    assert square["area_um2"] == pytest.approx(25 * 0.25)
    assert square["perimeter_um"] == pytest.approx(square["perimeter_px"] * 0.5)
    assert square["solidity"] == pytest.approx(1.0)
    assert square["hole_count"] == 0
    assert not bool(square["touches_border"])
    assert square["intensity_mean"] == pytest.approx(900.0)
    assert square["intensity_max"] == pytest.approx(900.0)
    assert square["intensity_integrated"] == pytest.approx(25 * 900.0)
    assert square["response_mean"] == pytest.approx(7.0)
    assert square["response_confidence"] == pytest.approx(2.0)
    assert square["saturated_fraction"] == pytest.approx(0.0)

    round_object = table[table["area_px"] == int(disk(5).sum())].iloc[0]
    assert round_object["circularity"] > 0.9
    assert round_object["solidity"] > 0.85

    line = table[table["bbox_height_px"] == 1].iloc[0]
    assert line["area_px"] == 20
    assert line["bbox_width_px"] == 20

    annulus = table[table["hole_count"] == 1].iloc[0]
    assert annulus["hole_area_px"] == 9
    assert annulus["hole_fraction"] == pytest.approx(9 / (annulus["area_px"] + 9))
    assert annulus["solidity"] < 1.0

    border = table[table["touches_border"].astype(bool)].iloc[0]
    assert border["area_px"] == 15


def test_measure_objects_response_reference_defaults_to_the_seed_threshold():
    mask = np.zeros((20, 20), bool)
    mask[5:7, 5:7] = True
    plane = np.full((20, 20), 100.0)
    response = np.zeros((20, 20))
    response[5:7, 5:7] = 7.0
    default = inspect.signature(ld.measure_objects).parameters["response_reference"].default
    # The default is the detector's own seed threshold, not a second copy of its value.
    assert default == ld.DetectorParams.seed_threshold
    table = ld.measure_objects(mask, plane, response, 1.0)
    assert table.iloc[0]["response_confidence"] == pytest.approx(
        7.0 / ld.DetectorParams().seed_threshold
    )


def test_measure_objects_counts_saturated_pixels():
    mask = np.zeros((20, 20), bool)
    mask[5:7, 5:7] = True
    plane = np.full((20, 20), 100.0)
    plane[5, 5] = 65535.0
    table = ld.measure_objects(
        mask, plane, np.zeros((20, 20)), 1.0, saturation_level=65000.0
    )
    assert table.iloc[0]["saturated_fraction"] == pytest.approx(0.25)


def test_measure_objects_connectivity():
    mask = np.zeros((20, 20), bool)
    mask[5, 5] = True
    mask[6, 6] = True
    plane = np.zeros((20, 20))
    assert len(ld.measure_objects(mask, plane, plane, 1.0, connectivity=8)) == 1
    assert len(ld.measure_objects(mask, plane, plane, 1.0, connectivity=4)) == 2


def test_measure_objects_empty_mask():
    table = ld.measure_objects(
        np.zeros((20, 20), bool), np.zeros((20, 20)), np.zeros((20, 20)), 0.5
    )
    assert table.empty
    assert "area_um2" in table.columns

# --------------------------------------------------------------------------- parameters

def test_detector_params_rescaled_scales_every_length():
    params = ld.DetectorParams()
    doubled = params.rescaled(2.0)
    assert doubled.candidate_radius_px == 2 * params.candidate_radius_px
    assert doubled.ring_outer_radius_px == 2 * params.ring_outer_radius_px
    assert doubled.noise_tile_px == 2 * params.noise_tile_px
    assert doubled.min_ring_px == 4 * params.min_ring_px  # an area scales with the square
    assert doubled.dog_sigma_px[0][0] == pytest.approx(2 * params.dog_sigma_px[0][0])
    assert doubled.seed_window_px % 2 == 1
    # Non-spatial rules are untouched.
    assert doubled.seed_threshold == params.seed_threshold
    assert doubled.low_fraction == params.low_fraction
    with pytest.raises(ValueError):
        params.rescaled(0.0)


def test_detector_params_from_micrometres_converts_with_the_pixel_size():
    pixel_size_um = 0.2
    params = ld.DetectorParams.from_micrometres(
        pixel_size_um,
        dog_sigma_um=[(0.2, 0.6)],
        noise_sigma_um=0.2,
        seed_window_um=1.0,
        candidate_radius_um=2.0,
        noise_tile_um=10.0,
        ring_outer_radius_um=1.2,
        ring_inner_radius_um=0.4,
        reference_sigma_um=0.8,
        min_ring_um2=0.84,
        seed_threshold=4.0,
    )
    assert params.candidate_radius_px == 10
    assert params.seed_window_px == 5
    assert params.noise_tile_px == 50
    assert params.ring_outer_radius_px == 6
    assert params.ring_inner_radius_px == 2
    assert params.min_ring_px == 21
    assert params.dog_sigma_px[0] == pytest.approx((1.0, 3.0))
    assert params.seed_threshold == 4.0
    assert params.scale_um(pixel_size_um)["candidate_radius_um"] == pytest.approx(2.0)


def test_detect_candidates_follows_a_rescaled_parameter_set():
    image, _ = _synthetic_field(61)
    fine = ld.detect_candidates(image)
    coarse = ld.detect_candidates(image, params=ld.DetectorParams().rescaled(2.0))
    assert len(coarse.candidates) > 0
    assert list(coarse.features.columns) == list(ld.FEATURE_NAMES)
    # Wider filters and a wider growth window give every candidate more pixels.
    fine_area = np.mean([candidate.area_px for candidate in fine.candidates])
    coarse_area = np.mean([candidate.area_px for candidate in coarse.candidates])
    assert coarse_area > fine_area
