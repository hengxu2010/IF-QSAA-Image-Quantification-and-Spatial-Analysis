"""Tests for if_qsaa.segmentation on synthetic images with known ground truth."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from if_qsaa import segmentation as seg

PIXEL_SIZE_UM = 0.5


def disc(shape: tuple[int, int], centre: tuple[float, float], radius: float) -> np.ndarray:
    rows, columns = np.mgrid[0 : shape[0], 0 : shape[1]]
    return (rows - centre[0]) ** 2 + (columns - centre[1]) ** 2 <= radius * radius


def noisy_field(
    shape: tuple[int, int] = (150, 150),
    level: float = 50.0,
    sigma: float = 2.0,
    objects: tuple[tuple[float, float, float, float], ...] = (),
    seed: int = 0,
) -> np.ndarray:
    """A flat background with Gaussian noise plus discs of known radius and amplitude."""
    rng = np.random.default_rng(seed)
    image = rng.normal(level, sigma, shape).astype(np.float32)
    for row, column, radius, amplitude in objects:
        image[disc(shape, (row, column), radius)] += amplitude
    return image


# --------------------------------------------------------------------------- scale helpers

def test_scale_helpers_follow_the_original_rounding():
    assert seg.scale_length_px(0.0, 0.5) == 0
    assert seg.scale_length_px(10.0, 0.5) == 5
    assert seg.scale_length_px(5.0, 0.5) == 3  # floor(2.5 + 0.5)
    assert seg.scale_length_px(1.0, 0.1, minimum_if_enabled=1) == 1
    assert seg.scale_area_px(0.0, 2.0) == 0
    assert seg.scale_area_px(10.0, 0.5) == 2
    assert seg.scale_area_px(1.0, 0.1) == 1
    assert seg.scale_window_px(51.0, 0.5) == 27
    assert seg.scale_window_px(1.0, 0.1) == 3
    assert seg.scale_window_px(50.0, 1.0) % 2 == 1


# --------------------------------------------------------------------------- SNR response

def test_robust_snr_response_measures_contrast_in_noise_sigmas():
    image = noisy_field(objects=((75, 75, 6.0, 20.0),), sigma=2.0, seed=1)
    response = seg.robust_snr_response(
        image, PIXEL_SIZE_UM, background_radius_um=15.0
    )
    assert response.dtype == np.float32
    assert float(response.min()) >= 0.0
    # 20 grey levels on a noise of 2 is of the order of ten sigmas at the object.
    assert 6.0 < float(response[75, 75]) < 12.0
    # The background stays far below the object.
    assert float(response[10:40, 10:40].max()) < 5.0


def test_robust_snr_response_is_deterministic():
    image = noisy_field(objects=((75, 75, 5.0, 15.0),), seed=2)
    first = seg.robust_snr_response(image, PIXEL_SIZE_UM, background_radius_um=15.0)
    second = seg.robust_snr_response(image, PIXEL_SIZE_UM, background_radius_um=15.0)
    assert np.array_equal(first, second)


def test_robust_snr_response_cancels_gain_and_offset():
    image = noisy_field(objects=((75, 75, 6.0, 20.0),), sigma=2.0, seed=3)
    plain = seg.robust_snr_response(image, PIXEL_SIZE_UM, background_radius_um=15.0)
    amplified = seg.robust_snr_response(
        (image * 3.0 + 100.0).astype(np.float32),
        PIXEL_SIZE_UM,
        background_radius_um=15.0,
    )
    assert float(amplified[75, 75]) == pytest.approx(float(plain[75, 75]), rel=0.05)


def test_robust_snr_response_shared_noise_scale_sets_the_divisor():
    image = noisy_field(objects=((75, 75, 6.0, 20.0),), sigma=2.0, seed=4)
    own = seg.robust_snr_response(image, PIXEL_SIZE_UM, background_radius_um=15.0)
    shared = seg.robust_snr_response(
        image, PIXEL_SIZE_UM, background_radius_um=15.0, noise_scale=4.0
    )
    assert float(shared[75, 75]) == pytest.approx(float(own[75, 75]) / 2.0, rel=0.1)


def test_robust_snr_response_of_a_constant_field_is_empty():
    flat = np.full((60, 60), 7.0, dtype=np.float32)
    response = seg.robust_snr_response(flat, PIXEL_SIZE_UM, background_radius_um=5.0)
    assert float(response.max()) == 0.0


def test_robust_snr_response_inverts_dark_objects():
    image = noisy_field(objects=((75, 75, 6.0, 20.0),), sigma=2.0, seed=5)
    inverted = (300.0 - image).astype(np.float32)
    response = seg.robust_snr_response(
        inverted, PIXEL_SIZE_UM, background_radius_um=15.0, invert=True
    )
    assert float(response[75, 75]) > 5.0


def test_robust_snr_response_handles_non_finite_pixels_and_denoising():
    image = noisy_field(objects=((75, 75, 6.0, 20.0),), seed=6)
    image[0, 0] = np.nan
    response = seg.robust_snr_response(
        image,
        PIXEL_SIZE_UM,
        background_radius_um=15.0,
        denoise_method="median",
        denoise_um=0.5,
    )
    assert np.isfinite(response).all()
    with pytest.raises(ValueError):
        seg.robust_snr_response(
            image,
            PIXEL_SIZE_UM,
            background_radius_um=15.0,
            denoise_method="wiener",
            denoise_um=1.0,
        )


# --------------------------------------------------------------------------- thresholds

def test_global_threshold_separates_two_levels():
    response = np.zeros((40, 40), dtype=np.float32)
    response[10:20, 10:20] = 10.0
    for method in ("otsu", "li", "triangle"):
        value = seg.global_threshold(response, method)
        assert 0.0 < value < 10.0
    with pytest.raises(ValueError):
        seg.global_threshold(response, "yen")


def test_global_threshold_without_positive_pixels_is_infinite():
    assert math.isinf(seg.global_threshold(np.zeros((8, 8), dtype=np.float32)))


def test_global_threshold_falls_back_on_a_constant_positive_plane():
    constant = np.full((8, 8), 3.0, dtype=np.float32)
    assert seg.global_threshold(constant) == pytest.approx(3.0)


def test_sensitivity_and_seed_levels():
    assert seg.sensitivity_factor(50.0) == pytest.approx(1.0)
    assert seg.sensitivity_factor(85.0) == pytest.approx(0.5)
    assert seg.sensitivity_factor(15.0) == pytest.approx(2.0)
    assert seg.sensitivity_factor(0.0) <= 3.20
    assert seg.sensitivity_factor(100.0) >= 0.30
    assert seg.snr_seed_level(5.0, 50.0) == pytest.approx(5.0)
    assert seg.snr_seed_level(5.0, 120.0) == pytest.approx(5.0 * 2.0 ** (-1.0))


def test_hysteresis_keeps_only_components_holding_a_seed():
    response = np.zeros((40, 60), dtype=np.float32)
    response[10:20, 5:15] = 3.0  # faint, with one strong pixel
    response[14, 9] = 9.0
    response[10:20, 40:50] = 3.0  # faint and alone
    mask = seg.hysteresis_mask(response, low=2.0, high=8.0)
    assert int(mask[10:20, 5:15].sum()) == 100
    assert int(mask[10:20, 40:50].sum()) == 0


def test_hysteresis_without_a_seed_is_empty():
    response = np.full((10, 10), 1.0, dtype=np.float32)
    assert int(seg.hysteresis_mask(response, low=0.5, high=5.0).sum()) == 0
    assert int(seg.hysteresis_mask(response, low=0.5, high=math.inf).sum()) == 0


def test_threshold_mask_snr_hysteresis_recovers_the_object():
    image = noisy_field(objects=((75, 75, 6.0, 25.0),), sigma=2.0, seed=7)
    response = seg.robust_snr_response(
        image, PIXEL_SIZE_UM, background_radius_um=15.0
    )
    mask, high, low, reference = seg.threshold_mask(
        response, method="snr_hysteresis", seed_sigma=5.0, grow_ratio=0.5
    )
    assert high == pytest.approx(5.0)
    assert low == pytest.approx(2.5)
    assert reference == pytest.approx(high)
    assert bool(mask[75, 75])
    assert int(mask.sum()) < 400  # the object only, not half the frame


@pytest.mark.parametrize("method", ["otsu", "li", "triangle", "hysteresis_otsu"])
def test_threshold_mask_global_methods(method):
    response = np.zeros((60, 60), dtype=np.float32)
    response[disc((60, 60), (30, 30), 8.0)] = 1.0
    mask, high, low, reference = seg.threshold_mask(response, method=method)
    assert int(mask.sum()) == int(disc((60, 60), (30, 30), 8.0).sum())
    assert math.isfinite(high) and math.isfinite(low)
    if method in ("otsu", "hysteresis_otsu"):
        assert reference is not None
    else:
        assert reference is None


def test_threshold_mask_fixed_value():
    response = np.zeros((30, 30), dtype=np.float32)
    response[5:10, 5:10] = 4.0
    response[5:10, 20:25] = 1.0
    mask, high, low, reference = seg.threshold_mask(
        response, method="value", value=3.0, grow_ratio=1.0
    )
    assert high == pytest.approx(3.0)
    assert low == pytest.approx(3.0)
    assert reference is None
    assert int(mask.sum()) == 25
    with pytest.raises(ValueError):
        seg.threshold_mask(response, method="value")


def two_level_patch() -> np.ndarray:
    """A patch above an absolute level, joined to a fainter patch below it."""
    response = np.zeros((30, 40), dtype=np.float32)
    response[10:14, 10:14] = 120.0  # above the level
    response[10:14, 14:19] = 70.0  # below it, but connected to the patch above
    return response


def test_threshold_mask_fixed_value_is_literal_by_default():
    # A level the caller computed means that level: nothing below it is foreground
    # unless growth is asked for explicitly.
    response = two_level_patch()
    mask, high, low, _ = seg.threshold_mask(response, method="value", value=100.0)
    assert high == pytest.approx(100.0)
    assert low == pytest.approx(100.0)
    assert int(mask.sum()) == 16  # the strong patch only
    assert float(response[mask].min()) >= 100.0
    explicit = seg.threshold_mask(
        response, method="value", value=100.0, grow_ratio=1.0
    )[0]
    assert np.array_equal(explicit, mask)


def test_threshold_mask_fixed_value_still_grows_when_asked():
    # The growing behaviour stays reachable through an explicit ratio.
    response = two_level_patch()
    mask, high, low, _ = seg.threshold_mask(
        response, method="value", value=100.0, grow_ratio=0.60
    )
    assert high == pytest.approx(100.0)
    assert low == pytest.approx(60.0)
    assert int(mask.sum()) == 36  # the faint patch is pulled in by hysteresis


@pytest.mark.parametrize("method", ["snr_hysteresis", "hysteresis_otsu"])
def test_threshold_mask_relative_rules_keep_growing_by_default(method):
    # Only the absolute route changed: a level this module derives itself still grows.
    response = np.zeros((60, 60), dtype=np.float32)
    response[disc((60, 60), (30, 30), 8.0)] = 6.0
    _, high, low, _ = seg.threshold_mask(response, method=method, seed_sigma=5.0)
    assert low == pytest.approx(high * 0.60)


def test_threshold_mask_local_mean_std_follows_an_uneven_background():
    rows, columns = np.mgrid[0:120, 0:120]
    response = (0.05 + 0.004 * columns).astype(np.float32)
    for row, column in ((30, 20), (30, 100)):
        response[disc((120, 120), (row, column), 4.0)] += 0.5
    mask, high, low, reference = seg.threshold_mask(
        response,
        method="local_mean_std",
        local_window_um=10.0,
        pixel_size_um=PIXEL_SIZE_UM,
        local_k=1.0,
        grow_ratio=1.0,
        clip_range=(0.0, 1.0),
    )
    assert reference is None
    assert low == pytest.approx(high)
    # Both objects are found although their absolute brightness differs.
    assert bool(mask[30, 20]) and bool(mask[30, 100])
    assert float(mask.mean()) < 0.05  # the ramp itself is not thresholded


def test_threshold_mask_local_method_needs_its_window():
    response = np.zeros((20, 20), dtype=np.float32)
    with pytest.raises(ValueError):
        seg.threshold_mask(response, method="local_mean_std")
    with pytest.raises(ValueError):
        seg.threshold_mask(response, method="sauvola")


def test_threshold_mask_local_growth_extends_the_mask():
    rows, columns = np.mgrid[0:80, 0:80]
    response = np.zeros((80, 80), dtype=np.float32)
    response[disc((80, 80), (40, 40), 5.0)] = 0.9
    response[disc((80, 80), (40, 40), 9.0) & ~disc((80, 80), (40, 40), 5.0)] = 0.3
    strong, _, _, _ = seg.threshold_mask(
        response,
        method="local_mean_std",
        local_window_um=8.0,
        pixel_size_um=PIXEL_SIZE_UM,
        grow_ratio=1.0,
    )
    grown, _, _, _ = seg.threshold_mask(
        response,
        method="local_mean_std",
        local_window_um=8.0,
        pixel_size_um=PIXEL_SIZE_UM,
        grow_ratio=0.3,
    )
    assert int(grown.sum()) >= int(strong.sum())


# --------------------------------------------------------------------------- evidence filter

def test_evidence_filter_drops_a_weak_grown_component():
    response = np.zeros((40, 80), dtype=np.float32)
    response[10:20, 5:15] = 5.0  # a confident object
    response[10:20, 40:50] = 3.0  # exactly at the candidate level
    mask = response > 0
    filtered = seg.filter_components_by_evidence(mask, response, high=5.0, low=3.0)
    assert int(filtered[10:20, 5:15].sum()) == 100
    assert int(filtered[10:20, 40:50].sum()) == 0


def test_evidence_filter_on_an_empty_mask():
    empty = np.zeros((10, 10), dtype=bool)
    response = np.zeros((10, 10), dtype=np.float32)
    assert int(seg.filter_components_by_evidence(empty, response, 5.0, 3.0).sum()) == 0


# --------------------------------------------------------------------------- clean-up

def test_fill_small_holes_respects_the_area_limit_and_the_frame():
    mask = np.zeros((30, 30), dtype=bool)
    mask[5:25, 5:25] = True
    mask[10:12, 10:12] = False  # 4 px hole
    mask[16:22, 16:22] = False  # 36 px hole
    assert int(mask.sum()) == 400 - 4 - 36
    assert int(seg.fill_small_holes(mask, 4).sum()) == 400 - 36
    assert int(seg.fill_small_holes(mask, 40).sum()) == 400
    assert int(seg.fill_small_holes(mask, 0).sum()) == int(mask.sum())
    # Background touching the frame edge is outside, not a hole.
    assert int(seg.fill_small_holes(np.zeros((6, 6), dtype=bool), 100).sum()) == 0


def test_clean_mask_opening_removes_a_speck_and_closing_bridges_a_gap():
    mask = np.zeros((30, 30), dtype=bool)
    mask[10:20, 10:20] = True
    mask[2, 2] = True
    opened = seg.clean_mask(mask, 1.0, open_um=1.0)
    assert not bool(opened[2, 2])
    assert int(opened.sum()) > 80

    gapped = np.zeros((30, 30), dtype=bool)
    gapped[10:20, 5:14] = True
    gapped[10:20, 16:25] = True
    closed = seg.clean_mask(gapped, 1.0, close_um=2.0)
    labels = seg.label_objects(closed, 1.0)
    assert int(labels.max()) == 1


def test_clean_mask_erode_and_dilate_keep_border_objects():
    mask = np.zeros((20, 20), dtype=bool)
    mask[0:6, 0:6] = True  # an object in the frame corner
    eroded = seg.clean_mask(mask, 1.0, erode_um=1.0)
    assert bool(eroded[0, 0])  # outside the frame counts as foreground
    dilated = seg.clean_mask(mask, 1.0, dilate_um=1.0)
    assert int(dilated.sum()) > int(mask.sum())


def test_clean_mask_can_remove_border_objects():
    mask = np.zeros((40, 40), dtype=bool)
    mask[0:6, 10:16] = True
    mask[20:26, 20:26] = True
    cleaned = seg.clean_mask(mask, 1.0, remove_border=True)
    assert int(cleaned.sum()) == 36
    assert int(seg.remove_border_objects(mask).sum()) == 36
    assert int(seg.remove_border_objects(np.zeros((5, 5), dtype=bool)).sum()) == 0


def test_clean_mask_leaves_an_empty_mask_empty():
    empty = np.zeros((12, 12), dtype=bool)
    cleaned = seg.clean_mask(
        empty, 1.0, open_um=2.0, close_um=2.0, erode_um=1.0, dilate_um=1.0,
        max_hole_area_um2=10.0, remove_border=True,
    )
    assert int(cleaned.sum()) == 0


# --------------------------------------------------------------------------- mask offset

def test_offset_mask_expands_by_an_exact_distance():
    mask = np.zeros((41, 41), dtype=bool)
    mask[20, 20] = True
    grown = seg.offset_mask(mask, 3)
    rows, columns = np.mgrid[0:41, 0:41]
    distance = np.sqrt((rows - 20.0) ** 2 + (columns - 20.0) ** 2)
    assert np.array_equal(grown, distance <= 3.0)


def test_offset_mask_contracts_and_counts_the_frame_edge_as_a_boundary():
    mask = np.zeros((21, 21), dtype=bool)
    mask[5:16, 5:16] = True
    shrunk = seg.offset_mask(mask, -2)
    assert int(shrunk.sum()) == 7 * 7
    full = np.ones((11, 11), dtype=bool)
    assert int(seg.offset_mask(full, -1).sum()) == 9 * 9


def test_offset_mask_zero_copies_and_rejects_non_integers():
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:5, 2:5] = True
    same = seg.offset_mask(mask, 0)
    assert np.array_equal(same, mask) and same is not mask
    assert int(seg.offset_mask(np.zeros((8, 8), dtype=bool), 4).sum()) == 0
    with pytest.raises(TypeError):
        seg.offset_mask(mask, True)
    with pytest.raises(TypeError):
        seg.offset_mask(mask, 1.5)


def test_offset_mask_um_rounds_to_whole_pixels():
    mask = np.zeros((41, 41), dtype=bool)
    mask[20, 20] = True
    assert np.array_equal(
        seg.offset_mask_um(mask, 1.5, PIXEL_SIZE_UM), seg.offset_mask(mask, 3)
    )
    square = np.zeros((21, 21), dtype=bool)
    square[5:16, 5:16] = True
    assert np.array_equal(
        seg.offset_mask_um(square, -1.0, PIXEL_SIZE_UM), seg.offset_mask(square, -2)
    )


# --------------------------------------------------------------------------- splitting

def test_split_separates_two_touching_discs_and_keeps_every_pixel():
    shape = (60, 80)
    mask = disc(shape, (30, 26), 9.0) | disc(shape, (30, 42), 9.0)
    labels = seg.split_touching_objects(mask, 1.0, peak_distance_um=5.0)
    assert int(labels.max()) == 2
    areas = [int((labels == i).sum()) for i in (1, 2)]
    assert sum(areas) == int(mask.sum())  # the partition loses nothing
    assert min(areas) > 0.4 * int(mask.sum())  # both halves are real objects


def test_split_rejects_the_artificial_peak_of_three_touching_circles():
    shape = (70, 80)
    mask = np.zeros(shape, dtype=bool)
    for centre in ((25, 30), (25, 46), (39, 38)):
        mask |= disc(shape, centre, 8.0)
    labels = seg.split_touching_objects(mask, 1.0, peak_distance_um=5.0)
    assert int(labels.max()) == 3
    assert int((labels > 0).sum()) == int(mask.sum())


def test_split_leaves_a_single_object_alone():
    shape = (60, 80)
    mask = disc(shape, (30, 40), 8.0)
    labels = seg.split_touching_objects(mask, 1.0, peak_distance_um=5.0)
    assert int(labels.max()) == 1
    assert np.array_equal(labels > 0, mask)


def test_split_protects_a_ring_by_its_hole():
    shape = (60, 80)
    mask = disc(shape, (30, 40), 15.0) & ~disc(shape, (30, 40), 10.0)
    assert int(seg.split_touching_objects(mask, 1.0, peak_distance_um=3.0).max()) == 1


def test_split_protects_an_elongated_structure():
    mask = np.zeros((40, 80), dtype=bool)
    mask[20:23, 5:75] = True
    assert int(seg.split_touching_objects(mask, 1.0, peak_distance_um=3.0).max()) == 1


def test_split_ties_on_a_distance_plateau_give_one_object():
    # Every pixel of the centre line of a bar is an equally deep maximum: the tied
    # peaks form one connected seed, so the bar is not cut.
    mask = np.zeros((40, 60), dtype=bool)
    mask[15:22, 20:41] = True
    labels = seg.split_touching_objects(mask, 1.0, peak_distance_um=3.0)
    assert int(labels.max()) == 1
    assert int((labels > 0).sum()) == int(mask.sum())


def test_split_protect_continuous_switch_keeps_everything_whole():
    shape = (60, 80)
    mask = disc(shape, (30, 26), 9.0) | disc(shape, (30, 42), 9.0)
    labels = seg.split_touching_objects(
        mask, 1.0, peak_distance_um=5.0, protect_continuous=True
    )
    assert int(labels.max()) == 1


def test_split_on_an_empty_mask_and_a_border_object():
    assert int(
        seg.split_touching_objects(
            np.zeros((10, 10), dtype=bool), 1.0, peak_distance_um=2.0
        ).max()
    ) == 0
    corner = np.zeros((30, 30), dtype=bool)
    corner[0:8, 0:8] = True
    labels = seg.split_touching_objects(corner, 1.0, peak_distance_um=3.0)
    assert int(labels.max()) == 1
    assert int((labels > 0).sum()) == 64


def test_label_objects_plain_and_split():
    shape = (60, 80)
    mask = disc(shape, (30, 26), 9.0) | disc(shape, (30, 42), 9.0)
    assert int(seg.label_objects(mask, 1.0).max()) == 1
    assert (
        int(seg.label_objects(mask, 1.0, split=True, peak_distance_um=5.0).max()) == 2
    )
    with pytest.raises(ValueError):
        seg.label_objects(mask, 1.0, split=True)


def test_label_objects_validates_split_arguments_on_an_empty_mask():
    # A typo in a splitting parameter must fail on every image, not only on the ones
    # that happen to hold an object.
    empty = np.zeros((12, 12), dtype=bool)
    labels = seg.label_objects(empty, 1.0, split=True, peak_distance_um=3.0)
    assert labels.dtype == np.int32
    assert int(labels.max()) == 0
    with pytest.raises(ValueError):
        seg.label_objects(empty, 1.0, split=True)
    with pytest.raises(TypeError):
        seg.label_objects(
            empty, 1.0, split=True, peak_distance_um=3.0, peak_heigth=0.3
        )


# --------------------------------------------------------------------------- measurement

def test_measure_objects_recovers_rectangle_geometry_and_contrast():
    image = np.full((60, 80), 10.0, dtype=np.float32)
    labels = np.zeros((60, 80), dtype=np.int32)
    labels[20:26, 30:50] = 1  # 6 rows by 20 columns
    image[20:26, 30:50] = 35.0
    table = seg.measure_objects(labels, image, 0.25, noise_sigma=5.0)
    assert len(table) == 1
    row = table.iloc[0]
    assert int(row["area_px"]) == 120
    assert row["area_um2"] == pytest.approx(120 * 0.25**2)
    assert row["centroid_x_px"] == pytest.approx(39.5)
    assert row["centroid_y_px"] == pytest.approx(22.5)
    assert int(row["bbox_x_px"]) == 30 and int(row["bbox_width_px"]) == 20
    # Axes are 4 * sqrt(eigenvalue) of the regularised pixel covariance.
    assert row["major_axis_um"] == pytest.approx(4.0 * 20.0 / math.sqrt(12.0) * 0.25)
    assert row["aspect_ratio"] == pytest.approx(20.0 / 6.0, rel=1e-6)
    assert row["equivalent_diameter_um"] == pytest.approx(
        math.sqrt(4.0 * 120 / math.pi) * 0.25
    )
    assert row["solidity"] == pytest.approx(1.0)
    assert row["mean_intensity"] == pytest.approx(35.0)
    assert row["integrated_intensity"] == pytest.approx(35.0 * 120)
    assert row["local_background"] == pytest.approx(10.0)
    assert row["boundary_contrast"] == pytest.approx(25.0)
    assert row["snr"] == pytest.approx(5.0)
    assert not bool(row["touches_border"])
    assert list(table.columns) == list(seg.MEASUREMENT_COLUMNS)


def test_measure_objects_recovers_disc_geometry():
    shape = (80, 80)
    mask = disc(shape, (40, 40), 10.0)
    labels = mask.astype(np.int32)
    image = np.where(mask, 60.0, 5.0).astype(np.float32)
    row = seg.measure_objects(labels, image, PIXEL_SIZE_UM).iloc[0]
    assert row["area_px"] == int(mask.sum())
    assert row["equivalent_diameter_um"] == pytest.approx(2 * 10.0 * PIXEL_SIZE_UM, rel=0.03)
    assert row["aspect_ratio"] == pytest.approx(1.0, abs=0.05)
    # The convex hull of a digitised disc is slightly larger than the disc itself.
    assert row["solidity"] > 0.9
    assert row["perimeter_um"] == pytest.approx(
        2 * math.pi * 10.0 * PIXEL_SIZE_UM, rel=0.1
    )


def test_measure_objects_ring_excludes_the_neighbour():
    image = np.full((60, 60), 10.0, dtype=np.float32)
    labels = np.zeros((60, 60), dtype=np.int32)
    labels[28:32, 20:24] = 1
    labels[28:32, 26:30] = 2
    image[labels == 1] = 40.0
    image[labels == 2] = 200.0  # a very bright neighbour nearby
    table = seg.measure_objects(labels, image, PIXEL_SIZE_UM)
    first = table.loc[table["id"] == 1].iloc[0]
    # The neighbour's pixels are excluded from the ring, so the background is clean.
    assert first["local_background"] == pytest.approx(10.0)
    assert first["boundary_contrast"] == pytest.approx(30.0)


def test_measure_objects_flags_a_border_object_and_keeps_a_usable_ring():
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[0:4, 5:9] = 1
    image = np.where(labels > 0, 50.0, 8.0).astype(np.float32)
    row = seg.measure_objects(labels, image, PIXEL_SIZE_UM).iloc[0]
    assert bool(row["touches_border"])
    assert int(row["ring_area_px"]) > 0
    assert row["local_background"] == pytest.approx(8.0)


def test_measure_objects_single_pixel_object():
    labels = np.zeros((11, 11), dtype=np.int32)
    labels[5, 5] = 1
    image = np.full((11, 11), 2.0, dtype=np.float32)
    image[5, 5] = 9.0
    row = seg.measure_objects(labels, image, 1.0).iloc[0]
    assert int(row["area_px"]) == 1
    assert row["major_axis_um"] == pytest.approx(4.0 / math.sqrt(12.0))
    assert row["aspect_ratio"] == pytest.approx(1.0)
    assert row["solidity"] == pytest.approx(1.0)
    assert row["boundary_contrast"] == pytest.approx(7.0)


def test_measure_objects_empty_label_image():
    table = seg.measure_objects(
        np.zeros((8, 8), dtype=np.int32), np.zeros((8, 8), dtype=np.float32), 1.0
    )
    assert table.empty
    assert list(table.columns) == list(seg.MEASUREMENT_COLUMNS)
    assert table["area_px"].dtype == np.int64


def test_measure_objects_response_columns():
    labels = np.zeros((30, 30), dtype=np.int32)
    labels[10:14, 10:14] = 1
    image = np.where(labels > 0, 30.0, 4.0).astype(np.float32)
    response = np.where(labels > 0, 8.0, 0.0).astype(np.float32)
    row = seg.measure_objects(
        labels, image, 1.0, response=response, response_reference=4.0
    ).iloc[0]
    assert row["response_mean"] == pytest.approx(8.0)
    assert row["response_max"] == pytest.approx(8.0)
    assert row["response_confidence"] == pytest.approx(2.0)
    without = seg.measure_objects(labels, image, 1.0).iloc[0]
    assert math.isnan(without["response_mean"])
    assert math.isnan(without["snr"])


def test_measure_objects_checks_shapes():
    with pytest.raises(ValueError):
        seg.measure_objects(
            np.zeros((5, 5), dtype=np.int32), np.zeros((6, 6), dtype=np.float32), 1.0
        )


# --------------------------------------------------------------------------- filters

def build_three_objects() -> tuple[np.ndarray, np.ndarray]:
    """One small, one large and one border object, with known areas in pixels."""
    labels = np.zeros((40, 60), dtype=np.int32)
    labels[20:22, 10:12] = 1  # 4 px
    labels[10:20, 30:40] = 2  # 100 px
    labels[0:5, 50:55] = 3  # 25 px, touching the frame edge
    image = np.where(labels > 0, 20.0, 3.0).astype(np.float32)
    image[labels == 1] = 100.0
    return labels, image


def test_filter_objects_by_area_and_renumbering():
    labels, image = build_three_objects()
    table = seg.measure_objects(labels, image, 1.0)
    kept_labels, kept = seg.filter_objects(labels, table, 1.0, min_area_um2=10.0)
    assert kept["id"].tolist() == [1, 2]
    assert kept["area_px"].tolist() == [100, 25]
    assert int(kept_labels.max()) == 2
    # The renumbered label image agrees with the table.
    for new_id, area in zip(kept["id"], kept["area_px"]):
        assert int((kept_labels == new_id).sum()) == int(area)


def test_filter_objects_border_intensity_and_shape_criteria():
    labels, image = build_three_objects()
    table = seg.measure_objects(labels, image, 1.0, noise_sigma=1.0)
    _, no_border = seg.filter_objects(labels, table, 1.0, remove_border=True)
    assert no_border["area_px"].tolist() == [4, 100]
    _, bright = seg.filter_objects(labels, table, 1.0, min_mean_intensity=50.0)
    assert bright["area_px"].tolist() == [4]
    _, large_only = seg.filter_objects(labels, table, 1.0, max_area_um2=50.0)
    assert sorted(large_only["area_px"].tolist()) == [4, 25]
    _, solid = seg.filter_objects(labels, table, 1.0, min_solidity=0.99)
    assert len(solid) == 3
    _, round_only = seg.filter_objects(labels, table, 1.0, max_aspect_ratio=1.01)
    assert len(round_only) == 3
    _, strong = seg.filter_objects(labels, table, 1.0, min_snr=50.0)
    assert strong["area_px"].tolist() == [4]


def test_filter_objects_refuses_a_criterion_the_measurement_never_populated():
    # measure_objects without noise_sigma leaves snr undefined in every row, so the
    # criterion would reject every object; that is reported, not returned as empty.
    labels, image = build_three_objects()
    table = seg.measure_objects(labels, image, 1.0)
    assert bool(table["snr"].isna().all())
    with pytest.raises(ValueError, match="min_snr"):
        seg.filter_objects(labels, table, 1.0, min_snr=1.0)
    assert bool(table["response_confidence"].isna().all())
    with pytest.raises(ValueError, match="min_response_confidence"):
        seg.filter_objects(labels, table, 1.0, min_response_confidence=0.5)
    # A column that is missing altogether is refused in the same place.
    with pytest.raises(ValueError, match="min_snr"):
        seg.filter_objects(labels, table.drop(columns=["snr"]), 1.0, min_snr=1.0)


def test_filter_objects_still_drops_single_undefined_rows():
    # One object without a usable value is dropped as before; only an entirely
    # undefined column is a mistake in the measurement call.
    labels, image = build_three_objects()
    table = seg.measure_objects(labels, image, 1.0, noise_sigma=1.0)
    table.loc[table.index[0], "snr"] = float("nan")
    _, kept = seg.filter_objects(labels, table, 1.0, min_snr=-1e6)
    assert len(kept) == len(table) - 1


def test_filter_objects_can_reject_everything():
    labels, image = build_three_objects()
    table = seg.measure_objects(labels, image, 1.0)
    kept_labels, kept = seg.filter_objects(labels, table, 1.0, min_area_um2=1e6)
    assert kept.empty
    assert int(kept_labels.max()) == 0


def test_filter_objects_on_an_empty_table():
    empty_labels = np.zeros((6, 6), dtype=np.int32)
    table = seg.measure_objects(empty_labels, np.zeros((6, 6), dtype=np.float32), 1.0)
    kept_labels, kept = seg.filter_objects(empty_labels, table, 1.0, min_area_um2=1.0)
    assert kept.empty
    assert int(kept_labels.max()) == 0


# --------------------------------------------------------------------------- pipelines

def test_segment_recovers_known_objects():
    objects = (
        (40, 40, 6.0, 30.0),
        (40, 110, 6.0, 30.0),
        (110, 40, 6.0, 30.0),
        (110, 110, 6.0, 30.0),
    )
    image = noisy_field((150, 150), objects=objects, sigma=2.0, seed=11)
    labels, table = seg.segment(
        image,
        PIXEL_SIZE_UM,
        background_radius_um=15.0,
        denoise_um=0.4,
        close_um=0.5,
        max_hole_area_um2=1.0,
        min_area_um2=2.0,
        object_noise_sigma=2.0,
    )
    assert len(table) == 4
    assert int(labels.max()) == 4
    truth_area_px = int(disc((150, 150), (40, 40), 6.0).sum())
    for area in table["area_px"]:
        assert 0.5 * truth_area_px <= area <= 1.6 * truth_area_px
    assert (table["snr"] > 3.0).all()
    assert isinstance(table, pd.DataFrame)


def test_segment_is_reproducible():
    image = noisy_field(objects=((75, 75, 6.0, 25.0),), seed=12)
    first = seg.segment(image, PIXEL_SIZE_UM, background_radius_um=15.0, min_area_um2=2.0)
    second = seg.segment(image, PIXEL_SIZE_UM, background_radius_um=15.0, min_area_um2=2.0)
    assert np.array_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])


def test_segment_background_correction_mode():
    rows, columns = np.mgrid[0:150, 0:150]
    image = (20.0 + 0.3 * columns).astype(np.float32)
    for centre in ((40, 40), (100, 110)):
        image[disc((150, 150), centre, 5.0)] += 60.0
    labels, table = seg.segment(
        image,
        PIXEL_SIZE_UM,
        background_radius_um=6.0,
        response_mode="background_correction",
        background_method="top_hat",
        threshold_method="otsu",
        min_area_um2=1.0,
    )
    assert len(table) == 2
    assert int(labels.max()) == 2
    with pytest.raises(ValueError):
        seg.segment(image, PIXEL_SIZE_UM, background_radius_um=6.0, response_mode="rank")


def test_segment_splits_touching_objects_when_asked():
    image = noisy_field((120, 120), sigma=1.0, seed=13)
    image[disc((120, 120), (60, 54), 9.0)] += 40.0
    image[disc((120, 120), (60, 70), 9.0)] += 40.0
    merged = seg.segment(
        image, PIXEL_SIZE_UM, background_radius_um=15.0, min_area_um2=5.0
    )[1]
    split = seg.segment(
        image,
        PIXEL_SIZE_UM,
        background_radius_um=15.0,
        min_area_um2=5.0,
        split=True,
        peak_distance_um=4.0,
    )[1]
    assert len(merged) == 1
    assert len(split) == 2


def test_segment_removes_border_objects_when_asked():
    image = noisy_field((120, 120), sigma=1.0, seed=14)
    image[disc((120, 120), (60, 60), 7.0)] += 40.0
    image[disc((120, 120), (2, 60), 7.0)] += 40.0
    kept = seg.segment(
        image, PIXEL_SIZE_UM, background_radius_um=15.0, min_area_um2=3.0
    )[1]
    trimmed = seg.segment(
        image,
        PIXEL_SIZE_UM,
        background_radius_um=15.0,
        min_area_um2=3.0,
        remove_border=True,
    )[1]
    assert len(kept) == 2
    assert len(trimmed) == 1
    assert not trimmed["touches_border"].any()


def test_segment_on_an_empty_frame_returns_an_empty_table():
    flat = np.full((40, 40), 12.0, dtype=np.float32)
    labels, table = seg.segment(flat, PIXEL_SIZE_UM, background_radius_um=5.0)
    assert int(labels.max()) == 0
    assert table.empty
    assert list(table.columns) == list(seg.MEASUREMENT_COLUMNS)


@pytest.mark.parametrize(
    "gate", [{"compartment_mode": "absolute", "compartment_ceiling": 60.0},
             {"compartment_mode": "auto"}]
)
def test_segment_compartment_gate_excludes_a_bright_compartment(gate):
    # A broad glow around one of the two objects, like an autofluorescent compartment.
    rows, columns = np.mgrid[0:140, 0:140]
    image = noisy_field((140, 140), level=20.0, sigma=1.0, seed=15)
    image += (
        150.0 * np.exp(-(((rows - 40) ** 2 + (columns - 110) ** 2) / (2 * 28.0**2)))
    ).astype(np.float32)
    image[disc((140, 140), (40, 30), 6.0)] += 40.0
    image[disc((140, 140), (40, 110), 6.0)] += 40.0
    without = seg.segment(
        image, PIXEL_SIZE_UM, background_radius_um=12.0, min_area_um2=3.0
    )[1]
    with_gate = seg.segment(
        image, PIXEL_SIZE_UM, background_radius_um=12.0, min_area_um2=3.0, **gate
    )[1]
    assert (without["centroid_x_px"] > 70).any()
    assert len(with_gate) == 1
    # Only the object outside the bright compartment survives the gate.
    assert with_gate.iloc[0]["centroid_x_px"] == pytest.approx(30.0, abs=2.0)


def test_measure_external_mask_measures_every_component():
    image = noisy_field((100, 100), sigma=1.0, seed=16)
    mask = disc((100, 100), (30, 30), 8.0) | disc((100, 100), (70, 70), 2.0)
    image[mask] += 50.0
    labels, table = seg.measure_external_mask(image, mask, PIXEL_SIZE_UM)
    assert int(labels.max()) == 2
    assert len(table) == 2  # no filter is applied, the tiny object survives
    assert set(table["area_px"]) == {
        int(disc((100, 100), (30, 30), 8.0).sum()),
        int(disc((100, 100), (70, 70), 2.0).sum()),
    }


def test_measure_external_mask_can_split_and_checks_shapes():
    image = noisy_field((100, 100), sigma=1.0, seed=17)
    mask = disc((100, 100), (50, 44), 9.0) | disc((100, 100), (50, 60), 9.0)
    labels, table = seg.measure_external_mask(
        image, mask, PIXEL_SIZE_UM, split=True, peak_distance_um=4.0
    )
    assert int(labels.max()) == 2
    assert len(table) == 2
    with pytest.raises(ValueError):
        seg.measure_external_mask(image, mask[:50, :50], PIXEL_SIZE_UM)
