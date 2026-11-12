"""Tests for if_qsaa.background on synthetic images with known ground truth."""

from __future__ import annotations

import numpy as np
import pytest

from if_qsaa import background as bg

PIXEL_SIZE_UM = 0.5


def ramp_with_spot(
    shape: tuple[int, int] = (100, 120),
    amplitude: float = 90.0,
) -> tuple[np.ndarray, np.ndarray]:
    """A linear background ramp plus one small bright square of known amplitude."""
    rows, columns = np.mgrid[0 : shape[0], 0 : shape[1]]
    ramp = 20.0 + 0.4 * columns + 0.2 * rows
    image = ramp.astype(np.float32).copy()
    image[50:54, 60:64] += amplitude
    return image, ramp.astype(np.float32)


def disc(shape: tuple[int, int], centre: tuple[int, int], radius: float) -> np.ndarray:
    rows, columns = np.mgrid[0 : shape[0], 0 : shape[1]]
    return (rows - centre[0]) ** 2 + (columns - centre[1]) ** 2 <= radius * radius


# --------------------------------------------------------------------------- sampling

def test_finite_sample_drops_non_finite_and_subsamples():
    values = np.arange(400, dtype=np.float64).reshape(20, 20)
    values[0, 0] = np.nan
    values[1, 1] = np.inf
    sample = bg.finite_sample(values)
    assert np.isfinite(sample).all()
    assert sample.size == 398
    big = np.ones((3000, 3000), dtype=np.float32)
    assert bg.finite_sample(big, max_pixels=10_000).size <= 10_000 * 4


def test_finite_sample_falls_back_to_zero():
    assert bg.finite_sample(np.full((4, 4), np.nan)).tolist() == [0.0]


# --------------------------------------------------------------------------- rolling ball

def test_rolling_ball_shrink_schedule():
    assert bg.rolling_ball_shrink_factor(1.0) == 1
    assert bg.rolling_ball_shrink_factor(10.0) == 1
    assert bg.rolling_ball_shrink_factor(10.5) == 2
    assert bg.rolling_ball_shrink_factor(30.0) == 2
    assert bg.rolling_ball_shrink_factor(100.0) == 4
    assert bg.rolling_ball_shrink_factor(101.0) == 8


def test_rolling_ball_leaves_a_flat_image_unchanged():
    flat = np.full((50, 50), 12.0, dtype=np.float32)
    background = bg.rolling_ball_background(flat, 5.0, PIXEL_SIZE_UM)
    assert np.allclose(background, 12.0)


def test_rolling_ball_removes_a_ramp_and_keeps_the_object():
    image, _ramp = ramp_with_spot()
    background = bg.rolling_ball_background(image, 10.0, PIXEL_SIZE_UM)
    assert background.dtype == np.float32
    # The background never rises above the image it was measured on.
    assert bool((background <= image + 1e-4).all())
    residual = image - background
    # Away from the object and the frame edge the residual of a plane is flat,
    # so the ramp is gone.
    background_residual = residual[25:45, 25:55]
    assert float(background_residual.std()) < 0.05
    # The object keeps its amplitude above that flat level.
    assert float(residual[51, 61] - background_residual.mean()) == pytest.approx(
        90.0, abs=1.0
    )


def test_rolling_ball_radius_in_um_matches_the_pixel_version():
    image, _ = ramp_with_spot(shape=(60, 60))
    by_um = bg.rolling_ball_background(image, 8.0, 0.4)
    by_px = bg.rolling_ball_background_px(image, 20.0)
    assert np.allclose(by_um, by_px)


def test_rolling_ball_rejects_a_non_positive_radius():
    with pytest.raises(ValueError):
        bg.rolling_ball_background_px(np.zeros((8, 8), dtype=np.float32), 0.0)


def test_rolling_ball_residual_prefilters():
    image, _ = ramp_with_spot(shape=(60, 60))
    image[10, 10] = 5000.0  # one hot pixel
    plain = bg.rolling_ball_residual(image, 8.0, PIXEL_SIZE_UM)
    median = bg.rolling_ball_residual(image, 8.0, PIXEL_SIZE_UM, prefilter="median")
    smoothed = bg.rolling_ball_residual(image, 8.0, PIXEL_SIZE_UM, prefilter="gaussian")
    assert plain[10, 10] > 1000.0
    assert median[10, 10] < 100.0
    assert smoothed[10, 10] < plain[10, 10]
    with pytest.raises(ValueError):
        bg.rolling_ball_residual(image, 8.0, PIXEL_SIZE_UM, prefilter="box")


def test_rolling_ball_works_on_a_single_row_image():
    line = np.full((1, 40), 3.0, dtype=np.float32)
    line[0, 20] = 30.0
    background = bg.rolling_ball_background_px(line, 6.0)
    assert background.shape == (1, 40)
    assert float(background.max()) <= 30.0


def test_rolling_ball_estimate_is_biased_low_by_more_on_a_noisier_image():
    # The documented property: the block-minimum reduction puts the ball below the
    # true background, by a share of the noise, so the offset grows with the noise.
    level = 400.0
    offsets = []
    for sigma in (2.0, 16.0):
        field = np.random.default_rng(3).normal(level, sigma, (160, 160)).astype(np.float32)
        estimate = bg.rolling_ball_background(field, 8.0, PIXEL_SIZE_UM)
        offset = level - float(np.mean(estimate))
        assert offset > 0.0  # biased low, never high
        offsets.append(offset / sigma)
    assert offsets[1] > offsets[0] * 0.5  # a comparable share of the noise either way


def test_a_curved_background_inflates_the_ball_residual_noise_scale():
    # The documented property: the reduced and bilinearly enlarged ball background
    # cannot follow curvature, and the mismatch stays in its residual, so a noise
    # scale read from it runs above the noise of the image. A Gaussian residual,
    # which does follow smooth curvature, stays close to it.
    shape = (200, 200)
    rows, columns = np.mgrid[0 : shape[0], 0 : shape[1]]
    curved = 400.0 + 200.0 * np.exp(
        -(((rows - 60) ** 2 + (columns - 140) ** 2) / (2.0 * 50.0**2))
    )
    sigma = 12.0
    field = np.random.default_rng(5).normal(curved, sigma, shape).astype(np.float32)

    ball_residual = field - bg.rolling_ball_background(field, 8.0, PIXEL_SIZE_UM)
    _, ball_sigma = bg.robust_noise(ball_residual.ravel())
    gaussian_residual = bg.gaussian_residual(field, 8.0, PIXEL_SIZE_UM)
    _, gaussian_sigma = bg.robust_noise(np.asarray(gaussian_residual).ravel())

    assert gaussian_sigma == pytest.approx(sigma, rel=0.05)
    assert ball_sigma > gaussian_sigma
    # The two background models therefore do not give one shared number.
    assert ball_sigma > sigma


# --------------------------------------------------------------------------- local window

def test_local_background_window_is_odd_and_scales_with_pixel_size():
    assert bg.local_background_window_px(12.0, 0.5) == 25
    assert bg.local_background_window_px(12.0, 1.0) == 13
    assert bg.local_background_window_px(0.1, 1.0) == 3  # never below one pixel radius


# --------------------------------------------------------------------------- smooth models

def test_gaussian_background_and_residual():
    image = np.full((80, 80), 4.0, dtype=np.float32)
    image[disc((80, 80), (40, 40), 3.0)] = 40.0
    background = bg.gaussian_background(image, 10.0, PIXEL_SIZE_UM)
    assert float(background[0, 0]) == pytest.approx(4.0, abs=0.5)
    residual = bg.gaussian_residual(image, 10.0, PIXEL_SIZE_UM)
    assert float(residual[40, 40]) > 20.0
    # A non-positive radius removes the global median instead of a local background.
    global_only = bg.gaussian_residual(image, 0.0, PIXEL_SIZE_UM)
    assert float(global_only[0, 0]) == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize(
    "method", ["none", "top_hat", "gaussian", "dog", "flat_field", "multiscale"]
)
def test_background_correction_is_non_negative_and_keeps_a_small_object(method):
    rows, columns = np.mgrid[0:80, 0:80]
    image = (0.1 + 0.002 * columns + 0.001 * rows).astype(np.float32)
    image[disc((80, 80), (40, 40), 4.0)] += 0.6
    response = bg.background_correction(
        image, 8.0, PIXEL_SIZE_UM, method=method, strength=1.0
    )
    assert response.shape == image.shape
    assert float(response.min()) >= 0.0
    assert float(response[40, 40]) > float(response[5:20, 5:20].mean())


def test_background_correction_top_hat_flattens_a_ramp():
    rows, columns = np.mgrid[0:90, 0:90]
    image = (0.05 + 0.004 * columns + 0.004 * rows).astype(np.float32)
    image[disc((90, 90), (45, 45), 3.0)] += 0.5
    response = bg.background_correction(image, 6.0, PIXEL_SIZE_UM, method="top_hat")
    assert float(response[5:35, 5:35].max()) < 0.02
    assert float(response[45, 45]) > 0.4


def test_background_correction_off_switches():
    image = np.linspace(-1.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    for response in (
        bg.background_correction(image, 0.0, PIXEL_SIZE_UM),
        bg.background_correction(image, 5.0, PIXEL_SIZE_UM, strength=0.0),
        bg.background_correction(image, 5.0, PIXEL_SIZE_UM, method="none"),
    ):
        assert np.allclose(response, np.maximum(image, 0.0))
    with pytest.raises(ValueError):
        bg.background_correction(image, 5.0, PIXEL_SIZE_UM, method="wavelet")


def test_background_correction_wide_radius_uses_the_smooth_surrogate():
    rng = np.random.default_rng(1)
    image = (rng.random((120, 120)) * 0.2).astype(np.float32)
    exact = bg.background_correction(
        image, 20.0, PIXEL_SIZE_UM, method="top_hat", max_disk_radius_px=128
    )
    surrogate = bg.background_correction(
        image, 20.0, PIXEL_SIZE_UM, method="top_hat", max_disk_radius_px=4
    )
    assert exact.shape == surrogate.shape
    assert not np.allclose(exact, surrogate)


def test_robust_rescale_maps_to_the_unit_interval():
    rng = np.random.default_rng(2)
    response = (rng.random((50, 50)) * 7.0).astype(np.float32)
    response[0, 0] = 1000.0  # one extreme pixel must not compress the rest
    scaled = bg.robust_rescale(response)
    assert float(scaled.min()) >= 0.0
    assert float(scaled.max()) <= 1.0
    assert float(np.median(scaled)) > 0.2
    assert np.allclose(bg.robust_rescale(np.full((10, 10), 3.0, dtype=np.float32)), 0.0)


# --------------------------------------------------------------------------- compartment gate

def make_compartment_frame() -> tuple[np.ndarray, np.ndarray]:
    """A frame whose right half is one bright broad compartment, with an object in each half."""
    signal = np.full((120, 120), 10.0, dtype=np.float32)
    signal[:, 60:] += 120.0
    response = np.zeros((120, 120), dtype=np.float32)
    response[30:34, 20:24] = 8.0
    response[30:34, 90:94] = 8.0
    return response, signal


def test_compartment_gate_auto_removes_the_bright_compartment():
    response, signal = make_compartment_frame()
    gated = bg.compartment_gate(response, signal, 10.0, PIXEL_SIZE_UM, mode="auto")
    assert float(gated[30:34, 20:24].sum()) == pytest.approx(128.0)
    assert float(gated[30:34, 90:94].sum()) == 0.0


def test_compartment_gate_absolute_uses_the_given_ceiling():
    response, signal = make_compartment_frame()
    gated = bg.compartment_gate(
        response, signal, 10.0, PIXEL_SIZE_UM, mode="absolute", ceiling=50.0
    )
    assert float(gated[30:34, 20:24].sum()) == pytest.approx(128.0)
    assert float(gated[30:34, 90:94].sum()) == 0.0
    # A ceiling above every compartment keeps everything.
    kept = bg.compartment_gate(
        response, signal, 10.0, PIXEL_SIZE_UM, mode="absolute", ceiling=1e6
    )
    assert np.allclose(kept, response)


def test_compartment_gate_off_and_bad_arguments():
    response, signal = make_compartment_frame()
    assert np.allclose(
        bg.compartment_gate(response, signal, 10.0, PIXEL_SIZE_UM, mode="off"), response
    )
    with pytest.raises(ValueError):
        bg.compartment_gate(response, signal, 10.0, PIXEL_SIZE_UM, mode="absolute")
    with pytest.raises(ValueError):
        bg.compartment_gate(response, signal, 10.0, PIXEL_SIZE_UM, mode="otsu2")


def test_compartment_gate_auto_keeps_a_uniform_frame():
    response = np.zeros((40, 40), dtype=np.float32)
    response[10:14, 10:14] = 5.0
    signal = np.full((40, 40), 7.0, dtype=np.float32)
    gated = bg.compartment_gate(response, signal, 5.0, PIXEL_SIZE_UM, mode="auto")
    assert np.allclose(gated, response)


# --------------------------------------------------------------------------- noise

def test_robust_noise_recovers_a_known_scale():
    rng = np.random.default_rng(3)
    centre, sigma = bg.robust_noise(rng.normal(5.0, 2.0, 200_000))
    assert centre == pytest.approx(5.0, abs=0.02)
    assert sigma == pytest.approx(2.0, abs=0.03)


def test_robust_noise_never_returns_zero():
    _, sigma = bg.robust_noise(np.zeros(100), floor=1e-6)
    assert sigma == pytest.approx(1e-6)
    with pytest.raises(ValueError):
        bg.robust_noise(np.array([]))


def test_censored_noise_recovers_an_unclipped_normal():
    rng = np.random.default_rng(4)
    mu, sigma, clipped = bg.censored_noise(rng.normal(40.0, 3.0, 200_000))
    assert mu == pytest.approx(40.0, abs=0.1)
    assert sigma == pytest.approx(3.0, abs=0.1)
    assert clipped == 0.0


def test_censored_noise_recovers_a_clipped_normal():
    rng = np.random.default_rng(5)
    values = np.clip(rng.normal(1.0, 3.0, 200_000), 0.0, None)
    mu, sigma, clipped = bg.censored_noise(values)
    # The MAD of this sample is useless; the two-quantile fit still finds the truth.
    assert mu == pytest.approx(1.0, abs=0.15)
    assert sigma == pytest.approx(3.0, abs=0.15)
    assert clipped == pytest.approx(0.43, abs=0.03)


def test_censored_noise_heavy_clipping_branch():
    rng = np.random.default_rng(6)
    values = np.clip(rng.normal(-3.0, 3.0, 200_000), 0.0, None)
    mu, sigma, clipped = bg.censored_noise(values)
    assert clipped > 0.6
    assert mu == pytest.approx(-3.0, abs=0.3)
    assert sigma == pytest.approx(3.0, abs=0.3)


def test_censored_noise_respects_the_sigma_floor_and_rejects_empty_input():
    constant = np.zeros(1000)
    _, sigma, clipped = bg.censored_noise(constant)
    assert sigma == pytest.approx(0.5)
    assert clipped == 1.0
    assert bg.censored_noise(constant, sigma_floor=2.0)[1] == pytest.approx(2.0)
    with pytest.raises(ValueError):
        bg.censored_noise(np.array([]))


def test_noise_threshold_takes_the_larger_of_the_rule_and_the_floor():
    assert bg.noise_threshold(2.0, 1.0, k=4.0) == pytest.approx(6.0)
    assert bg.noise_threshold(2.0, 1.0, k=4.0, floor=20.0) == pytest.approx(20.0)
    assert bg.noise_threshold(2.0, 10.0, k=4.0, floor=20.0) == pytest.approx(42.0)


def test_censored_noise_threshold_uses_only_the_masked_pixels():
    rng = np.random.default_rng(7)
    residual = rng.normal(0.0, 1.0, (200, 200)).astype(np.float32)
    residual[:, 100:] += 500.0  # outside the region of interest
    mask = np.zeros((200, 200), dtype=bool)
    mask[:, :100] = True
    threshold, mu, sigma, clipped = bg.censored_noise_threshold(
        residual, mask=mask, k=4.0, sigma_floor=0.0
    )
    assert mu == pytest.approx(0.0, abs=0.1)
    assert sigma == pytest.approx(1.0, abs=0.1)
    assert threshold == pytest.approx(mu + 4.0 * sigma)
    assert clipped == pytest.approx(0.69, abs=0.03)
    whole_frame = bg.censored_noise_threshold(residual, k=4.0, sigma_floor=0.0)[0]
    assert whole_frame > threshold


def test_censored_noise_threshold_floor_wins_on_a_clipped_channel():
    rng = np.random.default_rng(8)
    residual = np.clip(rng.normal(0.0, 0.9, (150, 150)), 0.0, None).astype(np.float32)
    threshold, _mu, _sigma, _clipped = bg.censored_noise_threshold(
        residual, k=4.0, floor=25.0
    )
    assert threshold == pytest.approx(25.0)


# --------------------------------------------------------------------------- shared scale

def test_estimate_shared_noise_scale_recovers_a_known_sigma():
    rng = np.random.default_rng(9)
    images = [rng.normal(0.0, 2.5, (200, 200)).astype(np.float32) for _ in range(3)]
    scale = bg.estimate_shared_noise_scale(images, 10.0, PIXEL_SIZE_UM)
    assert scale == pytest.approx(2.5, rel=0.1)


def test_estimate_shared_noise_scale_ignores_frame_brightness():
    # The local residual removes a uniform offset, so frames of very different
    # brightness but equal noise must give the same shared scale.
    rng = np.random.default_rng(10)
    images = [
        (rng.normal(0.0, 1.5, (150, 150)) + offset).astype(np.float32)
        for offset in (0.0, 50.0, 200.0)
    ]
    scale = bg.estimate_shared_noise_scale(images, 10.0, PIXEL_SIZE_UM)
    assert scale == pytest.approx(1.5, rel=0.1)
    single = bg.estimate_shared_noise_scale(images[:1], 10.0, PIXEL_SIZE_UM)
    assert scale == pytest.approx(single, rel=0.1)


def test_estimate_shared_noise_scale_refuses_unusable_input():
    with pytest.raises(ValueError):
        bg.estimate_shared_noise_scale([], 10.0, PIXEL_SIZE_UM)
    with pytest.raises(ValueError):
        bg.estimate_shared_noise_scale(
            [np.full((40, 40), 5.0, dtype=np.float32)], 10.0, PIXEL_SIZE_UM
        )
