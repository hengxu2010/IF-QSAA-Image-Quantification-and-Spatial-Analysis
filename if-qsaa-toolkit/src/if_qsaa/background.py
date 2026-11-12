"""Background estimation, background correction and background-noise statistics.

The module covers the background models the method uses before any object is
segmented, plus the noise statistics that turn a background into a threshold:

* a rolling-ball (Sternberg) background with the ImageJ shrink/enlarge scheme,
* top-hat, Gaussian, difference-of-Gaussians, flat-field and multiscale corrections,
* a broad-background compartment gate that excludes large bright compartments,
* censored background-noise estimation and the threshold rule
  ``background mean + k * noise sigma``, optionally raised to an absolute floor.

The rolling-ball background affects threshold selection in two ways. Its background
estimate is biased low by an amount that grows with the
noise of the image, so an absolute threshold applied to a ball residual is not
comparable between images of different noise levels; and on a curved background a
noise scale read from a ball residual runs above the noise of the image, because the
reduced and bilinearly enlarged background cannot follow the curvature and the
mismatch stays in the residual. The ``k`` of ``background mean + k * noise sigma``
therefore belongs to one residual model: keep the scale and the thresholded plane on
the same model, and prefer :func:`estimate_shared_noise_scale`, which pools Gaussian
residuals, wherever a Gaussian residual is usable.
:func:`rolling_ball_background_px`, :func:`rolling_ball_residual` and
:func:`noise_threshold` give the details.

Every spatial parameter of the public API is a physical length in micrometres or an
area in square micrometres and is converted to pixels with an explicit
``pixel_size_um``. All functions are pure: they take arrays and return arrays or
plain numbers, they never read or write files and they hold no state.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
from scipy import ndimage as ndi
from scipy import stats

from ._util import as_float, check_2d, check_same_shape, disk, mad, um_to_px

__all__ = [
    "finite_sample",
    "rolling_ball_shrink_factor",
    "rolling_ball_background_px",
    "rolling_ball_background",
    "rolling_ball_residual",
    "local_background_window_px",
    "gaussian_background",
    "gaussian_residual",
    "background_correction",
    "robust_rescale",
    "broad_background",
    "compartment_gate",
    "robust_noise",
    "censored_noise",
    "noise_threshold",
    "censored_noise_threshold",
    "estimate_shared_noise_scale",
]


# --------------------------------------------------------------------------- sampling

def finite_sample(image: np.ndarray, max_pixels: int = 1_000_000) -> np.ndarray:
    """Return the finite values of a strided subsample of an image.

    Robust statistics (median, MAD, percentiles) are read from a subsample so that
    their cost does not grow with the frame size. The subsample keeps every
    ``stride``-th row and column, with ``stride = ceil(sqrt(N / max_pixels))``, which
    preserves the spatial spread of the frame instead of favouring one corner.

    Parameters
    ----------
    image : ndarray
        Image of any shape. Non-finite values are dropped.
    max_pixels : int, optional
        Target size of the subsample in pixels (dimensionless count).

    Returns
    -------
    ndarray
        1-D ``float64`` array of finite values; ``[0.0]`` when nothing is finite.
    """
    arr = np.asarray(image)
    height, width = arr.shape[:2]
    count = int(height) * int(width)
    if count <= int(max_pixels):
        sampled = arr
    else:
        stride = max(1, int(math.ceil(math.sqrt(count / float(max_pixels)))))
        sampled = arr[::stride, ::stride]
    flat = np.asarray(sampled, dtype=np.float64).ravel()
    finite = flat[np.isfinite(flat)]
    return finite if finite.size else np.asarray([0.0], dtype=np.float64)


def _radius_px(length_um: float, pixel_size_um: float, *, minimum: int = 0) -> int:
    """Round a length in micrometres to an integer pixel radius.

    Rounds with ``floor(x + 0.5)``, so a parameter expressed in micrometres gives
    the same pixel radius at any working resolution that is consistent with
    ``pixel_size_um``. A non-positive length stays 0 (the step is off).
    """
    if float(length_um) <= 0.0:
        return 0
    value = int(math.floor(um_to_px(length_um, pixel_size_um) + 0.5))
    return max(int(minimum), value)


# --------------------------------------------------------------------------- rolling ball

def rolling_ball_shrink_factor(radius_px: float) -> int:
    """Shrink factor of the ImageJ rolling-ball scheme for a ball radius in pixels.

    The ball is rolled on an image reduced by this factor, which is the step that
    makes a large ball affordable. The schedule is the ImageJ one: 1 up to 10 px,
    2 up to 30 px, 4 up to 100 px and 8 above that.

    Parameters
    ----------
    radius_px : float
        Ball radius in pixels.

    Returns
    -------
    int
        Shrink factor (dimensionless, one of 1, 2, 4, 8).
    """
    r = float(radius_px)
    if r <= 10.0:
        return 1
    if r <= 30.0:
        return 2
    if r <= 100.0:
        return 4
    return 8


def _ball_structure(radius_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Height profile and footprint of the upper half of a ball of radius ``radius_px``."""
    radius = float(radius_px)
    extent = int(math.ceil(radius))
    y, x = np.ogrid[-extent : extent + 1, -extent : extent + 1]
    squared = (x * x + y * y).astype(np.float64)
    height = np.sqrt(np.clip(radius * radius - squared, 0.0, None))
    footprint = np.sqrt(squared) <= radius
    # The apex of the ball sits at the centre, so subtracting the centre height makes
    # a grey erosion by this profile return the apex level directly.
    structure = height - float(height[extent, extent])
    structure[~footprint] = 0.0
    return structure, footprint


def _roll_ball(image: np.ndarray, radius_px: float) -> np.ndarray:
    """Roll a ball of radius ``radius_px`` under the image surface and return its apex.

    One grey erosion by the ball's height profile, with the outside of the frame
    treated as infinitely high so the ball is not pulled down at the border.
    """
    structure, footprint = _ball_structure(radius_px)
    return ndi.grey_erosion(
        image.astype(np.float32, copy=False),
        footprint=footprint,
        structure=structure.astype(np.float32),
        mode="constant",
        cval=np.inf,
    )


def _enlarge_bilinear(small: np.ndarray, shape: tuple[int, int], shrink: int) -> np.ndarray:
    """Bilinear enlargement by an integer factor, half-pixel-centre convention."""
    height, width = shape
    rows = (np.arange(height, dtype=np.float64) + 0.5) / float(shrink) - 0.5
    columns = (np.arange(width, dtype=np.float64) + 0.5) / float(shrink) - 0.5
    grid_y, grid_x = np.meshgrid(rows, columns, indexing="ij")
    return ndi.map_coordinates(
        small.astype(np.float32, copy=False),
        np.stack((grid_y, grid_x)),
        order=1,
        mode="nearest",
    ).astype(np.float32, copy=False)


def rolling_ball_background_px(
    image: np.ndarray,
    radius_px: float,
    *,
    shrink: int | None = None,
    smooth: bool = True,
) -> np.ndarray:
    """ImageJ-style rolling-ball background of a 2-D image, radius in pixels.

    Steps, in order: a 3x3 mean filter that keeps single noisy pixels from pushing
    the ball down; reduction by the shrink factor using the block minimum (the image
    is first edge-padded to a whole number of blocks); the ball rolled under the
    reduced surface with radius ``radius_px / shrink``; bilinear enlargement back to
    the padded size and a crop to the input size; and finally a pixel-wise minimum
    with the input, so the background never exceeds the image.

    Parameters
    ----------
    image : ndarray
        2-D image in the grey levels of the input.
    radius_px : float
        Ball radius in pixels. Choose it larger than the objects to be kept.
    shrink : int, optional
        Override the shrink factor (dimensionless). ``None`` uses
        :func:`rolling_ball_shrink_factor`.
    smooth : bool, optional
        Apply the 3x3 mean filter before rolling the ball (ImageJ default).

    Returns
    -------
    ndarray
        ``float32`` background in the grey levels of the input, same shape as ``image``.

    Notes
    -----
    This scheme estimates the background low, and by more on a noisier image. The
    reduction takes the block minimum, so the ball is rolled under a surface that is
    already the lower envelope of the noise, and the final pixel-wise minimum with the
    input can only lower the result further. The offset scales with the image noise
    standard deviation and is not corrected by this function.

    What follows for the threshold matters more than the offset itself. Reading the
    background mean and the noise scale from the residual, as
    :func:`censored_noise_threshold` does, is self-consistent, because the ``mu`` it
    estimates sits in the same residual and absorbs the offset. An absolute number
    applied to the same residual is not: a fixed ``floor`` in :func:`noise_threshold`,
    or a fixed ``value`` passed to
    :func:`if_qsaa.segmentation.threshold_mask`, stands at a different distance above
    the true background in a quiet frame and in a noisy one, so it is not comparable
    across images whose noise levels differ. Keep such a number inside one batch
    imaged under one set of conditions, or state the threshold through the residual's
    own background mean and noise scale instead.
    """
    work = as_float(check_2d(image), np.float32)
    radius = float(radius_px)
    if radius <= 0.0:
        raise ValueError("radius_px must be positive")
    factor = int(shrink) if shrink is not None else rolling_ball_shrink_factor(radius)
    if factor < 1:
        raise ValueError("shrink must be >= 1")

    smoothed = ndi.uniform_filter(work, 3) if smooth else work
    height, width = smoothed.shape
    padded_height = -(-height // factor) * factor
    padded_width = -(-width // factor) * factor
    padded = np.pad(
        smoothed,
        ((0, padded_height - height), (0, padded_width - width)),
        mode="edge",
    )
    reduced = padded.reshape(
        padded_height // factor, factor, padded_width // factor, factor
    ).min(axis=(1, 3))
    reduced_background = _roll_ball(reduced, radius / factor)
    background = _enlarge_bilinear(
        reduced_background, (padded_height, padded_width), factor
    )[:height, :width]
    return np.minimum(background, work)


def rolling_ball_background(
    image: np.ndarray,
    radius_um: float,
    pixel_size_um: float,
    *,
    shrink: int | None = None,
    smooth: bool = True,
) -> np.ndarray:
    """Rolling-ball background with the ball radius given in micrometres.

    Parameters
    ----------
    image : ndarray
        2-D image in the grey levels of the input.
    radius_um : float
        Ball radius in micrometres.
    pixel_size_um : float
        Pixel size in micrometres per pixel (square pixels assumed).
    shrink : int, optional
        Override the ImageJ shrink factor (dimensionless).
    smooth : bool, optional
        Apply the 3x3 mean filter before rolling the ball.

    Returns
    -------
    ndarray
        ``float32`` background in the grey levels of the input.

    Notes
    -----
    The estimate is biased low by an amount that grows with the noise of the image;
    see :func:`rolling_ball_background_px` for what that means for a threshold.
    """
    return rolling_ball_background_px(
        image,
        um_to_px(radius_um, pixel_size_um),
        shrink=shrink,
        smooth=smooth,
    )


def rolling_ball_residual(
    image: np.ndarray,
    radius_um: float,
    pixel_size_um: float,
    *,
    prefilter: str | None = None,
    median_size_px: int = 3,
    gaussian_sigma_px: float = 1.0,
    shrink: int | None = None,
) -> np.ndarray:
    """Background-subtracted residual of one channel: filtered image minus its ball background.

    The optional pre-filter is the per-channel smoothing the method applies before
    the ball: a small median filter removes isolated hot pixels from a punctate
    readout channel, a small Gaussian removes pixel noise from a diffuse marker
    channel, and ``None`` leaves the channel as it is.

    Parameters
    ----------
    image : ndarray
        2-D channel in the grey levels of the input.
    radius_um : float
        Rolling-ball radius in micrometres.
    pixel_size_um : float
        Pixel size in micrometres per pixel.
    prefilter : {None, 'median', 'gaussian'}, optional
        Pre-filter applied before the ball.
    median_size_px : int, optional
        Side of the square median window in pixels, used when ``prefilter='median'``.
    gaussian_sigma_px : float, optional
        Gaussian standard deviation in pixels, used when ``prefilter='gaussian'``.
    shrink : int, optional
        Override the ImageJ shrink factor (dimensionless).

    Returns
    -------
    ndarray
        ``float32`` residual in the grey levels of the input (zero or negative where
        the image sits at its local background).

    Notes
    -----
    A noise scale read from this residual can run above the noise of the image
    itself, and by more the more the background curves. The ball is rolled on a
    reduced copy and the result is enlarged bilinearly, so the estimated background is
    a piecewise-bilinear surface: it follows a flat or evenly sloping background
    closely, but it cannot follow curvature, and the mismatch is left in the residual
    on top of the noise. The residual of :func:`gaussian_residual` does not carry that
    term, because a Gaussian background follows smooth curvature. The two supported
    background models therefore give different scales for the same images, and the
    difference is a property of the models rather than of a particular frame.

    The ``k`` of :func:`noise_threshold` is a number of standard deviations of one
    particular residual, so the scale and the plane that is thresholded have to come
    from the same background model; a ``k`` carried from one model to the other is a
    different criterion. Where a Gaussian residual is usable, prefer
    :func:`estimate_shared_noise_scale`, which pools Gaussian residuals and so stays
    closer to the noise of the images. Where the ball is the model being thresholded,
    read the scale from this residual and calibrate ``k`` against it rather than
    reusing a ``k`` chosen elsewhere. The low bias of the ball background itself is
    described in :func:`rolling_ball_background_px`.
    """
    work = as_float(check_2d(image), np.float32)
    if prefilter is None or prefilter == "none":
        filtered = work
    elif prefilter == "median":
        filtered = ndi.median_filter(work, size=int(median_size_px), mode="reflect")
    elif prefilter == "gaussian":
        filtered = ndi.gaussian_filter(work, float(gaussian_sigma_px), mode="reflect")
    else:
        raise ValueError("prefilter must be None, 'median' or 'gaussian'")
    background = rolling_ball_background(
        filtered, radius_um, pixel_size_um, shrink=shrink
    )
    return (filtered - background).astype(np.float32, copy=False)


# --------------------------------------------------------------------------- local window

def local_background_window_px(window_um: float, pixel_size_um: float) -> int:
    """Side of the sliding window in pixels for a window given in micrometres.

    The window is kept odd and centred: the radius is the window side in pixels
    halved and rounded, and the side is ``2 * radius + 1``.

    Parameters
    ----------
    window_um : float
        Nominal window side in micrometres.
    pixel_size_um : float
        Pixel size in micrometres per pixel.

    Returns
    -------
    int
        Odd window side in pixels (at least 3).
    """
    radius = int(round(um_to_px(window_um, pixel_size_um) / 2.0))
    return 2 * max(1, radius) + 1


# --------------------------------------------------------------------------- smooth models

def gaussian_background(
    image: np.ndarray,
    radius_um: float,
    pixel_size_um: float,
    *,
    sigma_factor: float = 0.55,
) -> np.ndarray:
    """Low-pass (Gaussian) background at a given background scale.

    The Gaussian standard deviation is ``max(1 px, sigma_factor * radius_px)``, so the
    background scale is named by the same radius as the morphological models.

    Parameters
    ----------
    image : ndarray
        2-D image in the grey levels of the input.
    radius_um : float
        Background scale in micrometres.
    pixel_size_um : float
        Pixel size in micrometres per pixel.
    sigma_factor : float, optional
        Ratio of the Gaussian standard deviation to the background radius
        (dimensionless).

    Returns
    -------
    ndarray
        ``float32`` background in the grey levels of the input.
    """
    work = as_float(check_2d(image), np.float32)
    radius = _radius_px(radius_um, pixel_size_um, minimum=1)
    sigma = max(1.0, float(radius) * float(sigma_factor))
    return ndi.gaussian_filter(work, sigma, mode="reflect").astype(
        np.float32, copy=False
    )


def gaussian_residual(
    image: np.ndarray,
    radius_um: float,
    pixel_size_um: float,
    *,
    sigma_factor: float = 0.55,
) -> np.ndarray:
    """Image minus its Gaussian background (a local high-pass residual).

    Parameters
    ----------
    image : ndarray
        2-D image in the grey levels of the input.
    radius_um : float
        Background scale in micrometres. Non-positive removes only the global
        median instead.
    pixel_size_um : float
        Pixel size in micrometres per pixel.
    sigma_factor : float, optional
        Ratio of the Gaussian standard deviation to the background radius.

    Returns
    -------
    ndarray
        ``float32`` residual in the grey levels of the input.
    """
    work = as_float(check_2d(image), np.float32)
    if float(radius_um) <= 0.0:
        return (work - np.float32(np.median(finite_sample(work)))).astype(
            np.float32, copy=False
        )
    background = gaussian_background(
        work, radius_um, pixel_size_um, sigma_factor=sigma_factor
    )
    return (work - background).astype(np.float32, copy=False)


def background_correction(
    image: np.ndarray,
    radius_um: float,
    pixel_size_um: float,
    *,
    method: str = "gaussian",
    strength: float = 1.0,
    max_disk_radius_px: int = 128,
    flat_field_epsilon: float = 0.02,
) -> np.ndarray:
    """Background-corrected response of an image, clipped at zero.

    All methods share one background radius and one strength, and all return a
    non-negative response in the units of the input.

    ``'none'``
        The input, clipped at zero.
    ``'top_hat'``
        White top-hat: the input minus ``strength`` times its grey opening by a flat
        disk of the background radius. Keeps objects smaller than the disk.
    ``'gaussian'``
        The input minus ``strength`` times its Gaussian background.
    ``'dog'``
        Difference of Gaussians: a fine Gaussian of standard deviation
        ``clip(0.08 * radius_px, 0.5, 4)`` px minus ``strength`` times the coarse
        Gaussian background.
    ``'flat_field'``
        Multiplicative correction ``(image + eps) / (background + eps) - 1``, scaled
        by ``strength``. Expects an image normalised to the unit interval, which is
        the scale the epsilon refers to.
    ``'multiscale'``
        Pixel-wise maximum of the ``'top_hat'`` and ``'gaussian'`` responses, which
        keeps both small and broad objects.

    A disk wider than ``max_disk_radius_px`` is replaced by a Gaussian of standard
    deviation ``0.50 * radius_px`` (``0.32 * radius_px`` for the fine scale of
    ``'multiscale'``), because an exact disk of that size costs far more than the
    difference it makes.

    Parameters
    ----------
    image : ndarray
        2-D image. ``'flat_field'`` expects the unit interval; the other methods
        work in any intensity units.
    radius_um : float
        Background scale in micrometres. Non-positive returns the clipped input.
    pixel_size_um : float
        Pixel size in micrometres per pixel.
    method : {'none', 'top_hat', 'gaussian', 'dog', 'flat_field', 'multiscale'}, optional
        Background model.
    strength : float, optional
        Fraction of the background that is subtracted (dimensionless). Non-positive
        returns the clipped input.
    max_disk_radius_px : int, optional
        Largest exact disk radius in pixels; wider radii use the Gaussian surrogate.
    flat_field_epsilon : float, optional
        Stabiliser of the flat-field ratio, in the units of the input.

    Returns
    -------
    ndarray
        ``float32`` non-negative response in the units of the input.
    """
    work = as_float(check_2d(image), np.float32)
    radius = _radius_px(radius_um, pixel_size_um, minimum=1)
    if method == "none" or radius <= 0 or float(strength) <= 0.0:
        return np.maximum(work, np.float32(0.0))
    gain = np.float32(strength)

    def _disk_opening(fallback_factor: float) -> np.ndarray:
        if radius <= int(max_disk_radius_px):
            return ndi.grey_opening(
                work, footprint=disk(radius), mode="reflect"
            ).astype(np.float32, copy=False)
        sigma = max(1.0, float(radius) * float(fallback_factor))
        return ndi.gaussian_filter(work, sigma, mode="reflect").astype(
            np.float32, copy=False
        )

    if method == "top_hat":
        return np.maximum(work - gain * _disk_opening(0.50), np.float32(0.0))

    coarse = gaussian_background(work, radius_um, pixel_size_um)
    if method == "gaussian":
        return np.maximum(work - gain * coarse, np.float32(0.0))
    if method == "dog":
        fine_sigma = max(0.5, min(4.0, float(radius) * 0.08))
        fine = ndi.gaussian_filter(work, fine_sigma, mode="reflect")
        return np.maximum(fine.astype(np.float32) - gain * coarse, np.float32(0.0))
    if method == "flat_field":
        epsilon = np.float32(flat_field_epsilon)
        ratio = (work + epsilon) / (coarse + epsilon)
        response = np.maximum(ratio - np.float32(1.0), np.float32(0.0))
        if float(strength) != 1.0:
            response = response * gain
        return response.astype(np.float32, copy=False)
    if method == "multiscale":
        fine = np.maximum(work - gain * _disk_opening(0.32), np.float32(0.0))
        broad = np.maximum(work - gain * coarse, np.float32(0.0))
        return np.maximum(fine, broad)
    raise ValueError(f"unknown background method: {method!r}")


def robust_rescale(response: np.ndarray, *, upper_percentile: float = 99.9) -> np.ndarray:
    """Rescale a non-negative response to the unit interval on robust limits.

    The lower limit is the sample minimum clipped at zero and the upper limit is the
    given upper percentile of the sample, so a few extreme pixels cannot compress the
    rest of the response. The result is clipped to ``[0, 1]``.

    Parameters
    ----------
    response : ndarray
        Response image in any intensity units.
    upper_percentile : float, optional
        Upper percentile used as the top of the scale, 0 to 100.

    Returns
    -------
    ndarray
        ``float32`` image in the unit interval (dimensionless).
    """
    work = as_float(response, np.float32)
    sample = finite_sample(work)
    low, high = (float(v) for v in np.percentile(sample, (0.0, float(upper_percentile))))
    low = max(0.0, low)
    if not math.isfinite(high) or high <= low:
        high = float(sample.max(initial=1.0))
    if high <= low:
        return np.zeros(work.shape, dtype=np.float32)
    scaled = (work - np.float32(low)) / np.float32(high - low)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32, copy=False)


# --------------------------------------------------------------------------- compartment gate

def broad_background(
    signal: np.ndarray,
    scale_um: float,
    pixel_size_um: float,
) -> np.ndarray:
    """Background at the scale of a whole compartment, not of a single object.

    A Gaussian of standard deviation ``max(2 px, 1.4 * scale_px)``: wide enough that
    an individual object does not raise it, so what it measures is the brightness of
    the compartment an object sits in.

    Parameters
    ----------
    signal : ndarray
        2-D image in the grey levels of the input.
    scale_um : float
        Compartment scale in micrometres.
    pixel_size_um : float
        Pixel size in micrometres per pixel.

    Returns
    -------
    ndarray
        ``float32`` broad background in the grey levels of the input.
    """
    work = as_float(check_2d(signal), np.float32)
    scale = _radius_px(scale_um, pixel_size_um, minimum=1)
    sigma = max(2.0, float(scale) * 1.4)
    return ndi.gaussian_filter(work, sigma, mode="reflect").astype(
        np.float32, copy=False
    )


def _otsu_level_8bit(values: np.ndarray) -> int:
    """Otsu split of an 8-bit histogram; returns the level ``t`` for ``value <= t``."""
    counts = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return 0
    levels = np.arange(256, dtype=np.float64)
    weight_low = np.cumsum(counts)[:-1]
    weight_high = total - weight_low
    sum_low = np.cumsum(counts * levels)[:-1]
    sum_all = float((counts * levels).sum())
    valid = (weight_low > 0) & (weight_high > 0)
    if not bool(valid.any()):
        return 0
    mean_low = np.divide(sum_low, weight_low, out=np.zeros_like(sum_low), where=valid)
    mean_high = np.divide(
        sum_all - sum_low, weight_high, out=np.zeros_like(sum_low), where=valid
    )
    between = weight_low * weight_high * (mean_low - mean_high) ** 2
    between[~valid] = -1.0
    return int(np.argmax(between))


def compartment_gate(
    response: np.ndarray,
    signal: np.ndarray,
    scale_um: float,
    pixel_size_um: float,
    *,
    mode: str = "auto",
    ceiling: float | None = None,
) -> np.ndarray:
    """Zero the response inside bright broad-background compartments.

    Broad glow belongs to a compartment (tissue folds, thickened edges, strongly
    autofluorescent regions), not to any object inside it, so the gate judges a pixel
    by the brightness of its compartment and never by how it ranks in its own frame.

    ``'off'``
        Return the response unchanged.
    ``'absolute'``
        Keep pixels whose broad background is at or below ``ceiling``, an absolute
        grey level shared by a whole batch. A non-positive ceiling leaves the
        response unchanged.
    ``'auto'``
        Split this frame's broad background into two modes with Otsu (on the
        background quantised to 8 bits between its 0.5th and 99.5th percentiles) and
        keep the lower mode. A frame with no bright compartment yields a cut at its
        own extreme, so almost nothing is excluded; a frame with no dynamic range in
        its background is left unchanged.

    Parameters
    ----------
    response : ndarray
        2-D response to be gated, in any intensity units.
    signal : ndarray
        2-D image the broad background is measured on, in grey levels, same shape as
        ``response``.
    scale_um : float
        Compartment scale in micrometres.
    pixel_size_um : float
        Pixel size in micrometres per pixel.
    mode : {'off', 'absolute', 'auto'}, optional
        Gate rule.
    ceiling : float, optional
        Absolute ceiling on the broad background, in the grey levels of ``signal``.
        Required by ``mode='absolute'``; it is a property of a batch's staining and
        exposure and has no generic value.

    Returns
    -------
    ndarray
        ``float32`` gated response, zero where the compartment is too bright.
    """
    work = as_float(check_2d(response), np.float32)
    plane = as_float(check_2d(signal), np.float32)
    check_same_shape(work, plane)
    if mode == "off":
        return work.copy()
    if mode not in {"absolute", "auto"}:
        raise ValueError("mode must be 'off', 'absolute' or 'auto'")
    broad = broad_background(plane, scale_um, pixel_size_um)
    if mode == "absolute":
        if ceiling is None:
            raise ValueError("mode='absolute' needs an explicit ceiling in grey levels")
        if float(ceiling) <= 0.0:
            return work.copy()
        allowed = broad <= np.float32(ceiling)
    else:
        sample = finite_sample(broad, max_pixels=400_000)
        low, high = (float(v) for v in np.percentile(sample, (0.5, 99.5)))
        span = high - low
        if not math.isfinite(span) or span <= 0.0:
            return work.copy()
        quantised = np.clip((broad - np.float32(low)) / np.float32(span), 0.0, 1.0)
        level = _otsu_level_8bit((quantised * 255.0).astype(np.uint8))
        allowed = broad <= np.float32(low + (level / 255.0) * span)
    if not bool(np.any(allowed)):
        return np.zeros(work.shape, dtype=np.float32)
    gated = work.copy()
    gated[~allowed] = np.float32(0.0)
    return gated


# --------------------------------------------------------------------------- noise and threshold

def robust_noise(values: Iterable[float], *, floor: float = 1e-6) -> tuple[float, float]:
    """Median and MAD-based noise scale of a set of values.

    The scale is ``1.4826 * MAD``, which estimates the standard deviation of a normal
    distribution. Use it on data whose background is not clipped; on clipped data the
    MAD can be exactly zero, and :func:`censored_noise` is the right estimator.

    Parameters
    ----------
    values : array_like
        Values in the grey levels of the input (for example the background residual
        inside a tissue mask).
    floor : float, optional
        Smallest scale returned, in the same grey levels, so the scale is never zero.

    Returns
    -------
    centre : float
        Median of the values, in grey levels.
    sigma : float
        Noise scale in grey levels, at least ``floor``.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("robust_noise needs at least one finite value")
    centre = float(np.median(arr))
    sigma = mad(arr, centre=centre)
    if not math.isfinite(sigma):
        sigma = 0.0
    return centre, max(float(sigma), float(floor))


def censored_noise(
    values: np.ndarray,
    *,
    clip_level: float = 0.5,
    sigma_floor: float = 0.5,
) -> tuple[float, float, float]:
    """Background location and noise scale when the dark side of the histogram is clipped.

    A normal distribution is fitted to two quantiles of the *unclipped lower part* of
    the distribution, which is the part that still describes background only. Writing
    ``p0`` for the clipped fraction, the two quantiles are
    ``q1 = max(0.25, p0 + 0.05)`` and ``q2 = min(q1 + 0.25, 0.85)``; when more than
    60 % of the values are clipped, the first 12 % above the clip is used instead
    (``q1 = p0 + 0.02``, ``q2 = min(p0 + 0.12, 0.95)``). With ``x1, x2`` the empirical
    quantiles and ``z1, z2`` the standard-normal quantiles,
    ``sigma = max((x2 - x1) / (z2 - z1), sigma_floor)`` and ``mu = x1 - z1 * sigma``.

    Using only the lower part keeps the estimate stable when the signal covers a large
    share of the frame, and fitting quantiles rather than taking a MAD keeps it usable
    when the background is clipped to a constant and the MAD is zero.

    Parameters
    ----------
    values : ndarray
        Background-residual values in the grey levels of the input, for example the
        residual inside a tissue mask. Shape is irrelevant; the array is flattened.
    clip_level : float, optional
        Values at or below this level count as clipped, in grey levels. Half a grey
        level is the natural cut for integer-valued images.
    sigma_floor : float, optional
        Smallest noise scale returned, in grey levels. Half a grey level is the
        quantisation limit of an integer-valued image.

    Returns
    -------
    mu : float
        Background mean in grey levels.
    sigma : float
        Noise scale (one standard deviation) in grey levels.
    clipped_fraction : float
        Fraction of values at or below ``clip_level`` (dimensionless, 0 to 1).

    Notes
    -----
    When so much of the distribution is clipped that the two quantiles coincide,
    there is no observable spread left: the location is read from the upper quantile
    and the scale is ``sigma_floor``.

    ``clipped_fraction`` is a plain count of values at or below ``clip_level`` and
    reads differently depending on what was passed in. On a raw integer image whose
    dark side is truncated it does measure clipped data, and that is the case the
    estimator exists for. On a background residual, which is a float plane centred
    near zero, roughly half the background sits below any small positive
    ``clip_level``, so a large fraction is the ordinary lower half of the background
    distribution and says nothing about the detector. Its effect is on the estimator,
    not on the data: it selects which quantiles are fitted, as described above. Read
    it as "how much of the input the fit had to skip", and compare it only between
    inputs of the same kind.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("censored_noise needs at least one finite value")
    clipped_fraction = float(np.mean(arr <= float(clip_level)))
    if clipped_fraction > 0.6:
        q1 = clipped_fraction + 0.02
        q2 = min(clipped_fraction + 0.12, 0.95)
    else:
        q1 = max(0.25, clipped_fraction + 0.05)
        q2 = min(q1 + 0.25, 0.85)
    q1 = min(q1, 1.0)
    q2 = min(q2, 1.0)
    if q2 <= q1:
        return float(np.quantile(arr, q2)), float(sigma_floor), clipped_fraction
    x1, x2 = (float(v) for v in np.quantile(arr, [q1, q2]))
    z1, z2 = (float(v) for v in stats.norm.ppf([q1, q2]))
    sigma = max((x2 - x1) / (z2 - z1), float(sigma_floor))
    return x1 - z1 * sigma, float(sigma), clipped_fraction


def noise_threshold(
    background_mean: float,
    noise_sigma: float,
    *,
    k: float = 4.0,
    floor: float = 0.0,
) -> float:
    """Threshold rule ``max(background mean + k * noise sigma, floor)``.

    The noise criterion is the rule of the method; the absolute floor is the optional
    second condition that keeps a threshold meaningful when the noise criterion
    collapses, for instance on a channel whose background is clipped. A floor is a
    property of one batch's images and has no generic value, so it is off by default.

    Parameters
    ----------
    background_mean : float
        Background mean in grey levels.
    noise_sigma : float
        Noise scale (one standard deviation) in grey levels.
    k : float, optional
        Number of noise standard deviations above the background (dimensionless).
    floor : float, optional
        Absolute lower bound on the threshold, in grey levels. 0 switches it off.

    Returns
    -------
    float
        Threshold in grey levels.

    Notes
    -----
    ``k`` counts standard deviations of whichever residual ``noise_sigma`` was
    measured on, so the scale and the plane the threshold is applied to must come from
    the same background model. The two models supported here do not agree: on a curved
    background a scale read from a rolling-ball residual runs above the noise of the
    image, because the reduced and bilinearly enlarged background cannot follow the
    curvature and the mismatch stays in the residual, while a Gaussian residual of the
    same images does not carry that term. Where a Gaussian residual is usable, prefer
    :func:`estimate_shared_noise_scale` for the scale; otherwise calibrate ``k``
    against the ball residual itself and keep that pairing. The ``floor`` is an
    absolute grey level and is only comparable between images whose background
    estimate carries the same offset; see :func:`rolling_ball_background_px`.
    """
    return max(float(background_mean) + float(k) * float(noise_sigma), float(floor))


def censored_noise_threshold(
    residual: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    k: float = 4.0,
    floor: float = 0.0,
    clip_level: float = 0.5,
    sigma_floor: float = 0.5,
) -> tuple[float, float, float, float]:
    """Censored noise estimate of a residual image and the threshold it implies.

    Combines :func:`censored_noise` with :func:`noise_threshold`.

    Parameters
    ----------
    residual : ndarray
        2-D background residual in the grey levels of the input.
    mask : ndarray, optional
        2-D boolean mask of the pixels the noise is estimated on (for example a
        tissue mask). ``None`` uses every pixel.
    k : float, optional
        Number of noise standard deviations above the background (dimensionless).
    floor : float, optional
        Absolute lower bound on the threshold, in grey levels. 0 switches it off.
    clip_level : float, optional
        Values at or below this level count as clipped, in grey levels.
    sigma_floor : float, optional
        Smallest noise scale returned, in grey levels.

    Returns
    -------
    threshold : float
        Threshold in grey levels.
    mu : float
        Background mean in grey levels.
    sigma : float
        Noise scale in grey levels.
    clipped_fraction : float
        Fraction of values at or below ``clip_level`` (dimensionless). On a residual
        this is the lower part of the background distribution and not a detector
        diagnostic; see :func:`censored_noise`.
    """
    work = as_float(check_2d(residual), np.float64)
    if mask is None:
        values = work.ravel()
    else:
        selection = check_2d(mask, "mask").astype(bool, copy=False)
        check_same_shape(work, selection)
        values = work[selection]
    mu, sigma, clipped_fraction = censored_noise(
        values, clip_level=clip_level, sigma_floor=sigma_floor
    )
    threshold = noise_threshold(mu, sigma, k=k, floor=floor)
    return threshold, mu, sigma, clipped_fraction


def estimate_shared_noise_scale(
    images: Sequence[np.ndarray],
    radius_um: float,
    pixel_size_um: float,
    *,
    max_pixels: int = 400_000,
) -> float:
    """One residual-noise scale, in grey levels, shared by a batch of images.

    The local residual of every image is pooled and one robust scale is taken from
    the pool. Pooling rather than averaging per-image estimates is deliberate: the
    point of a shared scale is that it must not be pulled around by how much textured
    background an individual frame happens to contain. Calibrate it on images that
    represent the batch's background and then apply the same number to every frame,
    so that counts from a quiet frame and from a textured frame sit on one axis.

    Parameters
    ----------
    images : sequence of ndarray
        2-D images in the grey levels of the input. Images that are not 2-D or are
        empty are skipped.
    radius_um : float
        Background scale in micrometres. Non-positive subtracts the global median of
        each image instead of a local background.
    pixel_size_um : float
        Pixel size in micrometres per pixel.
    max_pixels : int, optional
        Pixels contributed by each image to the pool (dimensionless count).

    Returns
    -------
    float
        Noise scale (one standard deviation) in grey levels.

    Notes
    -----
    The pool is built from *Gaussian* residuals (:func:`gaussian_residual`), whose
    spread stays close to the noise of the images because a Gaussian background
    follows a smooth background, curvature included. A scale taken from a rolling-ball
    residual of the same images is not the same number: that scheme's background is
    reduced and bilinearly enlarged, so it cannot follow curvature, and the mismatch
    adds to the spread of its residual. The two are therefore not interchangeable.
    This value is the one to use with the Gaussian route, including the
    ``noise_scale`` of :func:`if_qsaa.segmentation.robust_snr_response`, whose
    response is built from a Gaussian residual. When a rolling-ball residual is the
    plane being thresholded, read its scale from that residual (for instance with
    :func:`censored_noise`) and calibrate the ``k`` of :func:`noise_threshold` against
    it.

    Either way, estimate the scale on background: a whole-frame estimate taken from a
    field whose objects cover a noticeable share of the pixels is pulled up by those
    objects, which is a larger effect than the difference between the two background
    models. Pass a background mask to :func:`censored_noise_threshold`, or calibrate
    here on frames that carry the batch's background and few objects.
    """
    residuals: list[np.ndarray] = []
    for image in images:
        work = np.asarray(image)
        if work.ndim != 2 or work.size == 0:
            continue
        residual = gaussian_residual(work, radius_um, pixel_size_um)
        residuals.append(finite_sample(residual, max_pixels=max_pixels))
    if not residuals:
        raise ValueError("no usable 2-D image to calibrate a shared noise scale on")
    pooled = np.concatenate(residuals)
    centre = float(np.median(pooled))
    scale = mad(pooled, centre=centre)
    if math.isfinite(scale) and scale > 0.0:
        return float(scale)
    low, high = (float(v) for v in np.percentile(pooled, (16.0, 84.0)))
    fallback = (high - low) * 0.5
    if not math.isfinite(fallback) or fallback <= 0.0:
        raise ValueError(
            "the calibration images carry no measurable residual variation"
        )
    return fallback
