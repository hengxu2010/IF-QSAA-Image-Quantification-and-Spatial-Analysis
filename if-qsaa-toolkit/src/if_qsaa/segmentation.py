"""Object segmentation core: response, threshold, clean-up, splitting and measurement.

The module turns one image plane into object masks and a per-object measurement
table. It provides:

* a robust local-SNR response, the local background residual expressed in robust
  noise standard deviations,
* a threshold family: global methods (Otsu, Li, triangle), a fixed value, and a
  local mean plus ``k`` times the local standard deviation, each able to grow from
  its seeds by hysteresis,
* an SNR evidence filter that rejects grown components with too little aggregate
  evidence,
* mask clean-up (opening, closing, erosion, dilation, small-hole fill, removal of
  objects touching the frame edge) and a signed mask boundary offset,
* splitting of touching objects by distance-transform peaks with a constrained
  Voronoi partition and a connectivity repair pass,
* per-object measurement, object filters, one end-to-end pipeline, and
  re-measurement of a mask supplied from outside.

Foreground connectivity is 8 and background connectivity is 4 throughout.
Every spatial parameter of the public API is a length in
micrometres or an area in square micrometres and is converted to pixels with an
explicit ``pixel_size_um``; working at a reduced resolution therefore only means
passing the working pixel size, and :func:`scale_length_px`, :func:`scale_area_px`
and :func:`scale_window_px` scale a parameter set written in pixels onto a working
grid with the rounding this core uses. All functions are pure and
deterministic: no file access, no global state and no random numbers.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import filters as sk_filters
from skimage import measure as sk_measure
from skimage import morphology as sk_morphology

from . import background as bg
from ._util import (
    area_px_to_um2,
    area_um2_to_px,
    as_bool_mask,
    as_float,
    check_2d,
    check_same_shape,
    disk,
    label_mask,
    px_to_um,
    um_to_px,
)

__all__ = [
    "MEASUREMENT_COLUMNS",
    "scale_length_px",
    "scale_area_px",
    "scale_window_px",
    "robust_snr_response",
    "global_threshold",
    "snr_seed_level",
    "sensitivity_factor",
    "hysteresis_mask",
    "threshold_mask",
    "filter_components_by_evidence",
    "fill_small_holes",
    "clean_mask",
    "remove_border_objects",
    "offset_mask",
    "offset_mask_um",
    "split_touching_objects",
    "label_objects",
    "measure_objects",
    "filter_objects",
    "segment",
    "measure_external_mask",
]

_EPS32 = float(np.finfo(np.float32).eps)
MEASUREMENT_COLUMNS = (
    "id",
    "area_px",
    "area_um2",
    "centroid_x_px",
    "centroid_y_px",
    "bbox_x_px",
    "bbox_y_px",
    "bbox_width_px",
    "bbox_height_px",
    "equivalent_diameter_um",
    "major_axis_um",
    "minor_axis_um",
    "aspect_ratio",
    "perimeter_um",
    "solidity",
    "touches_border",
    "mean_intensity",
    "max_intensity",
    "integrated_intensity",
    "local_background",
    "boundary_contrast",
    "ring_area_px",
    "snr",
    "response_mean",
    "response_max",
    "response_confidence",
)


# --------------------------------------------------------------------------- scale

def scale_length_px(length_px: float, scale: float, *, minimum_if_enabled: int = 0) -> int:
    """Scale a length given in pixels onto a working grid.

    A disabled length (0 or less) stays disabled, and an enabled length becomes ``floor(length_px * scale + 0.5)``, never below
    ``minimum_if_enabled``.

    Parameters
    ----------
    length_px : float
        Length in pixels of the grid the parameter is expressed on.
    scale : float
        Linear ratio of the working grid to that grid (dimensionless).
    minimum_if_enabled : int, optional
        Smallest value in pixels for a length that is enabled.

    Returns
    -------
    int
        Length in pixels of the working grid.
    """
    if float(length_px) <= 0.0:
        return 0
    value = int(math.floor(float(length_px) * float(scale) + 0.5))
    return max(int(minimum_if_enabled), value)


def scale_area_px(area_px: float, scale: float) -> int:
    """Scale an area given in pixels onto a working grid.

    Areas scale with the square of the linear ratio; an enabled area never falls
    below one pixel.

    Parameters
    ----------
    area_px : float
        Area in pixels of the grid the parameter is expressed on.
    scale : float
        Linear ratio of the working grid to that grid (dimensionless).

    Returns
    -------
    int
        Area in pixels of the working grid.
    """
    if float(area_px) <= 0.0:
        return 0
    return max(1, int(round(float(area_px) * float(scale) ** 2)))


def scale_window_px(window_px: float, scale: float) -> int:
    """Scale a sliding-window side onto a working grid, keeping it odd and at least 3.

    Parameters
    ----------
    window_px : float
        Window side in pixels of the grid the parameter is expressed on.
    scale : float
        Linear ratio of the working grid to that grid (dimensionless).

    Returns
    -------
    int
        Odd window side in pixels of the working grid.
    """
    window = max(3, int(round(float(window_px) * float(scale))))
    return window + 1 if window % 2 == 0 else window


def _radius_px(length_um: float, pixel_size_um: float, *, minimum: int = 0) -> int:
    """Integer pixel radius of a length in micrometres, rounded as ``floor(x + 0.5)``."""
    if float(length_um) <= 0.0:
        return 0
    return max(int(minimum), int(math.floor(um_to_px(length_um, pixel_size_um) + 0.5)))


def _area_px(area_um2: float, pixel_size_um: float) -> int:
    """Integer pixel area of an area in square micrometres (at least one pixel)."""
    if float(area_um2) <= 0.0:
        return 0
    return max(1, int(round(area_um2_to_px(area_um2, pixel_size_um))))


# --------------------------------------------------------------------------- response

def _denoise(image: np.ndarray, method: str, size_px: float) -> np.ndarray:
    """Denoise a float plane; ``size_px`` is a Gaussian sigma or a median radius in pixels."""
    if method == "none" or float(size_px) <= 0.0:
        return image.astype(np.float32, copy=False)
    if method == "gaussian":
        return ndi.gaussian_filter(
            image, max(float(size_px), 0.05), mode="reflect"
        ).astype(np.float32, copy=False)
    if method == "median":
        radius = max(1, int(round(float(size_px))))
        return ndi.median_filter(
            image, size=2 * radius + 1, mode="reflect"
        ).astype(np.float32, copy=False)
    raise ValueError("denoise_method must be 'none', 'gaussian' or 'median'")


def robust_snr_response(
    signal: np.ndarray,
    pixel_size_um: float,
    *,
    background_radius_um: float,
    denoise_method: str = "gaussian",
    denoise_um: float = 0.0,
    invert: bool = False,
    noise_scale: float | None = None,
) -> np.ndarray:
    """Positive local evidence in robust background-noise standard deviations.

    The response is the local background residual divided by a robust scale of that
    residual, clipped at zero. A uniform offset is removed by the local background
    and a positive gain is cancelled by the residual scale, so the response measures
    contrast and not brightness: a frame with no measurable residual variation gives
    an empty response rather than a thresholded version of its own texture.

    The residual is the denoised plane minus its Gaussian background (minus its
    global median when the background radius is not positive). Its centre is the
    median of the residual and its scale is ``1.4826 * MAD``, with
    ``(P84 - P16) / 2`` as a fallback when the MAD is exactly zero and a small
    numerical floor when both are zero but the residual still has dynamic range.

    Pass ``noise_scale`` to divide by one fixed scale shared by a whole batch instead
    of by each frame's own texture: the seed level then names an absolute contrast,
    identical in every frame, which is what places counts from a quiet frame and from
    a textured frame on one axis. Such a scale is a property of a batch's images and
    has no generic value, so it must be calibrated with
    :func:`if_qsaa.background.estimate_shared_noise_scale`.

    Parameters
    ----------
    signal : ndarray
        2-D image plane in the grey levels of the input.
    pixel_size_um : float
        Pixel size in micrometres per pixel.
    background_radius_um : float
        Local background scale in micrometres. Not positive subtracts the global
        median instead.
    denoise_method : {'none', 'gaussian', 'median'}, optional
        Pre-threshold denoising filter.
    denoise_um : float, optional
        Filter size in micrometres: a Gaussian standard deviation or a median radius.
        0 switches denoising off.
    invert : bool, optional
        Treat dark objects on a bright background by negating the plane first.
    noise_scale : float, optional
        Shared absolute noise scale in grey levels. ``None`` uses this frame's own
        robust residual scale.

    Returns
    -------
    ndarray
        ``float32`` response in noise standard deviations, zero or greater.
    """
    work = as_float(check_2d(signal), np.float32)
    if invert:
        work = np.negative(work)
    sample = bg.finite_sample(work)
    centre_guess = float(np.median(sample))
    if not bool(np.all(np.isfinite(work))):
        work = np.nan_to_num(
            work, copy=True, nan=centre_guess, posinf=centre_guess, neginf=centre_guess
        )

    denoise_px = um_to_px(denoise_um, pixel_size_um) if float(denoise_um) > 0.0 else 0.0
    denoised = _denoise(work, denoise_method, denoise_px)
    residual = bg.gaussian_residual(denoised, background_radius_um, pixel_size_um)

    finite_residual = bg.finite_sample(residual)
    centre = float(np.median(finite_residual))
    scale = float(1.4826 * np.median(np.abs(finite_residual - centre)))
    if not (math.isfinite(scale) and scale > 0.0):
        low, high = (float(v) for v in np.percentile(finite_residual, (16.0, 84.0)))
        scale = (high - low) * 0.5
    if not (math.isfinite(scale) and scale > 0.0):
        # A quantised field can hold a real sparse object while most of the residual
        # is exactly zero. Keep only a numerical floor; a constant field stays empty.
        residual_min = float(finite_residual.min(initial=0.0))
        residual_max = float(finite_residual.max(initial=0.0))
        span = residual_max - residual_min
        if not math.isfinite(span) or span <= 0.0:
            return np.zeros(work.shape, dtype=np.float32)
        magnitude = max(abs(residual_min), abs(residual_max), 1.0)
        scale = max(span * 1e-6, _EPS32 * magnitude * 16.0)
    if noise_scale is not None and float(noise_scale) > 0.0:
        scale = abs(float(noise_scale))

    response = (residual - np.float32(centre)) / np.float32(scale)
    np.maximum(response, np.float32(0.0), out=response)
    if not bool(np.all(np.isfinite(response))):
        response = np.nan_to_num(
            response, copy=False, nan=0.0, posinf=float(np.finfo(np.float32).max), neginf=0.0
        )
    return response.astype(np.float32, copy=False)


# --------------------------------------------------------------------------- thresholds

def global_threshold(response: np.ndarray, method: str = "otsu") -> float:
    """Global automatic threshold of a response plane.

    Otsu, Li and the triangle method are taken from scikit-image and applied to the
    response values directly. When the method returns a non-positive level, or the
    plane has no dynamic range, the 95th percentile of the positive pixels is used
    instead, so an all-background frame cannot be split at zero. A plane with no
    positive pixel has no threshold and returns infinity.

    Parameters
    ----------
    response : ndarray
        Response plane in any intensity units.
    method : {'otsu', 'li', 'triangle'}, optional
        Global threshold rule.

    Returns
    -------
    float
        Threshold in the units of ``response``, or ``inf`` when none exists.
    """
    work = as_float(response, np.float64)
    finite = work[np.isfinite(work)]
    positive = finite[finite > 0.0]
    if positive.size == 0:
        return float("inf")
    if float(finite.max()) <= float(finite.min()):
        return float(np.percentile(positive, 95.0))
    if method == "otsu":
        value = float(sk_filters.threshold_otsu(finite))
    elif method == "li":
        value = float(sk_filters.threshold_li(finite))
    elif method == "triangle":
        value = float(sk_filters.threshold_triangle(finite))
    else:
        raise ValueError("method must be 'otsu', 'li' or 'triangle'")
    if not math.isfinite(value) or value <= 0.0:
        return float(np.percentile(positive, 95.0))
    return value


def sensitivity_factor(sensitivity: float = 50.0, *, exponent_scale: float = 35.0) -> float:
    """Multiplier a global threshold is scaled by for a given sensitivity.

    ``2 ** ((50 - sensitivity) / exponent_scale)``, clipped to ``[0.30, 3.20]``: 50 is
    neutral, higher values lower the threshold and recover fainter objects.

    Parameters
    ----------
    sensitivity : float, optional
        Sensitivity on a 0 to 100 scale (dimensionless), 50 being neutral.
    exponent_scale : float, optional
        Sensitivity units per halving of the threshold (dimensionless).

    Returns
    -------
    float
        Multiplier (dimensionless).
    """
    factor = 2.0 ** ((50.0 - float(sensitivity)) / float(exponent_scale))
    return float(np.clip(factor, 0.30, 3.20))


def snr_seed_level(seed_sigma: float = 5.0, sensitivity: float = 50.0) -> float:
    """Seed level of the robust-SNR threshold, in noise standard deviations.

    ``seed_sigma * 2 ** ((50 - sensitivity) / 70)``. It depends only on the recipe and
    never on the image, so it is also the correct reference for per-object evidence
    confidence at any working resolution.

    Parameters
    ----------
    seed_sigma : float, optional
        Seed contrast in noise standard deviations (dimensionless).
    sensitivity : float, optional
        Sensitivity on a 0 to 100 scale (dimensionless), 50 being neutral.

    Returns
    -------
    float
        Seed level in noise standard deviations.
    """
    return float(seed_sigma) * 2.0 ** ((50.0 - float(sensitivity)) / 70.0)


def hysteresis_mask(
    response: np.ndarray,
    low: float,
    high: float,
    *,
    connectivity: int = 8,
) -> np.ndarray:
    """Grow a mask from seeds: keep candidate components that contain a seed.

    Seeds are the pixels at or above ``high`` and candidates the pixels at or above
    ``low``. A candidate component survives whole if at least one of its pixels is a
    seed, so a faint rim is kept when it belongs to a confident object and dropped
    when it stands alone.

    Parameters
    ----------
    response : ndarray
        2-D response plane in any intensity units.
    low : float
        Candidate level in the units of ``response``.
    high : float
        Seed level in the units of ``response``.
    connectivity : int, optional
        Foreground connectivity, 4 or 8.

    Returns
    -------
    ndarray
        Boolean mask.
    """
    work = as_float(check_2d(response), np.float32)
    if not math.isfinite(high):
        return np.zeros(work.shape, dtype=bool)
    seeds = work >= np.float32(high)
    if not bool(seeds.any()):
        return np.zeros(work.shape, dtype=bool)
    candidates = work >= np.float32(low)
    labels, count = label_mask(candidates, connectivity)
    if count == 0:
        return np.zeros(work.shape, dtype=bool)
    keep = np.zeros(count + 1, dtype=bool)
    keep[np.unique(labels[seeds])] = True
    keep[0] = False
    return keep[labels]


def _grow_from_map(
    response: np.ndarray,
    threshold_map: np.ndarray,
    local_mean: np.ndarray,
    local_std: np.ndarray,
    grow_ratio: float,
    connectivity: int,
    clip_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, float]:
    """Hysteresis growth from a spatially varying threshold map."""
    positive = response > np.float32(_EPS32)
    strong = positive & (response >= threshold_map)
    if float(grow_ratio) >= 0.999:
        return strong, float(np.mean(threshold_map, dtype=np.float64))
    # Multiplying a low local threshold directly can connect most of an uneven frame
    # at once, so grow by a margin that follows the local contrast instead.
    margin = np.maximum(np.abs(threshold_map - local_mean), np.float32(0.35) * local_std)
    low_map = threshold_map - np.float32(1.0 - float(grow_ratio)) * margin
    if clip_range is not None:
        low_map = np.clip(low_map, clip_range[0], clip_range[1]).astype(np.float32)
    candidates = positive & (response >= low_map)
    labels, count = label_mask(candidates, connectivity)
    if count == 0:
        return np.zeros(response.shape, dtype=bool), float(
            np.mean(low_map, dtype=np.float64)
        )
    keep = np.zeros(count + 1, dtype=bool)
    keep[np.unique(labels[strong])] = True
    keep[0] = False
    return keep[labels], float(np.mean(low_map, dtype=np.float64))


_DEFAULT_GROW_RATIO = 0.60


def threshold_mask(
    response: np.ndarray,
    *,
    method: str = "snr_hysteresis",
    seed_sigma: float = 5.0,
    sensitivity: float = 50.0,
    grow_ratio: float | None = None,
    value: float | None = None,
    local_window_um: float | None = None,
    pixel_size_um: float | None = None,
    local_k: float = 0.20,
    clip_range: tuple[float, float] | None = None,
    connectivity: int = 8,
) -> tuple[np.ndarray, float, float, float | None]:
    """Threshold a response plane and return the mask, the two levels and a reference.

    ``'snr_hysteresis'``
        Seed at :func:`snr_seed_level` noise standard deviations and grow by
        hysteresis down to ``grow_ratio`` of the seed. The seed level is also the
        evidence reference.
    ``'otsu'``, ``'li'``, ``'triangle'``
        Global level from :func:`global_threshold`, scaled by
        :func:`sensitivity_factor`, applied without growth.
    ``'hysteresis_otsu'``
        The same global Otsu level as the seed, grown by hysteresis.
    ``'value'``
        A fixed level in the units of the response, applied literally: a pixel is
        foreground when it reaches that level. Growth is off on this route unless an
        explicit ``grow_ratio`` below 1 asks for it, because a level computed outside
        this module (``background + k * noise``, say) names the contrast the caller
        means to accept and nothing below it.
    ``'local_mean_std'``
        A spatially varying level ``local mean + k * local SD`` over a square window,
        where ``k`` is ``local_k`` plus a sensitivity offset of ``0.9 * (50 - s) / 100``,
        grown by a contrast-aware margin.

    Parameters
    ----------
    response : ndarray
        2-D response plane. For ``'snr_hysteresis'`` it must be in noise standard
        deviations; the other methods work in any intensity units.
    method : str, optional
        One of the rules above.
    seed_sigma : float, optional
        Seed contrast in noise standard deviations, used by ``'snr_hysteresis'``.
    sensitivity : float, optional
        Sensitivity on a 0 to 100 scale (dimensionless), 50 being neutral.
    grow_ratio : float, optional
        Candidate level as a fraction of the seed level (dimensionless, 0 to 1).
        1 switches growth off. ``None`` uses the default of the chosen rule: 1 for
        ``'value'``, which therefore thresholds at the level it is given, and 0.60 for
        every rule whose level is relative, where growing from the seed is part of the
        recipe. Pass a ratio below 1 explicitly to grow from a fixed level as well.
    value : float, optional
        Fixed level in the units of the response, required by ``method='value'``.
    local_window_um : float, optional
        Side of the local window in micrometres, required by ``'local_mean_std'``.
    pixel_size_um : float, optional
        Pixel size in micrometres per pixel, required by ``'local_mean_std'``.
    local_k : float, optional
        Number of local standard deviations above the local mean (dimensionless).
    clip_range : tuple of float, optional
        Clip the local threshold map to this range, in the units of the response.
        Use it when the response is known to live on a fixed scale.
    connectivity : int, optional
        Foreground connectivity, 4 or 8.

    Returns
    -------
    mask : ndarray
        Boolean mask.
    high : float
        Seed level, or the mean of the threshold map for the local method.
    low : float
        Candidate level, or the mean of the grown map for the local method.
    reference : float or None
        Evidence reference for per-object confidence: the seed level for
        ``'snr_hysteresis'``, the unscaled Otsu level where one was computed, and
        ``None`` otherwise.
    """
    work = as_float(check_2d(response), np.float32)
    # A level this module derives itself is relative, and growing from it is part of
    # the recipe; a level the caller computed is absolute and is applied as given.
    if grow_ratio is None:
        ratio = 1.0 if method == "value" else _DEFAULT_GROW_RATIO
    else:
        ratio = float(grow_ratio)

    if method == "snr_hysteresis":
        high = snr_seed_level(seed_sigma, sensitivity)
        low = high * ratio
        return hysteresis_mask(work, low, high, connectivity=connectivity), high, low, high

    if method in {"otsu", "li", "triangle", "hysteresis_otsu"}:
        base = "otsu" if method == "hysteresis_otsu" else method
        auto = global_threshold(work, base)
        reference = auto if base == "otsu" else None
        high = auto * sensitivity_factor(sensitivity)
        low = high * ratio
        if method == "hysteresis_otsu":
            mask = hysteresis_mask(work, low, high, connectivity=connectivity)
            return mask, float(high), float(low), reference
        if not math.isfinite(high):
            return np.zeros(work.shape, dtype=bool), float(high), float(high), reference
        return work >= np.float32(high), float(high), float(high), reference

    if method == "value":
        if value is None:
            raise ValueError("method='value' needs an explicit threshold value")
        high = float(value)
        low = high * ratio
        if ratio < 0.999:
            mask = hysteresis_mask(work, low, high, connectivity=connectivity)
            return mask, high, low, None
        return work >= np.float32(high), high, high, None

    if method != "local_mean_std":
        raise ValueError(f"unknown threshold method: {method!r}")

    if local_window_um is None or pixel_size_um is None:
        raise ValueError(
            "method='local_mean_std' needs local_window_um and pixel_size_um"
        )
    window = bg.local_background_window_px(local_window_um, pixel_size_um)
    local_mean = ndi.uniform_filter(work, size=window, mode="reflect")
    square_mean = ndi.uniform_filter(work * work, size=window, mode="reflect")
    local_std = np.sqrt(np.maximum(square_mean - local_mean * local_mean, 0.0)).astype(
        np.float32
    )
    effective_k = float(local_k) + (50.0 - float(sensitivity)) / 100.0 * 0.9
    threshold_map = local_mean + np.float32(effective_k) * local_std
    if clip_range is not None:
        threshold_map = np.clip(threshold_map, clip_range[0], clip_range[1]).astype(
            np.float32
        )
    mask, low_value = _grow_from_map(
        work,
        threshold_map,
        local_mean,
        local_std,
        ratio,
        connectivity,
        clip_range,
    )
    return mask, float(np.mean(threshold_map, dtype=np.float64)), low_value, None


def filter_components_by_evidence(
    mask: np.ndarray,
    response: np.ndarray,
    high: float,
    low: float,
    *,
    mean_fraction: float = 0.48,
    p90_fraction: float = 0.64,
    connectivity: int = 8,
) -> np.ndarray:
    """Reject grown components whose aggregate evidence is too weak.

    Hysteresis lets a component reach far below the seed level, so each component is
    judged as a whole: its mean response must reach ``max(low, mean_fraction * high)``
    and its 90th percentile must reach ``p90_fraction * high``. A component that is
    merely a large patch of slightly raised background fails both.

    Parameters
    ----------
    mask : ndarray
        2-D boolean mask to filter.
    response : ndarray
        2-D response plane the mask was thresholded on, same shape as ``mask``.
    high : float
        Seed level in the units of the response.
    low : float
        Candidate level in the units of the response.
    mean_fraction : float, optional
        Fraction of the seed level a component's mean response must reach
        (dimensionless).
    p90_fraction : float, optional
        Fraction of the seed level a component's 90th percentile must reach
        (dimensionless).
    connectivity : int, optional
        Foreground connectivity, 4 or 8.

    Returns
    -------
    ndarray
        Boolean mask with the weak components removed.
    """
    work_mask = as_bool_mask(mask)
    work = as_float(check_2d(response), np.float32)
    check_same_shape(work_mask, work)
    if not bool(work_mask.any()):
        return np.zeros(work_mask.shape, dtype=bool)
    labels, count = label_mask(work_mask, connectivity)
    if count == 0:
        return np.zeros(work_mask.shape, dtype=bool)
    mean_floor = max(float(low), float(mean_fraction) * float(high))
    p90_floor = float(p90_fraction) * float(high)
    keep = np.zeros(count + 1, dtype=bool)
    means = ndi.mean(work, labels=labels, index=np.arange(1, count + 1))
    keep[1:] = np.asarray(means, dtype=np.float64) >= mean_floor
    boxes = ndi.find_objects(labels)
    for label_id in np.flatnonzero(keep).tolist():
        if label_id == 0:
            continue
        window = boxes[label_id - 1]
        values = work[window][labels[window] == label_id]
        if values.size == 0 or float(np.percentile(values, 90.0)) < p90_floor:
            keep[label_id] = False
    return keep[labels]


# --------------------------------------------------------------------------- clean-up

def fill_small_holes(mask: np.ndarray, max_hole_area_px: int) -> np.ndarray:
    """Fill enclosed background components up to a given area.

    Background is labelled with 4-connectivity, the complement of the 8-connected
    foreground, so a closed rim cannot leak diagonally. Components that touch the
    frame edge are outside, not holes, and are never filled.

    Parameters
    ----------
    mask : ndarray
        2-D boolean mask.
    max_hole_area_px : int
        Largest hole filled, in pixels. 0 switches the step off.

    Returns
    -------
    ndarray
        Boolean mask.
    """
    work = as_bool_mask(mask)
    if int(max_hole_area_px) <= 0 or not bool(work.any()):
        return work.copy()
    labels, count = label_mask(~work, connectivity=4)
    if count == 0:
        return work.copy()
    border = np.unique(
        np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    )
    is_border = np.zeros(count + 1, dtype=bool)
    is_border[border] = True
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    fill = (areas <= int(max_hole_area_px)) & (~is_border)
    fill[0] = False
    return work | fill[labels]


def _binary_open(mask: np.ndarray, radius_px: int) -> np.ndarray:
    """Opening by a flat disk; the frame outside counts as foreground during erosion."""
    element = disk(radius_px)
    eroded = ndi.binary_erosion(mask, structure=element, border_value=1)
    return ndi.binary_dilation(eroded, structure=element, border_value=0)


def _binary_close(mask: np.ndarray, radius_px: int) -> np.ndarray:
    """Closing by a flat disk; the frame outside counts as background during dilation."""
    element = disk(radius_px)
    dilated = ndi.binary_dilation(mask, structure=element, border_value=0)
    return ndi.binary_erosion(dilated, structure=element, border_value=1)


def clean_mask(
    mask: np.ndarray,
    pixel_size_um: float,
    *,
    open_um: float = 0.0,
    close_um: float = 0.0,
    erode_um: float = 0.0,
    dilate_um: float = 0.0,
    max_hole_area_um2: float = 0.0,
    remove_border: bool = False,
) -> np.ndarray:
    """Clean up a binary mask in a fixed order.

    Opening, then closing, then erosion, then dilation, each by a flat disk of the
    given radius, then the small-hole fill, and finally the optional removal of whole
    objects touching the frame edge. A step whose radius is not positive is skipped.
    Outside the frame counts as foreground during erosion and as background during
    dilation, so an object touching the edge is not eaten away from outside.

    Parameters
    ----------
    mask : ndarray
        2-D boolean mask.
    pixel_size_um : float
        Pixel size in micrometres per pixel.
    open_um : float, optional
        Opening radius in micrometres; removes specks smaller than the disk.
    close_um : float, optional
        Closing radius in micrometres; bridges gaps narrower than the disk.
    erode_um : float, optional
        Erosion radius in micrometres.
    dilate_um : float, optional
        Dilation radius in micrometres.
    max_hole_area_um2 : float, optional
        Largest enclosed hole filled, in square micrometres. 0 switches it off.
    remove_border : bool, optional
        Remove whole objects touching any frame edge.

    Returns
    -------
    ndarray
        Boolean mask.
    """
    work = as_bool_mask(mask)
    steps = (
        ("open", _radius_px(open_um, pixel_size_um)),
        ("close", _radius_px(close_um, pixel_size_um)),
        ("erode", _radius_px(erode_um, pixel_size_um)),
        ("dilate", _radius_px(dilate_um, pixel_size_um)),
    )
    for name, radius in steps:
        if radius <= 0 or not bool(work.any()):
            continue
        if name == "open":
            work = _binary_open(work, radius)
        elif name == "close":
            work = _binary_close(work, radius)
        elif name == "erode":
            work = ndi.binary_erosion(work, structure=disk(radius), border_value=1)
        else:
            work = ndi.binary_dilation(work, structure=disk(radius), border_value=0)
    work = fill_small_holes(work, _area_px(max_hole_area_um2, pixel_size_um))
    if remove_border:
        work = remove_border_objects(work)
    return work


def remove_border_objects(mask: np.ndarray, *, connectivity: int = 8) -> np.ndarray:
    """Remove whole objects that touch any frame edge.

    A clipped object has an unknown true size and shape, so the whole parent component
    is removed, before any splitting.

    Parameters
    ----------
    mask : ndarray
        2-D boolean mask.
    connectivity : int, optional
        Foreground connectivity, 4 or 8.

    Returns
    -------
    ndarray
        Boolean mask.

    Notes
    -----
    Three routes remove border objects and they are not interchangeable. This
    function and :func:`clean_mask` with ``remove_border=True`` work on the mask and
    drop whole connected components, before any splitting, which is what a clipped
    object asks for: its true size and shape are unknown, so none of its parts should
    be measured. :func:`filter_objects` with ``remove_border=True`` works on the
    measured table and drops the labelled objects whose ``touches_border`` is true,
    which after splitting is only the piece that reaches the edge and not the rest of
    its parent. Use the mask route when a clipped object must not contribute at all,
    and the table route when the label image comes from elsewhere and only measured
    rows can be selected. :func:`segment` applies both for its single
    ``remove_border`` flag: the mask route before labelling and the table route after
    measurement.
    """
    work = as_bool_mask(mask)
    if not bool(work.any()):
        return work.copy()
    labels, count = label_mask(work, connectivity)
    if count == 0:
        return work.copy()
    border = np.unique(
        np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    )
    remove = np.zeros(count + 1, dtype=bool)
    remove[border] = True
    remove[0] = False
    return work & (~remove[labels])


def offset_mask(mask: np.ndarray, signed_px: int) -> np.ndarray:
    """Expand or contract a mask boundary by an exact pixel distance.

    Positive values expand the foreground and negative values contract it. Distances
    are Euclidean between pixel centres. Growth is clipped at the frame; during
    contraction the frame edge counts as a boundary, so a mask that fills the frame
    contracts from the edge like any other.

    Parameters
    ----------
    mask : ndarray
        2-D boolean mask.
    signed_px : int
        Signed offset in pixels: positive expands, negative contracts, 0 copies.

    Returns
    -------
    ndarray
        Boolean mask.
    """
    work = as_bool_mask(mask)
    if isinstance(signed_px, (bool, np.bool_)):
        raise TypeError("signed_px must be an integer number of pixels")
    delta = int(signed_px)
    if delta != float(signed_px):
        raise TypeError("signed_px must be an integer number of pixels")
    if delta == 0 or not bool(work.any()):
        return work.copy()
    radius = abs(delta)
    if delta > 0:
        distance = ndi.distance_transform_edt(~work)
        return work | (distance <= float(radius))
    padded = np.pad(work, ((1, 1), (1, 1)), mode="constant", constant_values=False)
    distance = ndi.distance_transform_edt(padded)[1:-1, 1:-1]
    return work & (distance > float(radius))


def offset_mask_um(mask: np.ndarray, offset_um: float, pixel_size_um: float) -> np.ndarray:
    """Expand or contract a mask boundary by a signed distance in micrometres.

    The distance is rounded to whole pixels before :func:`offset_mask` is applied.

    Parameters
    ----------
    mask : ndarray
        2-D boolean mask.
    offset_um : float
        Signed offset in micrometres: positive expands, negative contracts.
    pixel_size_um : float
        Pixel size in micrometres per pixel.

    Returns
    -------
    ndarray
        Boolean mask.
    """
    signed = int(round(um_to_px(abs(float(offset_um)), pixel_size_um)))
    if float(offset_um) < 0.0:
        signed = -signed
    return offset_mask(mask, signed)


# --------------------------------------------------------------------------- splitting

def _padded_distance(component: np.ndarray) -> np.ndarray:
    """Euclidean distance to the outside of a component, measured on a zero-padded copy."""
    padded = np.zeros(
        (component.shape[0] + 2, component.shape[1] + 2), dtype=bool
    )
    padded[1:-1, 1:-1] = component
    return ndi.distance_transform_edt(padded)[1:-1, 1:-1].astype(np.float32)


def _hole_area(component: np.ndarray) -> int:
    """Total area in pixels of the enclosed holes of a component."""
    padded = np.zeros(
        (component.shape[0] + 2, component.shape[1] + 2), dtype=bool
    )
    padded[1:-1, 1:-1] = component
    labels, count = label_mask(~padded, connectivity=4)
    if count <= 1:
        return 0
    outside = int(labels[0, 0])
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    areas[0] = 0
    if outside > 0:
        areas[outside] = 0
    return int(areas.sum())


def _thickness(component: np.ndarray, distance: np.ndarray, area: int) -> tuple[float, float, float]:
    """Characteristic width, maximum width and a curvature-safe elongation index."""
    if area <= 0:
        return 0.0, 0.0, 0.0
    values = distance[component]
    # Four times the mean interior distance approximates the width of a long strip;
    # area / width^2 stays high for a curved fibre, unlike a covariance aspect ratio.
    mean_thickness = max(1.0, 4.0 * float(values.mean()))
    max_thickness = max(1.0, 2.0 * float(values.max(initial=0.0)))
    elongation = float(area / max(mean_thickness * mean_thickness, 1.0))
    return mean_thickness, max_thickness, elongation


def _covariance_aspect(component: np.ndarray) -> float:
    """Aspect ratio from the pixel coordinate covariance, regularised by the unit square."""
    y, x = np.nonzero(component)
    if x.size >= 2:
        covariance = np.asarray(np.cov(np.vstack((x, y)), bias=True), dtype=np.float64)
    else:
        covariance = np.zeros((2, 2), dtype=np.float64)
    covariance = covariance + np.eye(2, dtype=np.float64) / 12.0
    eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(covariance), 0.0))[::-1]
    return math.sqrt(float(eigenvalues[0]) / max(float(eigenvalues[1]), 1e-12))


def _pixel_solidity(component: np.ndarray) -> float:
    """Filled area divided by convex-hull area, both as pixel counts."""
    area = int(np.count_nonzero(component))
    if area < 3:
        return 1.0
    hull = int(np.count_nonzero(sk_morphology.convex_hull_image(component)))
    if hull <= 0:
        return 1.0
    return min(1.0, area / float(hull))


def _repair_partition_connectivity(
    local_mask: np.ndarray,
    nearest: np.ndarray,
    active_seeds: np.ndarray,
) -> np.ndarray:
    """Make every partition singly connected without losing any mask pixel."""
    nearest_ids = np.unique(nearest[local_mask])
    partition = np.zeros(local_mask.shape, dtype=np.int32)
    for partition_id, nearest_id in enumerate(nearest_ids.tolist(), start=1):
        partition[local_mask & (nearest == nearest_id)] = partition_id

    # A Euclidean cell can meet a concave object in several islands. Keep the piece
    # that owns the seed; the rest is reassigned below along paths inside the object.
    for partition_id in range(1, len(nearest_ids) + 1):
        region = partition == partition_id
        pieces, piece_count = label_mask(region, connectivity=8)
        if piece_count <= 1:
            continue
        areas = np.bincount(pieces.ravel(), minlength=piece_count + 1)
        areas[0] = 0
        with_seeds = np.unique(pieces[region & active_seeds])
        with_seeds = with_seeds[with_seeds > 0]
        if with_seeds.size:
            main = int(with_seeds[int(np.argmax(areas[with_seeds]))])
        else:
            main = int(1 + np.argmax(areas[1:]))
        partition[region & (pieces != main)] = 0

    missing = local_mask & (partition == 0)
    if not bool(missing.any()):
        return partition

    height, width = local_mask.shape
    queue: deque[tuple[int, int, int]] = deque()
    frontier: list[tuple[int, int, int]] = []
    for y, x in np.argwhere(missing).tolist():
        neighbours = partition[
            max(0, y - 1) : min(height, y + 2), max(0, x - 1) : min(width, x + 2)
        ]
        positive = neighbours[neighbours > 0]
        if positive.size:
            frontier.append((y, x, int(positive.min())))
    for y, x, partition_id in frontier:
        if partition[y, x] == 0:
            partition[y, x] = partition_id
            queue.append((y, x, partition_id))
    offsets = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    while queue:
        y, x, partition_id = queue.popleft()
        for dy, dx in offsets:
            ny, nx = y + dy, x + dx
            if (
                0 <= ny < height
                and 0 <= nx < width
                and local_mask[ny, nx]
                and partition[ny, nx] == 0
            ):
                partition[ny, nx] = partition_id
                queue.append((ny, nx, partition_id))
    if bool(np.any(local_mask & (partition == 0))):
        raise RuntimeError("the connectivity repair did not cover the whole object")
    return partition


def split_touching_objects(
    mask: np.ndarray,
    pixel_size_um: float,
    *,
    peak_distance_um: float,
    peak_height: float = 0.28,
    protect_continuous: bool = False,
    hole_protection: float = 0.025,
    elongation_protection: float = 8.0,
    min_outer_solidity: float = 0.60,
    straight_solidity: float = 0.92,
    straight_aspect: float = 3.0,
    thickness_ratio: float = 1.18,
    unsupported_area_ratio: float = 0.38,
) -> np.ndarray:
    """Split touching objects by distance-transform peaks and a constrained Voronoi partition.

    Each 8-connected component is handled inside its own bounding box, so a partition
    can never cross into a neighbouring component. Per component:

    1. Protection. A component is kept whole when its enclosed holes make up at least
       ``hole_protection`` of its filled area, when ``protect_continuous`` is set, or
       when it is a continuous structure rather than a cluster of objects: elongation
       at least ``elongation_protection`` together with either a low solidity
       (at most ``min_outer_solidity``, a bent fibre) or a straight, uniformly thick
       strip (aspect at least ``straight_aspect``, solidity at least
       ``straight_solidity`` and a maximum-to-mean thickness ratio at most
       ``thickness_ratio``).
    2. Peaks. The peaks of the Euclidean distance transform: pixels that are a local
       maximum within ``peak_distance_um`` and at least ``peak_height`` of the
       component's maximum distance. When no pixel qualifies, the single deepest
       pixel is used. Fewer than two peak components means no split.
    3. Partition. Every object pixel is assigned to its nearest peak (Euclidean
       Voronoi restricted to the component).
    4. Unsupported peaks. With three or more cells, a cell whose area is less than
       ``unsupported_area_ratio`` of the disk implied by its peak radius is an
       artefact of several lobes overlapping, not an object. Those peaks are dropped
       and the partition recomputed, never below two parts.
    5. Repair. Each cell is reduced to the connected piece holding its peak and the
       remaining pixels are flooded from their neighbours, so no object pixel is lost
       and no cell is left in several pieces.

    Parameters
    ----------
    mask : ndarray
        2-D boolean mask.
    pixel_size_um : float
        Pixel size in micrometres per pixel.
    peak_distance_um : float
        Smallest distance in micrometres between two peaks, that is the smallest
        centre-to-centre distance of two objects that can still be separated.
    peak_height : float, optional
        Smallest peak depth as a fraction of the component's maximum distance
        (dimensionless).
    protect_continuous : bool, optional
        Keep every component whole; use it for fibrous, patchy or ring-shaped
        structures, which are continuous and must not be cut into pieces.
    hole_protection : float, optional
        Hole fraction above which a component is kept whole (dimensionless).
    elongation_protection : float, optional
        Elongation index above which the continuity tests apply (dimensionless).
    min_outer_solidity : float, optional
        Solidity at or below which an elongated component is treated as a bent
        continuous structure (dimensionless).
    straight_solidity : float, optional
        Solidity at or above which an elongated component may be a straight strip
        (dimensionless).
    straight_aspect : float, optional
        Aspect ratio at or above which an elongated component may be a straight strip
        (dimensionless).
    thickness_ratio : float, optional
        Maximum-to-mean thickness ratio at or below which a strip counts as uniformly
        thick (dimensionless).
    unsupported_area_ratio : float, optional
        Smallest ratio of a cell's area to the disk implied by its peak radius
        (dimensionless).

    Returns
    -------
    ndarray
        ``int32`` label image, 0 in the background and objects numbered from 1.
    """
    work = as_bool_mask(mask)
    components, count = label_mask(work, connectivity=8)
    output = np.zeros(work.shape, dtype=np.int32)
    if count == 0:
        return output
    radius = max(1, _radius_px(peak_distance_um, pixel_size_um, minimum=1))
    peak_footprint = disk(radius)
    boxes = ndi.find_objects(components)
    next_label = 1

    for component_id in range(1, count + 1):
        window = boxes[component_id - 1]
        local_mask = components[window] == component_id
        target = output[window]
        area = int(np.count_nonzero(local_mask))
        distance = _padded_distance(local_mask)
        hole_area = _hole_area(local_mask)
        hole_fraction = float(hole_area / max(area + hole_area, 1))
        mean_thickness, max_thickness, elongation = _thickness(local_mask, distance, area)

        aspect = 1.0
        solidity = 1.0
        if not protect_continuous and elongation >= float(elongation_protection):
            aspect = _covariance_aspect(local_mask)
            solidity = _pixel_solidity(local_mask)
        keep_whole = (
            protect_continuous
            or hole_fraction >= float(hole_protection)
            or (
                elongation >= float(elongation_protection)
                and (
                    solidity <= float(min_outer_solidity)
                    or (
                        aspect >= float(straight_aspect)
                        and solidity >= float(straight_solidity)
                        and max_thickness / max(mean_thickness, 1.0)
                        <= float(thickness_ratio)
                    )
                )
            )
        )
        if keep_whole:
            target[local_mask] = next_label
            next_label += 1
            continue

        component_max = float(distance.max(initial=0.0))
        local_peak = ndi.grey_dilation(
            distance, footprint=peak_footprint, mode="constant", cval=-np.inf
        )
        peaks = (
            local_mask
            & (distance >= local_peak - np.float32(1e-5))
            & (distance >= np.float32(component_max * float(peak_height)))
            & (distance > 0)
        )
        if not bool(peaks.any()):
            deepest = np.unravel_index(int(np.argmax(distance)), distance.shape)
            peaks[deepest] = True
        seed_labels, seed_count = label_mask(peaks, connectivity=8)
        if seed_count <= 1:
            target[local_mask] = next_label
            next_label += 1
            continue

        nearest = _nearest_seed(seed_labels)
        nearest_ids = np.unique(nearest[local_mask])
        active_seeds = seed_labels > 0
        if nearest_ids.size >= 3:
            peak_radius: dict[int, float] = {}
            for seed_id in range(1, seed_count + 1):
                seed_y, seed_x = np.nonzero(seed_labels == seed_id)
                if seed_y.size == 0:
                    continue
                best = int(np.argmax(distance[seed_y, seed_x]))
                nearest_id = int(nearest[seed_y[best], seed_x[best]])
                peak_radius[nearest_id] = max(
                    peak_radius.get(nearest_id, 0.0),
                    float(distance[seed_y[best], seed_x[best]]),
                )
            unsupported = {
                int(nearest_id)
                for nearest_id in nearest_ids.tolist()
                if int(np.count_nonzero(local_mask & (nearest == nearest_id)))
                / max(math.pi * peak_radius.get(int(nearest_id), 0.0) ** 2, 1.0)
                < float(unsupported_area_ratio)
            }
            if unsupported and nearest_ids.size - len(unsupported) >= 2:
                retained = np.isin(seed_labels, sorted(set(nearest_ids.tolist()) - unsupported))
                seed_kept = np.where(retained, seed_labels, 0)
                nearest = _nearest_seed(seed_kept)
                nearest_ids = np.unique(nearest[local_mask])
                active_seeds = retained

        partition = _repair_partition_connectivity(local_mask, nearest, active_seeds)
        for local_label in range(1, int(partition.max(initial=0)) + 1):
            selected = partition == local_label
            if bool(selected.any()):
                target[selected] = next_label
                next_label += 1
    return output


def _nearest_seed(seed_labels: np.ndarray) -> np.ndarray:
    """Label of the nearest seed pixel for every pixel (Euclidean Voronoi)."""
    _, indices = ndi.distance_transform_edt(seed_labels == 0, return_indices=True)
    return seed_labels[indices[0], indices[1]]


def label_objects(
    mask: np.ndarray,
    pixel_size_um: float,
    *,
    split: bool = False,
    peak_distance_um: float | None = None,
    **split_kwargs: float | bool,
) -> np.ndarray:
    """Label a mask, either as plain 8-connected components or with touching objects split.

    Parameters
    ----------
    mask : ndarray
        2-D boolean mask.
    pixel_size_um : float
        Pixel size in micrometres per pixel.
    split : bool, optional
        Split touching objects with :func:`split_touching_objects`.
    peak_distance_um : float, optional
        Smallest distance in micrometres between two peaks, required when
        ``split=True``.
    **split_kwargs
        Further keyword arguments of :func:`split_touching_objects`.

    Returns
    -------
    ndarray
        ``int32`` label image.

    Notes
    -----
    With ``split=True`` the splitting arguments are checked whether or not the mask
    holds anything: an empty mask still yields an empty label image, but a missing
    ``peak_distance_um`` or a misspelled name in ``**split_kwargs`` fails on every
    image rather than only on those that happen to contain an object.
    """
    work = as_bool_mask(mask)
    if split:
        if peak_distance_um is None:
            raise ValueError("split=True needs peak_distance_um")
        return split_touching_objects(
            work, pixel_size_um, peak_distance_um=peak_distance_um, **split_kwargs
        )
    labels, _ = label_mask(work, connectivity=8)
    return labels.astype(np.int32, copy=False)


# --------------------------------------------------------------------------- measurement

def _ring_mask(
    component: np.ndarray,
    occupied: np.ndarray,
    inner_px: float,
    outer_px: float,
    min_ring_px: int,
) -> np.ndarray:
    """Ring just outside an object, excluding every object pixel.

    The ring runs from ``inner_px`` to ``outer_px`` outside the object's own boundary.
    Pixels belonging to any object are removed, so a neighbour cannot be mistaken for
    this object's background. When too few pixels survive, the gap is dropped and the
    whole outer collar is used instead.
    """
    outer = ndi.binary_dilation(component, structure=disk(max(1.0, round(outer_px))))
    inner = ndi.binary_dilation(component, structure=disk(max(1.0, round(inner_px))))
    ring = outer & (~inner) & (~occupied)
    if int(np.count_nonzero(ring)) < int(min_ring_px):
        ring = outer & (~component)
    return ring


def measure_objects(
    labels: np.ndarray,
    intensity: np.ndarray,
    pixel_size_um: float,
    *,
    response: np.ndarray | None = None,
    response_reference: float | None = None,
    noise_sigma: float | None = None,
    ring_inner_um: float = 1.0,
    ring_outer_um: float = 3.0,
    min_ring_px: int = 5,
) -> pd.DataFrame:
    """Measure every labelled object: size, shape, intensity and local contrast.

    The centroid is the mean of the object's pixel coordinates. The axes are ``4 * sqrt(eigenvalue)`` of the pixel
    coordinate covariance regularised by the unit-square term ``I / 12``, so a single
    pixel has a finite width. Solidity is the object's area divided by the area of its
    convex hull, both as pixel counts. The equivalent diameter is the diameter of the
    disk of the same area.

    Local contrast is measured against a ring just outside the object, between
    ``ring_inner_um`` and ``ring_outer_um``, from which every object pixel is removed:
    ``local_background`` is the mean intensity in that ring and
    ``boundary_contrast`` is the object's mean intensity minus it. Given a noise
    scale, ``snr`` is that contrast in noise standard deviations.

    Parameters
    ----------
    labels : ndarray
        2-D ``int`` label image, 0 in the background.
    intensity : ndarray
        2-D intensity plane in the grey levels of the input, same shape as ``labels``.
    pixel_size_um : float
        Pixel size in micrometres per pixel.
    response : ndarray, optional
        2-D response plane the objects were detected on, in its own units. Adds the
        response columns.
    response_reference : float, optional
        Level the response is compared with for ``response_confidence``, in the units
        of the response (the seed level of the threshold that produced the mask).
    noise_sigma : float, optional
        Noise scale in the grey levels of ``intensity``, used for ``snr``. It is a
        property of the images and has no generic value.
    ring_inner_um : float, optional
        Inner radius of the contrast ring in micrometres, measured outward from the
        object boundary.
    ring_outer_um : float, optional
        Outer radius of the contrast ring in micrometres.
    min_ring_px : int, optional
        Smallest usable ring in pixels before the inner gap is dropped.

    Returns
    -------
    pandas.DataFrame
        One row per label in ascending label order, with the columns of
        ``MEASUREMENT_COLUMNS``: areas in pixels and square micrometres, pixel
        coordinates and bounding box in pixels, lengths in micrometres,
        dimensionless shape ratios, intensities in the grey levels of ``intensity``
        and the response columns.
    """
    label_image = check_2d(labels, "labels").astype(np.int32, copy=False)
    plane = as_float(check_2d(intensity, "intensity"), np.float64)
    check_same_shape(label_image, plane)
    if response is not None:
        response_plane = as_float(check_2d(response, "response"), np.float64)
        check_same_shape(label_image, response_plane)
    else:
        response_plane = None

    inner_px = um_to_px(ring_inner_um, pixel_size_um)
    outer_px = um_to_px(ring_outer_um, pixel_size_um)
    pad = int(math.ceil(outer_px)) + 2
    occupied_full = label_image > 0
    height, width = label_image.shape
    boxes = ndi.find_objects(label_image)
    rows: list[dict[str, object]] = []

    for index, window in enumerate(boxes):
        label_id = index + 1
        if window is None:
            continue
        y0, y1 = window[0].start, window[0].stop
        x0, x1 = window[1].start, window[1].stop
        py0, py1 = max(0, y0 - pad), min(height, y1 + pad)
        px0, px1 = max(0, x0 - pad), min(width, x1 + pad)
        crop = label_image[py0:py1, px0:px1]
        component = crop == label_id
        area = int(np.count_nonzero(component))
        if area == 0:
            continue
        yy, xx = np.nonzero(component)
        values = plane[py0:py1, px0:px1][component]

        centroid_y = float(yy.mean()) + py0
        centroid_x = float(xx.mean()) + px0
        if area >= 2:
            covariance = np.asarray(
                np.cov(np.vstack((xx, yy)), bias=True), dtype=np.float64
            )
        else:
            covariance = np.zeros((2, 2), dtype=np.float64)
        covariance = covariance + np.eye(2, dtype=np.float64) / 12.0
        eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(covariance), 0.0))[::-1]
        major_px = 4.0 * math.sqrt(float(eigenvalues[0]))
        minor_px = 4.0 * math.sqrt(float(eigenvalues[1]))

        ring = _ring_mask(
            component,
            occupied_full[py0:py1, px0:px1],
            inner_px,
            outer_px,
            min_ring_px,
        )
        ring_values = plane[py0:py1, px0:px1][ring]
        ring_area = int(ring_values.size)
        local_bg = float(ring_values.mean()) if ring_area else float("nan")
        mean_intensity = float(values.mean())
        contrast = mean_intensity - local_bg if ring_area else float("nan")

        row: dict[str, object] = {
            "id": int(label_id),
            "area_px": area,
            "area_um2": float(area_px_to_um2(area, pixel_size_um)),
            "centroid_x_px": centroid_x,
            "centroid_y_px": centroid_y,
            "bbox_x_px": int(x0),
            "bbox_y_px": int(y0),
            "bbox_width_px": int(x1 - x0),
            "bbox_height_px": int(y1 - y0),
            "equivalent_diameter_um": float(
                px_to_um(math.sqrt(4.0 * area / math.pi), pixel_size_um)
            ),
            "major_axis_um": float(px_to_um(major_px, pixel_size_um)),
            "minor_axis_um": float(px_to_um(minor_px, pixel_size_um)),
            "aspect_ratio": float(major_px / max(minor_px, 1e-6)),
            "perimeter_um": float(
                px_to_um(float(sk_measure.perimeter(component)), pixel_size_um)
            ),
            "solidity": _pixel_solidity(component),
            "touches_border": bool(x0 == 0 or y0 == 0 or x1 >= width or y1 >= height),
            "mean_intensity": mean_intensity,
            "max_intensity": float(values.max()),
            "integrated_intensity": float(np.sum(values, dtype=np.float64)),
            "local_background": local_bg,
            "boundary_contrast": contrast,
            "ring_area_px": ring_area,
            "snr": (
                float(contrast / float(noise_sigma))
                if noise_sigma is not None and float(noise_sigma) > 0.0 and ring_area
                else float("nan")
            ),
        }
        if response_plane is not None:
            response_values = response_plane[py0:py1, px0:px1][component]
            response_mean = float(response_values.mean())
            row["response_mean"] = response_mean
            row["response_max"] = float(response_values.max())
            row["response_confidence"] = (
                float(response_mean / float(response_reference))
                if response_reference is not None
                and math.isfinite(float(response_reference))
                and float(response_reference) > 1e-8
                else float("nan")
            )
        else:
            row["response_mean"] = float("nan")
            row["response_max"] = float("nan")
            row["response_confidence"] = float("nan")
        rows.append(row)

    table = pd.DataFrame(rows, columns=list(MEASUREMENT_COLUMNS))
    if table.empty:
        table = table.astype(
            {
                "id": "int64",
                "area_px": "int64",
                "bbox_x_px": "int64",
                "bbox_y_px": "int64",
                "bbox_width_px": "int64",
                "bbox_height_px": "int64",
                "ring_area_px": "int64",
                "touches_border": "bool",
            }
        )
    return table.reset_index(drop=True)


def _require_measured(
    table: pd.DataFrame, column: str, criterion: str, measured_with: str
) -> None:
    """Refuse a criterion whose column the measurement step never populated.

    A column that is undefined in every row compares false against any bound, so the
    criterion would quietly reject every object instead of selecting among them. That
    is a mistake in the measurement call, so it is reported here rather than returned
    as an empty table.
    """
    if column not in table.columns:
        raise ValueError(
            f"{criterion} needs the {column!r} column, which is missing from the "
            f"table; measure the objects with measure_objects({measured_with})"
        )
    values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=np.float64)
    if values.size and not bool(np.isfinite(values).any()):
        raise ValueError(
            f"{criterion} cannot be applied: {column!r} is undefined in every row, "
            f"which is what measure_objects returns when it is called without "
            f"{measured_with}; applying the criterion would reject every object. "
            f"Re-measure with measure_objects({measured_with}), or drop the criterion"
        )


def filter_objects(
    labels: np.ndarray,
    table: pd.DataFrame,
    pixel_size_um: float,
    *,
    min_area_um2: float = 0.0,
    max_area_um2: float = 0.0,
    max_aspect_ratio: float | None = None,
    min_solidity: float = 0.0,
    min_mean_intensity: float | None = None,
    min_snr: float | None = None,
    min_response_confidence: float = 0.0,
    remove_border: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Keep the objects that pass every criterion and renumber the survivors.

    A criterion left at its neutral value is not applied. Survivors are renumbered
    1 to n in ascending order of their original label, so the label image and the
    table stay in step.

    Two criteria read a column that :func:`measure_objects` fills only when it is
    given the corresponding input: ``min_snr`` needs ``noise_sigma`` and
    ``min_response_confidence`` needs ``response`` together with
    ``response_reference``. Without them the column is undefined in every row and the
    criterion would reject every object, so it raises instead of returning an empty
    table. An individual row whose value is undefined (an object with no usable
    contrast ring, for instance) is dropped by the criterion as before.

    Parameters
    ----------
    labels : ndarray
        2-D label image the table was measured on.
    table : pandas.DataFrame
        Measurement table from :func:`measure_objects`.
    pixel_size_um : float
        Pixel size in micrometres per pixel.
    min_area_um2 : float, optional
        Smallest object area in square micrometres. 0 switches it off.
    max_area_um2 : float, optional
        Largest object area in square micrometres. 0 switches it off.
    max_aspect_ratio : float, optional
        Largest ratio of major to minor axis (dimensionless).
    min_solidity : float, optional
        Smallest ratio of area to convex-hull area (dimensionless).
    min_mean_intensity : float, optional
        Smallest mean intensity, in the grey levels the table was measured in.
    min_snr : float, optional
        Smallest boundary contrast in noise standard deviations (dimensionless).
        Needs a table measured with ``noise_sigma``.
    min_response_confidence : float, optional
        Smallest mean response as a fraction of the threshold reference
        (dimensionless). 0 switches it off. Needs a table measured with ``response``
        and ``response_reference``.
    remove_border : bool, optional
        Drop objects touching any frame edge.

    Returns
    -------
    labels : ndarray
        ``int32`` label image with the survivors numbered 1 to n.
    table : pandas.DataFrame
        The surviving rows, with ``id`` renumbered to match.

    Raises
    ------
    ValueError
        If ``min_snr`` or ``min_response_confidence`` is asked for on a table whose
        column the measurement step never populated.
    """
    label_image = check_2d(labels, "labels").astype(np.int32, copy=False)
    if table.empty:
        return np.zeros(label_image.shape, dtype=np.int32), table.reset_index(drop=True)
    keep = pd.Series(True, index=table.index)
    min_area_px = _area_px(min_area_um2, pixel_size_um)
    if min_area_px > 0:
        keep &= table["area_px"] >= min_area_px
    if float(max_area_um2) > 0.0:
        keep &= table["area_px"] <= _area_px(max_area_um2, pixel_size_um)
    if max_aspect_ratio is not None:
        keep &= table["aspect_ratio"] <= float(max_aspect_ratio)
    if float(min_solidity) > 0.0:
        keep &= table["solidity"] >= float(min_solidity)
    if min_mean_intensity is not None:
        keep &= table["mean_intensity"] >= float(min_mean_intensity)
    if min_snr is not None:
        _require_measured(table, "snr", "min_snr", "noise_sigma=...")
        keep &= table["snr"] >= float(min_snr)
    if float(min_response_confidence) > 0.0:
        _require_measured(
            table,
            "response_confidence",
            "min_response_confidence",
            "response=..., response_reference=...",
        )
        keep &= table["response_confidence"] >= float(min_response_confidence)
    if remove_border:
        keep &= ~table["touches_border"].astype(bool)

    survivors = table.loc[keep.to_numpy(dtype=bool)].sort_values("id")
    max_label = int(label_image.max(initial=0))
    mapping = np.zeros(max_label + 1, dtype=np.int32)
    kept_ids = survivors["id"].to_numpy(dtype=np.int64)
    if kept_ids.size:
        mapping[kept_ids] = np.arange(1, kept_ids.size + 1, dtype=np.int32)
    final_labels = mapping[label_image]
    final_table = survivors.copy().reset_index(drop=True)
    final_table["id"] = np.arange(1, len(final_table) + 1, dtype=np.int64)
    return final_labels, final_table


# --------------------------------------------------------------------------- pipelines

def segment(
    image: np.ndarray,
    pixel_size_um: float,
    *,
    background_radius_um: float,
    response_mode: str = "snr",
    denoise_method: str = "gaussian",
    denoise_um: float = 0.0,
    invert: bool = False,
    noise_scale: float | None = None,
    background_method: str = "gaussian",
    background_strength: float = 1.0,
    compartment_mode: str = "off",
    compartment_ceiling: float | None = None,
    compartment_scale_um: float | None = None,
    threshold_method: str = "snr_hysteresis",
    seed_sigma: float = 5.0,
    sensitivity: float = 50.0,
    grow_ratio: float | None = None,
    threshold_value: float | None = None,
    local_window_um: float | None = None,
    local_k: float = 0.20,
    evidence_filter: bool | None = None,
    evidence_mean_fraction: float = 0.48,
    evidence_p90_fraction: float = 0.64,
    open_um: float = 0.0,
    close_um: float = 0.0,
    erode_um: float = 0.0,
    dilate_um: float = 0.0,
    max_hole_area_um2: float = 0.0,
    remove_border: bool = False,
    split: bool = False,
    peak_distance_um: float | None = None,
    peak_height: float = 0.28,
    protect_continuous: bool = False,
    ring_inner_um: float = 1.0,
    ring_outer_um: float = 3.0,
    object_noise_sigma: float | None = None,
    min_area_um2: float = 0.0,
    max_area_um2: float = 0.0,
    max_aspect_ratio: float | None = None,
    min_solidity: float = 0.0,
    min_mean_intensity: float | None = None,
    min_snr: float | None = None,
    min_response_confidence: float = 0.0,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Segment one image plane end to end and measure the objects.

    The pipeline is: response, compartment gate, threshold, mask clean-up, evidence
    filter, optional border removal, labelling (plain or split), measurement, object
    filters. It is deterministic and uses no random numbers.

    ``response_mode='snr'`` builds the response with :func:`robust_snr_response`, so
    the response is in noise standard deviations and the default threshold seeds at a
    fixed contrast. ``response_mode='background_correction'`` builds it with
    :func:`if_qsaa.background.background_correction` on the denoised plane and
    rescales it to the unit interval with
    :func:`if_qsaa.background.robust_rescale`, which is the path the global and local
    threshold rules expect.

    Parameters
    ----------
    image : ndarray
        2-D image plane in the grey levels of the input.
    pixel_size_um : float
        Pixel size in micrometres per pixel.
    background_radius_um : float
        Local background scale in micrometres. It is the physical scale of the
        background the objects sit on and has no generic value.
    response_mode : {'snr', 'background_correction'}, optional
        How the response is built.
    denoise_method : {'none', 'gaussian', 'median'}, optional
        Pre-threshold denoising filter.
    denoise_um : float, optional
        Denoising filter size in micrometres. 0 switches it off.
    invert : bool, optional
        Segment dark objects on a bright background.
    noise_scale : float, optional
        Shared absolute noise scale in grey levels for ``response_mode='snr'``.
    background_method : str, optional
        Background model for ``response_mode='background_correction'``; see
        :func:`if_qsaa.background.background_correction`.
    background_strength : float, optional
        Fraction of the background subtracted (dimensionless).
    compartment_mode : {'off', 'absolute', 'auto'}, optional
        Broad-background compartment gate; see
        :func:`if_qsaa.background.compartment_gate`.
    compartment_ceiling : float, optional
        Absolute ceiling on the broad background in grey levels, for
        ``compartment_mode='absolute'``.
    compartment_scale_um : float, optional
        Compartment scale in micrometres. Defaults to ``background_radius_um``.
    threshold_method : str, optional
        Threshold rule; see :func:`threshold_mask`.
    seed_sigma : float, optional
        Seed contrast in noise standard deviations (dimensionless).
    sensitivity : float, optional
        Sensitivity on a 0 to 100 scale (dimensionless), 50 being neutral.
    grow_ratio : float, optional
        Candidate level as a fraction of the seed level (dimensionless). ``None``
        uses the default of the chosen rule; see :func:`threshold_mask`.
    threshold_value : float, optional
        Fixed level for ``threshold_method='value'``, in the units of the response.
        It is applied literally unless ``grow_ratio`` is set below 1.
    local_window_um : float, optional
        Local window side in micrometres for ``threshold_method='local_mean_std'``.
    local_k : float, optional
        Number of local standard deviations above the local mean (dimensionless).
    evidence_filter : bool, optional
        Apply the SNR evidence filter. ``None`` applies it for
        ``response_mode='snr'`` only, the path whose seed level it needs.
    evidence_mean_fraction, evidence_p90_fraction : float, optional
        Fractions of the seed level used by :func:`filter_components_by_evidence`.
    open_um, close_um, erode_um, dilate_um : float, optional
        Clean-up radii in micrometres.
    max_hole_area_um2 : float, optional
        Largest enclosed hole filled, in square micrometres.
    remove_border : bool, optional
        Remove whole objects touching any frame edge.
    split : bool, optional
        Split touching objects.
    peak_distance_um : float, optional
        Smallest distance in micrometres between two peaks, required by ``split``.
    peak_height : float, optional
        Smallest peak depth as a fraction of the maximum distance (dimensionless).
    protect_continuous : bool, optional
        Keep continuous structures whole when splitting.
    ring_inner_um, ring_outer_um : float, optional
        Radii of the local-contrast ring in micrometres.
    object_noise_sigma : float, optional
        Noise scale in grey levels used for the per-object ``snr`` column. It is
        required by ``min_snr``, which has nothing to compare against without it.
    min_area_um2, max_area_um2 : float, optional
        Object area limits in square micrometres.
    max_aspect_ratio, min_solidity, min_mean_intensity, min_snr, min_response_confidence
        Object filters; see :func:`filter_objects`.

    Returns
    -------
    labels : ndarray
        ``int32`` label image of the surviving objects, numbered 1 to n.
    table : pandas.DataFrame
        Measurement table of the surviving objects.
    """
    plane = as_float(check_2d(image), np.float32)
    if response_mode == "snr":
        response = robust_snr_response(
            plane,
            pixel_size_um,
            background_radius_um=background_radius_um,
            denoise_method=denoise_method,
            denoise_um=denoise_um,
            invert=invert,
            noise_scale=noise_scale,
        )
    elif response_mode == "background_correction":
        work = np.negative(plane) if invert else plane
        denoise_px = (
            um_to_px(denoise_um, pixel_size_um) if float(denoise_um) > 0.0 else 0.0
        )
        denoised = _denoise(work, denoise_method, denoise_px)
        response = bg.robust_rescale(
            bg.background_correction(
                denoised,
                background_radius_um,
                pixel_size_um,
                method=background_method,
                strength=background_strength,
            )
        )
    else:
        raise ValueError("response_mode must be 'snr' or 'background_correction'")

    if compartment_mode != "off":
        response = bg.compartment_gate(
            response,
            plane,
            background_radius_um if compartment_scale_um is None else compartment_scale_um,
            pixel_size_um,
            mode=compartment_mode,
            ceiling=compartment_ceiling,
        )

    mask, high, low, reference = threshold_mask(
        response,
        method=threshold_method,
        seed_sigma=seed_sigma,
        sensitivity=sensitivity,
        grow_ratio=grow_ratio,
        value=threshold_value,
        local_window_um=local_window_um,
        pixel_size_um=pixel_size_um,
        local_k=local_k,
        clip_range=(0.0, 1.0) if response_mode == "background_correction" else None,
    )
    mask = clean_mask(
        mask,
        pixel_size_um,
        open_um=open_um,
        close_um=close_um,
        erode_um=erode_um,
        dilate_um=dilate_um,
        max_hole_area_um2=max_hole_area_um2,
    )
    use_evidence = (response_mode == "snr") if evidence_filter is None else bool(evidence_filter)
    if use_evidence:
        mask = filter_components_by_evidence(
            mask,
            response,
            high,
            low,
            mean_fraction=evidence_mean_fraction,
            p90_fraction=evidence_p90_fraction,
        )
    if remove_border:
        mask = remove_border_objects(mask)

    labels = label_objects(
        mask,
        pixel_size_um,
        split=split,
        peak_distance_um=peak_distance_um,
        peak_height=peak_height,
        protect_continuous=protect_continuous,
    )
    table = measure_objects(
        labels,
        plane,
        pixel_size_um,
        response=response,
        response_reference=reference,
        noise_sigma=object_noise_sigma,
        ring_inner_um=ring_inner_um,
        ring_outer_um=ring_outer_um,
    )
    return filter_objects(
        labels,
        table,
        pixel_size_um,
        min_area_um2=min_area_um2,
        max_area_um2=max_area_um2,
        max_aspect_ratio=max_aspect_ratio,
        min_solidity=min_solidity,
        min_mean_intensity=min_mean_intensity,
        min_snr=min_snr,
        min_response_confidence=min_response_confidence,
        remove_border=remove_border,
    )


def measure_external_mask(
    image: np.ndarray,
    mask: np.ndarray,
    pixel_size_um: float,
    *,
    response: np.ndarray | None = None,
    response_reference: float | None = None,
    noise_sigma: float | None = None,
    split: bool = False,
    peak_distance_um: float | None = None,
    peak_height: float = 0.28,
    protect_continuous: bool = False,
    ring_inner_um: float = 1.0,
    ring_outer_um: float = 3.0,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Label and measure a mask supplied from outside, applying no object filter.

    Use it to bring your own masks, hand-curated or produced by another tool, into the
    same measurement table as :func:`segment`. Every component is measured and kept;
    pass the table through :func:`filter_objects` if criteria should be applied.

    Parameters
    ----------
    image : ndarray
        2-D intensity plane in the grey levels of the input.
    mask : ndarray
        2-D boolean mask of the same shape.
    pixel_size_um : float
        Pixel size in micrometres per pixel.
    response : ndarray, optional
        2-D response plane to measure alongside, in its own units.
    response_reference : float, optional
        Level the response is compared with, in the units of the response.
    noise_sigma : float, optional
        Noise scale in grey levels used for the per-object ``snr`` column.
    split : bool, optional
        Split touching objects instead of labelling plain components.
    peak_distance_um : float, optional
        Smallest distance in micrometres between two peaks, required by ``split``.
    peak_height : float, optional
        Smallest peak depth as a fraction of the maximum distance (dimensionless).
    protect_continuous : bool, optional
        Keep continuous structures whole when splitting.
    ring_inner_um, ring_outer_um : float, optional
        Radii of the local-contrast ring in micrometres.

    Returns
    -------
    labels : ndarray
        ``int32`` label image of the supplied mask.
    table : pandas.DataFrame
        Measurement table of every component.
    """
    plane = as_float(check_2d(image), np.float32)
    work = as_bool_mask(mask)
    check_same_shape(plane, work)
    labels = label_objects(
        work,
        pixel_size_um,
        split=split,
        peak_distance_um=peak_distance_um,
        peak_height=peak_height,
        protect_continuous=protect_continuous,
    )
    table = measure_objects(
        labels,
        plane,
        pixel_size_um,
        response=response,
        response_reference=response_reference,
        noise_sigma=noise_sigma,
        ring_inner_um=ring_inner_um,
        ring_outer_um=ring_outer_um,
    )
    return labels, table
