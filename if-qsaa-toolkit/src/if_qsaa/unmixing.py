"""Spectral cross-talk handling for multichannel fluorescence images of tissue sections.

Three models of cross-talk are implemented, together with the object filters and the
robustness switch that belong with them.

Fixed-coefficient subtraction
    The model of the Methods: for every detection channel ``c`` other than the channel
    carrying the bright donor signal,

    ``I_corrected = max(0, I_observed - a[c] * I_source)``

    pixel-wise on co-registered images acquired with identical exposure, detector gain
    and offset. The coefficients ``a[c]`` are properties of the dye, the filter set and
    the acquisition settings, so they are always supplied by the caller
    (:func:`subtract_crosstalk`, :func:`unmix_fixed`). A background pedestal in the
    source channel is scaled by the coefficient like any other source intensity, so it
    is subtracted from the whole observed channel as a constant offset; remove it from
    the source first (``source_background``, in grey levels, or
    ``source_background_percentile``) when the images still carry one.

Coefficient calibration
    The Methods calibrate the apparent channel-coupling coefficients on single-source
    and co-stained sections: in a channel that carries no signal of its own, the
    observed intensity is proportional to the source intensity, and the coefficient is
    the slope of that relation over source-positive, unsaturated pixels. Both a
    least-squares and a robust estimator are offered, because a co-stained section
    always contains some pixels with genuine signal in the observed channel
    (:func:`calibrate_coupling_coefficient`, :func:`calibrate_coupling`).

Per-field empirical estimation and two-channel linear unmixing
    When no single-source control of the same section exists, the three coefficients of
    a readout / marker / nuclear triple can be estimated from the field itself and the
    readout-marker pair inverted as a 2x2 linear system. The order of the steps is

    1. ``delta`` (readout into the nuclear channel) from the lower envelope of the
       nuclear-to-readout ratio over bright, unsaturated readout pixels, and
       ``nuclear_corr = nuclear_res - delta * readout_res``
       (:func:`estimate_readout_to_nuclear_bleed`, :func:`subtract_bleed`). This step
       is applied in every field, so that nuclei are masked on a bleed-free channel.
    2. nuclei are masked on ``nuclear_corr`` (segmentation, not part of this module).
    3. ``gamma`` (nuclear into readout) and ``alpha`` (marker into readout) from the
       median, respectively a quantile, of the within-structure robust slopes of the
       observed channel on the source channel, and ``beta`` (readout into marker) from
       the lower envelope of the marker-to-readout ratio over bright readout pixels
       (:func:`estimate_crosstalk`).
    4. inversion of the symmetric 2x2 mixing ``readout' = readout + alpha * marker``,
       ``marker' = marker + beta * readout`` after the nuclear term has been removed
       (:func:`unmix_two_channel`).

    A field with only two channels, a readout and a marker, takes the same route with
    ``nuclear_corr=None``: steps 1 and 2 do not apply, ``gamma`` stays zero and the
    readout-marker pair is inverted as before.

    Object and marker thresholds are *not* recomputed after this unmixing: they stay in
    the noise units of the uncorrected residuals, as in the method this module follows.

Residual cross-talk object filters
    Linear unmixing leaves two kinds of residual artefact, both removed at object level:
    marker components that lie almost entirely inside a dilated readout footprint and
    carry no marker morphology of their own (:func:`remove_marker_shadows`), and readout
    objects that are brighter in another channel than in their own, judged by the
    contrast against a surrounding ring (:func:`object_contrast_table`,
    :func:`spectral_gate`).

Robustness switch
    :func:`run_variants` and :func:`compare_unmixing` re-run a caller-supplied analysis
    with and without the cross-talk handling, so that a conclusion can be reported with
    and without the correction.

Intensity parameters (saturation levels, intensity floors, minimum contrasts) have no
defaults: they depend on the detector scaling and the export of the user's own images.
Dimensionless rules of the estimators (quantiles, minimum numbers of pixels or
structures, the coefficient bound) keep the values of the method.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy import stats

from ._util import (
    as_bool_mask,
    as_float,
    check_2d,
    check_same_shape,
    default_rng,
    disk,
    label_mask,
    pixel_area_um2,
    um_to_px,
)
from .background import censored_noise

__all__ = [
    # fixed-coefficient subtraction
    "native_full_scale",
    "crosstalk_component",
    "subtract_crosstalk",
    "clipped_fraction",
    "unmix_fixed",
    # calibration
    "COUPLING_ESTIMATORS",
    "calibrate_coupling_coefficient",
    "calibrate_coupling",
    # per-field estimation and 2x2 unmixing
    "unsaturated_mask",
    "subtract_bleed",
    "estimate_readout_to_nuclear_bleed",
    "nuclei_mask",
    "component_slopes",
    "estimate_crosstalk",
    "unmix_two_channel",
    # residual cross-talk object filters
    "remove_marker_shadows",
    "footprint_beta_correction",
    "surround_ring",
    "ring_contrast",
    "contrast_ratio",
    "spectral_gate",
    "object_contrast_table",
    # robustness switch
    "unmixing_variants",
    "unmixing_off_overrides",
    "run_variants",
    "compare_unmixing",
]

# Guard against division by a vanishing denominator; not a tuned value.
_DIV_EPS = 1e-3


# --------------------------------------------------------------------------- private

def _disk_px(radius_px: float) -> np.ndarray:
    """Flat disk whose radius is rounded to whole pixels, at least one pixel.

    Rounding the radius keeps a dilation by a length in micrometres reproducible on
    images of different pixel size.
    """
    return disk(float(max(1, int(round(float(radius_px))))))


def _remove_small_components(mask: np.ndarray, min_px: int, connectivity: int = 4) -> np.ndarray:
    """Drop connected components of fewer than ``min_px`` pixels from a binary mask."""
    m = as_bool_mask(mask, "mask")
    if min_px <= 1 or not m.any():
        return m.copy()
    labels, n = label_mask(m, connectivity)
    if n == 0:
        return m.copy()
    areas = np.bincount(labels.ravel())
    keep = np.flatnonzero(areas >= int(min_px))
    keep = keep[keep > 0]
    return np.isin(labels, keep)


def _check_quantile(value: float, name: str) -> float:
    q = float(value)
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {q}")
    return q


def _check_positive(value: float, name: str) -> float:
    v = float(value)
    if not np.isfinite(v) or v <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value}")
    return v


def _noise_location_scale(
    values: np.ndarray, estimator: Callable[[np.ndarray], Sequence[float]] | None
) -> tuple[float, float]:
    """Background location and scale of a 1-D sample, in the intensity units of the input."""
    fn = censored_noise if estimator is None else estimator
    out = fn(np.asarray(values).ravel())
    return float(out[0]), float(out[1])


# ------------------------------------------------------- fixed-coefficient subtraction

def native_full_scale(image: np.ndarray) -> float:
    """Full-scale value of an image, for pairing channels stored at different bit depths.

    Parameters
    ----------
    image : ndarray
        Image of any dtype.

    Returns
    -------
    float
        Maximum of the dtype for integer images, 1 for boolean images, and for floating
        images 1 when no value exceeds 1 (the common 0-1 convention) or else the largest
        absolute value present.

    Notes
    -----
    Pass this value as ``observed_full_scale`` / ``source_full_scale`` when an 8-bit
    channel is paired with a 16-bit channel, so that the coefficient is not applied
    across a hidden scale mismatch.
    """
    arr = np.asarray(image)
    if arr.dtype == np.bool_:
        return 1.0
    if np.issubdtype(arr.dtype, np.integer):
        return max(1.0, float(np.iinfo(arr.dtype).max))
    finite = arr[np.isfinite(arr)]
    observed = float(np.max(np.abs(finite))) if finite.size else 0.0
    return 1.0 if observed <= 1.0 + 1e-6 else observed


def crosstalk_component(
    source: np.ndarray,
    coefficient: float,
    *,
    source_background: float | None = None,
    source_background_percentile: float | None = None,
    source_full_scale: float | None = None,
    observed_full_scale: float | None = None,
) -> np.ndarray:
    """Estimated cross-talk component ``a * I_source`` of one detection channel.

    Parameters
    ----------
    source : ndarray
        2-D image of the channel that carries the donor signal, in its own intensity
        units.
    coefficient : float
        Channel-coupling coefficient ``a`` (dimensionless ratio of intensities), as
        calibrated on single-source or co-stained sections.
    source_background : float or None, optional
        Background level of the source channel, in the grey levels of ``source``, which
        is subtracted from it and clipped at zero before the coefficient is applied.
        This is the pedestal remedy in the units of every other intensity parameter of
        this module; take the level from :mod:`if_qsaa.background` or from the offset of
        your own detector.
    source_background_percentile : float or None, optional
        The same correction expressed as a percentile of the source image (per cent, 0
        to 50), for a field whose background level is not known separately. Pass at most
        one of the two terms. Both are off by default, because the subtraction model of
        the method has no background term; a source image that still carries a pedestal
        needs one of them (see the Notes of :func:`subtract_crosstalk`).
    source_full_scale, observed_full_scale : float or None, optional
        Full-scale intensity of the source and of the observed channel, in their own
        units; the source is rescaled by their ratio. Pass both or neither
        (see :func:`native_full_scale`).

    Returns
    -------
    ndarray
        2-D float image of the estimated component, in the intensity units of the
        observed channel.
    """
    src = as_float(check_2d(source, "source"))
    coef = float(coefficient)
    if not np.isfinite(coef) or coef < 0:
        raise ValueError("coefficient must be a finite non-negative number")
    if (source_full_scale is None) != (observed_full_scale is None):
        raise ValueError("pass both source_full_scale and observed_full_scale, or neither")
    if source_background is not None and source_background_percentile is not None:
        raise ValueError(
            "pass at most one of source_background (grey levels) and "
            "source_background_percentile (per cent)"
        )
    if source_background is not None:
        level = float(source_background)
        if not np.isfinite(level) or level < 0:
            raise ValueError(
                "source_background must be a finite non-negative number of grey levels"
            )
        src = np.maximum(src - level, 0.0)
    if source_full_scale is not None:
        src = src / _check_positive(source_full_scale, "source_full_scale") * _check_positive(
            observed_full_scale, "observed_full_scale"
        )
    if source_background_percentile is not None:
        pct = float(source_background_percentile)
        if not 0.0 <= pct <= 50.0:
            raise ValueError("source_background_percentile must lie in [0, 50]")
        src = np.maximum(src - float(np.percentile(src, pct)), 0.0)
    return coef * src


def subtract_crosstalk(
    observed: np.ndarray,
    source: np.ndarray,
    coefficient: float,
    *,
    clip_negative: bool = True,
    source_background: float | None = None,
    source_background_percentile: float | None = None,
    source_full_scale: float | None = None,
    observed_full_scale: float | None = None,
) -> np.ndarray:
    """Fixed-coefficient cross-talk subtraction ``max(0, I_observed - a * I_source)``.

    Parameters
    ----------
    observed : ndarray
        2-D image of the detection channel to correct, in its own intensity units.
    source : ndarray
        2-D co-registered image of the channel that carries the donor signal, same
        shape. No registration is attempted here.
    coefficient : float
        Channel-coupling coefficient ``a`` (dimensionless), calibrated by the user.
    clip_negative : bool, optional
        Clip the result at zero, as in the subtraction model of the method.
    source_background, source_background_percentile, source_full_scale, \
observed_full_scale
        See :func:`crosstalk_component`.

    Returns
    -------
    ndarray
        2-D float image of the corrected channel, in the intensity units of
        ``observed``.

    Notes
    -----
    The model assumes that both images were acquired with identical exposure, detector
    gain and offset, which is what makes a single coefficient valid for a whole set.

    A background pedestal in the source channel is scaled by the coefficient like any
    other source intensity: a source background of ``b`` grey levels subtracts a
    constant ``a * b`` from the whole observed channel, which lowers its background, the
    noise threshold derived from it and every intensity measured on it, and drives
    background pixels to zero once the result is clipped. Remove the pedestal from the
    source first, with ``source_background`` (grey levels, like every other intensity
    parameter here) or ``source_background_percentile`` (per cent), or give both
    channels as background residuals, in which case neither term is needed.
    """
    obs = as_float(check_2d(observed, "observed"))
    src = check_2d(source, "source")
    check_same_shape(obs, src)
    component = crosstalk_component(
        src,
        coefficient,
        source_background=source_background,
        source_background_percentile=source_background_percentile,
        source_full_scale=source_full_scale,
        observed_full_scale=observed_full_scale,
    )
    out = obs - component
    if clip_negative:
        out = np.maximum(out, 0.0)
    return out


def clipped_fraction(
    observed: np.ndarray,
    source: np.ndarray,
    coefficient: float,
    *,
    source_background: float | None = None,
    source_background_percentile: float | None = None,
    source_full_scale: float | None = None,
    observed_full_scale: float | None = None,
) -> float:
    """Fraction of pixels driven below zero by a fixed-coefficient subtraction.

    Parameters
    ----------
    observed, source, coefficient
        See :func:`subtract_crosstalk`. The diagnostic is meant for raw images, with
        their background pedestal still in place (see the Notes).
    source_background, source_background_percentile, source_full_scale, \
observed_full_scale
        See :func:`crosstalk_component`.

    Returns
    -------
    float
        Fraction of pixels (dimensionless, 0 to 1) whose corrected value would be
        negative before clipping. On raw images a large fraction means the coefficient
        is too high for this image set.

    Notes
    -----
    The reading depends on which input convention ``observed`` follows. On raw images,
    whose background sits well above zero, a pixel can only go negative when more is
    subtracted than the channel holds, so the fraction measures an over-correction. On
    background residuals, which are centred on zero, about half of the background
    pixels fall below zero for any positive coefficient, however well it is calibrated,
    so a fraction near one half carries no information about the coefficient. The
    per-field route of this module (:func:`estimate_crosstalk`,
    :func:`unmix_two_channel`) works on residuals and keeps negative values on purpose;
    judge it by the coefficients and the spread of their per-structure slopes instead.
    """
    obs = as_float(check_2d(observed, "observed"))
    src = check_2d(source, "source")
    check_same_shape(obs, src)
    component = crosstalk_component(
        src,
        coefficient,
        source_background=source_background,
        source_background_percentile=source_background_percentile,
        source_full_scale=source_full_scale,
        observed_full_scale=observed_full_scale,
    )
    return float(np.count_nonzero(obs - component < 0.0) / max(1, obs.size))


def unmix_fixed(
    channels: Mapping[str, np.ndarray],
    source_channel: str,
    coefficients: Mapping[str, float],
    *,
    clip_negative: bool = True,
    source_background: float | None = None,
    source_background_percentile: float | None = None,
    full_scales: Mapping[str, float] | None = None,
) -> tuple[dict, dict]:
    """Apply fixed-coefficient subtraction to several detection channels at once.

    Parameters
    ----------
    channels : mapping of str to ndarray
        Co-registered 2-D images of all detection channels, keyed by channel name, each
        in its own intensity units.
    source_channel : str
        Key of the channel that carries the donor signal; it is never corrected.
    coefficients : mapping of str to float
        Channel-coupling coefficient ``a[c]`` (dimensionless) for each channel to
        correct. Channels absent from this mapping are returned unchanged.
    clip_negative : bool, optional
        Clip each corrected channel at zero, as in the subtraction model of the method.
    source_background, source_background_percentile : float or None, optional
        Pedestal of the source channel, in its grey levels respectively as a percentile
        of the source image; see :func:`crosstalk_component` and the Notes of
        :func:`subtract_crosstalk`.
    full_scales : mapping of str to float or None, optional
        Full-scale intensity per channel (own units), used when channels are stored at
        different bit depths. Must contain the source and every corrected channel.

    Returns
    -------
    corrected : dict of str to ndarray
        One float image per input channel: corrected where a coefficient was given,
        a float copy otherwise.
    components : dict of str to ndarray
        The estimated cross-talk component ``a[c] * I_source`` of each corrected
        channel, in the units of that channel, for display of the subtracted part.
    """
    if source_channel not in channels:
        raise KeyError(f"source channel {source_channel!r} is not in channels")
    if source_channel in coefficients:
        raise ValueError("the source channel cannot be corrected against itself")
    missing = [c for c in coefficients if c not in channels]
    if missing:
        raise KeyError(f"no image for channel(s) {sorted(missing)}")
    source = check_2d(channels[source_channel], "source channel")
    source_scale = None if full_scales is None else full_scales[source_channel]

    corrected: dict = {}
    components: dict = {}
    for name, image in channels.items():
        img = as_float(check_2d(image, f"channel {name!r}"))
        if name not in coefficients:
            corrected[name] = img
            continue
        check_same_shape(img, source)
        observed_scale = None if full_scales is None else full_scales[name]
        components[name] = crosstalk_component(
            source,
            coefficients[name],
            source_background=source_background,
            source_background_percentile=source_background_percentile,
            source_full_scale=source_scale,
            observed_full_scale=observed_scale,
        )
        out = img - components[name]
        corrected[name] = np.maximum(out, 0.0) if clip_negative else out
    return corrected, components


# ------------------------------------------------------------------------ calibration

COUPLING_ESTIMATORS = (
    "least_squares",
    "least_squares_through_origin",
    "theil_sen",
    "median_ratio",
)


def calibrate_coupling_coefficient(
    observed: np.ndarray,
    source: np.ndarray,
    *,
    source_min_level: float,
    observed_max_level: float | None = None,
    saturation_level: float | None = None,
    mask: np.ndarray | None = None,
    estimator: str = "least_squares",
    max_pixels: int | None = None,
    seed: int = 0,
) -> dict:
    """Calibrate one channel-coupling coefficient from a control section.

    The method calibrates the apparent coupling of a bright source into every other
    detection channel on single-source and co-stained sections acquired with identical
    settings. In a channel with no signal of its own the observed intensity is
    proportional to the source intensity, so the coefficient is the slope of the
    observed channel on the source channel over source-positive, unsaturated pixels.

    Parameters
    ----------
    observed : ndarray
        2-D image of the detection channel being calibrated, in its own intensity units.
    source : ndarray
        2-D co-registered image of the source channel, same shape, in its own units.
    source_min_level : float
        Pixels are used where the source exceeds this level, in the intensity units of
        ``source``. Required: the level depends on the user's detector scaling and on
        the background of their control section.
    observed_max_level : float or None, optional
        Pixels are used only where the observed channel is at or below this level, in
        its own intensity units. This is how a co-stained section is handled: it
        excludes pixels that carry genuine signal in the observed channel. ``None``
        suits a single-source section, where no such signal exists.
    saturation_level : float or None, optional
        Pixels at or above this level in either channel are excluded, in the intensity
        units of the images (both channels are tested against the same level).
        ``None`` disables the test.
    mask : ndarray or None, optional
        2-D boolean mask restricting the calibration, for example a tissue mask.
    estimator : {'least_squares', 'least_squares_through_origin', 'theil_sen', \
'median_ratio'}, optional
        ``'least_squares'`` is the ordinary slope and intercept; ``'theil_sen'`` is the
        robust median-of-pairwise-slopes estimator and ``'median_ratio'`` the median of
        the per-pixel ratios, both of which tolerate a minority of pixels with genuine
        signal in the observed channel; the through-origin variant forces a zero
        intercept, appropriate when the background has already been subtracted.
    max_pixels : int or None, optional
        Random subsample size (pixels) when more pixels than this are selected. The
        robust estimators are quadratic in the number of pixels, so a subsample keeps
        them affordable on whole fields.
    seed : int, optional
        Seed of the subsampling generator; a fixed seed makes the calibration
        reproducible.

    Returns
    -------
    dict
        ``coefficient`` (dimensionless slope), ``intercept`` (intensity units of
        ``observed``), ``n_pixels`` (number of pixels the fit used), ``n_selected``
        (number of pixels that passed the selection before subsampling),
        ``estimator`` and ``slope_ci`` (lower and upper confidence limits of the slope
        for the robust estimator, ``None`` otherwise).

    Notes
    -----
    The coefficient is only meaningful for images acquired with identical exposure,
    detector gain and offset; applying it to another acquisition setting requires a new
    calibration.
    """
    if estimator not in COUPLING_ESTIMATORS:
        raise ValueError(f"estimator must be one of {COUPLING_ESTIMATORS}, got {estimator!r}")
    obs = as_float(check_2d(observed, "observed"))
    src = as_float(check_2d(source, "source"))
    check_same_shape(obs, src)

    sel = np.isfinite(obs) & np.isfinite(src) & (src > float(source_min_level))
    if observed_max_level is not None:
        sel &= obs <= float(observed_max_level)
    if saturation_level is not None:
        level = float(saturation_level)
        sel &= (src < level) & (obs < level)
    if mask is not None:
        m = as_bool_mask(mask, "mask")
        check_same_shape(obs, m)
        sel &= m

    x = src[sel]
    y = obs[sel]
    n_selected = int(x.size)
    if n_selected < 2:
        raise ValueError(
            "fewer than two pixels passed the selection; lower source_min_level or "
            "widen the mask"
        )
    if max_pixels is not None and n_selected > int(max_pixels):
        take = default_rng(seed).choice(n_selected, int(max_pixels), replace=False)
        x, y = x[take], y[take]

    slope_ci = None
    if estimator == "least_squares":
        fit = stats.linregress(x, y)
        slope, intercept = float(fit.slope), float(fit.intercept)
    elif estimator == "least_squares_through_origin":
        denominator = float(x @ x)
        if denominator <= 0:
            raise ValueError("the selected source values are all zero")
        slope, intercept = float(x @ y) / denominator, 0.0
    elif estimator == "theil_sen":
        fit = stats.theilslopes(y, x)
        slope, intercept = float(fit.slope), float(fit.intercept)
        slope_ci = (float(fit.low_slope), float(fit.high_slope))
    else:  # median_ratio
        if np.any(x == 0):
            raise ValueError("median_ratio needs a strictly positive source_min_level")
        slope, intercept = float(np.median(y / x)), 0.0
    return dict(
        coefficient=slope,
        intercept=intercept,
        n_pixels=int(x.size),
        n_selected=n_selected,
        estimator=estimator,
        slope_ci=slope_ci,
    )


def calibrate_coupling(
    channels: Mapping[str, np.ndarray],
    source_channel: str,
    *,
    source_min_level: float,
    detectors: Sequence[str] | None = None,
    observed_max_level: float | Mapping[str, float] | None = None,
    **kwargs,
) -> dict:
    """Calibrate the coupling of one source into every other detection channel.

    Parameters
    ----------
    channels : mapping of str to ndarray
        Co-registered 2-D images of all detection channels of one control section.
    source_channel : str
        Key of the channel that carries the source signal.
    source_min_level : float
        Source level above which pixels are used, in the units of the source image
        (see :func:`calibrate_coupling_coefficient`).
    detectors : sequence of str or None, optional
        Channels to calibrate, in the order they should appear in the result. ``None``
        takes every channel except the source, in the order of ``channels``.
    observed_max_level : float, mapping of str to float, or None, optional
        Own-signal level of the observed channels, as one value for all or per channel
        (intensity units of each channel).
    **kwargs
        Passed to :func:`calibrate_coupling_coefficient`.

    Returns
    -------
    dict of str to dict
        One result of :func:`calibrate_coupling_coefficient` per detector.
    """
    if source_channel not in channels:
        raise KeyError(f"source channel {source_channel!r} is not in channels")
    names = [c for c in channels if c != source_channel] if detectors is None else list(detectors)
    source = channels[source_channel]
    out: dict = {}
    for name in names:
        if name not in channels:
            raise KeyError(f"no image for channel {name!r}")
        if isinstance(observed_max_level, Mapping):
            own_level = observed_max_level.get(name)
        else:
            own_level = observed_max_level
        out[name] = calibrate_coupling_coefficient(
            channels[name],
            source,
            source_min_level=source_min_level,
            observed_max_level=own_level,
            **kwargs,
        )
    return out


# ------------------------------------------ per-field estimation and 2x2 unmixing

def unsaturated_mask(
    channels: Sequence[np.ndarray],
    saturation_level: float,
    *,
    within: np.ndarray | None = None,
) -> np.ndarray:
    """Pixels below the saturation level in every given channel.

    Parameters
    ----------
    channels : sequence of ndarray
        2-D raw (not background-subtracted) channel images of one field, same shape.
    saturation_level : float
        Pixels at or above this level in any channel are excluded, in the intensity
        units of the raw images. Required: the level depends on the bit depth and the
        export of the user's own images.
    within : ndarray or None, optional
        2-D boolean mask the result is restricted to, for example a tissue mask.

    Returns
    -------
    ndarray
        2-D boolean mask of the pixels usable for coefficient estimation. A saturated
        pixel carries no slope information, so it must not enter a regression.
    """
    arrays = [as_float(check_2d(c, "channel")) for c in channels]
    if not arrays:
        raise ValueError("at least one channel is required")
    check_same_shape(*arrays)
    out = np.ones(arrays[0].shape, dtype=bool)
    for arr in arrays:
        out &= arr < float(saturation_level)
    if within is not None:
        mask = as_bool_mask(within, "within")
        check_same_shape(out, mask)
        out &= mask
    return out


def subtract_bleed(
    observed: np.ndarray,
    source: np.ndarray,
    coefficient: float,
    *,
    clip_negative: bool = False,
) -> np.ndarray:
    """Subtract a bleed-through term ``coefficient * source`` from a channel.

    Parameters
    ----------
    observed : ndarray
        2-D background residual of the channel to correct, in its intensity units.
    source : ndarray
        2-D background residual of the channel that bleeds into it, same shape and
        units.
    coefficient : float
        Bleed-through coefficient (dimensionless).
    clip_negative : bool, optional
        Clip the result at zero. The per-field unmixing does not clip, so that the
        residual keeps its noise distribution around zero; the fixed-coefficient model
        of the method does clip (see :func:`subtract_crosstalk`).

    Returns
    -------
    ndarray
        2-D float image of the corrected channel.
    """
    obs = as_float(check_2d(observed, "observed"))
    src = as_float(check_2d(source, "source"))
    check_same_shape(obs, src)
    out = obs - float(coefficient) * src
    return np.maximum(out, 0.0) if clip_negative else out


def estimate_readout_to_nuclear_bleed(
    nuclear_res: np.ndarray,
    readout_res: np.ndarray,
    *,
    usable: np.ndarray,
    readout_thr: float,
    readout_sigma: float,
    min_level: float,
    quantile: float = 0.10,
    sel_k: float = 2.0,
    min_pixels: int = 200,
    coef_max: float = 0.9,
) -> dict:
    """Estimate the bleed-through of the readout channel into the nuclear channel.

    The coefficient is the lower envelope (a low quantile) of the nuclear-to-readout
    intensity ratio over bright, unsaturated readout pixels. Genuine nuclear signal
    under a readout object can only raise that ratio, so the lower envelope measures
    the bleed-through and not the overlap.

    Parameters
    ----------
    nuclear_res : ndarray
        2-D background residual of the nuclear channel, in its intensity units.
    readout_res : ndarray
        2-D background residual of the readout channel, same shape and units.
    usable : ndarray
        2-D boolean mask of pixels eligible for estimation: unsaturated in both
        channels and inside the tissue mask (see :func:`unsaturated_mask`).
    readout_thr : float
        Readout object threshold, in the intensity units of ``readout_res``.
    readout_sigma : float
        Noise scale of the readout residual, same units.
    min_level : float
        Absolute lower bound of the bright-pixel selection, same units. Required: it
        depends on the detector scaling of the user's images.
    quantile : float, optional
        Quantile of the ratio taken as the lower envelope (dimensionless).
    sel_k : float, optional
        Pixels are used where the readout exceeds ``readout_thr + sel_k * readout_sigma``
        (noise units above the object threshold).
    min_pixels : int, optional
        Fewer selected pixels than this give a coefficient of zero rather than a value
        estimated from too little signal.
    coef_max : float, optional
        Upper bound of the coefficient (dimensionless); the estimate is clipped to
        ``[0, coef_max]``.

    Returns
    -------
    dict
        ``delta`` (dimensionless bleed-through coefficient), ``n_pixels`` (pixels the
        quantile used) and ``selection_level`` (the intensity level the selection used).

    Notes
    -----
    The method subtracts this bleed-through in every field, whether or not the field is
    unmixed, so that nuclei are masked on a bleed-free nuclear channel.
    """
    nuc = as_float(check_2d(nuclear_res, "nuclear_res"))
    read = as_float(check_2d(readout_res, "readout_res"))
    ok = as_bool_mask(usable, "usable")
    check_same_shape(nuc, read, ok)
    q = _check_quantile(quantile, "quantile")
    level = max(float(readout_thr) + float(sel_k) * float(readout_sigma), float(min_level))
    if level <= 0:
        raise ValueError("the bright-pixel selection level must be positive")
    sel = ok & (read >= level)
    n_pixels = int(sel.sum())
    if n_pixels < int(min_pixels):
        return dict(delta=0.0, n_pixels=n_pixels, selection_level=level)
    delta = float(np.clip(np.quantile(nuc[sel] / read[sel], q), 0.0, float(coef_max)))
    return dict(delta=delta, n_pixels=n_pixels, selection_level=level)


def nuclei_mask(
    nuclear_corr: np.ndarray,
    pixel_size_um: float,
    *,
    area_um2: tuple[float, float],
    tissue: np.ndarray | None = None,
    open_radius_px: float = 1.0,
    threshold: float | None = None,
    n_bins: int = 256,
) -> np.ndarray:
    """Mask of nuclei in a nuclear-stain channel from which the bleed-through was removed.

    The rule is a triangle threshold computed on the channel inside the measured area, a
    binary opening that removes single bright pixels, and a size gate that keeps only
    components whose area falls in the range a nucleus of the user's own material occupies.
    Run it on the corrected channel, that is after :func:`subtract_bleed` has removed the
    readout bleed-through estimated by :func:`estimate_readout_to_nuclear_bleed`, otherwise
    bright readout objects are masked as nuclei.

    Parameters
    ----------
    nuclear_corr : ndarray
        2-D nuclear-channel residual after bleed-through subtraction, in its intensity
        units.
    pixel_size_um : float
        Side of one pixel in micrometres.
    area_um2 : tuple of (float, float)
        Smallest and largest area a nucleus may have, in square micrometres, both ends
        included. There is no default: nuclear size is a property of the user's material
        and of the plane of section.
    tissue : ndarray, optional
        2-D boolean mask of the measured area. The threshold is computed inside it and
        the mask is restricted to it. ``None`` (the default) measures the whole frame.
    open_radius_px : float, optional
        Radius in pixels of the disk used for the opening. Default 1 pixel, the smallest
        isotropic element.
    threshold : float, optional
        Use this intensity threshold instead of the triangle threshold, in the units of
        ``nuclear_corr``. Default ``None``.
    n_bins : int, optional
        Number of histogram bins of the triangle threshold. Default 256.

    Returns
    -------
    numpy.ndarray
        2-D boolean mask of the nuclei that pass the size gate.

    Notes
    -----
    The size gate is applied to 8-connected components of the opened mask, so two nuclei
    that touch are gated as one component; separate them with
    :func:`if_qsaa.segmentation.split_touching_objects` first when that matters.
    """
    from skimage import filters as _filters

    plane = as_float(check_2d(nuclear_corr, "nuclear_corr"))
    if tissue is None:
        measured = np.ones(plane.shape, dtype=bool)
    else:
        measured = as_bool_mask(tissue, "tissue")
        check_same_shape(plane, measured)
    if not measured.any():
        raise ValueError("tissue holds no pixel, so no threshold can be computed")
    low, high = (float(area_um2[0]), float(area_um2[1]))
    if not np.isfinite(low) or not np.isfinite(high) or low < 0 or high < low:
        raise ValueError("area_um2 must be a finite (smallest, largest) pair with 0 <= smallest <= largest")

    level = (float(_filters.threshold_triangle(plane[measured].astype(np.float32), nbins=n_bins))
             if threshold is None else float(threshold))
    opened = ndi.binary_opening(plane > level, structure=_disk_px(open_radius_px)) & measured
    labels, n = label_mask(opened, connectivity=8)
    if n == 0:
        return np.zeros(plane.shape, dtype=bool)
    areas = np.bincount(labels.ravel()) * pixel_area_um2(pixel_size_um)
    keep = np.flatnonzero((areas >= low) & (areas <= high))
    keep = keep[keep > 0]
    return np.isin(labels, keep)


def component_slopes(
    labels: np.ndarray,
    source: np.ndarray,
    observed: np.ndarray,
    usable: np.ndarray,
    min_px: int,
    seed: int | np.random.Generator = 0,
    *,
    min_source_range: float,
    max_px: int = 400,
) -> np.ndarray:
    """Robust slope of an observed channel on a source channel inside each structure.

    Cross-talk makes the observed channel follow the profile of the source structure
    pixel by pixel, so the coupling coefficient is read off as the slope of the observed
    channel on the source channel *within* a source structure. A Theil-Sen (median of
    pairwise slopes) fit is used, so that pixels carrying genuine signal in the observed
    channel do not set the slope.

    Parameters
    ----------
    labels : ndarray
        2-D label image of the source structures (0 is background); structures are
        visited in increasing label order.
    source : ndarray
        2-D source-channel residual, in its intensity units.
    observed : ndarray
        2-D observed-channel residual, same shape and units.
    usable : ndarray
        2-D boolean mask of pixels eligible for the fit (see :func:`unsaturated_mask`).
    min_px : int
        Structures with fewer usable pixels than this are skipped (pixels).
    seed : int or Generator, optional
        Seed of the generator that subsamples large structures; a fixed seed makes the
        slopes reproducible, as everywhere else in this package. A
        ``numpy.random.Generator`` is accepted as well and is then used as given, which
        is how :func:`estimate_crosstalk` lets one stream serve its ``gamma`` and its
        ``alpha`` fit.
    min_source_range : float
        Structures whose usable source values span less than this are skipped, in the
        intensity units of ``source``. Required: it depends on the detector scaling of
        the user's images. Without a range of source intensities a slope is not
        identifiable.
    max_px : int, optional
        Structures larger than this are subsampled to ``max_px`` pixels without
        replacement, which bounds the cost of the robust fit.

    Returns
    -------
    ndarray
        1-D array of slopes (dimensionless), one per structure that passed the two
        size and range tests, in label order.
    """
    lab = check_2d(labels, "labels")
    src = as_float(check_2d(source, "source"))
    obs = as_float(check_2d(observed, "observed"))
    ok = as_bool_mask(usable, "usable")
    check_same_shape(lab, src, obs, ok)
    lab = lab.astype(np.intp, copy=False)
    rng = default_rng(seed)

    slopes: list = []
    for index, window in enumerate(ndi.find_objects(lab), start=1):
        if window is None:
            continue
        inside = (lab[window] == index) & ok[window]
        coords = np.argwhere(inside)
        if coords.shape[0] < int(min_px):
            continue
        if coords.shape[0] > int(max_px):
            coords = coords[rng.choice(coords.shape[0], int(max_px), replace=False)]
        rows = coords[:, 0] + window[0].start
        cols = coords[:, 1] + window[1].start
        x = src[rows, cols]
        if np.ptp(x) < float(min_source_range):
            continue
        slopes.append(float(stats.theilslopes(obs[rows, cols], x).slope))
    return np.asarray(slopes, dtype=float)


def estimate_crosstalk(
    readout_res: np.ndarray,
    marker_res: np.ndarray,
    nuclear_corr: np.ndarray | None = None,
    *,
    tissue: np.ndarray | None = None,
    usable: np.ndarray,
    nuclei: np.ndarray | None = None,
    marker_thr: float,
    beta_min_level: float,
    min_source_range: float,
    valid: np.ndarray | None = None,
    marker_min_px: int = 30,
    gamma_min_px: int = 20,
    alpha_min_px: int = 30,
    max_px: int = 400,
    min_structures: int = 20,
    alpha_quantile: float = 0.50,
    beta_quantile: float = 0.10,
    beta_k: float = 6.0,
    beta_min_pixels: int = 200,
    coef_max: float = 0.9,
    estimate_nuclear: bool = True,
    enabled: bool = True,
    noise_estimator: Callable[[np.ndarray], Sequence[float]] | None = None,
    seed: int = 0,
) -> dict:
    """Estimate the three cross-talk coefficients of a readout / marker / nuclear field.

    All three are estimated from the field itself, which is what the method does when no
    single-source control of the same section exists. In order:

    ``gamma`` (nuclear into readout)
        median of the within-nucleus robust slopes of the readout residual on the
        bleed-corrected nuclear residual. Genuine readout objects occupy a minority of
        nuclei, so the median across structures is not set by them.
    ``alpha`` (marker into readout)
        a quantile of the within-marker-structure robust slopes of the readout residual,
        after the nuclear term has been removed, on the marker residual.
    ``beta`` (readout into marker)
        lower envelope (a low quantile) of the marker-to-readout ratio over bright,
        unsaturated readout pixels. Readout objects are small and often saturated, so
        within-object slopes are unreliable; genuine marker signal under an object can
        only raise the ratio above the cross-talk floor.

    A two-channel field, a readout and a marker with no nuclear channel, is estimated by
    leaving ``nuclear_corr`` and ``nuclei`` out: ``gamma`` is then zero, ``alpha`` and
    ``beta`` are estimated as above and no placeholder image has to be built.

    Parameters
    ----------
    readout_res : ndarray
        2-D background residual of the readout channel, in its intensity units.
    marker_res : ndarray
        2-D background residual of the cellular-marker channel, same shape and units.
    nuclear_corr : ndarray or None, optional
        2-D background residual of the nuclear channel *after* the readout bleed-through
        has been subtracted (see :func:`estimate_readout_to_nuclear_bleed` and
        :func:`subtract_bleed`), same shape and units. ``None`` for a field without a
        nuclear channel: ``gamma`` is zero and the nuclear term is left out of the
        unmixing. Pass it together with ``nuclei`` or leave both out.
    tissue : ndarray or None, optional
        2-D boolean tissue mask; the readout noise of the bright-pixel selection is
        estimated inside it. ``None`` uses every pixel of the field.
    usable : ndarray
        2-D boolean mask of pixels eligible for estimation: unsaturated in every raw
        channel and inside the tissue (see :func:`unsaturated_mask`).
    nuclei : ndarray or None, optional
        2-D boolean nucleus mask, obtained from ``nuclear_corr`` by segmentation. The
        nuclei are the source structures of ``gamma``. ``None`` only together with
        ``nuclear_corr=None``.
    marker_thr : float
        Marker threshold in the intensity units of ``marker_res``; the preliminary
        marker mask is ``marker_res > marker_thr``. Required: it comes from the noise of
        the user's own images.
    beta_min_level : float
        Absolute lower bound of the bright-readout selection used for ``beta``, in the
        intensity units of ``readout_res``. Required for the same reason.
    min_source_range : float
        Minimum span of source intensity within a structure, same units; see
        :func:`component_slopes`.
    valid : ndarray or None, optional
        2-D boolean mask of pixels whose intensities are trustworthy, used when building
        the preliminary marker mask. ``None`` accepts every pixel.
    marker_min_px : int, optional
        Preliminary marker components smaller than this are dropped before the slopes
        are fitted (pixels, 4-connected components), so that noise specks do not count
        as structures.
    gamma_min_px, alpha_min_px : int, optional
        Minimum usable pixels per source structure (pixels) for ``gamma`` and ``alpha``.
    max_px : int, optional
        Subsample size per structure (pixels); see :func:`component_slopes`.
    min_structures : int, optional
        Fewer usable structures than this give a coefficient of zero rather than one
        estimated from too few structures.
    alpha_quantile : float, optional
        Quantile of the per-structure slopes taken as ``alpha`` (0.5 is the median).
    beta_quantile : float, optional
        Quantile of the marker-to-readout ratio taken as the lower envelope for
        ``beta``.
    beta_k : float, optional
        Bright readout pixels are those at or above ``mu + beta_k * sigma`` of the
        readout background, but never below ``beta_min_level`` (noise units).
    beta_min_pixels : int, optional
        Fewer bright pixels than this give ``beta = 0``.
    coef_max : float, optional
        Upper bound of every coefficient (dimensionless); estimates are clipped to
        ``[0, coef_max]``.
    estimate_nuclear : bool, optional
        False forces ``gamma = 0`` and leaves the nuclear channel out of the unmixing.
        It has no effect when no nuclear channel is given, where ``gamma`` is zero
        anyway.
    enabled : bool, optional
        False returns zero coefficients, which makes :func:`unmix_two_channel` the
        identity. This is the per-field switch of the method: fields without the bright
        donor signal are not unmixed. The readout-to-nuclear bleed-through is subtracted
        regardless, outside this function.
    noise_estimator : callable or None, optional
        Maps the readout values inside the tissue to a (location, scale) pair in
        intensity units. ``None`` uses the censored two-quantile normal fit of
        :mod:`if_qsaa.background`, which is the estimator the thresholds use.
    seed : int, optional
        Seed of the subsampling generator; one generator serves ``gamma`` and then
        ``alpha``, so a fixed seed makes the whole estimate reproducible.

    Returns
    -------
    dict
        ``alpha``, ``beta``, ``gamma`` (dimensionless coefficients) and the diagnostics
        ``n_nuclear_structures``, ``gamma_slope_quartiles``, ``n_marker_structures``,
        ``alpha_slope_quartiles``, ``n_beta_pixels`` and ``beta_select_level``. The
        quartile entries are ``[q25, q75]`` of the per-structure slopes, or ``None``
        when no structure was fitted; a wide spread means the coefficient is poorly
        determined in that field.

    Notes
    -----
    Object and marker thresholds are not recomputed after unmixing: they stay in the
    noise units of the uncorrected residuals.
    """
    read = as_float(check_2d(readout_res, "readout_res"))
    mark = as_float(check_2d(marker_res, "marker_res"))
    if (nuclear_corr is None) != (nuclei is None):
        raise ValueError(
            "pass both nuclear_corr and nuclei, or neither: the nucleus mask is the "
            "set of source structures of gamma, and a field without a nuclear channel "
            "is estimated with nuclear_corr=None and nuclei=None"
        )
    has_nuclear = nuclear_corr is not None
    nucl = as_float(check_2d(nuclear_corr, "nuclear_corr")) if has_nuclear else None
    nuc = as_bool_mask(nuclei, "nuclei") if has_nuclear else None
    tis = (
        np.ones(read.shape, dtype=bool) if tissue is None
        else as_bool_mask(tissue, "tissue")
    )
    ok = as_bool_mask(usable, "usable")
    check_same_shape(read, mark, tis, ok, *([nucl, nuc] if has_nuclear else []))
    alpha_q = _check_quantile(alpha_quantile, "alpha_quantile")
    beta_q = _check_quantile(beta_quantile, "beta_quantile")
    bound = float(coef_max)
    if bound <= 0:
        raise ValueError("coef_max must be positive")

    out = dict(
        alpha=0.0,
        beta=0.0,
        gamma=0.0,
        n_nuclear_structures=0,
        gamma_slope_quartiles=None,
        n_marker_structures=0,
        alpha_slope_quartiles=None,
        n_beta_pixels=0,
        beta_select_level=None,
    )
    if not enabled:
        return out

    rng = default_rng(seed)

    # gamma: nuclei are the source structures; the readout channel follows their profile
    if has_nuclear:
        nuc_labels, _ = label_mask(nuc, 8)
        gamma_slopes = component_slopes(
            nuc_labels, nucl, read, ok, gamma_min_px, rng,
            min_source_range=min_source_range, max_px=max_px,
        )
        gamma = (
            float(np.median(gamma_slopes))
            if gamma_slopes.size >= int(min_structures) and estimate_nuclear
            else 0.0
        )
        gamma = float(min(max(gamma, 0.0), bound))
        readout_corr = read - gamma * nucl
    else:
        gamma_slopes = np.zeros(0, dtype=float)
        gamma = 0.0
        readout_corr = read

    # alpha: marker structures are the source; fitted on the nuclear-corrected readout
    marker_mask = mark > float(marker_thr)
    if valid is not None:
        valid_mask = as_bool_mask(valid, "valid")
        check_same_shape(marker_mask, valid_mask)
        marker_mask = marker_mask & valid_mask
    marker_mask = _remove_small_components(marker_mask, int(marker_min_px), connectivity=4)
    marker_labels, _ = label_mask(marker_mask, 8)
    alpha_slopes = component_slopes(
        marker_labels, mark, readout_corr, ok, alpha_min_px, rng,
        min_source_range=min_source_range, max_px=max_px,
    )
    alpha = (
        float(np.quantile(alpha_slopes, alpha_q))
        if alpha_slopes.size >= int(min_structures)
        else 0.0
    )

    # beta: lower envelope of the marker-to-readout ratio on bright readout pixels
    mu, sigma = _noise_location_scale(readout_corr[tis], noise_estimator)
    level = max(mu + float(beta_k) * sigma, float(beta_min_level))
    if level <= 0:
        raise ValueError("the bright-readout selection level must be positive")
    sel = ok & (readout_corr >= level)
    n_beta = int(sel.sum())
    beta = (
        float(np.quantile(mark[sel] / readout_corr[sel], beta_q))
        if n_beta >= int(beta_min_pixels)
        else 0.0
    )

    out.update(
        alpha=float(min(max(alpha, 0.0), bound)),
        beta=float(min(max(beta, 0.0), bound)),
        gamma=gamma,
        n_nuclear_structures=int(gamma_slopes.size),
        gamma_slope_quartiles=(
            np.quantile(gamma_slopes, [0.25, 0.75]).tolist() if gamma_slopes.size else None
        ),
        n_marker_structures=int(alpha_slopes.size),
        alpha_slope_quartiles=(
            np.quantile(alpha_slopes, [0.25, 0.75]).tolist() if alpha_slopes.size else None
        ),
        n_beta_pixels=n_beta,
        beta_select_level=float(level),
    )
    return out


def unmix_two_channel(
    readout_res: np.ndarray,
    marker_res: np.ndarray,
    nuclear_corr: np.ndarray | None,
    alpha: float,
    beta: float,
    gamma: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert the 2x2 readout / marker mixing after removing the nuclear term.

    The mixing model is ``readout' = readout + alpha * marker`` and
    ``marker' = marker + beta * readout``, where ``readout'`` is the readout residual
    with the nuclear bleed-through already removed. Its exact inverse is

    ``readout_u = (readout' - alpha * marker') / (1 - alpha * beta)``,
    ``marker_u  = (marker'  - beta  * readout') / (1 - alpha * beta)``.

    Parameters
    ----------
    readout_res : ndarray
        2-D background residual of the readout channel, in its intensity units.
    marker_res : ndarray
        2-D background residual of the marker channel, same shape and units.
    nuclear_corr : ndarray or None
        2-D bleed-corrected nuclear residual, same shape and units; it is multiplied by
        ``gamma`` and removed from the readout channel first. ``None`` for a two-channel
        field with no nuclear channel, as returned by :func:`estimate_crosstalk` without
        one; ``gamma`` must then be zero, so that nothing is silently dropped.
    alpha : float
        Marker-into-readout coefficient (dimensionless).
    beta : float
        Readout-into-marker coefficient (dimensionless).
    gamma : float, optional
        Nuclear-into-readout coefficient (dimensionless).

    Returns
    -------
    readout_u : ndarray
        2-D unmixed readout residual, in the units of the input.
    marker_u : ndarray
        2-D unmixed marker residual, in the units of the input.

    Notes
    -----
    Negative values are kept, so that the residual keeps its noise distribution around
    zero and the thresholds of the uncorrected residuals stay valid. All coefficients
    zero returns the inputs unchanged, which is how a field that is not unmixed is
    handled.
    """
    read = as_float(check_2d(readout_res, "readout_res"))
    mark = as_float(check_2d(marker_res, "marker_res"))
    a, b, g = float(alpha), float(beta), float(gamma)
    if nuclear_corr is None:
        if g != 0.0:
            raise ValueError(
                "gamma must be zero when nuclear_corr is None; pass the bleed-corrected "
                "nuclear residual to remove a nuclear term"
            )
        nucl = None
        check_same_shape(read, mark)
    else:
        nucl = as_float(check_2d(nuclear_corr, "nuclear_corr"))
        check_same_shape(read, mark, nucl)
    denominator = 1.0 - a * b
    if denominator <= 0:
        raise ValueError(
            "1 - alpha * beta must be positive; the mixing matrix is not invertible"
        )
    readout_corr = read if nucl is None else read - g * nucl
    readout_u = (readout_corr - a * mark) / denominator
    marker_u = (mark - b * readout_corr) / denominator
    return readout_u, marker_u


# --------------------------------------------- residual cross-talk object filters

def remove_marker_shadows(
    marker_mask: np.ndarray,
    object_labels: np.ndarray,
    pixel_size_um: float,
    *,
    dilate_um: float = 1.0,
    cover: float = 0.80,
) -> tuple[np.ndarray, int]:
    """Remove marker components that lie mostly inside a dilated readout footprint.

    A marker component covered almost entirely by the footprint of readout objects
    carries no marker morphology of its own: it is a residual copy of the readout
    signal left over by the linear unmixing. The method removes components covered by at
    least 80 % of their area.

    Parameters
    ----------
    marker_mask : ndarray
        2-D boolean mask of the cellular-marker territory.
    object_labels : ndarray
        2-D label image of the readout objects (0 is background), same shape.
    pixel_size_um : float
        Pixel size in micrometres per pixel.
    dilate_um : float, optional
        Dilation of the readout footprint before the coverage is measured, in
        micrometres.
    cover : float, optional
        Minimum covered fraction of a component's area for removal (dimensionless).
        A value of 1 disables the filter.

    Returns
    -------
    mask : ndarray
        2-D boolean marker mask with the shadow components removed.
    n_removed : int
        Number of marker components removed.

    Notes
    -----
    Components are 8-connected. The filter can only remove marker territory, so it can
    only lower an apparent association between objects and marker, never raise it.
    """
    mask = as_bool_mask(marker_mask, "marker_mask")
    labels = check_2d(object_labels, "object_labels")
    check_same_shape(mask, labels)
    occupied = np.asarray(labels) > 0
    if float(cover) >= 1.0 or not occupied.any():
        return mask.copy(), 0
    footprint = ndi.binary_dilation(
        occupied, structure=_disk_px(um_to_px(dilate_um, pixel_size_um))
    )
    component_labels, n = label_mask(mask, 8)
    if n == 0:
        return mask.copy(), 0
    flat = component_labels.ravel()
    area = np.bincount(flat)
    covered = np.bincount(flat, weights=footprint.ravel().astype(float))
    fraction = covered / np.maximum(area, 1)
    shadows = np.flatnonzero(fraction >= float(cover))
    shadows = shadows[shadows > 0]
    if shadows.size == 0:
        return mask.copy(), 0
    return mask & ~np.isin(component_labels, shadows), int(shadows.size)


def footprint_beta_correction(
    marker_res: np.ndarray,
    readout_corr: np.ndarray,
    marker_u: np.ndarray,
    object_labels: np.ndarray,
    pixel_size_um: float,
    *,
    usable: np.ndarray,
    readout_thr: float,
    marker_thr: float,
    alpha: float,
    beta: float,
    min_level: float,
    quantile: float | None = None,
    far_um: float = 3.0,
    min_pixels: int = 100,
    dilate_um: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Re-correct the marker channel inside readout footprints with an upper-bound coefficient.

    Inside the footprint of a readout object, marker intensity and readout cross-talk
    cannot be told apart from the field-wide lower envelope alone. This optional step
    replaces ``beta`` there by an upper bound: a high quantile of the marker-to-readout
    ratio measured over bright readout pixels that lie far from any marker territory, so
    that the ratio is cross-talk rather than co-localisation. It is a sensitivity
    variant, off by default.

    Parameters
    ----------
    marker_res : ndarray
        2-D background residual of the marker channel, in its intensity units.
    readout_corr : ndarray
        2-D readout residual with the nuclear term already removed, same shape and
        units.
    marker_u : ndarray
        2-D unmixed marker residual to be revised, same shape and units.
    object_labels : ndarray
        2-D label image of the readout objects (0 is background), same shape.
    pixel_size_um : float
        Pixel size in micrometres per pixel.
    usable : ndarray
        2-D boolean mask of pixels eligible for estimation (see
        :func:`unsaturated_mask`).
    readout_thr : float
        Readout object threshold, in the intensity units of ``readout_corr``.
    marker_thr : float
        Marker threshold, in the intensity units of ``marker_u``.
    alpha, beta : float
        Coefficients of the field-wide unmixing (dimensionless).
    min_level : float
        Absolute lower bound of the bright-readout selection, in the intensity units of
        ``readout_corr``. Required: it depends on the detector scaling of the user's
        images.
    quantile : float or None, optional
        Quantile of the ratio taken as the upper bound. ``None`` leaves the marker
        channel untouched, which is the default setting of the method.
    far_um : float, optional
        Minimum distance from any marker territory for a pixel to enter the estimate, in
        micrometres.
    min_pixels : int, optional
        Fewer selected pixels than this leave the coefficient at ``beta``.
    dilate_um : float, optional
        Dilation of the readout footprint inside which the revised coefficient is
        applied, in micrometres.

    Returns
    -------
    marker_corrected : ndarray
        2-D marker residual, revised inside the dilated readout footprints.
    beta_used : float
        The coefficient applied inside the footprints (dimensionless); it is never
        smaller than ``beta``.
    """
    mark = as_float(check_2d(marker_res, "marker_res"))
    read = as_float(check_2d(readout_corr, "readout_corr"))
    marker_out = as_float(check_2d(marker_u, "marker_u"))
    labels = check_2d(object_labels, "object_labels")
    ok = as_bool_mask(usable, "usable")
    check_same_shape(mark, read, marker_out, labels, ok)
    occupied = np.asarray(labels) > 0
    beta_used = float(beta)
    if quantile is None or not occupied.any():
        return marker_out.copy(), beta_used

    q = _check_quantile(quantile, "quantile")
    far = ndi.distance_transform_edt(~(marker_out > float(marker_thr)))
    far = far * float(pixel_size_um) >= float(far_um)
    sel = occupied & far & ok & (read >= max(float(readout_thr), float(min_level)))
    if int(sel.sum()) >= int(min_pixels):
        ratio = mark[sel] / np.maximum(read[sel], _DIV_EPS)
        beta_used = max(beta_used, float(np.quantile(ratio, q)))
    denominator = 1.0 - float(alpha) * beta_used
    if denominator <= 0:
        raise ValueError(
            "1 - alpha * beta must be positive; the revised coefficient is too large"
        )
    footprint = ndi.binary_dilation(
        occupied, structure=_disk_px(um_to_px(dilate_um, pixel_size_um))
    )
    revised = (mark - beta_used * read) / denominator
    return np.where(footprint, revised, marker_out), beta_used


def surround_ring(
    object_mask: np.ndarray,
    pixel_size_um: float,
    *,
    occupied: np.ndarray | None = None,
    ring_um: tuple = (1.0, 3.0),
    min_ring_px: int = 5,
) -> np.ndarray:
    """Ring of local background around an object, for a contrast measurement.

    Parameters
    ----------
    object_mask : ndarray
        2-D boolean mask of one object.
    pixel_size_um : float
        Pixel size in micrometres per pixel.
    occupied : ndarray or None, optional
        2-D boolean mask of all object pixels, same shape; these are excluded from the
        ring so that a neighbouring object does not enter the local background.
    ring_um : tuple of float, optional
        Inner and outer radius of the ring in micrometres, measured outward from the
        object boundary.
    min_ring_px : int, optional
        If the ring has fewer pixels than this, it falls back to the whole outer
        dilation minus the object, so that a contrast can still be measured for a small
        or crowded object.

    Returns
    -------
    ndarray
        2-D boolean mask of the ring.
    """
    obj = as_bool_mask(object_mask, "object_mask")
    inner_um, outer_um = float(ring_um[0]), float(ring_um[1])
    if not 0 < inner_um < outer_um:
        raise ValueError("ring_um must be an increasing pair of positive radii")
    outer = ndi.binary_dilation(obj, structure=_disk_px(um_to_px(outer_um, pixel_size_um)))
    inner = ndi.binary_dilation(obj, structure=_disk_px(um_to_px(inner_um, pixel_size_um)))
    ring = outer & ~inner
    if occupied is not None:
        other = as_bool_mask(occupied, "occupied")
        check_same_shape(ring, other)
        ring = ring & ~other
    if int(ring.sum()) < int(min_ring_px):
        ring = outer & ~obj
    return ring


def ring_contrast(
    object_mask: np.ndarray,
    images: Mapping[str, np.ndarray],
    pixel_size_um: float,
    *,
    occupied: np.ndarray | None = None,
    ring_um: tuple = (1.0, 3.0),
    min_ring_px: int = 5,
) -> dict:
    """Mean intensity of an object minus the mean of its surrounding ring, per channel.

    Parameters
    ----------
    object_mask : ndarray
        2-D boolean mask of one object.
    images : mapping of str to ndarray
        2-D channel images, same shape, each in its own intensity units.
    pixel_size_um : float
        Pixel size in micrometres per pixel.
    occupied, ring_um, min_ring_px
        See :func:`surround_ring`.

    Returns
    -------
    dict of str to float
        Local contrast of the object per channel, in the intensity units of that
        channel. NaN when the ring is empty.
    """
    obj = as_bool_mask(object_mask, "object_mask")
    ring = surround_ring(
        obj, pixel_size_um, occupied=occupied, ring_um=ring_um, min_ring_px=min_ring_px
    )
    out: dict = {}
    for name, image in images.items():
        arr = as_float(check_2d(image, f"image {name!r}"))
        check_same_shape(arr, obj)
        if not obj.any() or not ring.any():
            out[name] = float("nan")
        else:
            out[name] = float(arr[obj].mean() - arr[ring].mean())
    return out


def contrast_ratio(
    contrast_own: float,
    contrast_other: float,
    *,
    min_contrast: float,
) -> float:
    """Ratio of an object's contrast in its own channel to its contrast in another.

    Parameters
    ----------
    contrast_own : float
        Local contrast in the channel the object was segmented in, in intensity units.
    contrast_other : float
        Local contrast in the other channel, same units.
    min_contrast : float
        Smallest contrast treated as a usable denominator, in intensity units.
        Required: it depends on the detector scaling of the user's images. At or below
        it the other channel carries no contrast and the ratio is infinite.

    Returns
    -------
    float
        Dimensionless ratio, or infinity when the other channel has no contrast.
    """
    if not float(contrast_other) > float(min_contrast):
        return float("inf")
    return float(contrast_own) / float(contrast_other)


def spectral_gate(
    contrast_own: float,
    contrast_nuclear: float,
    contrast_marker: float = float("nan"),
    *,
    min_contrast: float,
    gate_ratio: float = 1.0,
    marker_gate_ratio: float = 1.0,
    use_marker_gate: bool = True,
) -> dict:
    """Decide whether an object is brighter in its own channel than in the others.

    A copy produced by cross-talk is, by construction, at most as bright in the channel
    it leaked into as in the channel it came from. The method therefore excludes objects
    whose readout contrast against a surrounding ring is lower than their contrast in
    the nuclear or the marker channel.

    Parameters
    ----------
    contrast_own : float
        Local contrast in the object's own channel, in intensity units.
    contrast_nuclear : float
        Local contrast in the nuclear channel, same units.
    contrast_marker : float, optional
        Local contrast in the marker channel, same units. Needed only when
        ``use_marker_gate`` is true.
    min_contrast : float
        Smallest usable denominator contrast, in intensity units (see
        :func:`contrast_ratio`).
    gate_ratio : float, optional
        Required ratio of own to nuclear contrast (dimensionless); 1 means the object
        must be at least as bright in its own channel.
    marker_gate_ratio : float, optional
        Required ratio of own to marker contrast (dimensionless).
    use_marker_gate : bool, optional
        False applies only the nuclear gate.

    Returns
    -------
    dict
        ``ratio_nuclear``, ``ratio_marker`` (dimensionless), the boolean flags
        ``gate_nuclear``, ``gate_marker`` and their conjunction ``gate_pass``. A ratio
        exactly equal to its threshold passes.

    Notes
    -----
    Contrasts are compared in intensity units rather than in noise units, because noise
    units are not comparable between channels that were scaled for display.
    """
    ratio_nuclear = contrast_ratio(contrast_own, contrast_nuclear, min_contrast=min_contrast)
    ratio_marker = contrast_ratio(contrast_own, contrast_marker, min_contrast=min_contrast)
    gate_nuclear = bool(ratio_nuclear >= float(gate_ratio))
    gate_marker = bool(ratio_marker >= float(marker_gate_ratio)) if use_marker_gate else True
    return dict(
        ratio_nuclear=ratio_nuclear,
        ratio_marker=ratio_marker,
        gate_nuclear=gate_nuclear,
        gate_marker=gate_marker,
        gate_pass=bool(gate_nuclear and gate_marker),
    )


def object_contrast_table(
    object_labels: np.ndarray,
    own: np.ndarray,
    nuclear: np.ndarray,
    marker: np.ndarray,
    pixel_size_um: float,
    *,
    min_contrast: float,
    ring_um: tuple = (1.0, 3.0),
    min_ring_px: int = 5,
    gate_ratio: float = 1.0,
    marker_gate_ratio: float = 1.0,
    use_marker_gate: bool = True,
    ratio_cap: float = 999.0,
) -> pd.DataFrame:
    """Per-object ring contrasts in three channels and the spectral gate they imply.

    Parameters
    ----------
    object_labels : ndarray
        2-D label image of the segmented readout objects (0 is background).
    own : ndarray
        2-D readout residual the objects were segmented from, in its intensity units.
        Pass the residual *before* unmixing: the gate asks whether the object is
        genuinely brighter in its own channel than in the channels it could have leaked
        in from, which the unmixing would already have removed.
    nuclear : ndarray
        2-D nuclear residual after the readout bleed-through has been subtracted, same
        shape and units.
    marker : ndarray
        2-D marker residual before unmixing, same shape and units.
    pixel_size_um : float
        Pixel size in micrometres per pixel.
    min_contrast : float
        Smallest usable denominator contrast, in intensity units (see
        :func:`contrast_ratio`).
    ring_um, min_ring_px
        See :func:`surround_ring`.
    gate_ratio, marker_gate_ratio, use_marker_gate
        See :func:`spectral_gate`.
    ratio_cap : float, optional
        Finite value stored in place of an infinite ratio (dimensionless). The gates
        are evaluated on the uncapped ratio.

    Returns
    -------
    DataFrame
        One row per object, with ``label``, ``area_px``, ``area_um2``, the contrasts
        ``contrast_own``, ``contrast_nuclear``, ``contrast_marker`` (intensity units),
        ``n_ring_px``, the ratios ``ratio_nuclear``, ``ratio_marker`` and the flags
        ``gate_nuclear``, ``gate_marker``, ``gate_pass``. Objects are in increasing
        label order.
    """
    labels = check_2d(object_labels, "object_labels").astype(np.intp, copy=False)
    own_img = as_float(check_2d(own, "own"))
    nuclear_img = as_float(check_2d(nuclear, "nuclear"))
    marker_img = as_float(check_2d(marker, "marker"))
    check_same_shape(labels, own_img, nuclear_img, marker_img)

    occupied = labels > 0
    margin = int(um_to_px(float(ring_um[1]), pixel_size_um)) + 2
    area_factor = pixel_area_um2(pixel_size_um)
    columns = [
        "label", "area_px", "area_um2", "contrast_own", "contrast_nuclear",
        "contrast_marker", "n_ring_px", "ratio_nuclear", "ratio_marker",
        "gate_nuclear", "gate_marker", "gate_pass",
    ]

    rows: list = []
    for index, window in enumerate(ndi.find_objects(labels), start=1):
        if window is None:
            continue
        crop = tuple(
            slice(max(0, w.start - margin), w.stop + margin) for w in window
        )
        obj = labels[crop] == index
        ring = surround_ring(
            obj, pixel_size_um, occupied=occupied[crop], ring_um=ring_um,
            min_ring_px=min_ring_px,
        )
        if ring.any():
            contrasts = {
                "contrast_own": float(own_img[crop][obj].mean() - own_img[crop][ring].mean()),
                "contrast_nuclear": float(
                    nuclear_img[crop][obj].mean() - nuclear_img[crop][ring].mean()
                ),
                "contrast_marker": float(
                    marker_img[crop][obj].mean() - marker_img[crop][ring].mean()
                ),
            }
        else:
            contrasts = dict(
                contrast_own=float("nan"),
                contrast_nuclear=float("nan"),
                contrast_marker=float("nan"),
            )
        gate = spectral_gate(
            contrasts["contrast_own"],
            contrasts["contrast_nuclear"],
            contrasts["contrast_marker"],
            min_contrast=min_contrast,
            gate_ratio=gate_ratio,
            marker_gate_ratio=marker_gate_ratio,
            use_marker_gate=use_marker_gate,
        )
        area_px = int(obj.sum())
        rows.append(
            dict(
                label=int(index),
                area_px=area_px,
                area_um2=float(area_px * area_factor),
                n_ring_px=int(ring.sum()),
                ratio_nuclear=float(min(gate["ratio_nuclear"], float(ratio_cap))),
                ratio_marker=float(min(gate["ratio_marker"], float(ratio_cap))),
                gate_nuclear=gate["gate_nuclear"],
                gate_marker=gate["gate_marker"],
                gate_pass=gate["gate_pass"],
                **contrasts,
            )
        )
    if not rows:
        return pd.DataFrame({name: pd.Series(dtype=object) for name in columns})
    return pd.DataFrame(rows)[columns]


# ------------------------------------------------------------- robustness switch

def unmixing_off_overrides() -> dict:
    """Parameter overrides that switch the whole cross-talk handling off.

    Returns
    -------
    dict
        ``unmix`` false, ``shadow_cover`` 1 (filter disabled) and ``use_marker_gate``
        false: the three steps that exist only because of spectral cross-talk. The names
        are the parameter names of :func:`if_qsaa.contact.analyse_contact_field`, so the
        mapping can be used as one setting of a
        :func:`if_qsaa.contact.run_contact_sensitivity` grid, for example
        ``{"without_crosstalk_correction": unmixing_off_overrides()}``. Running an
        analysis with and without these shows how much of a conclusion rests on the
        correction.

    Notes
    -----
    An override switches a step off only if the analysis reads that parameter under that
    name, so :func:`compare_unmixing` checks these three keys against the parameters it
    is given rather than injecting a key that would do nothing.
    """
    return {"unmix": False, "shadow_cover": 1.0, "use_marker_gate": False}


def unmixing_variants() -> dict:
    """Cross-talk-related settings of a robustness grid, as parameter overrides.

    Returns
    -------
    dict of str to dict
        One override mapping per setting: the correction switched off entirely, the
        nuclear term left out, each object filter switched off or tightened, the two
        estimator quantiles moved up and down, and the optional footprint upper-bound
        correction switched on.

    Notes
    -----
    The keys of each override are the parameter names of
    :func:`if_qsaa.contact.analyse_contact_field` (``unmix`` switches
    ``estimate_crosstalk(enabled=...)``, ``unmix_nuclear`` its ``estimate_nuclear`` and
    ``use_gate`` whether ``gate_pass`` filters objects), so the mapping can be passed as
    the grid of :func:`if_qsaa.contact.run_contact_sensitivity`. The two
    ``footprint_correction_*`` settings switch on the footprint upper-bound correction,
    which also needs ``footprint_min_level`` in the base parameters: a grey level of the
    user's readout channel, the lower bound of the bright readout pixels the upper-bound
    coefficient is read from, so it has no generic value and the analysis raises without it.
    Rename the keys, or pass a mapping of your own, if your pipeline uses different
    names.
    """
    return {
        "without_crosstalk_correction": unmixing_off_overrides(),
        "without_nuclear_unmixing": {"unmix_nuclear": False},
        "without_shadow_removal": {"shadow_cover": 1.0},
        "without_spectral_gate": {"use_gate": False},
        "without_marker_gate": {"use_marker_gate": False},
        "strict_spectral_gate": {"gate_ratio": 1.5},
        "strict_marker_gate": {"marker_gate_ratio": 1.5},
        "alpha_quantile_low": {"alpha_quantile": 0.25},
        "alpha_quantile_high": {"alpha_quantile": 0.75},
        "beta_quantile_low": {"beta_quantile": 0.05},
        "beta_quantile_high": {"beta_quantile": 0.25},
        "footprint_correction_high": {"footprint_beta_quantile": 0.90},
        "footprint_correction_mid": {"footprint_beta_quantile": 0.50},
    }


def run_variants(
    analyse: Callable[[Mapping], Mapping],
    base_params: Mapping,
    variants: Mapping[str, Mapping] | None = None,
    *,
    include_baseline: bool = True,
    baseline_name: str = "baseline",
    setting_column: str = "setting",
) -> pd.DataFrame:
    """Re-run an analysis under a set of parameter overrides and tabulate the results.

    Parameters
    ----------
    analyse : callable
        Called once per setting as ``analyse(params)`` and expected to return a mapping
        of headline numbers. It must not modify ``params``.
    base_params : mapping
        Parameters of the analysis. It is never modified; each setting receives a copy
        updated with its overrides.
    variants : mapping of str to mapping, or None, optional
        Setting name to parameter overrides. ``None`` uses
        :func:`unmixing_variants`.
    include_baseline : bool, optional
        Run ``base_params`` unchanged first.
    baseline_name : str, optional
        Name of that first row.
    setting_column : str, optional
        Name of the column holding the setting name.

    Returns
    -------
    DataFrame
        One row per setting, in the order the settings were given, with the setting name
        in the first column and the keys returned by ``analyse`` in the remaining ones.

    Notes
    -----
    The function is pure: the caller decides how the runs are executed, so an expensive
    analysis can be parallelised outside.
    """
    settings = unmixing_variants() if variants is None else variants
    jobs: list = []
    if include_baseline:
        jobs.append((baseline_name, {}))
    jobs.extend(settings.items())

    rows: list = []
    for name, overrides in jobs:
        params = dict(base_params)
        params.update(dict(overrides))
        result = analyse(params)
        row = {setting_column: name}
        row.update(dict(result))
        rows.append(row)
    return pd.DataFrame(rows)


def compare_unmixing(
    analyse: Callable[[Mapping], Mapping],
    base_params: Mapping,
    *,
    off_overrides: Mapping | None = None,
    setting_column: str = "setting",
    validate_overrides: bool = True,
) -> pd.DataFrame:
    """Run an analysis once with and once without the cross-talk handling.

    Parameters
    ----------
    analyse : callable
        See :func:`run_variants`.
    base_params : mapping
        Parameters of the analysis, with the cross-talk handling on. Every key of
        ``off_overrides`` must appear in it (see ``validate_overrides``).
    off_overrides : mapping or None, optional
        Overrides that switch the handling off. ``None`` uses
        :func:`unmixing_off_overrides`, whose keys are ``unmix``,
        ``shadow_cover`` and ``use_marker_gate``. An analysis that names those
        parameters differently must pass its own mapping.
    setting_column : str, optional
        Name of the column holding the setting name.
    validate_overrides : bool, optional
        Check that ``base_params`` already holds every key the overrides inject, and
        raise :class:`KeyError` naming the keys it does not. An override on a key the
        analysis never reads switches nothing off, so a misspelled or missing key would
        otherwise run the correction in both rows and return two identical rows with no
        warning. Pass false for an analysis that takes some of its parameters from
        elsewhere than ``base_params``.

    Returns
    -------
    DataFrame
        Two rows, ``with_unmixing`` and ``without_unmixing``.
    """
    overrides = unmixing_off_overrides() if off_overrides is None else off_overrides
    if validate_overrides:
        missing = [key for key in overrides if key not in base_params]
        if missing:
            raise KeyError(
                f"base_params has no entry for {sorted(missing)}, so switching "
                "the cross-talk handling off would change nothing and the two rows "
                "would be identical; add the parameter(s) to base_params, pass the "
                "names your analysis uses as off_overrides, or pass "
                "validate_overrides=False"
            )
    return run_variants(
        analyse,
        base_params,
        {"without_unmixing": overrides},
        include_baseline=True,
        baseline_name="with_unmixing",
        setting_column=setting_column,
    )
