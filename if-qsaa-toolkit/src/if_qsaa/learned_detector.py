"""Optional learned object detector: a logistic score over six interpretable features.

The detector looks for compact objects that are clearly brighter than their immediate
surroundings and that sit in a quiet neighbourhood. Candidates are proposed from a
multiscale difference-of-Gaussians local-contrast map and grown around each local
maximum; every candidate is then described by six features and accepted or rejected by
a linear logistic score.

The six features
----------------
``ringnear``
    Brightness of the candidate above the immediately surrounding ring, in units of the
    local noise.
``zmean``
    Mean local contrast inside the candidate (in noise units).
``zmax``
    Peak local contrast inside the candidate (in noise units).
``noiserel``
    Noise level of the candidate's neighbourhood divided by the median noise level of
    the whole image (a quiet neighbourhood is more likely to carry a real object).
``reftone``
    High-pass level of an optional reference channel at the candidate's position, in
    units of the reference channel's local noise (high values indicate bleed-through or
    a cross-channel artefact).
``aspect``
    Long-to-short axis ratio of the candidate's pixel set.

Why the features are normalised per image
-----------------------------------------
Real objects occupy a stable place in feature space, but the population of non-objects
differs enormously between images. Each feature is therefore turned into a robust score
relative to the candidate population of its own image before it is scored. The scale of
that normalisation is estimated from the background side of the distribution only
(median plus the median absolute deviation of the lower half): a full-range deviation is
inflated by the true signal, which would compress images rich in objects and erase the
real differences between images.

Fitting and inference take the same input: the raw feature table of one image, as
:func:`detect_candidates` measures it. :func:`score_candidates` normalises the one table
it is given, and :func:`fit_learned_model` takes one raw table per image (a sequence of
tables) and normalises each of them separately, with the very settings it writes into the
model. A table that was normalised beforehand is accepted only through
``fit_learned_model(..., already_normalized=True)``, which refuses a normalisation that
differs from the one the model records instead of letting fitting and inference diverge.

Model fitting
-------------
Fit a :class:`LearnedModel` on labelled candidates from your images with
:func:`fit_learned_model`. Use :func:`score_candidates` with the fitted model to score
candidates, then select a threshold with :func:`choose_threshold_max_f1`. Pass the fitted
model and threshold to :func:`segment_learned`. Re-check the model and threshold when
imaging or staining conditions change.

Spatial scale
-------------
:class:`DetectorParams` holds the detector's working scale. Its defaults are the generic
design scale of the method, expressed in pixels, and suit small punctate targets of
roughly a few up to a few tens of pixels across; they state no pixel size of their own.
For your own pixel size give every length in micrometres with
:meth:`DetectorParams.from_micrometres`, or scale a set whose pixel size you know with
:meth:`DetectorParams.rescaled`. Object measurements are reported in both pixels
and micrometres and therefore take ``pixel_size_um`` explicitly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy import optimize
from scipy.special import expit
from skimage import measure as _skmeasure
from skimage import morphology as _skmorphology

from ._util import (
    area_px_to_um2,
    as_bool_mask,
    as_float,
    check_2d,
    check_same_shape,
    label_mask,
    mad,
    px_to_um,
    um_to_px,
)

__all__ = [
    "FEATURE_NAMES",
    "LOG_FEATURES",
    "NORMALIZATION",
    "NORMALIZATION_KEY",
    "Candidate",
    "CandidateSet",
    "DetectorParams",
    "LearnedModel",
    "LearnedSegmentation",
    "to_float_plane",
    "subtract_reference",
    "dog_response",
    "propose_seeds",
    "grow_candidates",
    "local_noise_map",
    "reference_tone_map",
    "candidate_features",
    "detect_candidates",
    "normalize_per_image",
    "standardize_features",
    "logistic_score",
    "score_candidates",
    "paint_candidates",
    "segment_learned",
    "measure_objects",
    "fit_learned_model",
    "choose_threshold_max_f1",
    "save_model",
    "load_model",
]


#: Feature order used everywhere in this module and stored in a model file.
FEATURE_NAMES = ("ringnear", "zmean", "zmax", "noiserel", "reftone", "aspect")

#: Features that are skewed and enter the normalisation through ``log1p(max(x, 0))``.
LOG_FEATURES = ("ringnear", "zmean", "zmax", "reftone")

#: Name of the per-image normalisation rule; inference must repeat what fitting used.
NORMALIZATION = "median_lower_half_mad"

#: Key under which :func:`normalize_per_image` records the settings it used in the
#: ``attrs`` of the table it returns, so that fitting can refuse a table normalised with
#: settings other than the ones the model will carry.
NORMALIZATION_KEY = "if_qsaa_normalization"

#: Name of the feature that needs an optional reference channel.
REFERENCE_FEATURE = "reftone"

# Scales a median absolute deviation to a Gaussian standard deviation.
_MAD_SCALE = 1.4826

# Gaussian kernels are truncated at four standard deviations and mirrored at the border.
_GAUSS_TRUNCATE = 4.0
_GAUSS_MODE = "mirror"

# Bound and smallest lower half of the per-image normalisation. Fitting and inference must
# use the same values, so each one is written once here.
_DEFAULT_CLIP = 12.0
_DEFAULT_MIN_LOWER = 5


# --------------------------------------------------------------------------- parameters

@dataclass(frozen=True)
class DetectorParams:
    """Working scale and fixed rules of the candidate detector.

    All spatial fields are in pixels: the detector works at a fixed pixel scale, and the
    defaults are the generic design scale of the method (suited to small punctate targets
    of roughly a few up to a few tens of pixels across). The defaults state no pixel size
    of their own, so give every length in micrometres with :meth:`from_micrometres` for
    your own images, or scale a set whose pixel size you do know with :meth:`rescaled`.

    Parameters
    ----------
    full_scale : float
        Upper end of the detector's internal intensity range, in grey levels. Input
        images are scaled onto ``0 .. full_scale``.
    min_size_px : int
        Smallest accepted image side, in pixels.
    reference_offset_percentile : float
        Percentile of the reference channel, in per cent, that is treated as its own
        offset before it is subtracted from the target channel.
    reference_coefficient : float
        Dimensionless factor applied to the reference channel above its offset. The
        generic rule of the method subtracts it with unit weight; pass a coupling
        coefficient calibrated on single-stained control sections to subtract less or
        more.
    dog_sigma_px : tuple of (float, float)
        One pair of Gaussian standard deviations in pixels per difference-of-Gaussians
        band; the response is the pixel-wise maximum over the bands.
    noise_sigma_px : float
        Standard deviation in pixels of the smoothing used for the global noise estimate.
    noise_floor : float or None
        Lower bound of the global noise estimate, in grey levels of the scaled plane, or
        ``None`` for no bound (the default). A floor keeps a nearly noise-free plane from
        producing an unbounded local contrast, so it belongs to the user's own detector and
        bit depth; set it from the read noise of your own images rather than guessing.
    seed_threshold : float
        Local contrast, in noise units, that a local maximum must reach to become a seed.
    seed_window_px : int
        Side of the square neighbourhood, in pixels, in which a seed must be maximal.
    candidate_radius_px : int
        Half-side of the square growth window around a seed, in pixels.
    low_fraction : float
        Fraction of the seed's peak response used as the growth threshold.
    low_min_fraction : float
        Fraction of ``seed_threshold`` used as the lower bound of the growth threshold.
    low_floor : float
        Absolute lower bound of the growth threshold, in noise units.
    noise_tile_px : int
        Side of the square blocks, in pixels, over which the local noise is estimated.
    ring_outer_radius_px : int
        Outer radius of the surrounding background ring, in pixels.
    ring_inner_radius_px : int
        Inner radius of the surrounding background ring, in pixels; pixels this close to
        the candidate are excluded from the ring.
    min_ring_px : int
        Smallest number of ring pixels, in pixels, for which the ring contrast is
        evaluated; below it the ring feature is set to zero.
    reference_sigma_px : float
        Standard deviation in pixels of the high-pass filter applied to the reference
        channel before its level is read.
    """

    full_scale: float = 65535.0
    min_size_px: int = 64
    reference_offset_percentile: float = 10.0
    reference_coefficient: float = 1.0
    dog_sigma_px: tuple[tuple[float, float], ...] = ((0.65, 2.2), (1.15, 4.2))
    noise_sigma_px: float = 1.0
    noise_floor: float | None = None
    seed_threshold: float = 3.5
    seed_window_px: int = 5
    candidate_radius_px: int = 9
    low_fraction: float = 0.18
    low_min_fraction: float = 0.4
    low_floor: float = 0.8
    noise_tile_px: int = 50
    ring_outer_radius_px: int = 6
    ring_inner_radius_px: int = 2
    min_ring_px: int = 21
    reference_sigma_px: float = 4.0

    def rescaled(self, factor: float) -> "DetectorParams":
        """Return a copy with every spatial parameter scaled by ``factor``.

        Parameters
        ----------
        factor : float
            Dimensionless scale factor: every length is multiplied by it, radii and
            window sides are rounded to whole pixels and the ring pixel count scales with
            the square. Pass ``old_pixel_size_um / pixel_size_um`` to carry a parameter
            set from one pixel size to another, so that an object of the same physical
            size covers the same number of pixels.

            This needs the pixel size the current set was built for, which only you know:
            the shipped defaults are expressed in pixels and carry no pixel size, so
            there is no factor that converts them from "their" pixel size to yours. For
            those, either state your own lengths once with :meth:`from_micrometres`, or
            scale by the ratio of object sizes in pixels -- the defaults are made for
            punctate targets of roughly a few up to a few tens of pixels across, so a
            factor that brings your targets into that range is the right one. A set you
            built with :meth:`from_micrometres` does have a known pixel size, and
            ``DetectorParams.from_micrometres(new_pixel_size_um,
            **params.scale_um(old_pixel_size_um))`` rebuilds it at another one without
            listing the lengths again (non-spatial fields are not carried over and have
            to be passed again).

        Returns
        -------
        DetectorParams
            Rescaled parameter set.
        """
        f = float(factor)
        if f <= 0:
            raise ValueError("factor must be positive")
        return replace(
            self,
            dog_sigma_px=tuple((a * f, b * f) for a, b in self.dog_sigma_px),
            noise_sigma_px=self.noise_sigma_px * f,
            seed_window_px=_odd(max(3, int(round(self.seed_window_px * f)))),
            candidate_radius_px=max(1, int(round(self.candidate_radius_px * f))),
            noise_tile_px=max(2, int(round(self.noise_tile_px * f))),
            ring_outer_radius_px=max(1, int(round(self.ring_outer_radius_px * f))),
            ring_inner_radius_px=max(1, int(round(self.ring_inner_radius_px * f))),
            min_ring_px=max(1, int(round(self.min_ring_px * f * f))),
            reference_sigma_px=self.reference_sigma_px * f,
        )

    @classmethod
    def from_micrometres(
        cls,
        pixel_size_um: float,
        *,
        dog_sigma_um: Sequence[Sequence[float]],
        noise_sigma_um: float,
        seed_window_um: float,
        candidate_radius_um: float,
        noise_tile_um: float,
        ring_outer_radius_um: float,
        ring_inner_radius_um: float,
        reference_sigma_um: float,
        min_ring_um2: float,
        **other: float,
    ) -> "DetectorParams":
        """Build a parameter set from lengths in micrometres.

        Parameters
        ----------
        pixel_size_um : float
            Pixel size of the images, in micrometres per pixel.
        dog_sigma_um : sequence of (float, float)
            Difference-of-Gaussians bands, each a pair of standard deviations in
            micrometres.
        noise_sigma_um : float
            Smoothing standard deviation for the global noise estimate, in micrometres.
        seed_window_um : float
            Side of the local-maximum neighbourhood, in micrometres (rounded up to an
            odd number of pixels).
        candidate_radius_um : float
            Half-side of the growth window around a seed, in micrometres.
        noise_tile_um : float
            Side of the local-noise blocks, in micrometres.
        ring_outer_radius_um, ring_inner_radius_um : float
            Outer and inner radius of the surrounding background ring, in micrometres.
        reference_sigma_um : float
            High-pass standard deviation for the reference channel, in micrometres.
        min_ring_um2 : float
            Smallest ring area for which the ring contrast is evaluated, in square
            micrometres.
        **other
            Any non-spatial field of :class:`DetectorParams`, passed through unchanged.

        Returns
        -------
        DetectorParams
            Parameter set in pixels.
        """
        def to_px(value: float) -> float:
            return um_to_px(value, pixel_size_um)

        min_ring_px = max(1, int(round(float(min_ring_um2) / (float(pixel_size_um) ** 2))))
        return cls(
            dog_sigma_px=tuple((to_px(a), to_px(b)) for a, b in dog_sigma_um),
            noise_sigma_px=to_px(noise_sigma_um),
            seed_window_px=_odd(max(3, int(round(to_px(seed_window_um))))),
            candidate_radius_px=max(1, int(round(to_px(candidate_radius_um)))),
            noise_tile_px=max(2, int(round(to_px(noise_tile_um)))),
            ring_outer_radius_px=max(1, int(round(to_px(ring_outer_radius_um)))),
            ring_inner_radius_px=max(1, int(round(to_px(ring_inner_radius_um)))),
            min_ring_px=min_ring_px,
            reference_sigma_px=to_px(reference_sigma_um),
            **other,
        )

    def scale_um(self, pixel_size_um: float) -> dict[str, object]:
        """Return the spatial parameters expressed in micrometres.

        Parameters
        ----------
        pixel_size_um : float
            Pixel size of the images, in micrometres per pixel.

        Returns
        -------
        dict
            Lengths in micrometres and the minimum ring area in square micrometres.
        """
        return {
            "dog_sigma_um": tuple(
                (px_to_um(a, pixel_size_um), px_to_um(b, pixel_size_um))
                for a, b in self.dog_sigma_px
            ),
            "noise_sigma_um": px_to_um(self.noise_sigma_px, pixel_size_um),
            "seed_window_um": px_to_um(self.seed_window_px, pixel_size_um),
            "candidate_radius_um": px_to_um(self.candidate_radius_px, pixel_size_um),
            "noise_tile_um": px_to_um(self.noise_tile_px, pixel_size_um),
            "ring_outer_radius_um": px_to_um(self.ring_outer_radius_px, pixel_size_um),
            "ring_inner_radius_um": px_to_um(self.ring_inner_radius_px, pixel_size_um),
            "reference_sigma_um": px_to_um(self.reference_sigma_px, pixel_size_um),
            "min_ring_um2": area_px_to_um2(self.min_ring_px, pixel_size_um),
        }


@dataclass(frozen=True, eq=False)
class Candidate:
    """One candidate: the pixels grown around a seed.

    Attributes
    ----------
    ys, xs : numpy.ndarray
        Row and column indices of the candidate's pixels, in pixels, in row-major order.
    seed_y, seed_x : int
        Row and column of the seed, in pixels.
    """

    ys: np.ndarray
    xs: np.ndarray
    seed_y: int
    seed_x: int

    @property
    def area_px(self) -> int:
        """Number of pixels in the candidate."""
        return int(np.asarray(self.ys).size)


@dataclass(frozen=True, eq=False)
class CandidateSet:
    """Candidates of one image together with the planes they were measured on.

    Attributes
    ----------
    plane : numpy.ndarray
        Target channel scaled onto the detector's intensity range.
    corrected : numpy.ndarray
        Target channel after the optional reference subtraction; used for the ring and
        noise features only.
    response : numpy.ndarray
        Local-contrast response in noise units.
    global_noise : float
        Global noise estimate of the target channel, in grey levels of ``plane``.
    candidates : list of Candidate
        Candidates in the order they were proposed.
    features : pandas.DataFrame
        Raw features, one row per candidate, columns :data:`FEATURE_NAMES`.
    has_reference : bool
        Whether a reference channel was used.
    """

    plane: np.ndarray
    corrected: np.ndarray
    response: np.ndarray
    global_noise: float
    candidates: list[Candidate]
    features: pd.DataFrame
    has_reference: bool


@dataclass(frozen=True, eq=False)
class LearnedSegmentation:
    """Result of one learned segmentation.

    Attributes
    ----------
    main_mask : numpy.ndarray
        Boolean mask of the accepted candidates.
    review_mask : numpy.ndarray
        Boolean mask of the candidates in the review band, with the main mask removed.
    scores : numpy.ndarray
        Logistic score of every candidate (a logit, not a probability).
    threshold : float
        Score at or above which a candidate is accepted.
    review_margin : float
        Width of the review band below the threshold, in score units.
    n_accepted, n_review, n_candidates : int
        Number of accepted candidates, of review-band candidates and of all candidates.
        Candidate pixel sets may overlap, so these are candidate counts, not counts of
        connected components in the masks: two accepted candidates whose supports touch
        are one object in ``main_mask``. For the number of objects, measure the mask --
        :func:`measure_objects` returns one row per connected component of it.
    candidate_set : CandidateSet
        The candidates and planes the scores were computed from.
    """

    main_mask: np.ndarray
    review_mask: np.ndarray
    scores: np.ndarray
    threshold: float
    review_margin: float
    n_accepted: int
    n_review: int
    n_candidates: int
    candidate_set: CandidateSet


# --------------------------------------------------------------------------- small tools

def _odd(value: int) -> int:
    """Round a window side up to an odd number of pixels."""
    v = int(value)
    return v if v % 2 else v + 1


def _gauss(image: np.ndarray, sigma_px: float) -> np.ndarray:
    """Gaussian blur with a mirrored border, truncated at four standard deviations."""
    return ndi.gaussian_filter(
        image, float(sigma_px), mode=_GAUSS_MODE, truncate=_GAUSS_TRUNCATE
    )


def _ellipse_element(radius_px: int) -> np.ndarray:
    """Flat elliptical element of the given radius in pixels.

    Row ``dy`` of the element spans ``round(r * sqrt(1 - dy**2 / r**2))`` pixels on each
    side of the centre, which makes the element slightly fuller than a strict disk.
    """
    r = int(radius_px)
    if r < 1:
        raise ValueError("radius_px must be at least 1")
    size = 2 * r + 1
    element = np.zeros((size, size), dtype=bool)
    for row in range(size):
        dy = row - r
        half = int(np.floor(np.sqrt(max(r * r - dy * dy, 0)) + 0.5))
        element[row, max(r - half, 0) : min(r + half + 1, size)] = True
    return element


def _square_element(radius_px: int) -> np.ndarray:
    """Flat square element of the given radius in pixels."""
    r = int(radius_px)
    if r < 1:
        raise ValueError("radius_px must be at least 1")
    return np.ones((2 * r + 1, 2 * r + 1), dtype=bool)


def _resize_bilinear(small: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Bilinear upsampling of a coarse grid to ``shape``, replicating the border.

    Output pixel centres are mapped to input coordinates with the half-pixel convention
    ``(i + 0.5) * n_in / n_out - 0.5``.
    """
    rows, cols = small.shape
    height, width = int(shape[0]), int(shape[1])
    y = (np.arange(height, dtype=np.float64) + 0.5) * rows / height - 0.5
    x = (np.arange(width, dtype=np.float64) + 0.5) * cols / width - 0.5
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    return ndi.map_coordinates(
        small.astype(np.float64), [grid_y, grid_x], order=1, mode="nearest"
    ).astype(np.float32)


def _as_feature_frame(table, feature_names: Sequence[str] = FEATURE_NAMES) -> pd.DataFrame:
    """Return ``table`` as a DataFrame holding exactly the named feature columns."""
    names = list(feature_names)
    if isinstance(table, pd.DataFrame):
        missing = [name for name in names if name not in table.columns]
        if missing:
            raise ValueError(f"feature table is missing columns {missing}")
        return table.loc[:, names]
    arr = np.asarray(table, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != len(names):
        raise ValueError(f"feature table must have {len(names)} columns, got {arr.shape}")
    return pd.DataFrame(arr, columns=names)


def _is_feature_table(item) -> bool:
    """Whether ``item`` is one feature table rather than a row of one."""
    if isinstance(item, pd.DataFrame):
        return True
    return isinstance(item, np.ndarray) and item.ndim == 2


def _feature_tables(features) -> list:
    """Return the input as a list of one feature table per image.

    A ``DataFrame`` or a two-dimensional array is the table of one image. A list or tuple
    of such tables is one table per image; anything else (a nested list of numbers, for
    instance) is read as the single table of one image, as :func:`_as_feature_frame` does.
    """
    if isinstance(features, (list, tuple)) and features:
        if all(_is_feature_table(item) for item in features):
            return list(features)
    return [features]


# --------------------------------------------------------------------------- input planes

def to_float_plane(
    image: np.ndarray, *, full_scale: float = 65535.0, min_size_px: int = 64
) -> np.ndarray:
    """Scale an image onto the detector's internal intensity range.

    Integer images are stretched by ``full_scale / dtype_max``. Float images are taken as
    they are, unless their maximum is at most 1.5, in which case they are read as a
    normalised image and multiplied by ``full_scale``. An image with a channel axis is
    reduced to the pixel-wise maximum of its first three channels (any fourth channel is
    ignored). Non-finite values are replaced: not-a-number by zero, plus infinity by
    ``full_scale``, minus infinity by zero.

    Give the detector the channel as acquired. The optional reference subtraction of this
    module takes the place of an external cross-talk correction, so a plane that has
    already been unmixed should not be passed through it twice.

    Parameters
    ----------
    image : numpy.ndarray
        Two-dimensional image, or an image with a trailing channel axis, of any integer
        or floating dtype.
    full_scale : float
        Upper end of the internal intensity range, in grey levels.
    min_size_px : int
        Smallest accepted image side, in pixels.

    Returns
    -------
    numpy.ndarray
        Plane of dtype ``float32`` on the range ``0 .. full_scale``.
    """
    arr = np.asarray(image)
    if arr.ndim == 3:
        plane = arr[..., :3].max(axis=2) if arr.shape[2] >= 3 else arr[..., 0]
    elif arr.ndim == 2:
        plane = arr
    else:
        raise ValueError("image must be 2-D or 2-D with a trailing channel axis")

    scale = float(full_scale)
    if plane.dtype.kind in "ui":
        info = np.iinfo(plane.dtype)
        factor = 1.0 if info.max == int(scale) else scale / float(info.max)
        out = plane.astype(np.float32) * np.float32(factor)
    else:
        out = plane.astype(np.float32)
        top = float(np.nanmax(out)) if out.size else 1.0
        if top <= 1.5:
            out = out * np.float32(scale)

    if out.shape[0] < int(min_size_px) or out.shape[1] < int(min_size_px):
        raise ValueError(
            f"image is {out.shape}, smaller than the detector's minimum "
            f"{min_size_px}x{min_size_px} px"
        )
    return np.nan_to_num(out, nan=0.0, posinf=scale, neginf=0.0)


def subtract_reference(
    raw: np.ndarray,
    reference: np.ndarray | None,
    *,
    offset_percentile: float = 10.0,
    coefficient: float = 1.0,
) -> np.ndarray:
    """Subtract a reference channel from the target channel, clipped at zero.

    The reference channel above its own low percentile is multiplied by ``coefficient``
    and removed from the target: ``max(raw - coefficient * max(ref - p, 0), 0)``, with
    ``p`` the ``offset_percentile`` of the reference. Without a reference the target is
    returned unchanged.

    Parameters
    ----------
    raw : numpy.ndarray
        Target plane, in grey levels.
    reference : numpy.ndarray or None
        Reference plane of the same shape, in grey levels.
    offset_percentile : float
        Percentile of the reference treated as its own offset, in per cent.
    coefficient : float
        Dimensionless weight of the reference. Unit weight is the generic rule of the
        method; a coefficient calibrated on single-stained control sections can be passed
        instead.

    Returns
    -------
    numpy.ndarray
        Corrected plane of dtype ``float32``.
    """
    target = as_float(check_2d(raw, "raw"), np.float32)
    if reference is None:
        return target
    ref = as_float(check_2d(reference, "reference"), np.float32)
    check_same_shape(target, ref)
    offset = float(np.percentile(ref, float(offset_percentile)))
    bleed = np.maximum(ref - offset, 0.0)
    return np.maximum(target - float(coefficient) * bleed, 0.0).astype(np.float32)


# --------------------------------------------------------------- candidate generation

def dog_response(
    raw: np.ndarray,
    *,
    dog_sigma_px: Sequence[Sequence[float]] = ((0.65, 2.2), (1.15, 4.2)),
    noise_sigma_px: float = 1.0,
    noise_floor: float | None = None,
) -> tuple[np.ndarray, float]:
    """Multiscale difference-of-Gaussians local-contrast response, in noise units.

    The global noise is the median absolute deviation of the target plane from its own
    light blur, scaled to a standard deviation and, when ``noise_floor`` is given, bounded
    below by it; without a floor the estimate is whatever the plane itself supports. Each
    band is a difference of two Gaussian blurs divided by that noise, and the response is
    the pixel-wise maximum over the bands. The response is computed on the target plane
    as acquired, not on the reference-corrected plane.

    Parameters
    ----------
    raw : numpy.ndarray
        Target plane, in grey levels.
    dog_sigma_px : sequence of (float, float)
        One pair of Gaussian standard deviations in pixels per band; the first is the
        fine, the second the coarse blur.
    noise_sigma_px : float
        Standard deviation in pixels of the blur used for the noise estimate.
    noise_floor : float or None
        Lower bound of the noise estimate, in grey levels of the user's own images, or
        ``None`` for no bound.

    Returns
    -------
    response : numpy.ndarray
        Local contrast of dtype ``float32``, dimensionless (multiples of the noise).
    noise : float
        Global noise estimate, in grey levels.
    """
    plane = as_float(check_2d(raw, "raw"), np.float32)
    residual = plane - _gauss(plane, noise_sigma_px)
    noise = float(_MAD_SCALE * np.median(np.abs(residual)))
    if noise_floor is not None:
        noise = max(noise, float(noise_floor))
    if noise <= 0.0:
        # A plane whose residual vanishes carries no contrast to measure: the response is
        # zero everywhere, so nothing is proposed, rather than dividing by zero.
        return np.zeros_like(plane, dtype=np.float32), 0.0
    bands = []
    for fine_sigma, coarse_sigma in dog_sigma_px:
        bands.append((_gauss(plane, fine_sigma) - _gauss(plane, coarse_sigma)) / noise)
    if not bands:
        raise ValueError("dog_sigma_px must hold at least one pair of sigmas")
    response = bands[0]
    for band in bands[1:]:
        response = np.maximum(response, band)
    return response.astype(np.float32), noise


def propose_seeds(
    response: np.ndarray, *, seed_threshold: float = 3.5, window_px: int = 5
) -> np.ndarray:
    """Propose one seed per local maximum of the response.

    A pixel is a seed when it equals the maximum of its square neighbourhood and reaches
    ``seed_threshold``. Pixels of a flat plateau are grouped with 4-connectivity and one
    representative is kept per plateau: the pixel with the highest response, ties broken
    by the smallest row and then the smallest column. Seeds come out in plateau label
    order, which is the raster order of each plateau's first pixel.

    Parameters
    ----------
    response : numpy.ndarray
        Local-contrast response, in noise units.
    seed_threshold : float
        Smallest response of a seed, in noise units.
    window_px : int
        Side of the square neighbourhood, in pixels.

    Returns
    -------
    numpy.ndarray
        Integer array of shape ``(n_seeds, 2)`` with the row and column of every seed, in
        pixels.
    """
    resp = as_float(check_2d(response, "response"), np.float32)
    local_max = ndi.maximum_filter(resp, size=int(window_px), mode="reflect")
    maxima = (resp == local_max) & (resp >= float(seed_threshold))
    ys, xs = np.nonzero(maxima)
    if ys.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    plateaus, _ = label_mask(maxima, connectivity=4)
    ids = plateaus[ys, xs]
    order = np.lexsort((xs, ys, -resp[ys, xs], ids))
    sorted_ids = ids[order]
    keep = np.empty(order.size, dtype=bool)
    keep[0] = True
    np.not_equal(sorted_ids[1:], sorted_ids[:-1], out=keep[1:])
    chosen = order[keep]
    return np.stack([ys[chosen], xs[chosen]], axis=1).astype(np.int64)


def grow_candidates(
    response: np.ndarray,
    seeds: np.ndarray,
    *,
    radius_px: int = 9,
    seed_threshold: float = 3.5,
    low_fraction: float = 0.18,
    low_min_fraction: float = 0.4,
    low_floor: float = 0.8,
) -> list[Candidate]:
    """Grow one candidate pixel set around every seed.

    Growth runs inside a square window of half-side ``radius_px`` centred on the seed;
    pixels outside the image never join. The growth threshold is
    ``max(low_min_fraction * seed_threshold, low_fraction * peak, low_floor)`` with
    ``peak`` the seed's response, and the candidate is the 8-connected component of the
    pixels at or above that threshold which contains the seed. Candidates of different
    seeds may overlap; nothing is made exclusive and no randomness is involved.

    Parameters
    ----------
    response : numpy.ndarray
        Local-contrast response, in noise units.
    seeds : numpy.ndarray
        Integer array of shape ``(n_seeds, 2)`` with seed rows and columns, in pixels.
    radius_px : int
        Half-side of the growth window, in pixels.
    seed_threshold : float
        Seed threshold the lower bound of the growth threshold refers to, in noise units.
    low_fraction : float
        Fraction of the seed's peak response used as the growth threshold.
    low_min_fraction : float
        Fraction of ``seed_threshold`` used as the lower bound.
    low_floor : float
        Absolute lower bound of the growth threshold, in noise units.

    Returns
    -------
    list of Candidate
        One candidate per seed, in the order of ``seeds``.
    """
    resp = as_float(check_2d(response, "response"), np.float32)
    coords = np.asarray(seeds, dtype=np.int64).reshape(-1, 2)
    r = int(radius_px)
    if r < 1:
        raise ValueError("radius_px must be at least 1")
    size = 2 * r + 1
    height, width = resp.shape
    padded = np.full((height + 2 * r, width + 2 * r), -np.inf, dtype=np.float32)
    padded[r : r + height, r : r + width] = resp
    low_min = float(low_min_fraction) * float(seed_threshold)
    out = []
    for seed_y, seed_x in coords:
        peak = float(resp[seed_y, seed_x])
        low = np.float32(max(low_min, float(low_fraction) * peak, float(low_floor)))
        window = padded[seed_y : seed_y + size, seed_x : seed_x + size]
        labels, _ = label_mask(window >= low, connectivity=8)
        seed_label = int(labels[r, r])
        if seed_label == 0:
            continue
        dy, dx = np.nonzero(labels == seed_label)
        out.append(
            Candidate(
                ys=(dy + int(seed_y) - r).astype(np.int64),
                xs=(dx + int(seed_x) - r).astype(np.int64),
                seed_y=int(seed_y),
                seed_x=int(seed_x),
            )
        )
    return out


# --------------------------------------------------------------------------- features

def local_noise_map(image: np.ndarray, *, tile_px: int = 50) -> np.ndarray:
    """Per-pixel noise level estimated block by block, in grey levels.

    Horizontal neighbour differences are taken, the difference image is padded by
    reflection to whole blocks of ``tile_px``, and each block's median difference is
    scaled to a standard deviation (median absolute deviation scaling divided by the
    square root of two, because a difference of two pixels has twice the variance). The
    block grid is smoothed with a 3x3 median filter, bounded below by a quarter of the
    median of its positive blocks, upsampled bilinearly to the image size and bounded
    below by a small positive value.

    Parameters
    ----------
    image : numpy.ndarray
        Plane to estimate the noise of, in grey levels.
    tile_px : int
        Side of the square blocks, in pixels. It must be small enough for the difference
        image to be padded by reflection.

    Returns
    -------
    numpy.ndarray
        Strictly positive noise level per pixel, dtype ``float32``, in grey levels.
    """
    plane = as_float(check_2d(image, "image"), np.float32)
    tile = int(tile_px)
    if tile < 1:
        raise ValueError("tile_px must be at least 1")
    differences = np.abs(np.diff(plane, axis=1))
    pad_rows = (-differences.shape[0]) % tile
    pad_cols = (-differences.shape[1]) % tile
    if pad_rows or pad_cols:
        differences = np.pad(differences, ((0, pad_rows), (0, pad_cols)), mode="reflect")
    rows = differences.shape[0] // tile
    cols = differences.shape[1] // tile
    blocks = (
        differences.reshape(rows, tile, cols, tile)
        .transpose(0, 2, 1, 3)
        .reshape(rows, cols, -1)
    )
    grid = (np.median(blocks, axis=2) * _MAD_SCALE / np.sqrt(2.0)).astype(np.float32)
    grid = ndi.median_filter(grid, size=3, mode="nearest")
    positive = grid[grid > 0]
    floor = 0.25 * float(np.median(positive)) if positive.size else 0.0
    grid = np.maximum(grid, floor)
    return np.maximum(_resize_bilinear(grid, plane.shape), 1e-3)


def reference_tone_map(
    reference: np.ndarray, *, sigma_px: float = 4.0, tile_px: int = 50
) -> np.ndarray:
    """Reference-channel level in units of the reference channel's own local noise.

    The reference plane is high-passed by subtracting its Gaussian blur and divided by its
    local noise map, so the result says how much reference signal stands out locally.

    Parameters
    ----------
    reference : numpy.ndarray
        Reference plane, in grey levels.
    sigma_px : float
        Standard deviation of the high-pass blur, in pixels.
    tile_px : int
        Side of the local-noise blocks, in pixels.

    Returns
    -------
    numpy.ndarray
        Dimensionless map of dtype ``float32``.
    """
    ref = as_float(check_2d(reference, "reference"), np.float32)
    high_pass = ref - _gauss(ref, sigma_px)
    return (high_pass / local_noise_map(ref, tile_px=tile_px)).astype(np.float32)


def _axis_ratio(ys: np.ndarray, xs: np.ndarray) -> float:
    """Long-to-short axis ratio of a pixel set from its sample covariance."""
    count = int(ys.size)
    if count <= 2:
        return 1.0
    yf = ys.astype(np.float64)
    xf = xs.astype(np.float64)
    dy = yf - yf.mean()
    dx = xf - xf.mean()
    denominator = count - 1
    a = float(dy @ dy) / denominator
    b = float(dy @ dx) / denominator
    c = float(dx @ dx) / denominator
    half = 0.5 * (a + c)
    disc = float(np.sqrt(max(0.25 * (a - c) ** 2 + b * b, 0.0)))
    return float(np.sqrt(max(half + disc, 1e-9) / max(half - disc, 1e-9)))


def candidate_features(
    candidates: Sequence[Candidate],
    corrected: np.ndarray,
    response: np.ndarray,
    noise_map: np.ndarray,
    reference_tone: np.ndarray | None = None,
    *,
    ring_outer_radius_px: int = 6,
    ring_inner_radius_px: int = 2,
    min_ring_px: int = 21,
) -> pd.DataFrame:
    """Six features per candidate.

    ``ringnear`` is the candidate's mean corrected intensity minus the median intensity of
    the surrounding ring, divided by the candidate's mean local noise; the ring is the
    candidate dilated by ``ring_outer_radius_px`` with the candidate dilated by
    ``ring_inner_radius_px`` removed, clipped to the image, and the feature is set to zero
    when the ring holds fewer than ``min_ring_px`` pixels. ``zmean`` and ``zmax`` are the
    mean and the maximum response inside the candidate. ``noiserel`` is the candidate's
    mean local noise divided by the median of the noise map. ``reftone`` is the mean
    reference tone inside the candidate, zero without a reference channel. ``aspect`` is
    the long-to-short axis ratio of the candidate's pixel coordinates (unity for
    candidates of at most two pixels).

    Parameters
    ----------
    candidates : sequence of Candidate
        Candidates of one image.
    corrected : numpy.ndarray
        Target plane after the optional reference subtraction, in grey levels.
    response : numpy.ndarray
        Local-contrast response, in noise units.
    noise_map : numpy.ndarray
        Strictly positive local noise of ``corrected``, in grey levels.
    reference_tone : numpy.ndarray or None
        Dimensionless reference tone map, or None without a reference channel.
    ring_outer_radius_px, ring_inner_radius_px : int
        Outer and inner radius of the surrounding ring, in pixels.
    min_ring_px : int
        Smallest usable ring size, in pixels.

    Returns
    -------
    pandas.DataFrame
        One row per candidate, columns :data:`FEATURE_NAMES`, dtype ``float32``.
    """
    plane = as_float(check_2d(corrected, "corrected"), np.float32)
    resp = as_float(check_2d(response, "response"), np.float32)
    noise = as_float(check_2d(noise_map, "noise_map"), np.float32)
    check_same_shape(plane, resp, noise)
    if reference_tone is not None:
        tone = as_float(check_2d(reference_tone, "reference_tone"), np.float32)
        check_same_shape(plane, tone)
    else:
        tone = None
    if not len(candidates):
        return pd.DataFrame(
            np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32), columns=list(FEATURE_NAMES)
        )

    height, width = resp.shape
    outer = _ellipse_element(ring_outer_radius_px)
    inner = _square_element(ring_inner_radius_px)
    # The crop is widened by the outer ring radius, so the ring is complete except where
    # the image border cuts it off.
    margin = int(ring_outer_radius_px)
    noise_median = max(float(np.median(noise)), 1e-9)

    rows = np.zeros((len(candidates), len(FEATURE_NAMES)), dtype=np.float32)
    for index, candidate in enumerate(candidates):
        ys = np.asarray(candidate.ys, dtype=np.int64)
        xs = np.asarray(candidate.xs, dtype=np.int64)
        if ys.size == 0:
            continue
        y0 = max(0, int(ys.min()) - margin)
        y1 = min(height, int(ys.max()) + margin + 1)
        x0 = max(0, int(xs.min()) - margin)
        x1 = min(width, int(xs.max()) + margin + 1)
        local = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        local[ys - y0, xs - x0] = True
        ring = ndi.binary_dilation(local, structure=outer) & ~ndi.binary_dilation(
            local, structure=inner
        )
        ring_px = int(ring.sum())
        candidate_noise = float(noise[ys, xs].mean(dtype=np.float64))
        if ring_px >= int(min_ring_px):
            ring_median = float(np.median(plane[y0:y1, x0:x1][ring]))
            inside_mean = float(plane[ys, xs].mean(dtype=np.float64))
            ringnear = (inside_mean - ring_median) / candidate_noise
        else:
            ringnear = 0.0
        inside_response = resp[ys, xs]
        reftone = float(tone[ys, xs].mean(dtype=np.float64)) if tone is not None else 0.0
        rows[index] = (
            ringnear,
            float(inside_response.mean(dtype=np.float64)),
            float(inside_response.max()),
            candidate_noise / noise_median,
            reftone,
            _axis_ratio(ys, xs),
        )
    return pd.DataFrame(rows, columns=list(FEATURE_NAMES))


def detect_candidates(
    image: np.ndarray,
    reference: np.ndarray | None = None,
    *,
    params: DetectorParams = DetectorParams(),
) -> CandidateSet:
    """Run the whole candidate stage of the detector on one image.

    Scales the image (and the optional reference channel) onto the internal intensity
    range, subtracts the reference, computes the local-contrast response from the target
    plane, proposes seeds, grows a candidate around each seed and measures the six
    features.

    Parameters
    ----------
    image : numpy.ndarray
        Target channel, any integer or float dtype; see :func:`to_float_plane`.
    reference : numpy.ndarray or None
        Reference channel of the same shape, or None.
    params : DetectorParams
        Working scale and rules of the detector; spatial fields in pixels.

    Returns
    -------
    CandidateSet
        Candidates, their features and the planes they were measured on.
    """
    plane = to_float_plane(
        image, full_scale=params.full_scale, min_size_px=params.min_size_px
    )
    if reference is not None:
        ref = to_float_plane(
            reference, full_scale=params.full_scale, min_size_px=params.min_size_px
        )
        check_same_shape(plane, ref)
    else:
        ref = None

    corrected = subtract_reference(
        plane,
        ref,
        offset_percentile=params.reference_offset_percentile,
        coefficient=params.reference_coefficient,
    )
    response, global_noise = dog_response(
        plane,
        dog_sigma_px=params.dog_sigma_px,
        noise_sigma_px=params.noise_sigma_px,
        noise_floor=params.noise_floor,
    )
    seeds = propose_seeds(
        response, seed_threshold=params.seed_threshold, window_px=params.seed_window_px
    )
    candidates = grow_candidates(
        response,
        seeds,
        radius_px=params.candidate_radius_px,
        seed_threshold=params.seed_threshold,
        low_fraction=params.low_fraction,
        low_min_fraction=params.low_min_fraction,
        low_floor=params.low_floor,
    )
    noise = local_noise_map(corrected, tile_px=params.noise_tile_px)
    tone = (
        reference_tone_map(
            ref, sigma_px=params.reference_sigma_px, tile_px=params.noise_tile_px
        )
        if ref is not None
        else None
    )
    features = candidate_features(
        candidates,
        corrected,
        response,
        noise,
        tone,
        ring_outer_radius_px=params.ring_outer_radius_px,
        ring_inner_radius_px=params.ring_inner_radius_px,
        min_ring_px=params.min_ring_px,
    )
    return CandidateSet(
        plane=plane,
        corrected=corrected,
        response=response,
        global_noise=float(global_noise),
        candidates=candidates,
        features=features,
        has_reference=ref is not None,
    )


# --------------------------------------------------------------- normalisation, scoring

def _normalization_record(
    log_features: Sequence[str], clip: float, min_lower: int
) -> dict[str, object]:
    """Settings of one per-image normalisation, as recorded on a normalised table."""
    return {
        "rule": NORMALIZATION,
        "log_features": tuple(str(name) for name in log_features),
        "clip": float(clip),
        "min_lower": int(min_lower),
    }


def _normalization_stamp(table) -> dict[str, object] | None:
    """Return the normalisation settings recorded on ``table``, or None."""
    attrs = getattr(table, "attrs", None)
    if isinstance(attrs, dict):
        stamp = attrs.get(NORMALIZATION_KEY)
        if isinstance(stamp, dict):
            return dict(stamp)
    return None


def _describe_normalization(record: Mapping[str, object]) -> str:
    """One readable line per normalisation record, for an error message."""
    return (
        f"rule={record.get('rule')!r}, "
        f"log_features={tuple(record.get('log_features', ()))}, "
        f"clip={record.get('clip')}, min_lower={record.get('min_lower')}"
    )


def _check_clip_bound(frame: pd.DataFrame, clip: float, where: str) -> None:
    """Refuse a table that cannot have come out of a normalisation with this clip."""
    values = frame.to_numpy(dtype=np.float64)
    finite = values[np.isfinite(values)] if values.size else values
    if finite.size == 0:
        return
    peak = float(np.abs(finite).max())
    if peak > float(clip) * (1.0 + 1e-6):
        raise ValueError(
            f"{where} holds a value of {peak:.4g}, outside the +/-{float(clip):g} bound "
            "of the per-image normalisation, so it has not been normalised with this "
            "clip: pass the raw features and let the fit normalise them, or normalise "
            "with normalize_per_image and the same clip"
        )


def _same_normalization(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    """Whether two normalisation records describe the same rule and settings."""
    try:
        return (
            str(left["rule"]) == str(right["rule"])
            and tuple(str(name) for name in left["log_features"])
            == tuple(str(name) for name in right["log_features"])
            and float(left["clip"]) == float(right["clip"])
            and int(left["min_lower"]) == int(right["min_lower"])
        )
    except (KeyError, TypeError, ValueError):
        return False


def normalize_per_image(
    features,
    *,
    log_features: Sequence[str] = LOG_FEATURES,
    clip: float = _DEFAULT_CLIP,
    min_lower: int = _DEFAULT_MIN_LOWER,
) -> pd.DataFrame:
    """Robust per-image normalisation of a feature table, anchored on the background side.

    The skewed features named in ``log_features`` first pass through
    ``log1p(max(x, 0))``. Every column is then centred on its median and divided by the
    median absolute deviation of its lower half (the values at or below the median),
    scaled to a standard deviation; a column with fewer than ``min_lower`` values in its
    lower half keeps a scale of one. The result is clipped to ``[-clip, +clip]``. Taking
    the scale from the lower half only keeps the true signal in the upper half from
    inflating it, which would compress images rich in objects.

    This normalisation runs over all candidates of one image, labelled or not, and the
    same rule has to be used for fitting and for inference. Calling it by hand is not
    needed for either: :func:`fit_learned_model` and :func:`score_candidates` both take
    raw features and run it themselves. The settings used are recorded in the ``attrs`` of
    the returned table under :data:`NORMALIZATION_KEY`, so that fitting can tell a
    normalised table from a raw one and refuse settings other than its own.

    Parameters
    ----------
    features : pandas.DataFrame or array_like
        Raw features of one image, columns :data:`FEATURE_NAMES` (dimensionless or in
        noise units, as produced by :func:`candidate_features`).
    log_features : sequence of str
        Names of the features that go through the log transform.
    clip : float
        Bound of the normalised score, dimensionless.
    min_lower : int
        Smallest number of values in the lower half for which a scale is estimated.

    Returns
    -------
    pandas.DataFrame
        Normalised, dimensionless features with the same columns and index, carrying the
        settings used in ``attrs[NORMALIZATION_KEY]``.
    """
    frame = _as_feature_frame(features)
    record = _normalization_record(log_features, clip, min_lower)
    values = frame.to_numpy(dtype=np.float32, copy=True)
    if values.shape[0] == 0:
        empty = pd.DataFrame(values, columns=frame.columns, index=frame.index)
        empty.attrs[NORMALIZATION_KEY] = record
        return empty

    log_set = set(log_features)
    for position, name in enumerate(frame.columns):
        if name in log_set:
            values[:, position] = np.log1p(np.maximum(values[:, position], 0.0))

    median = np.median(values, axis=0)
    scale = np.empty(values.shape[1], dtype=np.float64)
    for position in range(values.shape[1]):
        column = values[:, position]
        lower = column[column <= median[position]]
        scale[position] = (
            mad(lower, centre=float(median[position]))
            if lower.size >= int(min_lower)
            else 1.0
        )
    normalized = np.clip(
        (values - median) / np.maximum(scale, 1e-9), -float(clip), float(clip)
    )
    out = pd.DataFrame(normalized, columns=frame.columns, index=frame.index)
    out.attrs[NORMALIZATION_KEY] = record
    return out


def standardize_features(normalized, model: "LearnedModel") -> np.ndarray:
    """Standardise per-image normalised features with a model's mean and scale.

    Parameters
    ----------
    normalized : pandas.DataFrame or array_like
        Per-image normalised features, dimensionless.
    model : LearnedModel
        Model carrying the mean and scale of the fitted training table.

    Returns
    -------
    numpy.ndarray
        Standardised features of shape ``(n_candidates, n_features)``, dimensionless.
    """
    frame = _as_feature_frame(normalized, model.feature_names)
    return (frame.to_numpy(dtype=np.float64) - model.mean) / model.scale


def logistic_score(standardized, coef, intercept: float) -> np.ndarray:
    """Linear logistic score of standardised features.

    The score is a logit, not a probability: ``features . coef + intercept``.

    Parameters
    ----------
    standardized : array_like
        Standardised features of shape ``(n_candidates, n_features)``, dimensionless.
    coef : array_like
        One coefficient per feature, dimensionless.
    intercept : float
        Intercept of the linear score, dimensionless.

    Returns
    -------
    numpy.ndarray
        Score per candidate, dtype ``float32``.
    """
    matrix = np.asarray(standardized, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    weights = np.asarray(coef, dtype=np.float64).ravel()
    if matrix.shape[1] != weights.size:
        raise ValueError("coef must hold one value per feature")
    return (matrix @ weights + float(intercept)).astype(np.float32)


def score_candidates(
    features, model: "LearnedModel", *, has_reference: bool = True
) -> np.ndarray:
    """Score the candidates of one image.

    Applies the per-image normalisation recorded in the model, standardises with the
    model's mean and scale and returns the linear logistic score. Without a reference
    channel the reference feature is set to the training mean (zero after
    standardisation), so it does not shift the score.

    Parameters
    ----------
    features : pandas.DataFrame or array_like
        Raw features of one image, columns :data:`FEATURE_NAMES`, as
        :attr:`CandidateSet.features` holds them. A table that has already been through
        :func:`normalize_per_image` is refused, because it would be normalised twice; to
        score one of those, call :func:`standardize_features` and :func:`logistic_score`
        yourself.
    model : LearnedModel
        Fitted model.
    has_reference : bool
        Whether a reference channel was used for the features.

    Returns
    -------
    numpy.ndarray
        Score per candidate, dtype ``float32``.

    Raises
    ------
    ValueError
        If the table carries the record of a per-image normalisation.
    """
    if _normalization_stamp(features) is not None:
        raise ValueError(
            "this feature table has already been normalised per image; pass the raw "
            "features of one image, which score_candidates normalises itself, or score a "
            "normalised table with standardize_features and logistic_score"
        )
    normalized = normalize_per_image(
        features,
        log_features=model.log_features,
        clip=model.clip,
        min_lower=_DEFAULT_MIN_LOWER,
    )
    standardized = standardize_features(normalized, model)
    if not has_reference:
        standardized[:, model.reference_index] = 0.0
    return logistic_score(standardized, model.coef, model.intercept)


# --------------------------------------------------------------------------- masks

def paint_candidates(
    shape: tuple, candidates: Sequence[Candidate], selected: np.ndarray
) -> tuple[np.ndarray, int]:
    """Paint the pixel sets of the selected candidates into a boolean mask.

    Parameters
    ----------
    shape : tuple of int
        Mask shape in pixels, ``(height, width)``.
    candidates : sequence of Candidate
        All candidates of the image.
    selected : array_like of bool
        One flag per candidate.

    Returns
    -------
    mask : numpy.ndarray
        Boolean mask of the union of the selected candidates.
    n_selected : int
        Number of selected candidates. Candidate pixel sets may overlap, so this is not
        the number of connected components in the mask.
    """
    mask = np.zeros((int(shape[0]), int(shape[1])), dtype=bool)
    flags = np.asarray(selected, dtype=bool).ravel()
    if flags.size != len(candidates):
        raise ValueError("selected must hold one flag per candidate")
    index = np.flatnonzero(flags)
    if index.size:
        ys = np.concatenate([np.asarray(candidates[i].ys) for i in index])
        xs = np.concatenate([np.asarray(candidates[i].xs) for i in index])
        mask[ys, xs] = True
    return mask, int(index.size)


def segment_learned(
    image: np.ndarray,
    model: "LearnedModel",
    reference: np.ndarray | None = None,
    *,
    params: DetectorParams = DetectorParams(),
    threshold: float | None = None,
    review_margin: float = 1.5,
) -> LearnedSegmentation:
    """Detect, score and paint objects in one image.

    A candidate is accepted when its score reaches the threshold, and falls into the
    review band when its score lies within ``review_margin`` below the threshold. The
    review mask never overlaps the main mask.

    Parameters
    ----------
    image : numpy.ndarray
        Target channel.
    model : LearnedModel
        Model fitted on the user's own labelled candidates.
    reference : numpy.ndarray or None
        Reference channel of the same shape, or None.
    params : DetectorParams
        Working scale and rules of the detector; spatial fields in pixels.
    threshold : float or None
        Score at or above which a candidate is accepted, dimensionless. None uses the
        model's own threshold.
    review_margin : float
        Width of the review band below the threshold, in score units.

    Returns
    -------
    LearnedSegmentation
        Masks, scores and counts.
    """
    candidate_set = detect_candidates(image, reference, params=params)
    cut = float(model.threshold if threshold is None else threshold)
    scores = score_candidates(
        candidate_set.features, model, has_reference=candidate_set.has_reference
    )
    accepted = scores >= cut
    in_review = (scores >= cut - float(review_margin)) & ~accepted
    shape = candidate_set.response.shape
    main_mask, n_accepted = paint_candidates(shape, candidate_set.candidates, accepted)
    review_mask, n_review = paint_candidates(shape, candidate_set.candidates, in_review)
    return LearnedSegmentation(
        main_mask=main_mask,
        review_mask=review_mask & ~main_mask,
        scores=scores,
        threshold=cut,
        review_margin=float(review_margin),
        n_accepted=n_accepted,
        n_review=n_review,
        n_candidates=len(candidate_set.candidates),
        candidate_set=candidate_set,
    )


# --------------------------------------------------------------------------- measurement

_OBJECT_COLUMNS = (
    "label",
    "area_px",
    "area_um2",
    "perimeter_px",
    "perimeter_um",
    "circularity",
    "solidity",
    "hole_count",
    "hole_area_px",
    "hole_fraction",
    "mean_thickness_px",
    "mean_thickness_um",
    "max_thickness_px",
    "max_thickness_um",
    "elongation_index",
    "centroid_y_px",
    "centroid_x_px",
    "centroid_y_um",
    "centroid_x_um",
    "bbox_y_px",
    "bbox_x_px",
    "bbox_height_px",
    "bbox_width_px",
    "touches_border",
    "intensity_mean",
    "intensity_max",
    "intensity_integrated",
    "response_mean",
    "response_max",
    "response_confidence",
    "saturated_fraction",
)


def measure_objects(
    mask: np.ndarray,
    plane: np.ndarray,
    response: np.ndarray,
    pixel_size_um: float,
    *,
    response_reference: float = DetectorParams.seed_threshold,
    saturation_level: float = 65000.0,
    connectivity: int = 8,
) -> pd.DataFrame:
    """Measure every connected object of a mask, without any filtering.

    Objects are the connected components of the mask (8-connectivity by default). Nothing
    is removed, smoothed or split: the accepted candidates are measured as they are.
    Shape terms follow the usual definitions: ``circularity = 4 * pi * area / perimeter^2``
    bounded at unity, ``solidity = area / convex hull area``, holes are the background
    components inside the object (4-connected), the mean thickness is four times the mean
    interior distance to the boundary and the maximum thickness twice the largest such
    distance, and ``elongation_index = area / mean thickness^2``.

    Parameters
    ----------
    mask : numpy.ndarray
        Boolean object mask.
    plane : numpy.ndarray
        Intensity plane the objects were detected on, in grey levels.
    response : numpy.ndarray
        Local-contrast response, in noise units.
    pixel_size_um : float
        Pixel size, in micrometres per pixel.
    response_reference : float
        Response, in noise units, that the mean response is divided by to give
        ``response_confidence`` (a value of one means the object is exactly at that
        reference contrast). The seed threshold the candidates were proposed with is the
        natural choice, and the default is the ``seed_threshold`` of
        :class:`DetectorParams` itself; pass ``params.seed_threshold`` when you detected
        with a parameter set that changed it, or the confidence refers to a threshold the
        objects were never held to.
    saturation_level : float
        Grey level at or above which a pixel counts as saturated.
    connectivity : int
        4 or 8; the neighbourhood that joins object pixels.

    Returns
    -------
    pandas.DataFrame
        One row per object, with areas in pixels and square micrometres, lengths in
        pixels and micrometres, and the intensity and response statistics.
    """
    binary = as_bool_mask(mask)
    intensity = as_float(check_2d(plane, "plane"), np.float64)
    resp = as_float(check_2d(response, "response"), np.float64)
    check_same_shape(binary, intensity, resp)
    labels, _ = label_mask(binary, connectivity=connectivity)
    height, width = binary.shape
    reference = float(response_reference)

    records = []
    for index, window in enumerate(ndi.find_objects(labels), start=1):
        if window is None:
            continue
        row_slice, col_slice = window
        component = labels[window] == index
        area_px = int(np.count_nonzero(component))
        if area_px == 0:
            continue
        local_y, local_x = np.nonzero(component)
        y0 = int(row_slice.start)
        x0 = int(col_slice.start)
        ys = local_y + y0
        xs = local_x + x0

        padded = np.zeros(
            (component.shape[0] + 2, component.shape[1] + 2), dtype=bool
        )
        padded[1:-1, 1:-1] = component
        # Boundary length through the 4-neighbourhood estimator, in pixels.
        perimeter_px = float(_skmeasure.perimeter(padded, 4))
        circularity = (
            min(1.0, 4.0 * np.pi * area_px / (perimeter_px * perimeter_px))
            if perimeter_px > 0
            else 0.0
        )
        hull_area = float(_skmorphology.convex_hull_image(padded).sum())
        solidity = min(1.0, area_px / hull_area) if hull_area > 0 else 1.0

        background, n_background = label_mask(~padded, connectivity=4)
        hole_area_px = 0
        hole_count = 0
        if n_background > 1:
            counts = np.bincount(background.ravel(), minlength=n_background + 1)
            outside = int(background[0, 0])
            hole_labels = [
                component_label
                for component_label in range(1, n_background + 1)
                if component_label != outside
            ]
            hole_count = len(hole_labels)
            hole_area_px = int(sum(int(counts[label]) for label in hole_labels))
        hole_fraction = hole_area_px / max(area_px + hole_area_px, 1)

        distance = ndi.distance_transform_edt(padded)[1:-1, 1:-1][component]
        mean_thickness_px = max(1.0, 4.0 * float(distance.mean()))
        max_thickness_px = max(1.0, 2.0 * float(distance.max()))
        elongation = area_px / max(mean_thickness_px * mean_thickness_px, 1.0)

        intensity_values = intensity[ys, xs]
        response_values = resp[ys, xs]
        response_mean = float(response_values.mean())
        confidence = response_mean / reference if reference > 1e-8 else 0.0
        centroid_y = float(ys.mean())
        centroid_x = float(xs.mean())
        touches_border = bool(
            y0 == 0
            or x0 == 0
            or int(row_slice.stop) >= height
            or int(col_slice.stop) >= width
        )

        records.append(
            {
                "label": index,
                "area_px": area_px,
                "area_um2": area_px_to_um2(area_px, pixel_size_um),
                "perimeter_px": perimeter_px,
                "perimeter_um": px_to_um(perimeter_px, pixel_size_um),
                "circularity": float(circularity),
                "solidity": float(solidity),
                "hole_count": hole_count,
                "hole_area_px": hole_area_px,
                "hole_fraction": float(hole_fraction),
                "mean_thickness_px": mean_thickness_px,
                "mean_thickness_um": px_to_um(mean_thickness_px, pixel_size_um),
                "max_thickness_px": max_thickness_px,
                "max_thickness_um": px_to_um(max_thickness_px, pixel_size_um),
                "elongation_index": float(elongation),
                "centroid_y_px": centroid_y,
                "centroid_x_px": centroid_x,
                "centroid_y_um": px_to_um(centroid_y, pixel_size_um),
                "centroid_x_um": px_to_um(centroid_x, pixel_size_um),
                "bbox_y_px": y0,
                "bbox_x_px": x0,
                "bbox_height_px": int(row_slice.stop) - y0,
                "bbox_width_px": int(col_slice.stop) - x0,
                "touches_border": touches_border,
                "intensity_mean": float(intensity_values.mean()),
                "intensity_max": float(intensity_values.max()),
                "intensity_integrated": float(intensity_values.sum()),
                "response_mean": response_mean,
                "response_max": float(response_values.max()),
                "response_confidence": float(confidence),
                "saturated_fraction": float(
                    np.count_nonzero(intensity_values >= float(saturation_level)) / area_px
                ),
            }
        )
    return pd.DataFrame(records, columns=list(_OBJECT_COLUMNS))


# --------------------------------------------------------------------------- the model

@dataclass(frozen=True, eq=False)
class LearnedModel:
    """Fitted detector model: standardisation, coefficients and the decision threshold.

    Attributes
    ----------
    feature_names : tuple of str
        Feature order the mean, scale and coefficients refer to.
    log_features : tuple of str
        Features that go through ``log1p(max(x, 0))`` before the per-image normalisation.
    clip : float
        Bound used by the per-image normalisation, dimensionless.
    mean, scale : numpy.ndarray
        Mean and scale of the per-image normalised training features, dimensionless.
    coef : numpy.ndarray
        One logistic coefficient per feature, dimensionless.
    intercept : float
        Intercept of the linear score, dimensionless.
    threshold : float
        Score at or above which a candidate is accepted, dimensionless.
    normalization : str
        Name of the per-image normalisation rule; inference refuses a rule it does not
        implement.
    """

    feature_names: tuple[str, ...]
    log_features: tuple[str, ...]
    clip: float
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    threshold: float
    normalization: str = NORMALIZATION

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.feature_names)
        if not names:
            raise ValueError("feature_names must not be empty")
        if str(self.normalization) != NORMALIZATION:
            raise ValueError(
                f"unknown normalisation {self.normalization!r}; this module implements "
                f"{NORMALIZATION!r}"
            )
        vectors = {}
        for field_name in ("mean", "scale", "coef"):
            vector = np.asarray(getattr(self, field_name), dtype=np.float64).ravel()
            if vector.size != len(names):
                raise ValueError(f"{field_name} must hold one value per feature")
            vectors[field_name] = vector
        if np.any(vectors["scale"] == 0):
            raise ValueError("scale must not contain zeros")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(
            self, "log_features", tuple(str(name) for name in self.log_features)
        )
        object.__setattr__(self, "clip", float(self.clip))
        object.__setattr__(self, "intercept", float(self.intercept))
        object.__setattr__(self, "threshold", float(self.threshold))
        object.__setattr__(self, "normalization", str(self.normalization))
        for field_name, vector in vectors.items():
            object.__setattr__(self, field_name, vector)

    @property
    def reference_index(self) -> int:
        """Position of the reference-channel feature in the feature order."""
        if REFERENCE_FEATURE in self.feature_names:
            return self.feature_names.index(REFERENCE_FEATURE)
        raise ValueError(f"the model has no {REFERENCE_FEATURE!r} feature")

    def with_threshold(self, threshold: float) -> "LearnedModel":
        """Return a copy of the model carrying another decision threshold."""
        return replace(self, threshold=float(threshold))

    def to_dict(self) -> dict[str, object]:
        """Return the model as a plain dictionary ready to be written as JSON."""
        return {
            "feats": list(self.feature_names),
            "log_feats": list(self.log_features),
            "clip": float(self.clip),
            "mean": [float(value) for value in self.mean],
            "scale": [float(value) for value in self.scale],
            "coef": [float(value) for value in self.coef],
            "intercept": float(self.intercept),
            "threshold": float(self.threshold),
            "norm": str(self.normalization),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "LearnedModel":
        """Build a model from the plain dictionary written by :meth:`to_dict`."""
        missing = [
            key
            for key in (
                "feats",
                "log_feats",
                "clip",
                "mean",
                "scale",
                "coef",
                "intercept",
                "threshold",
                "norm",
            )
            if key not in payload
        ]
        if missing:
            raise ValueError(f"model is missing the keys {missing}")
        return cls(
            feature_names=tuple(payload["feats"]),
            log_features=tuple(payload["log_feats"]),
            clip=float(payload["clip"]),
            mean=np.asarray(payload["mean"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
            coef=np.asarray(payload["coef"], dtype=np.float64),
            intercept=float(payload["intercept"]),
            threshold=float(payload["threshold"]),
            normalization=str(payload["norm"]),
        )


def save_model(model: LearnedModel, path) -> Path:
    """Write a fitted model to a JSON file.

    Parameters
    ----------
    model : LearnedModel
        Model to store.
    path : str or pathlib.Path
        Destination file.

    Returns
    -------
    pathlib.Path
        The path written.
    """
    target = Path(path)
    target.write_text(
        json.dumps(model.to_dict(), indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    return target


def load_model(path) -> LearnedModel:
    """Read a model from a JSON file written by :func:`save_model`.

    Parameters
    ----------
    path : str or pathlib.Path
        Model file.

    Returns
    -------
    LearnedModel
        The stored model.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return LearnedModel.from_dict(payload)


# --------------------------------------------------------------------------- fitting

def _fit_logistic(
    standardized: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    l2_penalty: float,
    max_iterations: int,
) -> tuple[np.ndarray, float]:
    """Fit a logistic regression by minimising the penalised negative log-likelihood."""
    n_features = standardized.shape[1]

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        coef = theta[:-1]
        intercept = theta[-1]
        logit = standardized @ coef + intercept
        loss = float(
            np.sum(weights * (np.logaddexp(0.0, logit) - labels * logit))
            + 0.5 * l2_penalty * float(coef @ coef)
        )
        residual = weights * (expit(logit) - labels)
        gradient = np.empty(n_features + 1, dtype=np.float64)
        gradient[:-1] = standardized.T @ residual + l2_penalty * coef
        gradient[-1] = float(residual.sum())
        return loss, gradient

    result = optimize.minimize(
        objective,
        np.zeros(n_features + 1, dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": int(max_iterations)},
    )
    if not result.success:
        raise RuntimeError(f"logistic fit did not converge: {result.message}")
    return result.x[:-1].copy(), float(result.x[-1])


def fit_learned_model(
    features,
    labels,
    *,
    already_normalized: bool = False,
    log_features: Sequence[str] = LOG_FEATURES,
    clip: float = _DEFAULT_CLIP,
    l2_penalty: float = 1.0,
    sample_weight=None,
    threshold: float = 0.0,
    max_iterations: int = 500,
) -> LearnedModel:
    """Fit the standardisation and the logistic coefficients on labelled candidates.

    Fitting takes the same input as inference: the raw feature table of one image, as
    :func:`detect_candidates` measures it and :func:`score_candidates` reads it, or one
    such table per image in a list. Every table is normalised on its own with
    :func:`normalize_per_image`, using the ``log_features`` and the ``clip`` that go into
    the model, so the fit cannot see features prepared differently from the ones inference
    will see. The normalised tables are concatenated in the order given, standardised by
    their own mean and standard deviation, and a logistic regression is fitted by
    minimising the negative log-likelihood plus ``0.5 * l2_penalty * ||coef||^2`` (the
    intercept is not penalised).

    ``labels`` is one flat array over the candidates of all tables, in the order the
    tables are given.

    Tables that you have normalised yourself are accepted through
    ``already_normalized=True``. The settings :func:`normalize_per_image` recorded on them
    must then be the ones the model will carry, and a table whose values lie outside the
    ``clip`` cannot have been normalised with it; either case raises instead of recording
    a normalisation that inference would not reproduce.

    The threshold is not part of the fit: pick it afterwards on labelled candidates with
    :func:`choose_threshold_max_f1` and attach it with :meth:`LearnedModel.with_threshold`.

    Parameters
    ----------
    features : pandas.DataFrame, array_like, or list of them
        Raw features of the labelled candidates, columns :data:`FEATURE_NAMES`: one table
        per image, or a single table for a single image. Give every candidate of an image,
        labelled or not, since the normalisation of an image is estimated from its whole
        candidate population. With ``already_normalized=True`` the tables are taken as
        per-image normalised and used as they are.
    labels : array_like
        Human label per candidate: 1 for an object, 0 for a non-object, one flat array
        over the tables in the order they are given.
    already_normalized : bool
        Read the tables as per-image normalised already instead of normalising them. This
        is the route for a table prepared outside this call; the default is to pass raw
        features and let the fit normalise them.
    log_features : sequence of str
        Features that go through the log transform before the per-image normalisation;
        used for the normalisation here and recorded in the model so inference repeats it.
    clip : float
        Bound used by the per-image normalisation here and recorded in the model.
    l2_penalty : float
        Weight of the L2 penalty on the coefficients, dimensionless. Zero fits an
        unpenalised model.
    sample_weight : array_like or None
        Optional weight per candidate, dimensionless; pass class-balancing weights here
        when objects are rare.
    threshold : float
        Decision threshold stored with the model. The default zero is the score at which
        the fitted logistic probability is one half.
    max_iterations : int
        Maximum number of optimiser iterations.

    Returns
    -------
    LearnedModel
        Model holding the feature order, the log-transformed features, the clip, the
        standardisation mean and scale, the coefficients, the intercept and the
        threshold.

    Raises
    ------
    ValueError
        If a raw table turns out to carry the record of a per-image normalisation (or, on
        the ``already_normalized`` route, a record that differs from the one the model
        would carry, or values outside the clip), if the labels do not hold one value per
        candidate, or if a feature value is not finite.
    """
    tables = _feature_tables(features)
    wanted = _normalization_record(log_features, clip, _DEFAULT_MIN_LOWER)
    frames = []
    for position, table in enumerate(tables):
        where = f"feature table {position}" if len(tables) > 1 else "the feature table"
        frame = _as_feature_frame(table)
        stamp = _normalization_stamp(table)
        if not already_normalized:
            if stamp is not None:
                raise ValueError(
                    f"{where} has already been normalised per image "
                    f"({_describe_normalization(stamp)}); pass the raw features of each "
                    "image, which this call normalises itself, or pass "
                    "already_normalized=True to fit on normalised tables"
                )
            frames.append(
                normalize_per_image(
                    frame,
                    log_features=log_features,
                    clip=clip,
                    min_lower=_DEFAULT_MIN_LOWER,
                )
            )
            continue
        if stamp is None:
            _check_clip_bound(frame, clip, where)
        elif not _same_normalization(stamp, wanted):
            raise ValueError(
                f"{where} was normalised with {_describe_normalization(stamp)}, but the "
                f"model would record {_describe_normalization(wanted)}; normalise with "
                "the settings the model is to carry, or pass the raw features and let "
                "this call normalise them"
            )
        frames.append(frame)

    joined = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    matrix = joined.to_numpy(dtype=np.float64)
    if matrix.size and not np.all(np.isfinite(matrix)):
        raise ValueError("features must be finite")
    target = np.asarray(labels, dtype=np.float64).ravel()
    if target.size != matrix.shape[0]:
        detail = (
            " in {} tables of {} rows".format(
                len(frames), ", ".join(str(len(part)) for part in frames)
            )
            if len(frames) > 1
            else ""
        )
        raise ValueError(
            f"labels must hold one value per candidate: {target.size} labels for "
            f"{matrix.shape[0]} candidates{detail}"
        )
    if matrix.shape[0] < 2:
        raise ValueError("fitting needs at least two labelled candidates")
    if not np.all((target == 0.0) | (target == 1.0)):
        raise ValueError("labels must be 0 or 1")
    if target.min() == target.max():
        raise ValueError("labels must contain both classes")
    if sample_weight is None:
        weights = np.ones(target.size, dtype=np.float64)
    else:
        weights = np.asarray(sample_weight, dtype=np.float64).ravel()
        if weights.size != target.size:
            raise ValueError("sample_weight must hold one value per candidate")
        if np.any(weights < 0):
            raise ValueError("sample_weight must not be negative")
    if float(l2_penalty) < 0:
        raise ValueError("l2_penalty must not be negative")

    mean = matrix.mean(axis=0)
    spread = matrix.std(axis=0)
    # A constant column carries no information; keep its scale at one.
    scale = np.where(spread <= 1e-12, 1.0, spread)
    standardized = (matrix - mean) / scale
    coef, intercept = _fit_logistic(
        standardized, target, weights, float(l2_penalty), int(max_iterations)
    )
    return LearnedModel(
        feature_names=tuple(joined.columns),
        log_features=tuple(log_features),
        clip=float(clip),
        mean=mean,
        scale=scale,
        coef=coef,
        intercept=intercept,
        threshold=float(threshold),
    )


def _threshold_candidates(scores: np.ndarray) -> np.ndarray:
    """Distinct scores, which are the only thresholds that change the accepted set."""
    return np.unique(np.asarray(scores, dtype=np.float64).ravel())


def choose_threshold_max_f1(scores, labels) -> float:
    """Threshold that maximises the F1 score on labelled candidates.

    This is the criterion the method states for the decision threshold. Candidates are
    accepted at or above the threshold; ties in F1 are broken towards the larger
    threshold, the more conservative cut.

    Parameters
    ----------
    scores : array_like
        Logistic score per candidate, dimensionless.
    labels : array_like
        Human label per candidate: 1 for an object, 0 for a non-object.

    Returns
    -------
    float
        Chosen threshold, dimensionless.
    """
    values = np.asarray(scores, dtype=np.float64).ravel()
    target = np.asarray(labels, dtype=np.float64).ravel()
    if values.size != target.size or values.size == 0:
        raise ValueError("scores and labels must have the same non-zero length")
    positives = target == 1.0
    best_threshold = float(values.max())
    best_f1 = -1.0
    for cut in _threshold_candidates(values):
        accepted = values >= cut
        true_positive = int(np.count_nonzero(accepted & positives))
        false_positive = int(np.count_nonzero(accepted & ~positives))
        false_negative = int(np.count_nonzero(~accepted & positives))
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = (2.0 * true_positive / denominator) if denominator else 0.0
        if f1 > best_f1 or (f1 == best_f1 and cut > best_threshold):
            best_f1 = f1
            best_threshold = float(cut)
    return best_threshold
