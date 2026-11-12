"""Contact between readout objects and cell-marker territories, with a randomisation null.

The module answers one question: do the segmented readout objects of a field touch the
territory of a cellular marker more often than objects of the same number, area and
shape would touch it by chance? Everything needed for that answer lives here, while the
steps shared with the rest of the package are imported rather than repeated.

Segmentation
    Cell-marker territories are segmented from the marker channel by background
    subtraction, a censored-noise threshold, a binary closing, an enclosed-hole fill and
    a minimum component area (:func:`segment_marker_territory`,
    :func:`marker_territory_mask`). Readout objects are thresholded on the (optionally
    unmixed) readout residual, their pinholes are filled and components below the
    minimum area are dropped (:func:`segment_readout_objects`). The background models,
    the noise estimator and the mask clean-up come from :mod:`if_qsaa.background` and
    :mod:`if_qsaa.segmentation`; the cross-talk estimation, the removal of marker
    components that are only shadows of a readout object and the per-object spectral
    gates come from :mod:`if_qsaa.unmixing`.

Valid tissue
    Randomisations must stay inside tissue, so dark voids (vessels, holes, tears) are
    detected on the smoothed total signal of all channels and excluded
    (:func:`tissue_void_mask`).

Contact and distance
    A marker territory is in contact with an object when it covers at least a given
    fraction of the ring of a given width immediately outside the object's boundary; the
    object's own footprint is never used, so a marker shadow underneath an object cannot
    create contact (:func:`object_rings`, :func:`contact_ring_coverage`). Independently,
    the signed distance from an object's centroid to the nearest edge of the marker mask
    is negative inside the mask and positive outside (:func:`signed_distance_um`).

Null model
    Object number, area and shape are held constant: every object, every object pixel
    and every ring pixel is moved by the same random rigid transformation, a rotation by
    a multiple of 90 degrees followed by a toroidal translation (an unrotated draw is
    redrawn until it translates farther than a given minimum, so that it is not nearly
    the identity), while the marker mask, the distance map and the tissue mask stay
    fixed (:func:`rigid_transform`, :func:`draw_rigid_transforms`). Contact enrichment
    is the observed contact fraction divided by the mean contact fraction over the
    randomisations, with a two-sided empirical P value from the same randomisations
    (:func:`contact_null_model`, :func:`enrichment`). The distance distribution is tested
    over the whole curve by the studentised envelope statistic
    ``sup|ECDF - mean ECDF| / SD`` over a user-given distance range
    (:func:`envelope_test`).

Whole field and robustness
    :func:`analyse_contact_field` runs the chain above on one field from a parameter
    mapping, and :func:`run_contact_sensitivity` repeats the whole analysis over a
    user-supplied grid of alternative settings and returns one row per setting. The
    analysis unit is the field: a randomisation P value is a statement about that field,
    so several fields are analysed one by one and combined by the caller.

Intensity parameters (grey-level floors, saturation levels, minimum contrasts) and the
background scale have no defaults, because they depend on the detector scaling, the
export and the resolution of the user's own images. Dimensionless rules and physical
lengths that the method states as rules keep their values.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from ._util import (
    area_um2_to_px,
    as_bool_mask,
    as_float,
    check_2d,
    check_same_shape,
    default_rng,
    disk,
    label_mask,
    um_to_px,
)
from .background import censored_noise_threshold, robust_noise, rolling_ball_residual
from .segmentation import clean_mask, fill_small_holes, label_objects
from .stats import empirical_p_two_sided
from .unmixing import (
    estimate_crosstalk,
    estimate_readout_to_nuclear_bleed,
    footprint_beta_correction,
    object_contrast_table,
    remove_marker_shadows,
    run_variants,
    subtract_bleed,
    unmix_two_channel,
    unsaturated_mask,
)

__all__ = [
    # valid tissue
    "tissue_void_mask",
    # segmentation
    "MarkerTerritory",
    "marker_territory_mask",
    "segment_marker_territory",
    "segment_readout_objects",
    "object_centroids",
    # distance
    "signed_distance_um",
    "distance_at_points",
    "distance_classes",
    "DISTANCE_CLASSES",
    "distance_ecdf_grid",
    "ecdf_on_grid",
    # contact
    "PixelSet",
    "object_pixels",
    "object_rings",
    "ring_coverage",
    "pooled_coverage",
    "contact_ring_coverage",
    # null model
    "rigid_transform",
    "draw_rigid_transforms",
    "enrichment",
    "NullModelResult",
    "contact_null_model",
    # whole-curve test
    "EnvelopeTestResult",
    "envelope_test",
    # whole field and robustness
    "ContactField",
    "ContactAnalysis",
    "default_contact_parameters",
    "required_contact_parameters",
    "analyse_contact_field",
    "contact_headline",
    "run_contact_sensitivity",
]

#: Codes returned by :func:`distance_classes`, in order.
DISTANCE_CLASSES = ("contained", "apposed", "separate")


# --------------------------------------------------------------------------- private

def _disk(radius_px: float) -> np.ndarray:
    """Flat disk whose radius is rounded to whole pixels, at least one pixel."""
    return disk(max(1.0, float(np.round(float(radius_px)))))


def _remove_small_components(
    mask: np.ndarray, min_px: int, *, connectivity: int = 4
) -> np.ndarray:
    """Keep the connected components of at least ``min_px`` pixels."""
    work = as_bool_mask(mask)
    if int(min_px) <= 1 or not bool(work.any()):
        return work.copy()
    labels, count = label_mask(work, connectivity=connectivity)
    if count == 0:
        return work.copy()
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    keep = areas >= int(min_px)
    keep[0] = False
    return keep[labels]


def _two_sided_p(observed: float, null_draws: np.ndarray) -> float:
    """Two-sided empirical P value, or NaN when the inputs are not all finite."""
    draws = np.asarray(null_draws, dtype=float).ravel()
    if draws.size == 0 or not np.isfinite(observed) or not np.isfinite(draws).all():
        return float("nan")
    return float(empirical_p_two_sided(float(observed), draws).P)


def _transform_yx(
    rows: np.ndarray,
    cols: np.ndarray,
    k: int,
    shift_row: int,
    shift_col: int,
    size_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate by ``k * 90`` degrees about the field centre, then shift with wrap-around."""
    size = int(size_px)
    y, x = rows, cols
    turn = int(k) % 4
    if turn == 1:
        y, x = size - 1 - x, y
    elif turn == 2:
        y, x = size - 1 - y, size - 1 - x
    elif turn == 3:
        y, x = x, size - 1 - y
    return (y + shift_row) % size, (x + shift_col) % size


def _square_side(image: np.ndarray, name: str) -> int:
    """Side of a square field in pixels; the 90-degree rotations need equal sides."""
    arr = check_2d(image, name)
    if arr.shape[0] != arr.shape[1]:
        raise ValueError(
            f"{name} must be square: the null model rotates the field by multiples of "
            f"90 degrees, got shape {arr.shape}"
        )
    return int(arr.shape[0])


# --------------------------------------------------------------------------- valid tissue

def tissue_void_mask(
    total_signal: np.ndarray,
    pixel_size_um: float,
    *,
    smoothing_um: float = 2.0,
    k: float = 3.0,
    min_void_area_um2: float = 60.0,
    dilate_um: float = 1.0,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Tissue and void masks of a field, from the total signal of all channels.

    A void is a region that is darker than the tissue in every channel at once: a
    vessel lumen, a hole or a tear. The total signal is smoothed, its background
    location and scale are taken as the median and the MAD-based scale, and a pixel is
    a void when the smoothed total lies more than ``k`` scales below that median.
    Void regions smaller than ``min_void_area_um2`` are noise and are dropped; the
    surviving voids are dilated, so that their soft rim is not left in the tissue.

    Parameters
    ----------
    total_signal : ndarray
        2-D sum of the channels of the field, in the grey levels of the input.
    pixel_size_um : float
        Pixel size in micrometres per pixel.
    smoothing_um : float, optional
        Gaussian smoothing applied before the test, in micrometres (one standard
        deviation).
    k : float, optional
        Number of noise scales below the background median at which a pixel counts as
        a void (dimensionless).
    min_void_area_um2 : float, optional
        Void regions of at most this area are dropped, in square micrometres.
    dilate_um : float, optional
        Dilation of the surviving voids, in micrometres.
    valid : ndarray or None, optional
        2-D boolean mask of pixels whose intensities are trustworthy, for example the
        complement of burned-in annotation. Pixels outside it are never called void.
        ``None`` accepts every pixel.

    Returns
    -------
    tissue : ndarray
        2-D boolean mask of valid tissue, the complement of the void mask.
    void : ndarray
        2-D boolean void mask.
    """
    image = as_float(check_2d(total_signal, "total_signal"))
    if valid is None:
        usable = np.ones(image.shape, dtype=bool)
    else:
        usable = as_bool_mask(valid, "valid")
        check_same_shape(image, usable)
    if not bool(usable.any()):
        raise ValueError("valid must contain at least one pixel")
    smoothed = ndi.gaussian_filter(image, um_to_px(smoothing_um, pixel_size_um))
    centre, sigma = robust_noise(smoothed[usable])
    void = (smoothed < centre - float(k) * sigma) & usable
    min_px = int(area_um2_to_px(min_void_area_um2, pixel_size_um)) + 1
    void = _remove_small_components(void, min_px)
    if bool(void.any()) and float(dilate_um) > 0:
        void = ndi.binary_dilation(
            void, structure=_disk(um_to_px(dilate_um, pixel_size_um))
        )
    return ~void, void


# --------------------------------------------------------------------------- segmentation

@dataclass(frozen=True)
class MarkerTerritory:
    """Cell-marker territory of one field and the threshold it was cut at.

    Attributes
    ----------
    mask : ndarray
        2-D boolean mask of the marker territory.
    residual : ndarray
        2-D background residual of the marker channel, in the grey levels of the input.
    threshold : float
        Threshold applied to the residual, in grey levels.
    background_mean : float
        Background mean of the residual, in grey levels.
    noise_sigma : float
        Background-noise scale of the residual, in grey levels.
    clipped_fraction : float
        Fraction of the residual values at or below the clip level (dimensionless).
    area_fraction : float
        Share of the measured area covered by the territory (dimensionless). The
        denominator is the area the noise was measured in: the pixels of the
        ``tissue`` mask when one was given, and every pixel of the field when it was
        ``None``. Compare two fields by this number only when both were measured
        against the same kind of denominator; ``mask.mean()`` is the share of the
        whole field whatever was passed.
    """

    mask: np.ndarray
    residual: np.ndarray
    threshold: float
    background_mean: float
    noise_sigma: float
    clipped_fraction: float
    area_fraction: float


def marker_territory_mask(
    plane: np.ndarray,
    threshold: float,
    pixel_size_um: float,
    *,
    fill_hole_um2: float,
    min_area_um2: float,
    closing_px: float = 2.0,
    closing_um: float | None = None,
) -> np.ndarray:
    """Binary cell-marker territory from a thresholded marker plane.

    The steps are the ones the method applies, in this order: threshold, a binary
    closing by a flat disk so that the sampling gaps inside a stained structure are
    bridged, a fill of the enclosed holes up to ``fill_hole_um2``, and the removal of
    components below ``min_area_um2``.

    Parameters
    ----------
    plane : ndarray
        2-D marker plane the territory is cut from, in grey levels: the background
        residual, or the residual after linear unmixing.
    threshold : float
        Threshold in the grey levels of ``plane``. The method keeps it in the noise
        units of the *uncorrected* residual, so an unmixed plane is cut at the
        threshold measured before unmixing.
    pixel_size_um : float
        Pixel size in micrometres per pixel.
    fill_hole_um2 : float
        Largest enclosed hole filled, in square micrometres. It is a property of the
        marker's morphology: a marker filling cell somata needs a larger value than a
        fine process network, so there is no generic value.
    min_area_um2 : float
        Smallest component kept, in square micrometres. Also a property of the marker's
        morphology, so again no generic value.
    closing_px : float, optional
        Closing radius in pixels, rounded to whole pixels. Two pixels is the smallest
        closing that bridges the sampling of a diffraction-limited edge.
    closing_um : float or None, optional
        Closing radius in micrometres; when given it replaces ``closing_px``. A radius
        that rounds to zero pixels switches the closing off.

    Returns
    -------
    ndarray
        2-D boolean mask of the marker territory.

    Notes
    -----
    Components are 8-connected, holes are enclosed background regions that do not reach
    the frame edge, and the area test is applied to 4-connected components, so that two
    specks touching only at a corner are judged separately.
    """
    image = as_float(check_2d(plane, "plane"))
    mask = image > float(threshold)
    radius_um = (
        float(closing_px) * float(pixel_size_um) if closing_um is None else float(closing_um)
    )
    mask = clean_mask(mask, pixel_size_um, close_um=radius_um)
    mask = fill_small_holes(mask, int(area_um2_to_px(fill_hole_um2, pixel_size_um)))
    min_px = max(1, int(round(area_um2_to_px(min_area_um2, pixel_size_um))))
    return _remove_small_components(mask, min_px)


def segment_marker_territory(
    channel: np.ndarray,
    pixel_size_um: float,
    *,
    background_radius_um: float,
    noise_k: float,
    fill_hole_um2: float,
    min_area_um2: float,
    tissue: np.ndarray | None = None,
    smoothing_sigma_px: float = 1.0,
    closing_px: float = 2.0,
    closing_um: float | None = None,
    clip_level: float = 0.5,
    sigma_floor: float = 0.5,
) -> MarkerTerritory:
    """Segment the cell-marker territory of one field from the marker channel.

    Background subtraction, threshold and mask clean-up in the order of the method: a
    small Gaussian pre-smoothing and a rolling-ball background give the residual, the
    threshold is ``background mean + noise_k * noise sigma`` of that residual inside the
    tissue, and the mask is closed, hole-filled and size-filtered by
    :func:`marker_territory_mask`.

    Parameters
    ----------
    channel : ndarray
        2-D marker channel in the grey levels of the input.
    pixel_size_um : float
        Pixel size in micrometres per pixel.
    background_radius_um : float
        Rolling-ball radius in micrometres. It must be larger than the structures to be
        kept, so it depends on the user's objects and resolution.
    noise_k : float
        Number of background-noise standard deviations above the background at which
        the territory is cut (dimensionless). The marker channel carries a diffuse
        signal, so the multiple is a choice of the user, not a rule of the method.
    fill_hole_um2, min_area_um2 : float
        See :func:`marker_territory_mask`.
    tissue : ndarray or None, optional
        2-D boolean tissue mask the noise is estimated inside, and the denominator of
        the returned ``area_fraction``. ``None`` uses every pixel.
    smoothing_sigma_px : float, optional
        Gaussian pre-smoothing of the channel in pixels (one standard deviation),
        applied before the ball; one pixel removes detector noise without moving edges.
    closing_px, closing_um : float, optional
        See :func:`marker_territory_mask`.
    clip_level : float, optional
        Residual values at or below this level count as clipped, in grey levels.
    sigma_floor : float, optional
        Smallest noise scale used, in grey levels.

    Returns
    -------
    MarkerTerritory
        Mask, residual, threshold and noise statistics.
    """
    residual = rolling_ball_residual(
        channel,
        background_radius_um,
        pixel_size_um,
        prefilter="gaussian",
        gaussian_sigma_px=smoothing_sigma_px,
    )
    threshold, mu, sigma, clipped = censored_noise_threshold(
        residual,
        mask=tissue,
        k=noise_k,
        floor=0.0,
        clip_level=clip_level,
        sigma_floor=sigma_floor,
    )
    mask = marker_territory_mask(
        residual,
        threshold,
        pixel_size_um,
        fill_hole_um2=fill_hole_um2,
        min_area_um2=min_area_um2,
        closing_px=closing_px,
        closing_um=closing_um,
    )
    if tissue is None:
        area_fraction = float(mask.mean())
    else:
        within = as_bool_mask(tissue, "tissue")
        area_fraction = float(mask[within].mean()) if bool(within.any()) else float("nan")
    return MarkerTerritory(
        mask=mask,
        residual=residual,
        threshold=float(threshold),
        background_mean=float(mu),
        noise_sigma=float(sigma),
        clipped_fraction=float(clipped),
        area_fraction=area_fraction,
    )


def segment_readout_objects(
    plane: np.ndarray,
    threshold: float,
    pixel_size_um: float,
    *,
    min_area_um2: float = 0.5,
    pinhole_area_um2: float = 1.0,
    split: bool = False,
    peak_distance_um: float | None = None,
    **split_kwargs: Any,
) -> np.ndarray:
    """Label the readout objects of one field above a threshold.

    The steps are: threshold the plane, fill only the pinholes (enclosed holes of at
    most ``pinhole_area_um2``, so that a product ringing a cell body is not filled into
    a disc whose centroid would then fall inside the cell), drop components below
    ``min_area_um2`` and label what is left with 8-connectivity.

    Parameters
    ----------
    plane : ndarray
        2-D readout plane in grey levels: the background residual, or the residual
        after linear unmixing.
    threshold : float
        Threshold in the grey levels of ``plane``.
    pixel_size_um : float
        Pixel size in micrometres per pixel.
    min_area_um2 : float, optional
        Smallest object area in square micrometres; the minimum candidate area of the
        method.
    pinhole_area_um2 : float, optional
        Largest enclosed hole filled, in square micrometres.
    split : bool, optional
        Split touching objects with :func:`if_qsaa.segmentation.split_touching_objects`
        instead of labelling plain connected components. Off by default: splitting cuts
        genuine objects that grow into clusters.
    peak_distance_um : float or None, optional
        Smallest centre-to-centre distance of two objects that can still be separated,
        in micrometres. Required when ``split`` is true.
    **split_kwargs
        Further keyword arguments of
        :func:`if_qsaa.segmentation.split_touching_objects`.

    Returns
    -------
    ndarray
        ``int32`` label image, 0 in the background and objects numbered from 1.

    Notes
    -----
    The area test is applied to 4-connected components and the labelling is
    8-connected, so two specks touching at a corner are rejected separately but
    survivors that touch at a corner are one object.
    """
    image = as_float(check_2d(plane, "plane"))
    mask = image > float(threshold)
    mask = fill_small_holes(mask, int(area_um2_to_px(pinhole_area_um2, pixel_size_um)))
    min_px = max(1, int(round(area_um2_to_px(min_area_um2, pixel_size_um))))
    mask = _remove_small_components(mask, min_px)
    labels = label_objects(
        mask,
        pixel_size_um,
        split=split,
        peak_distance_um=peak_distance_um,
        **split_kwargs,
    )
    if not split:
        return labels.astype(np.int32, copy=False)
    # splitting can leave fragments below the minimum area; drop them and renumber
    areas = np.bincount(labels.ravel())
    keep = np.flatnonzero(areas >= min_px)
    keep = keep[keep > 0]
    lookup = np.zeros(areas.size, dtype=np.int32)
    lookup[keep] = np.arange(1, keep.size + 1, dtype=np.int32)
    return lookup[labels]


def object_centroids(
    object_labels: np.ndarray,
    *,
    object_ids: Sequence[int] | np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Centroids of labelled objects, in pixel coordinates.

    Parameters
    ----------
    object_labels : ndarray
        2-D label image, 0 in the background.
    object_ids : sequence of int or None, optional
        Labels to measure, in the order of the output. ``None`` uses every label
        present, in ascending order.
    weights : ndarray or None, optional
        2-D non-negative weight image, same shape, in any units. Pass the readout plane
        clipped at zero for the intensity-weighted centroid the method uses; ``None``
        gives the unweighted centroid. An object whose weights sum to zero falls back
        to its unweighted centroid.

    Returns
    -------
    ndarray
        ``(n, 2)`` float array of row and column coordinates in pixels.
    """
    labels = check_2d(object_labels, "object_labels").astype(np.intp, copy=False)
    ids = _object_ids(labels, object_ids)
    if ids.size == 0:
        return np.zeros((0, 2), dtype=float)
    plain = np.asarray(
        ndi.center_of_mass(np.ones(labels.shape, dtype=float), labels, ids), dtype=float
    ).reshape(-1, 2)
    if weights is None:
        return plain
    weight = as_float(check_2d(weights, "weights"))
    check_same_shape(labels, weight)
    if float(weight.min()) < 0:
        raise ValueError("weights must be non-negative")
    totals = np.asarray(ndi.sum_labels(weight, labels, ids), dtype=float).ravel()
    weighted = plain.copy()
    carries_weight = totals > 0
    if bool(carries_weight.any()):
        weighted[carries_weight] = np.asarray(
            ndi.center_of_mass(weight, labels, ids[carries_weight]), dtype=float
        ).reshape(-1, 2)
    return weighted


def _object_ids(
    labels: np.ndarray, object_ids: Sequence[int] | np.ndarray | None
) -> np.ndarray:
    """Validate the requested labels, or list the labels present in ascending order."""
    if object_ids is None:
        present = np.unique(labels)
        return present[present > 0].astype(np.intp, copy=False)
    ids = np.asarray(object_ids, dtype=np.intp).ravel()
    if ids.size and (int(ids.min()) <= 0 or int(ids.max()) > int(labels.max())):
        raise ValueError("object_ids must be labels present in the label image")
    return ids


# --------------------------------------------------------------------------- distance

def signed_distance_um(mask: np.ndarray, pixel_size_um: float) -> np.ndarray:
    """Signed Euclidean distance to the edge of a mask, in micrometres.

    The distance is negative inside the mask and positive outside, and the boundary
    itself sits at zero: with ``d_in`` and ``d_out`` the Euclidean distance transforms
    of the mask and of its complement, the map is ``-(d_in - 0.5)`` inside and
    ``d_out - 0.5`` outside. The half-pixel offset puts the edge halfway between the
    last mask pixel and the first background pixel, so the outermost mask pixel is at
    ``-0.5`` pixels and its neighbour outside at ``+0.5`` pixels.

    Parameters
    ----------
    mask : ndarray
        2-D boolean mask, for example a cell-marker territory.
    pixel_size_um : float
        Pixel size in micrometres per pixel.

    Returns
    -------
    ndarray
        ``float32`` map of signed distances in micrometres.

    Notes
    -----
    The map of an empty mask carries no edge to measure from and is not meaningful;
    callers should treat an empty mask as "no territory" rather than as a far edge.
    """
    work = as_bool_mask(mask)
    outside = ndi.distance_transform_edt(~work)
    inside = ndi.distance_transform_edt(work)
    signed = np.where(work, -(inside - 0.5), outside - 0.5)
    return (signed * float(pixel_size_um)).astype(np.float32, copy=False)


def distance_at_points(
    distance_map: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """Read a distance map at point coordinates, rounded to the nearest pixel.

    Parameters
    ----------
    distance_map : ndarray
        2-D map, for example the output of :func:`signed_distance_um`.
    points : ndarray
        ``(n, 2)`` array of row and column coordinates in pixels; they are rounded to
        the nearest pixel and clipped to the frame.

    Returns
    -------
    ndarray
        ``(n,)`` float array of map values, in the units of the map.
    """
    arr = check_2d(distance_map, "distance_map")
    coords = np.asarray(points, dtype=float).reshape(-1, 2)
    rows = np.clip(np.rint(coords[:, 0]).astype(np.intp), 0, arr.shape[0] - 1)
    cols = np.clip(np.rint(coords[:, 1]).astype(np.intp), 0, arr.shape[1] - 1)
    return np.asarray(arr[rows, cols], dtype=float)


def distance_classes(
    distance_um: np.ndarray, apposed_um: float = 2.0
) -> np.ndarray:
    """Classify signed distances as contained, apposed or separate.

    The codes are the positions in :data:`DISTANCE_CLASSES`: 0 when the distance is at
    most zero, that is the centroid lies inside the marker mask; 1 when it is outside
    but at most ``apposed_um`` from the edge; 2 otherwise.

    Parameters
    ----------
    distance_um : array_like
        Signed distances in micrometres.
    apposed_um : float, optional
        Width of the apposed band outside the mask, in micrometres. It is the same
        width as the contact ring of the method.

    Returns
    -------
    ndarray
        ``int`` array of class codes.
    """
    values = np.asarray(distance_um, dtype=float)
    return np.where(values <= 0, 0, np.where(values <= float(apposed_um), 1, 2)).astype(
        np.intp, copy=False
    )


def distance_ecdf_grid(
    start_um: float = -10.0, stop_um: float = 30.0, step_um: float = 0.25
) -> np.ndarray:
    """Evenly spaced distance grid the cumulative distributions are evaluated on.

    The default grid runs from -10 to 30 micrometres in steps of a quarter micrometre,
    161 points in all. It is wider on purpose, so that one grid serves fields of
    different object densities and the curves of several fields can be averaged point
    by point; the price is that in a small field the far points lie beyond every
    observed distance, where every cumulative distribution sits at 1 and carries no
    information. Such points cost nothing and are left out of the whole-curve test by
    :func:`envelope_test`, which is why its ``n_points`` is normally smaller than the
    grid (see :class:`EnvelopeTestResult`).

    Parameters
    ----------
    start_um, stop_um : float
        First and last grid point in micrometres; both ends are included.
    step_um : float
        Spacing in micrometres.

    Returns
    -------
    ndarray
        1-D float array of distances in micrometres, rounded to three decimals so that
        a grid point can be compared exactly with a window limit.
    """
    if float(step_um) <= 0:
        raise ValueError("step_um must be positive")
    span = float(stop_um) - float(start_um)
    if span < 0:
        raise ValueError("stop_um must not be smaller than start_um")
    count = int(round(span / float(step_um))) + 1
    return np.round(
        np.linspace(float(start_um), float(start_um) + (count - 1) * float(step_um), count),
        3,
    )


def ecdf_on_grid(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Empirical cumulative distribution of values on a grid.

    Parameters
    ----------
    values : array_like
        Sample, in the units of the grid.
    grid : array_like
        Increasing grid points, in the same units.

    Returns
    -------
    ndarray
        Fraction of values at or below each grid point (dimensionless, 0 to 1).
    """
    sample = np.asarray(values, dtype=float).ravel()
    points = np.asarray(grid, dtype=float).ravel()
    if sample.size == 0:
        return np.full(points.size, np.nan)
    return np.searchsorted(np.sort(sample), points, side="right") / sample.size


# --------------------------------------------------------------------------- contact

@dataclass(frozen=True)
class PixelSet:
    """Pixels of several objects, flattened, with the owning object of every pixel.

    Attributes
    ----------
    rows, cols : ndarray
        ``(m,)`` integer pixel coordinates.
    owner : ndarray
        ``(m,)`` index into ``object_ids`` of the object each pixel belongs to.
    counts : ndarray
        ``(n,)`` number of pixels per object.
    object_ids : ndarray
        ``(n,)`` labels of the objects, in the order of ``counts``.
    """

    rows: np.ndarray
    cols: np.ndarray
    owner: np.ndarray
    counts: np.ndarray
    object_ids: np.ndarray

    @property
    def n_objects(self) -> int:
        """Number of objects in the set."""
        return int(self.object_ids.size)


def object_pixels(
    object_labels: np.ndarray,
    *,
    object_ids: Sequence[int] | np.ndarray | None = None,
) -> PixelSet:
    """Pixels of every labelled object, flattened with an owner index.

    Parameters
    ----------
    object_labels : ndarray
        2-D label image, 0 in the background.
    object_ids : sequence of int or None, optional
        Labels to collect, in the order of the output. ``None`` uses every label
        present, in ascending order.

    Returns
    -------
    PixelSet
        Pixel coordinates, owner index and the area of every object in pixels.
    """
    labels = check_2d(object_labels, "object_labels").astype(np.intp, copy=False)
    ids = _object_ids(labels, object_ids)
    lookup = np.full(int(labels.max()) + 1, -1, dtype=np.intp)
    lookup[ids] = np.arange(ids.size, dtype=np.intp)
    rows, cols = np.nonzero(labels > 0)
    owner = lookup[labels[rows, cols]]
    keep = owner >= 0
    rows, cols, owner = rows[keep], cols[keep], owner[keep]
    counts = np.bincount(owner, minlength=ids.size).astype(float)
    return PixelSet(rows=rows, cols=cols, owner=owner, counts=counts, object_ids=ids)


def object_rings(
    object_labels: np.ndarray,
    pixel_size_um: float,
    *,
    ring_um: float = 2.0,
    object_ids: Sequence[int] | np.ndarray | None = None,
) -> PixelSet:
    """Pixels of the ring immediately outside every object's boundary.

    The ring of an object is its dilation by a flat disk of radius ``ring_um`` minus its
    own pixels. Pixels of other objects are part of the ring, because the ring describes
    the neighbourhood of the object, and the ring is truncated where it would leave the
    frame.

    Parameters
    ----------
    object_labels : ndarray
        2-D label image, 0 in the background.
    pixel_size_um : float
        Pixel size in micrometres per pixel.
    ring_um : float, optional
        Ring width in micrometres, that is the dilation radius; the contact rule of the
        method uses the ring 2 micrometres outside the boundary.
    object_ids : sequence of int or None, optional
        Labels to measure, in the order of the output. ``None`` uses every label
        present, in ascending order.

    Returns
    -------
    PixelSet
        Ring pixel coordinates, owner index and the number of ring pixels per object.

    Notes
    -----
    The object's own footprint is excluded, so marker signal underneath an object can
    never be scored as contact.
    """
    labels = check_2d(object_labels, "object_labels").astype(np.intp, copy=False)
    ids = _object_ids(labels, object_ids)
    radius_px = um_to_px(ring_um, pixel_size_um)
    element = _disk(radius_px)
    pad = int(radius_px) + 2
    boxes = ndi.find_objects(labels)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    owner: list[np.ndarray] = []
    for index, label in enumerate(ids):
        box = boxes[int(label) - 1] if int(label) - 1 < len(boxes) else None
        if box is None:
            continue
        row0 = max(0, box[0].start - pad)
        row1 = min(labels.shape[0], box[0].stop + pad)
        col0 = max(0, box[1].start - pad)
        col1 = min(labels.shape[1], box[1].stop + pad)
        crop = labels[row0:row1, col0:col1] == label
        ring = ndi.binary_dilation(crop, structure=element) & ~crop
        ring_rows, ring_cols = np.nonzero(ring)
        rows.append(ring_rows + row0)
        cols.append(ring_cols + col0)
        owner.append(np.full(ring_rows.size, index, dtype=np.intp))
    if rows:
        all_rows = np.concatenate(rows)
        all_cols = np.concatenate(cols)
        all_owner = np.concatenate(owner)
    else:
        all_rows = np.zeros(0, dtype=np.intp)
        all_cols = np.zeros(0, dtype=np.intp)
        all_owner = np.zeros(0, dtype=np.intp)
    counts = np.bincount(all_owner, minlength=ids.size).astype(float)
    return PixelSet(
        rows=all_rows, cols=all_cols, owner=all_owner, counts=counts, object_ids=ids
    )


def ring_coverage(pixels: PixelSet, mask: np.ndarray) -> np.ndarray:
    """Share of each object's pixel set that lies on a mask.

    Parameters
    ----------
    pixels : PixelSet
        Pixel set from :func:`object_rings` (the ring coverage) or from
        :func:`object_pixels` (the share of the object itself on the mask).
    mask : ndarray
        2-D boolean mask, for example a cell-marker territory.

    Returns
    -------
    ndarray
        ``(n,)`` covered fraction per object (dimensionless, 0 to 1). An object without
        pixels gets 0.
    """
    territory = as_bool_mask(mask)
    values = territory[pixels.rows, pixels.cols].astype(float)
    total = np.bincount(pixels.owner, weights=values, minlength=pixels.n_objects)
    return total / np.maximum(pixels.counts, 1.0)


def pooled_coverage(
    pixels: PixelSet, mask: np.ndarray, *, select: np.ndarray | None = None
) -> float:
    """Share of all pixels of a set that lie on a mask, pooled over objects.

    Parameters
    ----------
    pixels : PixelSet
        Pixel set, usually from :func:`object_pixels`.
    mask : ndarray
        2-D boolean mask.
    select : ndarray or None, optional
        ``(n,)`` boolean selection of objects to pool over. ``None`` uses all of them.

    Returns
    -------
    float
        Covered fraction (dimensionless, 0 to 1), NaN when no pixel is selected.
    """
    territory = as_bool_mask(mask)
    values = territory[pixels.rows, pixels.cols]
    if select is not None:
        values = values[np.asarray(select, dtype=bool)[pixels.owner]]
    if values.size == 0:
        return float("nan")
    return float(values.mean())


def contact_ring_coverage(
    object_labels: np.ndarray,
    marker_mask: np.ndarray,
    pixel_size_um: float,
    *,
    ring_um: float = 2.0,
    min_cover: float = 0.05,
    object_ids: Sequence[int] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, PixelSet]:
    """Operational contact of every object with a cell-marker territory.

    An object is in contact when the marker territory covers at least ``min_cover`` of
    the ring of width ``ring_um`` immediately outside the object's boundary. A coverage
    exactly equal to ``min_cover`` counts as contact.

    Parameters
    ----------
    object_labels : ndarray
        2-D label image of the readout objects, 0 in the background.
    marker_mask : ndarray
        2-D boolean mask of the cell-marker territory, same shape.
    pixel_size_um : float
        Pixel size in micrometres per pixel.
    ring_um : float, optional
        Ring width in micrometres; the rule of the method is 2 micrometres.
    min_cover : float, optional
        Smallest covered fraction of the ring that counts as contact (dimensionless);
        the rule of the method is 5 per cent, which keeps single marker specks from
        creating contact.
    object_ids : sequence of int or None, optional
        Labels to measure, in the order of the output.

    Returns
    -------
    coverage : ndarray
        ``(n,)`` covered fraction of each ring (dimensionless).
    contact : ndarray
        ``(n,)`` boolean contact flags.
    rings : PixelSet
        The ring pixels, so that a null model can move them with their objects.
    """
    labels = check_2d(object_labels, "object_labels")
    territory = as_bool_mask(marker_mask, "marker_mask")
    check_same_shape(labels, territory)
    rings = object_rings(
        labels, pixel_size_um, ring_um=ring_um, object_ids=object_ids
    )
    coverage = ring_coverage(rings, territory)
    return coverage, coverage >= float(min_cover), rings


# --------------------------------------------------------------------------- null model

def rigid_transform(
    points: np.ndarray, k: int, shift_row: int, shift_col: int, size_px: int
) -> np.ndarray:
    """Rotate points by ``k * 90`` degrees in a square field, then shift with wrap-around.

    The rotation is the one of a quarter turn of the image about its centre, so the
    pattern keeps its shape and stays on the pixel grid; the translation is toroidal, so
    no object is lost at an edge and the number of objects is conserved.

    Parameters
    ----------
    points : ndarray
        ``(..., 2)`` array of row and column coordinates in pixels; floats are allowed,
        which is how object centroids are moved.
    k : int
        Number of quarter turns (0 to 3; larger values wrap).
    shift_row, shift_col : int
        Translation in pixels, applied after the rotation.
    size_px : int
        Side of the square field in pixels.

    Returns
    -------
    ndarray
        Array of the same shape as ``points`` with the transformed coordinates.
    """
    coords = np.asarray(points)
    rows, cols = _transform_yx(
        coords[..., 0], coords[..., 1], k, shift_row, shift_col, size_px
    )
    return np.stack([rows, cols], axis=-1)


def draw_rigid_transforms(
    size_px: int,
    n_randomisations: int,
    *,
    min_shift_px: float = 0.0,
    rng: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw random rigid transformations of a square field.

    Each transformation is a quarter turn drawn uniformly from the four rotations and a
    translation drawn uniformly over the field, applied with wrap-around. A placement
    that is nearly the identity carries no information about chance contact, so an
    unrotated draw whose toroidal translation is shorter than ``min_shift_px`` is
    redrawn until it is long enough. The toroidal length of a translation is
    ``hypot(min(dy, S - dy), min(dx, S - dx))``, the shortest distance a point travels
    on the torus.

    Parameters
    ----------
    size_px : int
        Side of the square field in pixels.
    n_randomisations : int
        Number of transformations to draw (dimensionless count).
    min_shift_px : float, optional
        Smallest toroidal translation length of an unrotated draw, in pixels.
    rng : int, sequence of int, numpy.random.Generator or None, optional
        Seed or generator; see :func:`if_qsaa._util.default_rng`.

    Returns
    -------
    turns : ndarray
        ``(n,)`` integer number of quarter turns.
    shift_rows, shift_cols : ndarray
        ``(n,)`` integer translations in pixels.

    Notes
    -----
    The rotations are drawn first, then the row shifts, then the column shifts, and the
    near-identity redraws come last, so one seed always gives the same set of
    transformations and alternative settings of an analysis can be compared on common
    random numbers.
    """
    size = int(size_px)
    if size <= 0:
        raise ValueError("size_px must be positive")
    count = int(n_randomisations)
    if count <= 0:
        raise ValueError("n_randomisations must be positive")
    limit = float(min_shift_px)
    longest = float(np.hypot(size // 2, size // 2))
    if limit > longest:
        raise ValueError(
            "min_shift_px exceeds the longest toroidal translation of this field, so no "
            "unrotated draw could ever be accepted"
        )
    generator = default_rng(rng)
    turns = generator.integers(0, 4, count)
    shift_rows = generator.integers(0, size, count)
    shift_cols = generator.integers(0, size, count)
    for index in range(count):
        while turns[index] == 0 and (
            np.hypot(
                min(shift_rows[index], size - shift_rows[index]),
                min(shift_cols[index], size - shift_cols[index]),
            )
            < limit
        ):
            shift_rows[index], shift_cols[index] = generator.integers(0, size, 2)
    return turns, shift_rows, shift_cols


def enrichment(observed: float, null_draws: np.ndarray) -> float:
    """Observed value divided by the mean of a randomisation distribution.

    Parameters
    ----------
    observed : float
        Observed value of the statistic, in its own unit.
    null_draws : array_like
        The same statistic over the randomisations, in the same unit.

    Returns
    -------
    float
        Ratio (dimensionless), NaN when the randomisation mean is not positive, because
        no ratio is defined then.
    """
    draws = np.asarray(null_draws, dtype=float).ravel()
    if draws.size == 0:
        return float("nan")
    mean = float(draws.mean())
    if not np.isfinite(mean) or mean <= 0:
        return float("nan")
    return float(observed) / mean


@dataclass
class NullModelResult:
    """Observed statistics of one field and their randomisation distributions.

    Every ``*_null`` array holds one value per randomisation, in the order the
    transformations were drawn, and every ``*_P`` is a two-sided empirical P value
    ``(r + 1) / (n + 1)`` from that array, without adjustment for multiplicity.

    Attributes
    ----------
    n_objects : int
        Number of objects the statistics are computed on.
    n_randomisations : int
        Number of random rigid transformations.
    contact_fraction : float
        Observed fraction of objects in contact with the marker territory.
    contact_count : int
        Observed number of objects in contact.
    ring_coverage : ndarray
        ``(n_objects,)`` observed covered fraction of each object's ring.
    contact : ndarray
        ``(n_objects,)`` observed contact flags.
    contact_null : ndarray
        ``(n_randomisations,)`` contact fraction of each randomisation, the distribution
        to plot the observed value against.
    contact_null_quantiles : ndarray
        2.5th, 50th and 97.5th percentiles of ``contact_null`` (dimensionless).
    contact_enrichment : float
        Observed contact fraction divided by the mean of ``contact_null``.
    contact_P : float
        Two-sided empirical P value of the contact fraction.
    distance_um : ndarray
        ``(n_objects,)`` signed centroid-to-edge distances in micrometres.
    contained_fraction, apposed_fraction : float
        Observed fractions of objects whose centroid is inside the marker mask, and
        within the apposed band of its edge.
    contained_null, apposed_null : ndarray
        The same fractions over the randomisations.
    containment_enrichment, apposed_enrichment : float
        Ratios of observed to mean randomised fraction (dimensionless).
    containment_P, apposed_P : float
        Two-sided empirical P values.
    class_fractions, class_null_mean : ndarray
        ``(3,)`` fractions of objects per class of :data:`DISTANCE_CLASSES`, observed
        and averaged over the randomisations.
    marker_overlap_fraction : float
        Observed share of all object pixels that lie on the marker territory.
    marker_overlap_null : ndarray
        The same share over the randomisations.
    marker_overlap_enrichment : float
        Ratio of observed to mean randomised share (dimensionless).
    marker_overlap_P : float
        Two-sided empirical P value.
    median_distance_um : float
        Observed median signed distance in micrometres.
    median_distance_null : ndarray
        Median signed distance of each randomisation, in micrometres.
    median_distance_P : float
        Two-sided empirical P value of the median distance.
    ecdf_grid_um : ndarray
        Distance grid of the cumulative distributions, in micrometres.
    ecdf_observed : ndarray
        Observed cumulative distribution on that grid (dimensionless).
    ecdf_null : ndarray
        ``(n_randomisations, grid)`` cumulative distributions of the randomisations,
        stored as ``float32``; empty when they were not kept.
    envelope : EnvelopeTestResult or None
        Studentised whole-curve test on the cumulative distributions.
    accepted_fraction_mean : float
        Mean share of objects whose transformed centroid landed inside tissue
        (dimensionless); objects landing in a void are left out of that randomisation.
    """

    n_objects: int
    n_randomisations: int
    contact_fraction: float
    contact_count: int
    ring_coverage: np.ndarray
    contact: np.ndarray
    contact_null: np.ndarray
    contact_null_quantiles: np.ndarray
    contact_enrichment: float
    contact_P: float
    distance_um: np.ndarray
    contained_fraction: float
    apposed_fraction: float
    contained_null: np.ndarray
    apposed_null: np.ndarray
    containment_enrichment: float
    apposed_enrichment: float
    containment_P: float
    apposed_P: float
    class_fractions: np.ndarray
    class_null_mean: np.ndarray
    marker_overlap_fraction: float
    marker_overlap_null: np.ndarray
    marker_overlap_enrichment: float
    marker_overlap_P: float
    median_distance_um: float
    median_distance_null: np.ndarray
    median_distance_P: float
    ecdf_grid_um: np.ndarray
    ecdf_observed: np.ndarray
    ecdf_null: np.ndarray
    envelope: "EnvelopeTestResult | None"
    accepted_fraction_mean: float


def contact_null_model(
    object_labels: np.ndarray,
    centroids: np.ndarray,
    marker_mask: np.ndarray,
    distance_map: np.ndarray,
    tissue: np.ndarray | None,
    pixel_size_um: float,
    *,
    object_ids: Sequence[int] | np.ndarray | None = None,
    n_randomisations: int = 10000,
    min_shift_um: float = 20.0,
    contact_ring_um: float = 2.0,
    contact_min_cover: float = 0.05,
    apposed_um: float = 2.0,
    ecdf_grid_um: np.ndarray | None = None,
    envelope_window_um: tuple[float, float] | None = (-5.0, 30.0),
    keep_null_ecdf: bool = True,
    seed: Any = None,
) -> NullModelResult:
    """Contact and distance statistics of one field against a rigid-transformation null.

    The null holds object number, area and shape constant: for every randomisation one
    rigid transformation (a quarter turn plus a toroidal translation longer than
    ``min_shift_um``) is applied to every object centroid, every object pixel and every
    ring pixel, while the marker territory, the distance map and the tissue mask stay
    where they are. An object whose transformed centroid falls outside the tissue is
    left out of that randomisation; if that would empty a randomisation, all objects are
    kept instead.

    Parameters
    ----------
    object_labels : ndarray
        2-D label image of the readout objects, 0 in the background. The field must be
        square, because the pattern is rotated by multiples of 90 degrees.
    centroids : ndarray
        ``(n, 2)`` centroids in pixel coordinates, in the order of ``object_ids``.
    marker_mask : ndarray
        2-D boolean mask of the cell-marker territory, same shape as the labels.
    distance_map : ndarray
        2-D signed distance to the edge of ``marker_mask`` in micrometres, from
        :func:`signed_distance_um`.
    tissue : ndarray or None
        2-D boolean mask of valid tissue; randomisations stay inside it. ``None``
        accepts every pixel, as ``tissue=None`` does in
        :func:`segment_marker_territory`, so a field without voids needs no
        hand-built mask of ones.
    pixel_size_um : float
        Pixel size in micrometres per pixel.
    object_ids : sequence of int or None, optional
        Labels of the objects to analyse, in the order of ``centroids``. ``None`` uses
        every label present, in ascending order.
    n_randomisations : int, optional
        Number of random rigid transformations; the method draws 10,000. The smallest
        P value any of the statistics can reach is ``1 / (n + 1)``, so a reduced count
        caps the P values a run can report.
    min_shift_um : float, optional
        Smallest toroidal translation of an unrotated draw, in micrometres.
    contact_ring_um, contact_min_cover : float, optional
        Contact rule, see :func:`contact_ring_coverage`.
    apposed_um : float, optional
        Width of the apposed band in micrometres, see :func:`distance_classes`.
    ecdf_grid_um : ndarray or None, optional
        Distance grid in micrometres the cumulative distributions are evaluated on.
        ``None`` uses :func:`distance_ecdf_grid`, whose default spans -10 to 30
        micrometres in steps of 0.25, that is 161 points. The grid is deliberately
        wider than the distances of a small field, which costs nothing but means that
        the far points carry no spread; see :class:`EnvelopeTestResult`.
    envelope_window_um : tuple of float or None, optional
        Distance range in micrometres over which the whole-curve statistic is taken.
        ``None`` skips that test. The default window is narrower than the default
        grid, so the test uses at most 141 of its 161 points.
    keep_null_ecdf : bool, optional
        Keep the randomised cumulative distributions. They are needed for the
        whole-curve test and for drawing the envelope; switching them off saves memory
        in a large robustness grid.
    seed : int, sequence of int, numpy.random.Generator or None, optional
        Seed or generator of the randomisation. The same seed gives the same
        transformations, so alternative settings can be compared on common random
        numbers; pass a different seed (or a sequence such as ``(seed, field_index)``)
        for each field.

    Returns
    -------
    NullModelResult
        Observed statistics, randomisation distributions, enrichments and P values.

    Notes
    -----
    Ring pixels are collected once on the observed pattern and then moved with their
    object, so every randomisation measures the same ring shape; a ring truncated at the
    frame edge keeps its truncated pixel count as the denominator.
    """
    size = _square_side(object_labels, "object_labels")
    labels = check_2d(object_labels, "object_labels").astype(np.intp, copy=False)
    territory = as_bool_mask(marker_mask, "marker_mask")
    distance = as_float(check_2d(distance_map, "distance_map"))
    valid_tissue = (
        np.ones(labels.shape, dtype=bool)
        if tissue is None
        else as_bool_mask(tissue, "tissue")
    )
    check_same_shape(labels, territory, distance, valid_tissue)
    centre = np.asarray(centroids, dtype=float).reshape(-1, 2)
    pixels = object_pixels(labels, object_ids=object_ids)
    n_objects = pixels.n_objects
    if n_objects == 0:
        raise ValueError("at least one object is required")
    if centre.shape[0] != n_objects:
        raise ValueError("centroids must have one row per object")
    grid = distance_ecdf_grid() if ecdf_grid_um is None else np.asarray(
        ecdf_grid_um, dtype=float
    ).ravel()
    rings = object_rings(
        labels, pixel_size_um, ring_um=contact_ring_um, object_ids=pixels.object_ids
    )

    # observed statistics
    coverage = ring_coverage(rings, territory)
    contact = coverage >= float(contact_min_cover)
    contact_fraction = float(contact.mean())
    distance_obs = distance_at_points(distance, centre)
    classes_obs = distance_classes(distance_obs, apposed_um)
    contained_fraction = float(np.mean(distance_obs <= 0))
    apposed_fraction = float(np.mean(distance_obs <= float(apposed_um)))
    overlap_fraction = pooled_coverage(pixels, territory)
    median_distance = float(np.median(distance_obs))
    ecdf_obs = ecdf_on_grid(distance_obs, grid)

    # randomisations
    turns, shift_rows, shift_cols = draw_rigid_transforms(
        size,
        n_randomisations,
        min_shift_px=um_to_px(min_shift_um, pixel_size_um),
        rng=seed,
    )
    count = int(turns.size)
    contact_null = np.empty(count, dtype=float)
    contained_null = np.empty(count, dtype=float)
    apposed_null = np.empty(count, dtype=float)
    overlap_null = np.empty(count, dtype=float)
    median_null = np.empty(count, dtype=float)
    class_null = np.empty((count, 3), dtype=float)
    accepted_counts = np.empty(count, dtype=float)
    ecdf_null = (
        np.empty((count, grid.size), dtype=np.float32)
        if keep_null_ecdf
        else np.zeros((0, grid.size), dtype=np.float32)
    )
    for index in range(count):
        turn = int(turns[index])
        shift_row = int(shift_rows[index])
        shift_col = int(shift_cols[index])
        moved_rows, moved_cols = _transform_yx(
            centre[:, 0], centre[:, 1], turn, shift_row, shift_col, size
        )
        rows = np.clip(np.rint(moved_rows).astype(np.intp), 0, size - 1)
        cols = np.clip(np.rint(moved_cols).astype(np.intp), 0, size - 1)
        accepted = valid_tissue[rows, cols]
        if not bool(accepted.any()):
            accepted = np.ones(n_objects, dtype=bool)
        accepted_counts[index] = float(accepted.sum())
        drawn = distance[rows[accepted], cols[accepted]]
        contained_null[index] = float(np.mean(drawn <= 0))
        apposed_null[index] = float(np.mean(drawn <= float(apposed_um)))
        median_null[index] = float(np.median(drawn))
        class_null[index] = (
            np.bincount(distance_classes(drawn, apposed_um), minlength=3) / drawn.size
        )
        pixel_rows, pixel_cols = _transform_yx(
            pixels.rows, pixels.cols, turn, shift_row, shift_col, size
        )
        on_object = accepted[pixels.owner]
        overlap_null[index] = float(
            territory[pixel_rows[on_object], pixel_cols[on_object]].mean()
        )
        ring_rows, ring_cols = _transform_yx(
            rings.rows, rings.cols, turn, shift_row, shift_col, size
        )
        covered = np.bincount(
            rings.owner,
            weights=territory[ring_rows, ring_cols].astype(float),
            minlength=n_objects,
        ) / np.maximum(rings.counts, 1.0)
        contact_null[index] = float(
            np.mean(covered[accepted] >= float(contact_min_cover))
        )
        if keep_null_ecdf:
            ecdf_null[index] = ecdf_on_grid(drawn, grid)

    envelope = None
    if keep_null_ecdf and envelope_window_um is not None:
        envelope = envelope_test(
            ecdf_obs, ecdf_null, grid, window_um=envelope_window_um
        )
    return NullModelResult(
        n_objects=n_objects,
        n_randomisations=count,
        contact_fraction=contact_fraction,
        contact_count=int(contact.sum()),
        ring_coverage=coverage,
        contact=contact,
        contact_null=contact_null,
        contact_null_quantiles=np.quantile(contact_null, [0.025, 0.5, 0.975]),
        contact_enrichment=enrichment(contact_fraction, contact_null),
        contact_P=_two_sided_p(contact_fraction, contact_null),
        distance_um=distance_obs,
        contained_fraction=contained_fraction,
        apposed_fraction=apposed_fraction,
        contained_null=contained_null,
        apposed_null=apposed_null,
        containment_enrichment=enrichment(contained_fraction, contained_null),
        apposed_enrichment=enrichment(apposed_fraction, apposed_null),
        containment_P=_two_sided_p(contained_fraction, contained_null),
        apposed_P=_two_sided_p(apposed_fraction, apposed_null),
        class_fractions=np.bincount(classes_obs, minlength=3) / n_objects,
        class_null_mean=class_null.mean(axis=0),
        marker_overlap_fraction=overlap_fraction,
        marker_overlap_null=overlap_null,
        marker_overlap_enrichment=enrichment(overlap_fraction, overlap_null),
        marker_overlap_P=_two_sided_p(overlap_fraction, overlap_null),
        median_distance_um=median_distance,
        median_distance_null=median_null,
        median_distance_P=_two_sided_p(median_distance, median_null),
        ecdf_grid_um=grid,
        ecdf_observed=ecdf_obs,
        ecdf_null=ecdf_null,
        envelope=envelope,
        accepted_fraction_mean=float(accepted_counts.mean() / n_objects),
    )


# --------------------------------------------------------------------------- whole curve

@dataclass(frozen=True)
class EnvelopeTestResult:
    """Studentised whole-curve test of an observed distribution against randomisations.

    Attributes
    ----------
    statistic : float
        Largest studentised departure of the observed cumulative distribution from the
        randomisation mean, ``sup|ECDF - mean ECDF| / SD`` (dimensionless).
    P : float
        ``(1 + number of randomisations whose own statistic is at least as large) /
        (n + 1)``.
    n : int
        Number of randomisations.
    n_points : int
        Grid points inside the window that carried a positive spread and therefore
        entered the statistic. It is normally smaller than the grid the cumulative
        distributions were evaluated on, for two reasons that are both correct
        behaviour: the window is usually narrower than the grid (the defaults of
        :func:`distance_ecdf_grid` and of ``window_um`` give 141 of 161 points), and
        a grid point where every randomisation gives the same value has no spread to
        studentise against and is dropped. Read it as the size of the test, not as a
        count of failures.
    window_um : tuple of float
        Distance range the statistic was taken over, in micrometres.
    """

    statistic: float
    P: float
    n: int
    n_points: int
    window_um: tuple


def envelope_test(
    ecdf_observed: np.ndarray,
    ecdf_null: np.ndarray,
    grid_um: np.ndarray,
    *,
    window_um: tuple[float, float] = (-5.0, 30.0),
    dtype: Any = np.float32,
) -> EnvelopeTestResult:
    """Test a cumulative distance distribution over the whole curve, not at one summary.

    The statistic is the largest studentised departure of the observed cumulative
    distribution from the mean of the randomised ones,
    ``sup|ECDF - mean ECDF| / SD``, taken over the grid points inside ``window_um``;
    it is the simultaneous form of the pointwise band the randomisations describe. Each
    randomisation is scored against the same mean and spread, which include its own
    curve, and the P value is ``(1 + number of randomisations whose statistic is at
    least as large as the observed one) / (n + 1)``.

    Parameters
    ----------
    ecdf_observed : ndarray
        ``(g,)`` observed cumulative distribution on the grid (dimensionless).
    ecdf_null : ndarray
        ``(n, g)`` cumulative distributions of the randomisations, same grid.
    grid_um : ndarray
        ``(g,)`` distance grid in micrometres.
    window_um : tuple of float, optional
        Lower and upper distance limit in micrometres, both included. It may be
        narrower than the grid, and the default is: with the default grid of
        :func:`distance_ecdf_grid` it selects 141 of the 161 grid points.
    dtype : data-type, optional
        Floating-point type the curves are compared in. The default matches the type the
        randomised curves are usually stored in, so that a stored and a recomputed test
        agree to the last digit.

    Returns
    -------
    EnvelopeTestResult
        Statistic, P value and the size of the test.

    Notes
    -----
    Grid points where the randomisations have no spread carry no information and are
    left out. When no point inside the window has spread, the statistic is NaN and the
    P value is 1.
    """
    grid = np.asarray(grid_um, dtype=float).ravel()
    observed = np.asarray(ecdf_observed, dtype=float).ravel()
    null = np.atleast_2d(np.asarray(ecdf_null, dtype=float))
    if observed.size != grid.size:
        raise ValueError("ecdf_observed and grid_um must have the same length")
    if null.size == 0 or null.shape[1] != grid.size:
        raise ValueError("ecdf_null must have one column per grid point")
    low, high = float(window_um[0]), float(window_um[1])
    inside = (grid >= low) & (grid <= high)
    if not bool(inside.any()):
        raise ValueError("no grid point lies inside window_um")
    obs = observed[inside].astype(dtype, copy=False)
    draws = null[:, inside].astype(dtype, copy=False)
    mean = draws.mean(axis=0)
    spread = draws.std(axis=0)
    usable = spread > 0
    n = int(draws.shape[0])
    if not bool(usable.any()):
        return EnvelopeTestResult(
            statistic=float("nan"), P=1.0, n=n, n_points=0, window_um=(low, high)
        )
    statistic = float(
        np.max(np.abs(obs[usable] - mean[usable]) / spread[usable])
    )
    per_draw = np.max(
        np.abs(draws[:, usable] - mean[usable]) / spread[usable], axis=1
    )
    return EnvelopeTestResult(
        statistic=statistic,
        P=float((1 + int(np.sum(per_draw >= statistic))) / (n + 1)),
        n=n,
        n_points=int(usable.sum()),
        window_um=(low, high),
    )


# --------------------------------------------------------------------------- whole field

@dataclass(frozen=True)
class ContactField:
    """The images of one field, as :func:`analyse_contact_field` takes them.

    Attributes
    ----------
    readout : ndarray
        2-D readout channel (the objects to analyse) in the grey levels of the input.
        The field must be square, because the null model rotates it by multiples of
        90 degrees.
    marker : ndarray
        2-D cell-marker channel, same shape and units.
    pixel_size_um : float
        Pixel size in micrometres per pixel, from the acquisition metadata.
    nuclear : ndarray or None
        2-D nuclear channel, same shape and units. ``None`` leaves the nuclear terms of
        the cross-talk handling out.
    nuclei : ndarray or None
        2-D boolean nucleus mask, segmented from the nuclear channel after the readout
        bleed-through has been removed. It is the source structure of the
        nuclear-into-readout coefficient; without it that coefficient stays zero.
    valid : ndarray or None
        2-D boolean mask of pixels whose intensities are trustworthy, for example the
        complement of burned-in annotation. ``None`` accepts every pixel.
    total : ndarray or None
        2-D total signal used to find tissue voids. ``None`` sums the channels given.
    """

    readout: np.ndarray
    marker: np.ndarray
    pixel_size_um: float
    nuclear: np.ndarray | None = None
    nuclei: np.ndarray | None = None
    valid: np.ndarray | None = None
    total: np.ndarray | None = None


@dataclass
class ContactAnalysis:
    """Everything one field's contact analysis produced.

    Attributes
    ----------
    objects : pandas.DataFrame
        One row per candidate object: ``label``, the centroid in pixels, areas in
        pixels and square micrometres, the ring contrasts and spectral-gate flags of
        :func:`if_qsaa.unmixing.object_contrast_table`, ``distance_um``,
        ``distance_class``, ``marker_overlap_fraction``, ``ring_marker_fraction``,
        ``contact``, ``in_tissue`` and ``analysed``.
    object_labels : ndarray
        2-D label image of the candidate objects.
    marker_mask : ndarray
        2-D boolean cell-marker territory after shadow removal.
    tissue, void : ndarray
        2-D boolean masks of valid tissue and of the tissue voids.
    signed_distance_um : ndarray
        2-D signed distance to the marker edge in micrometres.
    readout_unmixed, marker_unmixed : ndarray
        2-D readout and marker planes after the cross-talk handling, in grey levels.
    thresholds : dict
        Thresholds and noise statistics of both channels, in grey levels.
    crosstalk : dict
        Cross-talk coefficients and their diagnostics (dimensionless).
    n_objects_total, n_analysed, n_gate_rejected : int
        Candidate objects, objects entering the statistics, and objects the spectral
        gates rejected.
    n_marker_shadows_removed : int
        Marker components removed as shadows of a readout object.
    marker_area_fraction, object_area_fraction, void_fraction : float
        Shares of the tissue covered by marker territory and by objects, and the share
        of the field called void (dimensionless).
    null : NullModelResult or None
        The randomisation result, or ``None`` when no object was left to analyse.
    parameters : dict
        The parameters the analysis ran with, defaults included.
    """

    objects: pd.DataFrame
    object_labels: np.ndarray
    marker_mask: np.ndarray
    tissue: np.ndarray
    void: np.ndarray
    signed_distance_um: np.ndarray
    readout_unmixed: np.ndarray
    marker_unmixed: np.ndarray
    thresholds: dict
    crosstalk: dict
    n_objects_total: int
    n_analysed: int
    n_gate_rejected: int
    n_marker_shadows_removed: int
    marker_area_fraction: float
    object_area_fraction: float
    void_fraction: float
    null: NullModelResult | None
    parameters: dict = _field(default_factory=dict)


_DEFAULT_PARAMETERS: dict = {
    # noise estimation
    "clip_level": 0.5,
    "sigma_floor": 0.5,
    "readout_k": 4.0,
    # readout objects
    "min_object_area_um2": 0.5,
    "pinhole_area_um2": 1.0,
    "split_objects": False,
    "split_peak_distance_um": None,
    # marker territory
    "marker_smoothing_sigma_px": 1.0,
    "marker_closing_px": 2.0,
    "marker_closing_um": None,
    "shadow_cover": 0.80,
    "footprint_dilate_um": 1.0,
    # cross-talk handling
    "unmix": True,
    "unmix_nuclear": True,
    "bleed_quantile": 0.10,
    "alpha_quantile": 0.50,
    "beta_quantile": 0.10,
    "crosstalk_seed": 0,
    "footprint_beta_quantile": None,
    "footprint_min_level": None,
    # spectral gates
    "use_gate": True,
    "use_marker_gate": True,
    "gate_ratio": 1.0,
    "marker_gate_ratio": 1.0,
    "gate_ring_um": (1.0, 3.0),
    # valid tissue
    "use_void": True,
    "void_k": 3.0,
    "void_smoothing_um": 2.0,
    "min_void_area_um2": 60.0,
    "void_dilate_um": 1.0,
    # contact and distance
    "contact_ring_um": 2.0,
    "contact_min_cover": 0.05,
    "apposed_um": 2.0,
    "ecdf_start_um": -10.0,
    "ecdf_stop_um": 30.0,
    "ecdf_step_um": 0.25,
    "envelope_window_um": (-5.0, 30.0),
    # null model
    "n_randomisations": 10000,
    "min_shift_um": 20.0,
    "keep_null_ecdf": True,
    "seed": None,
}

_REQUIRED_PARAMETERS: tuple = (
    "background_radius_um",
    "readout_floor_grey",
    "marker_k",
    "marker_fill_hole_um2",
    "marker_min_area_um2",
    "saturation_level",
    "bright_min_level",
    "min_source_range",
    "gate_min_contrast",
)


def default_contact_parameters() -> dict:
    """Generic parameters of the contact analysis, as a fresh dictionary.

    Only rules of the method are defaults. Lengths are in micrometres, areas in square
    micrometres, intensities in the grey levels of the input, and everything else is
    dimensionless.

    Returns
    -------
    dict
        ``clip_level``, ``sigma_floor``
            Clip level and smallest noise scale of the censored noise estimator, in
            grey levels (half a grey level is the quantisation limit of an
            integer-valued image).
        ``readout_k``
            Noise multiple of the readout threshold: the threshold rule of the method is
            the background mean plus four noise standard deviations, raised to an
            absolute floor.
        ``min_object_area_um2``, ``pinhole_area_um2``
            Minimum candidate-object area of the method, and the largest enclosed hole
            filled inside an object.
        ``split_objects``, ``split_peak_distance_um``
            Splitting of touching objects, off in the method.
        ``marker_smoothing_sigma_px``, ``marker_closing_px``, ``marker_closing_um``
            Pre-smoothing and closing of the marker channel; one and two pixels are the
            smallest widths that act on the sampling rather than on the structures.
        ``shadow_cover``, ``footprint_dilate_um``
            A marker component covered by at least 80 per cent by the readout footprint
            dilated by 1 micrometre is a shadow of the readout signal and is removed.
        ``unmix``, ``unmix_nuclear``
            Per-field switches of the cross-talk handling: a field without the
            cross-emitting dye is not unmixed.
        ``bleed_quantile``, ``alpha_quantile``, ``beta_quantile``, ``crosstalk_seed``
            Estimator rules of the cross-talk coefficients and the seed of their pixel
            subsampling.
        ``footprint_beta_quantile``, ``footprint_min_level``
            Optional upper-bound re-correction of the marker channel inside readout
            footprints, off in the method.
        ``use_gate``, ``use_marker_gate``, ``gate_ratio``, ``marker_gate_ratio``,
        ``gate_ring_um``
            Spectral gates: an object is excluded when its readout contrast against the
            surrounding 1-to-3-micrometre ring is lower than its contrast in the nuclear
            or the marker channel, which is a ratio of one.
        ``use_void``, ``void_k``, ``void_smoothing_um``, ``min_void_area_um2``,
        ``void_dilate_um``
            Tissue-void detection, see :func:`tissue_void_mask`.
        ``contact_ring_um``, ``contact_min_cover``
            Contact rule of the method: marker territory covering at least 5 per cent of
            the 2-micrometre ring outside the object.
        ``apposed_um``
            Width of the apposed band, the same 2 micrometres.
        ``ecdf_start_um``, ``ecdf_stop_um``, ``ecdf_step_um``, ``envelope_window_um``
            Grid of the cumulative distance distributions, and the range of the
            whole-curve test, which the method takes from -5 to 30 micrometres.
        ``n_randomisations``, ``min_shift_um``, ``keep_null_ecdf``, ``seed``
            Null model: 10,000 random rigid transformations with toroidal translations
            longer than 20 micrometres.

    See Also
    --------
    required_contact_parameters : the parameters that have no generic value.
    """
    return dict(_DEFAULT_PARAMETERS)


def required_contact_parameters() -> tuple:
    """Parameters the caller must supply, because no value of them is generic.

    Returns
    -------
    tuple of str
        ``background_radius_um``
            Rolling-ball radius in micrometres; it must exceed the objects to be kept,
            so it follows the user's resolution and object size.
        ``readout_floor_grey``
            Absolute lower bound of the readout threshold in grey levels, the second
            half of the threshold rule of the method. It is a property of the detector
            scaling and the export of the user's images; pass 0 for no floor.
        ``marker_k``
            Noise multiple of the marker-territory threshold (dimensionless).
        ``marker_fill_hole_um2``, ``marker_min_area_um2``
            Largest enclosed hole filled and smallest component kept in the marker
            territory, in square micrometres; both follow the morphology of the marker.
        ``saturation_level``
            Grey level at or above which a pixel is saturated and carries no slope
            information; it depends on the bit depth and the export.
        ``bright_min_level``
            Absolute lower bound in grey levels of the bright-readout selections the
            bleed-through and marker-into-readout coefficients are estimated on.
        ``min_source_range``
            Smallest span of source intensity inside a structure, in grey levels, for a
            within-structure slope to be fitted.
        ``gate_min_contrast``
            Smallest usable denominator contrast of the spectral gates, in grey levels.
    """
    return tuple(_REQUIRED_PARAMETERS)


def _merge_parameters(params: Mapping[str, Any] | None) -> dict:
    """Defaults updated with the caller's parameters, with the required ones checked."""
    merged = dict(_DEFAULT_PARAMETERS)
    if params:
        unknown = set(params) - set(_DEFAULT_PARAMETERS) - set(_REQUIRED_PARAMETERS)
        if unknown:
            raise KeyError(f"unknown parameters: {sorted(unknown)}")
        merged.update(dict(params))
    missing = [name for name in _REQUIRED_PARAMETERS if merged.get(name) is None]
    if missing:
        raise KeyError(
            "these parameters have no generic value and must be supplied: "
            + ", ".join(missing)
        )
    return merged


def analyse_contact_field(
    field: ContactField, params: Mapping[str, Any] | None = None
) -> ContactAnalysis:
    """Run the whole contact analysis on one field.

    The chain is, in order: tissue voids, background residuals of every channel,
    censored-noise thresholds of the readout and marker channels, removal of the readout
    bleed-through from the nuclear channel, per-field cross-talk estimation and 2x2
    unmixing of the readout-marker pair, readout-object segmentation, optional
    upper-bound re-correction of the marker channel inside object footprints, marker
    territory with shadow removal, signed distance to the marker edge, per-object
    spectral gates, and the randomisation null of :func:`contact_null_model` on the
    objects that lie in tissue and pass the gates.

    Parameters
    ----------
    field : ContactField
        The images of one field and its pixel size.
    params : mapping or None, optional
        Parameters of the analysis; see :func:`default_contact_parameters` for the
        generic ones and :func:`required_contact_parameters` for those that must be
        supplied. Unknown keys are rejected, so a typo cannot pass silently.

    Returns
    -------
    ContactAnalysis
        Object table, masks, coefficients and the randomisation result.
    """
    p = _merge_parameters(params)
    pixel_size_um = float(field.pixel_size_um)
    readout = as_float(check_2d(field.readout, "readout"))
    marker = as_float(check_2d(field.marker, "marker"))
    _square_side(readout, "readout")
    check_same_shape(readout, marker)
    has_nuclear = field.nuclear is not None
    nuclear = (
        as_float(check_2d(field.nuclear, "nuclear"))
        if has_nuclear
        else np.zeros(readout.shape, dtype=float)
    )
    check_same_shape(readout, nuclear)
    valid = (
        np.ones(readout.shape, dtype=bool)
        if field.valid is None
        else as_bool_mask(field.valid, "valid")
    )
    check_same_shape(readout, valid)

    # valid tissue
    if p["use_void"]:
        total = (
            readout + marker + nuclear if field.total is None else as_float(
                check_2d(field.total, "total")
            )
        )
        tissue, void = tissue_void_mask(
            total,
            pixel_size_um,
            smoothing_um=p["void_smoothing_um"],
            k=p["void_k"],
            min_void_area_um2=p["min_void_area_um2"],
            dilate_um=p["void_dilate_um"],
            valid=valid,
        )
    else:
        tissue = np.ones(readout.shape, dtype=bool)
        void = np.zeros(readout.shape, dtype=bool)

    # background residuals: a small median filter on the punctate readout channel, a
    # small Gaussian on the diffuse marker channel, the nuclear channel unfiltered
    readout_res = rolling_ball_residual(
        readout, p["background_radius_um"], pixel_size_um, prefilter="median"
    )
    marker_res = rolling_ball_residual(
        marker,
        p["background_radius_um"],
        pixel_size_um,
        prefilter="gaussian",
        gaussian_sigma_px=p["marker_smoothing_sigma_px"],
    )
    nuclear_res = rolling_ball_residual(
        nuclear, p["background_radius_um"], pixel_size_um
    )

    # thresholds, in the noise units of the uncorrected residuals
    readout_thr, readout_mu, readout_sigma, readout_clipped = censored_noise_threshold(
        readout_res,
        mask=tissue,
        k=p["readout_k"],
        floor=p["readout_floor_grey"],
        clip_level=p["clip_level"],
        sigma_floor=p["sigma_floor"],
    )
    marker_thr, marker_mu, marker_sigma, marker_clipped = censored_noise_threshold(
        marker_res,
        mask=tissue,
        k=p["marker_k"],
        floor=0.0,
        clip_level=p["clip_level"],
        sigma_floor=p["sigma_floor"],
    )

    # readout bleed-through into the nuclear channel: removed in every field
    usable_nuclear = unsaturated_mask(
        [readout, nuclear], p["saturation_level"], within=tissue
    )
    if has_nuclear:
        bleed = estimate_readout_to_nuclear_bleed(
            nuclear_res,
            readout_res,
            usable=usable_nuclear,
            readout_thr=readout_thr,
            readout_sigma=readout_sigma,
            min_level=p["bright_min_level"],
            quantile=p["bleed_quantile"],
        )
    else:
        bleed = dict(delta=0.0, n_pixels=0, selection_level=float("nan"))
    nuclear_corr = subtract_bleed(nuclear_res, readout_res, bleed["delta"])

    # per-field cross-talk coefficients and the 2x2 unmixing
    usable = unsaturated_mask(
        [readout, marker, nuclear], p["saturation_level"], within=tissue
    )
    nuclei = (
        np.zeros(readout.shape, dtype=bool)
        if field.nuclei is None
        else as_bool_mask(field.nuclei, "nuclei")
    )
    check_same_shape(readout, nuclei)
    crosstalk = estimate_crosstalk(
        readout_res,
        marker_res,
        nuclear_corr,
        tissue=tissue,
        usable=usable,
        nuclei=nuclei,
        marker_thr=marker_thr,
        beta_min_level=p["bright_min_level"],
        min_source_range=p["min_source_range"],
        valid=valid,
        alpha_quantile=p["alpha_quantile"],
        beta_quantile=p["beta_quantile"],
        estimate_nuclear=bool(p["unmix_nuclear"] and field.nuclei is not None),
        enabled=bool(p["unmix"]),
        seed=int(p["crosstalk_seed"]),
    )
    crosstalk = dict(crosstalk)
    crosstalk["delta"] = float(bleed["delta"])
    crosstalk["n_delta_pixels"] = int(bleed["n_pixels"])
    readout_unmixed, marker_unmixed = unmix_two_channel(
        readout_res,
        marker_res,
        nuclear_corr,
        crosstalk["alpha"],
        crosstalk["beta"],
        crosstalk["gamma"],
    )

    # readout objects
    object_labels = segment_readout_objects(
        readout_unmixed,
        readout_thr,
        pixel_size_um,
        min_area_um2=p["min_object_area_um2"],
        pinhole_area_um2=p["pinhole_area_um2"],
        split=bool(p["split_objects"]),
        peak_distance_um=p["split_peak_distance_um"],
    )

    # optional upper-bound marker re-correction inside the object footprints
    readout_corr = subtract_bleed(readout_res, nuclear_corr, crosstalk["gamma"])
    if p["footprint_beta_quantile"] is not None and int(object_labels.max()) > 0:
        if p["footprint_min_level"] is None:
            raise KeyError(
                "footprint_min_level is required when footprint_beta_quantile is set"
            )
        marker_unmixed, beta_footprint = footprint_beta_correction(
            marker_res,
            readout_corr,
            marker_unmixed,
            object_labels,
            pixel_size_um,
            usable=unsaturated_mask(
                [readout, marker], p["saturation_level"], within=tissue
            ),
            readout_thr=readout_thr,
            marker_thr=marker_thr,
            alpha=crosstalk["alpha"],
            beta=crosstalk["beta"],
            min_level=p["footprint_min_level"],
            quantile=p["footprint_beta_quantile"],
            far_um=3.0,
            dilate_um=p["footprint_dilate_um"],
        )
        crosstalk["beta_footprint"] = float(beta_footprint)

    # marker territory and the removal of its readout shadows
    marker_mask = marker_territory_mask(
        marker_unmixed,
        marker_thr,
        pixel_size_um,
        fill_hole_um2=p["marker_fill_hole_um2"],
        min_area_um2=p["marker_min_area_um2"],
        closing_px=p["marker_closing_px"],
        closing_um=p["marker_closing_um"],
    )
    marker_mask, n_shadows = remove_marker_shadows(
        marker_mask,
        object_labels,
        pixel_size_um,
        dilate_um=p["footprint_dilate_um"],
        cover=p["shadow_cover"],
    )
    distance_map = signed_distance_um(marker_mask, pixel_size_um)

    # per-object table: spectral gates, centroids, distances, overlap
    table = object_contrast_table(
        object_labels,
        readout_res,
        nuclear_corr,
        marker_res,
        pixel_size_um,
        min_contrast=p["gate_min_contrast"],
        ring_um=tuple(p["gate_ring_um"]),
        gate_ratio=p["gate_ratio"],
        marker_gate_ratio=p["marker_gate_ratio"],
        use_marker_gate=bool(p["use_marker_gate"]),
    )
    ids = table["label"].to_numpy(dtype=np.intp) if len(table) else np.zeros(0, np.intp)
    centroids = object_centroids(
        object_labels, object_ids=ids, weights=np.clip(readout_unmixed, 0.0, None)
    )
    table = table.copy()
    if len(table):
        pixels = object_pixels(object_labels, object_ids=ids)
        table["centroid_row_px"] = centroids[:, 0]
        table["centroid_col_px"] = centroids[:, 1]
        table["distance_um"] = distance_at_points(distance_map, centroids)
        table["distance_class"] = [
            DISTANCE_CLASSES[code]
            for code in distance_classes(table["distance_um"].to_numpy(), p["apposed_um"])
        ]
        table["marker_overlap_fraction"] = ring_coverage(pixels, marker_mask)
        rows = np.clip(np.rint(centroids[:, 0]).astype(np.intp), 0, tissue.shape[0] - 1)
        cols = np.clip(np.rint(centroids[:, 1]).astype(np.intp), 0, tissue.shape[1] - 1)
        table["in_tissue"] = tissue[rows, cols]
        gate_pass = table["gate_pass"].to_numpy(dtype=bool)
        table["analysed"] = table["in_tissue"].to_numpy(dtype=bool) & (
            gate_pass if p["use_gate"] else np.ones(len(table), dtype=bool)
        )
    else:
        for column in (
            "centroid_row_px",
            "centroid_col_px",
            "distance_um",
            "marker_overlap_fraction",
        ):
            table[column] = np.zeros(0, dtype=float)
        table["distance_class"] = np.zeros(0, dtype=object)
        table["in_tissue"] = np.zeros(0, dtype=bool)
        table["analysed"] = np.zeros(0, dtype=bool)

    analysed = (
        table["analysed"].to_numpy(dtype=bool) if len(table) else np.zeros(0, dtype=bool)
    )
    grid = distance_ecdf_grid(p["ecdf_start_um"], p["ecdf_stop_um"], p["ecdf_step_um"])
    null = None
    table["ring_marker_fraction"] = np.nan
    table["contact"] = False
    if int(analysed.sum()) > 0:
        null = contact_null_model(
            object_labels,
            centroids[analysed],
            marker_mask,
            distance_map,
            tissue,
            pixel_size_um,
            object_ids=ids[analysed],
            n_randomisations=int(p["n_randomisations"]),
            min_shift_um=p["min_shift_um"],
            contact_ring_um=p["contact_ring_um"],
            contact_min_cover=p["contact_min_cover"],
            apposed_um=p["apposed_um"],
            ecdf_grid_um=grid,
            envelope_window_um=p["envelope_window_um"],
            keep_null_ecdf=bool(p["keep_null_ecdf"]),
            seed=p["seed"],
        )
        position = np.flatnonzero(analysed)
        table.loc[table.index[position], "ring_marker_fraction"] = null.ring_coverage
        table.loc[table.index[position], "contact"] = null.contact

    return ContactAnalysis(
        objects=table.reset_index(drop=True),
        object_labels=object_labels,
        marker_mask=marker_mask,
        tissue=tissue,
        void=void,
        signed_distance_um=distance_map,
        readout_unmixed=readout_unmixed,
        marker_unmixed=marker_unmixed,
        thresholds=dict(
            readout_threshold=float(readout_thr),
            readout_background_mean=float(readout_mu),
            readout_noise_sigma=float(readout_sigma),
            readout_clipped_fraction=float(readout_clipped),
            marker_threshold=float(marker_thr),
            marker_background_mean=float(marker_mu),
            marker_noise_sigma=float(marker_sigma),
            marker_clipped_fraction=float(marker_clipped),
        ),
        crosstalk=crosstalk,
        n_objects_total=int(len(table)),
        n_analysed=int(analysed.sum()),
        n_gate_rejected=(
            int((~table["gate_pass"].to_numpy(dtype=bool)).sum()) if len(table) else 0
        ),
        n_marker_shadows_removed=int(n_shadows),
        marker_area_fraction=(
            float(marker_mask[tissue].mean()) if bool(tissue.any()) else float("nan")
        ),
        object_area_fraction=(
            float((object_labels > 0)[tissue].mean())
            if bool(tissue.any())
            else float("nan")
        ),
        void_fraction=float(void.mean()),
        null=null,
        parameters=p,
    )


def contact_headline(analysis: ContactAnalysis) -> dict:
    """Headline numbers of one field's analysis, as one flat row.

    Parameters
    ----------
    analysis : ContactAnalysis
        Result of :func:`analyse_contact_field`.

    Returns
    -------
    dict
        Counts, area fractions, the cross-talk coefficients, the contact fraction with
        its randomisation mean, enrichment and P value, the containment and apposition
        statistics, the median distance, and the whole-curve statistic with its P value.
        Every entry is a number, so a grid of settings tabulates directly; a field with
        no analysed object gives NaN for every randomisation entry.
    """
    null = analysis.null
    nan = float("nan")
    row = dict(
        n_objects_total=analysis.n_objects_total,
        n_objects=analysis.n_analysed,
        n_gate_rejected=analysis.n_gate_rejected,
        n_marker_shadows_removed=analysis.n_marker_shadows_removed,
        marker_area_fraction=analysis.marker_area_fraction,
        object_area_fraction=analysis.object_area_fraction,
        void_fraction=analysis.void_fraction,
        readout_threshold=analysis.thresholds["readout_threshold"],
        marker_threshold=analysis.thresholds["marker_threshold"],
        alpha=float(analysis.crosstalk.get("alpha", nan)),
        beta=float(analysis.crosstalk.get("beta", nan)),
        gamma=float(analysis.crosstalk.get("gamma", nan)),
        delta=float(analysis.crosstalk.get("delta", nan)),
    )
    if null is None:
        row.update(
            contact_fraction=nan,
            contact_null_mean=nan,
            contact_enrichment=nan,
            contact_P=nan,
            contained_fraction=nan,
            containment_enrichment=nan,
            containment_P=nan,
            apposed_fraction=nan,
            apposed_enrichment=nan,
            apposed_P=nan,
            marker_overlap_fraction=nan,
            marker_overlap_enrichment=nan,
            median_distance_um=nan,
            median_distance_null_mean=nan,
            median_distance_P=nan,
            envelope_statistic=nan,
            envelope_P=nan,
            accepted_fraction_mean=nan,
            n_randomisations=0,
        )
        return row
    row.update(
        contact_fraction=null.contact_fraction,
        contact_null_mean=float(null.contact_null.mean()),
        contact_enrichment=null.contact_enrichment,
        contact_P=null.contact_P,
        contained_fraction=null.contained_fraction,
        containment_enrichment=null.containment_enrichment,
        containment_P=null.containment_P,
        apposed_fraction=null.apposed_fraction,
        apposed_enrichment=null.apposed_enrichment,
        apposed_P=null.apposed_P,
        marker_overlap_fraction=null.marker_overlap_fraction,
        marker_overlap_enrichment=null.marker_overlap_enrichment,
        median_distance_um=null.median_distance_um,
        median_distance_null_mean=float(null.median_distance_null.mean()),
        median_distance_P=null.median_distance_P,
        envelope_statistic=null.envelope.statistic if null.envelope else nan,
        envelope_P=null.envelope.P if null.envelope else nan,
        accepted_fraction_mean=null.accepted_fraction_mean,
        n_randomisations=null.n_randomisations,
    )
    return row


def run_contact_sensitivity(
    field: ContactField,
    base_params: Mapping[str, Any],
    grid: Mapping[str, Mapping[str, Any]] | Sequence[tuple],
    *,
    include_baseline: bool = True,
    baseline_name: str = "baseline",
    n_randomisations: int | None = None,
    headline: Callable[[ContactAnalysis], Mapping] = contact_headline,
) -> pd.DataFrame:
    """Repeat the whole analysis of one field over a grid of alternative settings.

    Every setting re-runs :func:`analyse_contact_field` from the raw channels, so a
    changed threshold or background scale propagates through segmentation, cross-talk
    handling and the null model exactly as it does in the primary analysis. With one
    seed in ``base_params`` all settings share the same random transformations, so the
    settings are compared on common random numbers rather than on independent noise.

    Parameters
    ----------
    field : ContactField
        The images of one field. The analysis unit is the field, so several fields are
        run one by one and the rows concatenated by the caller.
    base_params : mapping
        Parameters of the primary analysis; see :func:`analyse_contact_field`.
    grid : mapping of str to mapping, or sequence of (str, mapping)
        Setting name to the parameter overrides of that setting.
    include_baseline : bool, optional
        Run ``base_params`` unchanged as the first row.
    baseline_name : str, optional
        Name of that first row.
    n_randomisations : int or None, optional
        Number of randomisations for the grid, overriding ``base_params``. A grid
        tabulates enrichments and P values only, so fewer randomisations than the
        primary analysis are usually enough.
    headline : callable, optional
        Maps one :class:`ContactAnalysis` to the numbers of its row.

    Returns
    -------
    pandas.DataFrame
        One row per setting, in the order the settings were given, with the setting name
        in the first column.

    Notes
    -----
    The cost of a row is the cost of a whole field analysis, whatever the setting
    changes: background models, cross-talk estimation and segmentation are repeated
    even for a grid that only varies ``contact_ring_um``, ``contact_min_cover``,
    ``apposed_um``, the grid of the cumulative distributions or the null-model
    settings, because the function deliberately runs each setting through the same
    chain as the primary analysis rather than caching parts of it. For a scan over
    those late parameters alone, call :func:`contact_null_model` directly on the masks
    and centroids of one finished analysis, with one seed for all settings; that reuses
    everything the setting cannot change. ``n_randomisations`` keeps the grid itself
    affordable.
    """
    settings = dict(grid)
    params = dict(base_params)
    if n_randomisations is not None:
        params["n_randomisations"] = int(n_randomisations)
    return run_variants(
        lambda one: headline(analyse_contact_field(field, one)),
        params,
        settings,
        include_baseline=include_baseline,
        baseline_name=baseline_name,
    )
