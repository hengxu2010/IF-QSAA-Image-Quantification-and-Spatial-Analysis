"""Concentric boundary-distance profiling of one mask around the objects of another.

For every reference object, concentric bands are built outward from the object's own
boundary at user-given distances, and the percentage of each band that is covered by
the second (tested) mask is measured. The rule of the method is to compare that
coverage with the tested mask's occupancy of the whole field, which is the coverage
expected if the tested mask were spread evenly over the frame; the difference is the
excess coverage in percentage points.

Distances are signed, in micrometres, and measured from the reference object's
boundary:

* a pixel outside the object is at its Euclidean distance to the nearest pixel of the
  object, so the first ring outside the object sits at one pixel;
* the outermost pixel layer of the object itself is at distance 0, and pixels further
  inside are negative, so a band with negative edges is a shell inside the object;
* the whole footprint of the object is therefore the band of distances <= 0.

A band spans ``(inner, outer]``: the inner distance is excluded and the outer distance
included, and consecutive bands share an edge without overlapping. Band pixels that
fall outside the frame simply do not exist, so a coverage percentage is always taken
over the band pixels present in the frame.

Every band carries its identity -- the two edges, the footprint flag and a printable
``band_label`` -- through the per-object table and on through the per-unit and
across-unit summaries, so a summarised band never has to be matched back to its
geometry by hand. The footprint band has no finite inner edge (its ``inner_um`` is
``-inf``), which is why it is labelled rather than written as an interval.
"""

from __future__ import annotations

from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from ._util import as_bool_mask, check_same_shape, pixel_area_um2, px_to_um, um_to_px

__all__ = [
    "whole_field_occupancy_percent",
    "signed_boundary_distance_um",
    "concentric_bands",
    "band_coverage_percent",
    "object_profile",
    "profile_objects",
    "unit_profiles",
    "profile_summary",
    "profile_tests",
    "stratify_equal_count",
    "excess_summary",
]

_PROFILE_COLUMNS = [
    "field",
    "object_label",
    "object_area_um2",
    "band",
    "inner_um",
    "outer_um",
    "mid_um",
    "inside",
    "band_label",
    "band_px",
    "band_area_um2",
    "coverage_percent",
    "baseline_percent",
    "excess_percent",
]

_DEFAULT_VALUE_COLS = ("coverage_percent", "baseline_percent", "excess_percent")

#: Columns that identify a band, carried through :func:`unit_profiles` and
#: :func:`profile_summary` so that a summarised band can still be labelled.
_BAND_IDENTITY_COLS = ("inner_um", "outer_um", "inside", "band_label")

_P_KEYS = ("p_value", "pvalue", "p", "P")
_STAT_KEYS = ("statistic", "W", "w")


# --------------------------------------------------------------------------- baseline

def whole_field_occupancy_percent(
    mask: np.ndarray, valid: np.ndarray | None = None
) -> float:
    """Percentage of the field that one mask covers.

    This is the baseline of the method: the coverage a band would show if the mask were
    spread evenly over the field.

    Parameters
    ----------
    mask : ndarray
        2-D mask, non-zero = foreground.
    valid : ndarray of bool, optional
        Region that counts as field, same shape as ``mask``. Without it the whole frame
        counts, which is the convention of the method.

    Returns
    -------
    float
        100 x covered pixels / field pixels, in per cent; NaN when the field is empty.
    """
    m = as_bool_mask(mask, "mask")
    if valid is None:
        return 100.0 * float(m.mean()) if m.size else float("nan")
    v = as_bool_mask(valid, "valid")
    check_same_shape(m, v)
    n = int(v.sum())
    return 100.0 * float((m & v).sum()) / n if n else float("nan")


# --------------------------------------------------------------------------- distances

def signed_boundary_distance_um(
    object_mask: np.ndarray, pixel_size_um: float
) -> np.ndarray:
    """Signed Euclidean distance to the boundary of one object, in micrometres.

    Positive outside the object (distance to its nearest pixel), 0 on the object's
    outermost pixel layer and negative further inside. Pixels beyond the frame are not
    available, so near the frame border the distance is measured within the frame only.

    Parameters
    ----------
    object_mask : ndarray of bool
        Footprint of one reference object.
    pixel_size_um : float
        Pixel size in micrometres per pixel (square pixels).

    Returns
    -------
    ndarray of float
        Signed distance in micrometres, same shape as ``object_mask``.
    """
    obj = as_bool_mask(object_mask, "object_mask")
    if not obj.any():
        raise ValueError("object_mask holds no pixel")
    if float(pixel_size_um) <= 0:
        raise ValueError("pixel_size_um must be positive")
    outside = ndi.distance_transform_edt(~obj)
    inside = ndi.distance_transform_edt(obj)
    signed_px = np.where(obj, -(inside - 1.0), outside)
    return signed_px * px_to_um(1.0, pixel_size_um)


def _band_edges(
    edges_um: Sequence[float], include_inside: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Inner edge, outer edge, inside flag and midpoint of every band."""
    e = np.asarray([float(x) for x in edges_um], dtype=float)
    if e.size < 2:
        raise ValueError("edges_um needs at least two distances")
    if np.any(np.diff(e) <= 0):
        raise ValueError("edges_um must be strictly increasing")
    inner = list(e[:-1])
    outer = list(e[1:])
    inside = [False] * (e.size - 1)
    if include_inside:
        if e[0] != 0.0:
            raise ValueError(
                "with include_inside the first edge must be 0 um, so that the extra "
                "band is the object's own footprint"
            )
        inner.insert(0, -np.inf)
        outer.insert(0, 0.0)
        inside.insert(0, True)
    inner_a = np.asarray(inner, dtype=float)
    outer_a = np.asarray(outer, dtype=float)
    inside_a = np.asarray(inside, dtype=bool)
    mid = np.where(inside_a, 0.0, 0.5 * (inner_a + outer_a))
    return inner_a, outer_a, inside_a, mid


def _band_label(inner_um: float, outer_um: float, inside: bool) -> str:
    """Printable identity of one band: its edges in micrometres, or ``'footprint'``.

    The footprint band has no finite inner edge (every distance <= 0 belongs to it), so
    it is named instead of being written as an interval.
    """
    if inside:
        return "footprint"
    return f"({inner_um:g}, {outer_um:g}]"


def _bands_from_object(
    obj: np.ndarray,
    edges_um: Sequence[float],
    pixel_size_um: float,
    include_inside: bool,
    valid: np.ndarray | None,
    exclude: np.ndarray | None,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    signed = signed_boundary_distance_um(obj, pixel_size_um)
    inner, outer, inside, mid = _band_edges(edges_um, include_inside)
    bands = []
    for lo, hi in zip(inner, outer):
        band = (signed > lo) & (signed <= hi)
        if exclude is not None:
            band &= ~exclude
        if valid is not None:
            band &= valid
        bands.append(band)
    return bands, inner, outer, inside, mid


def concentric_bands(
    object_mask: np.ndarray,
    edges_um: Sequence[float],
    pixel_size_um: float,
    *,
    include_inside: bool = True,
    valid: np.ndarray | None = None,
    exclude: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Concentric bands around one reference object.

    Parameters
    ----------
    object_mask : ndarray of bool
        Footprint of the reference object.
    edges_um : sequence of float
        Band edges in micrometres, strictly increasing, measured from the object's
        boundary (negative = inside the object). ``n`` edges give ``n - 1`` bands.
    pixel_size_um : float
        Pixel size in micrometres per pixel (square pixels).
    include_inside : bool, optional
        Prepend a band that is the object's own footprint (distances <= 0). It requires
        the first edge to be 0 um.
    valid : ndarray of bool, optional
        Region the bands are restricted to, same shape as the object mask.
    exclude : ndarray of bool, optional
        Pixels removed from every band, for instance the other objects of the reference
        mask.

    Returns
    -------
    list of ndarray of bool
        One mask per band, in order of increasing distance; the footprint band comes
        first when ``include_inside`` is true.
    """
    bands, _, _, _, _ = _bands_from_object(
        as_bool_mask(object_mask, "object_mask"),
        edges_um,
        pixel_size_um,
        include_inside,
        None if valid is None else as_bool_mask(valid, "valid"),
        None if exclude is None else as_bool_mask(exclude, "exclude"),
    )
    return bands


def band_coverage_percent(
    bands: Iterable[np.ndarray], other_mask: np.ndarray
) -> np.ndarray:
    """Percentage of each band that the tested mask covers.

    Parameters
    ----------
    bands : iterable of ndarray of bool
        Band masks, for instance from :func:`concentric_bands`.
    other_mask : ndarray
        The tested mask, same shape as the bands, non-zero = foreground.

    Returns
    -------
    ndarray of float
        One percentage per band; NaN for a band without a pixel in the frame.
    """
    other = as_bool_mask(other_mask, "other_mask")
    out = []
    for band in bands:
        b = as_bool_mask(band, "band")
        check_same_shape(b, other)
        n = int(b.sum())
        out.append(100.0 * float((b & other).sum()) / n if n else np.nan)
    return np.asarray(out, dtype=float)


# --------------------------------------------------------------------------- profiles

def _crop(bbox, shape: tuple[int, int], margin_px: int) -> tuple[slice, slice]:
    ys, xs = bbox
    return (
        slice(max(0, ys.start - margin_px), min(shape[0], ys.stop + margin_px)),
        slice(max(0, xs.start - margin_px), min(shape[1], xs.stop + margin_px)),
    )


def _margin_px(edges_um: Sequence[float], pixel_size_um: float) -> int:
    """Crop margin that holds every band: the farthest edge, rounded up, plus a pixel."""
    reach = max(abs(float(x)) for x in edges_um)
    return int(np.ceil(um_to_px(reach, pixel_size_um))) + 1


def _profile_rows(
    labels_ref: np.ndarray,
    label: int,
    other: np.ndarray,
    edges_um: Sequence[float],
    pixel_size_um: float,
    include_inside: bool,
    exclude_other_reference_objects: bool,
    valid: np.ndarray | None,
    baseline_percent: float,
    field_id: object,
    bbox,
) -> list[dict]:
    """Per-band rows of one reference object, computed on a crop around the object."""
    window = _crop(bbox, labels_ref.shape, _margin_px(edges_um, pixel_size_um))
    lab = labels_ref[window]
    obj = lab == label
    exclude = (lab > 0) & ~obj if exclude_other_reference_objects else None
    bands, inner, outer, inside, mid = _bands_from_object(
        obj,
        edges_um,
        pixel_size_um,
        include_inside,
        None if valid is None else valid[window],
        exclude,
    )
    other_crop = other[window]
    px_area = pixel_area_um2(pixel_size_um)
    object_area_um2 = float(obj.sum()) * px_area
    rows = []
    for k, band in enumerate(bands):
        n = int(band.sum())
        cover = 100.0 * float((band & other_crop).sum()) / n if n else np.nan
        rows.append(
            {
                "field": field_id,
                "object_label": int(label),
                "object_area_um2": object_area_um2,
                "band": k,
                "inner_um": float(inner[k]),
                "outer_um": float(outer[k]),
                "mid_um": float(mid[k]),
                "inside": bool(inside[k]),
                "band_label": _band_label(
                    float(inner[k]), float(outer[k]), bool(inside[k])
                ),
                "band_px": n,
                "band_area_um2": n * px_area,
                "coverage_percent": cover,
                "baseline_percent": baseline_percent,
                "excess_percent": cover - baseline_percent,
            }
        )
    return rows


def object_profile(
    labels_ref: np.ndarray,
    label: int,
    other_mask: np.ndarray,
    edges_um: Sequence[float],
    pixel_size_um: float,
    *,
    include_inside: bool = True,
    exclude_other_reference_objects: bool = False,
    valid: np.ndarray | None = None,
    baseline_percent: float | None = None,
    field_id: object = None,
) -> pd.DataFrame:
    """Band coverage profile around one reference object.

    Parameters
    ----------
    labels_ref : ndarray of int
        Label image of the reference mask (0 = background).
    label : int
        Label of the reference object the bands are built around.
    other_mask : ndarray
        The tested mask, same shape, non-zero = foreground.
    edges_um : sequence of float
        Band edges in micrometres from the object's boundary, strictly increasing;
        negative edges lie inside the object.
    pixel_size_um : float
        Pixel size in micrometres per pixel (square pixels).
    include_inside : bool, optional
        Add the object's own footprint as the first band.
    exclude_other_reference_objects : bool, optional
        Remove the pixels of the other objects of the reference mask from the bands.
    valid : ndarray of bool, optional
        Region the bands and the baseline are restricted to.
    baseline_percent : float, optional
        Whole-field occupancy of the tested mask in per cent. Computed from
        ``other_mask`` (and ``valid``) when not given.
    field_id : object, optional
        Identifier of the field, copied into the ``field`` column.

    Returns
    -------
    DataFrame
        One row per band: ``field``, ``object_label``, ``object_area_um2`` (um^2),
        ``band`` (index), ``inner_um``, ``outer_um``, ``mid_um`` (um, the midpoint; 0
        for the footprint band), ``inside`` (footprint band flag), ``band_label``
        (printable band identity: ``'footprint'`` for the footprint band, whose
        ``inner_um`` is ``-inf``, and ``'(inner, outer]'`` in micrometres otherwise),
        ``band_px``, ``band_area_um2`` (um^2), ``coverage_percent`` (per cent of the
        band covered by the tested mask, NaN for an empty band), ``baseline_percent``
        and ``excess_percent`` (coverage - baseline, percentage points).

        The four identity columns ``inner_um``, ``outer_um``, ``inside`` and
        ``band_label`` are carried through :func:`unit_profiles`,
        :func:`profile_summary` and :func:`excess_summary`, so a summarised band can
        still be named without re-joining this table.
    """
    labels = np.asarray(labels_ref)
    other = as_bool_mask(other_mask, "other_mask")
    check_same_shape(labels, other)
    if valid is not None:
        valid = as_bool_mask(valid, "valid")
        check_same_shape(labels, valid)
    bboxes = ndi.find_objects(labels)
    if not 1 <= int(label) <= len(bboxes) or bboxes[int(label) - 1] is None:
        raise ValueError("label is not present in labels_ref")
    base = (
        whole_field_occupancy_percent(other, valid)
        if baseline_percent is None
        else float(baseline_percent)
    )
    rows = _profile_rows(
        labels,
        int(label),
        other,
        edges_um,
        pixel_size_um,
        include_inside,
        exclude_other_reference_objects,
        valid,
        base,
        field_id,
        bboxes[int(label) - 1],
    )
    return pd.DataFrame(rows, columns=_PROFILE_COLUMNS)


def profile_objects(
    labels_ref: np.ndarray,
    other_mask: np.ndarray,
    edges_um: Sequence[float],
    pixel_size_um: float,
    *,
    labels: Sequence[int] | None = None,
    include_inside: bool = True,
    exclude_other_reference_objects: bool = False,
    valid: np.ndarray | None = None,
    baseline_percent: float | None = None,
    field_id: object = None,
) -> pd.DataFrame:
    """Band coverage profiles of every reference object of one field.

    Parameters
    ----------
    labels_ref : ndarray of int
        Label image of the reference mask.
    other_mask : ndarray
        The tested mask, same shape.
    edges_um : sequence of float
        Band edges in micrometres, see :func:`object_profile`.
    pixel_size_um : float
        Pixel size in micrometres per pixel (square pixels).
    labels : sequence of int, optional
        Reference objects to profile; by default every label present in the image, in
        increasing order.
    include_inside, exclude_other_reference_objects, valid, baseline_percent, field_id
        As in :func:`object_profile`. The baseline is computed once for the field.

    Returns
    -------
    DataFrame
        One row per object and band, with the columns of :func:`object_profile`.
    """
    labels_img = np.asarray(labels_ref)
    other = as_bool_mask(other_mask, "other_mask")
    check_same_shape(labels_img, other)
    if valid is not None:
        valid = as_bool_mask(valid, "valid")
        check_same_shape(labels_img, valid)
    base = (
        whole_field_occupancy_percent(other, valid)
        if baseline_percent is None
        else float(baseline_percent)
    )
    bboxes = ndi.find_objects(labels_img)
    if labels is None:
        wanted = [int(v) for v in np.unique(labels_img) if int(v) > 0]
    else:
        wanted = [int(v) for v in labels]
    rows: list[dict] = []
    for label in wanted:
        if not 1 <= label <= len(bboxes) or bboxes[label - 1] is None:
            raise ValueError("a requested label is not present in labels_ref")
        rows.extend(
            _profile_rows(
                labels_img,
                label,
                other,
                edges_um,
                pixel_size_um,
                include_inside,
                exclude_other_reference_objects,
                valid,
                base,
                field_id,
                bboxes[label - 1],
            )
        )
    return pd.DataFrame(rows, columns=_PROFILE_COLUMNS)


# --------------------------------------------------------------------------- grouping

def _by_list(by) -> list[str]:
    return [by] if isinstance(by, str) else list(by)


def _need(frame: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")


def _identity_cols(
    frame: pd.DataFrame, band_cols: Sequence[str] | None, taken: Iterable[str]
) -> list[str]:
    """Band-identity columns to carry through a grouping."""
    if band_cols is None:
        wanted = [c for c in _BAND_IDENTITY_COLS if c in frame.columns]
    else:
        wanted = list(band_cols)
        _need(frame, wanted)
    seen = set(taken)
    return [c for c in wanted if c not in seen]


def _carried_identity(grouped, cols: Sequence[str]) -> pd.DataFrame:
    """One value of every band-identity column per group, checked to be unambiguous."""
    cols = list(cols)
    if not cols:
        return pd.DataFrame()
    counts = grouped[cols].nunique()
    ambiguous = [c for c in cols if len(counts) and int(counts[c].max()) > 1]
    if ambiguous:
        raise ValueError(
            f"the band identity columns {ambiguous} hold more than one value inside a "
            "band of the grouping, so the bands being summarised are not the same "
            "bands: profiles built with different band edges cannot be pooled by the "
            "band midpoint alone"
        )
    return grouped[cols].first()


def unit_profiles(
    objects: pd.DataFrame,
    unit_col: str,
    *,
    by: Sequence[str] = (),
    distance_col: str = "mid_um",
    value_cols: Sequence[str] = _DEFAULT_VALUE_COLS,
    exclude: Sequence = (),
    band_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Average a per-object profile table within each unit of analysis.

    The unit of analysis of the method is the subject: inside a unit the objects are
    averaged with equal weight, and the units are the sample for every summary and test
    that follows.

    A single field holds no such unit. Passing the object column (``object_label``) as
    ``unit_col`` is then the legitimate way to obtain per-band means and a spread, but
    it changes what the spread means: it describes the variation between the objects of
    that one field, not between subjects, and it supports no statement about a
    population of subjects. Report it as a within-field description and say which unit
    was used.

    Parameters
    ----------
    objects : DataFrame
        Per-object, per-band table, for instance from :func:`profile_objects` with the
        unit (and any grouping) columns joined to it.
    unit_col : str
        Column holding the unit of analysis -- the subject, or ``object_label`` for the
        single-field case described above.
    by : sequence of str, optional
        Further grouping columns kept separate, for instance a region or a stratum.
    distance_col : str, optional
        Column identifying the band; the band midpoint by default.
    value_cols : sequence of str, optional
        Columns to average.
    exclude : sequence, optional
        Units left out of this profile entirely.
    band_cols : sequence of str, optional
        Band-identity columns carried through unchanged, so that the result can still
        be labelled by band geometry. Default: whichever of ``inner_um``, ``outer_um``,
        ``inside`` and ``band_label`` the input holds. An empty sequence carries none of
        them and skips the consistency check below.

    Returns
    -------
    DataFrame
        One row per grouping, unit and band: the keys, the carried band-identity
        columns, ``n_objects`` and the mean of every value column.

    Raises
    ------
    ValueError
        If a band-identity column holds more than one value inside one band of the
        grouping, which means the pooled rows do not describe the same band.
    """
    keys = _by_list(by) + [unit_col, distance_col]
    _need(objects, keys + list(value_cols))
    identity = _identity_cols(objects, band_cols, keys + list(value_cols))
    frame = objects
    if len(exclude):
        frame = frame[~frame[unit_col].isin(list(exclude))]
    grouped = frame.groupby(keys, dropna=False, sort=True)
    out = grouped[list(value_cols)].mean().reset_index()
    carried = _carried_identity(grouped, identity)
    for offset, column in enumerate(identity):
        out.insert(len(keys) + offset, column, carried[column].to_numpy())
    out.insert(len(keys) + len(identity), "n_objects",
               grouped.size().reset_index(drop=True).to_numpy())
    return out


def profile_summary(
    units: pd.DataFrame,
    *,
    by: Sequence[str] = (),
    distance_col: str = "mid_um",
    value_cols: Sequence[str] = _DEFAULT_VALUE_COLS,
    ddof: int = 1,
    band_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Mean and standard deviation of a per-unit profile across units.

    The geometry of each band is carried through, so the summary can be read and
    printed on its own: the band identity does not have to be joined back from the
    per-object table. The unit of analysis is whatever :func:`unit_profiles` was given;
    with one field and the object column as the unit, the s.d. is across the objects of
    that field (see :func:`unit_profiles`).

    Parameters
    ----------
    units : DataFrame
        One row per unit and band, for instance from :func:`unit_profiles`.
    by : sequence of str, optional
        Grouping columns kept separate.
    distance_col : str, optional
        Column identifying the band.
    value_cols : sequence of str, optional
        Columns to summarise.
    ddof : int, optional
        Delta degrees of freedom of the standard deviation; 1 gives the sample s.d.,
        which is what the method reports.
    band_cols : sequence of str, optional
        Band-identity columns carried through unchanged. Default: whichever of
        ``inner_um``, ``outer_um``, ``inside`` and ``band_label`` the input holds; an
        empty sequence carries none of them. ``band_label`` is the safe one to print,
        because the footprint band's ``inner_um`` is ``-inf``.

    Returns
    -------
    DataFrame
        One row per grouping and band: the keys, the carried band-identity columns,
        ``n_units``, ``n_objects`` (summed when the column is present) and
        ``<col>_mean`` / ``<col>_sd`` per value column.

    Raises
    ------
    ValueError
        If a band-identity column holds more than one value inside one band of the
        grouping (see :func:`unit_profiles`).
    """
    keys = _by_list(by) + [distance_col]
    _need(units, keys + list(value_cols))
    identity = _identity_cols(units, band_cols, keys + list(value_cols))
    grouped = units.groupby(keys, dropna=False, sort=True)
    carried = _carried_identity(grouped, identity)
    data = {column: carried[column] for column in identity}
    data["n_units"] = grouped.size()
    if "n_objects" in units.columns:
        data["n_objects"] = grouped["n_objects"].sum()
    for col in value_cols:
        data[f"{col}_mean"] = grouped[col].mean()
        data[f"{col}_sd"] = grouped[col].std(ddof=ddof)
    return pd.DataFrame(data).reset_index()


def _test_value(result, keys: Sequence[str]) -> float:
    if isinstance(result, Mapping):
        for key in keys:
            if key in result:
                return float(result[key])
    else:
        for key in keys:
            value = getattr(result, key, None)
            if value is not None:
                return float(value)
    return float("nan")


def profile_tests(
    units: pd.DataFrame,
    *,
    distance_col: str = "mid_um",
    value_col: str = "coverage_percent",
    baseline_col: str = "baseline_percent",
    test: Callable | None = None,
    test_kwargs: Mapping | None = None,
    ddof: int = 1,
) -> pd.DataFrame:
    """Paired test of band coverage against the whole-field baseline, band by band.

    At every band the units are the pairs: the unit's mean coverage of that band against
    the unit's mean whole-field occupancy of the tested mask. The test is run once per
    band and no multiplicity correction is applied, as in the method.

    Parameters
    ----------
    units : DataFrame
        One row per unit and band, for instance from :func:`unit_profiles`.
    distance_col : str, optional
        Column identifying the band.
    value_col, baseline_col : str, optional
        The paired columns: band coverage and whole-field baseline, both in per cent.
    test : callable, optional
        ``test(x, y, **test_kwargs)`` returning a mapping with a P value and a
        statistic (keys ``p_value`` / ``pvalue`` / ``p`` / ``P`` and ``statistic`` /
        ``W``), or an object with those attributes. By default the exact signed-rank
        test of :mod:`if_qsaa.stats` is used.
    test_kwargs : mapping, optional
        Extra keyword arguments passed to ``test``.
    ddof : int, optional
        Delta degrees of freedom of the s.d. of the differences.

    Returns
    -------
    DataFrame
        One row per band: the band identifier, ``n`` (units with both values),
        ``n_above`` (units whose coverage exceeds their baseline), ``mean_diff`` and
        ``sd_diff`` (percentage points), ``statistic`` and ``p_value``.
    """
    _need(units, [distance_col, value_col, baseline_col])
    if test is None:
        from . import stats as _stats

        test = _stats.wilcoxon_signed_rank_exact
    kwargs = dict(test_kwargs or {})
    rows = []
    for distance, block in units.groupby(distance_col, dropna=False, sort=True):
        pair = block[[value_col, baseline_col]].dropna()
        x = pair[value_col].to_numpy(dtype=float)
        y = pair[baseline_col].to_numpy(dtype=float)
        diff = x - y
        if diff.size:
            result = test(x, y, **kwargs)
            p = _test_value(result, _P_KEYS)
            statistic = _test_value(result, _STAT_KEYS)
        else:
            p = statistic = float("nan")
        rows.append(
            {
                distance_col: distance,
                "n": int(diff.size),
                "n_above": int((diff > 0).sum()),
                "mean_diff": float(diff.mean()) if diff.size else np.nan,
                "sd_diff": float(diff.std(ddof=ddof)) if diff.size > ddof else np.nan,
                "statistic": statistic,
                "p_value": p,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- strata

def stratify_equal_count(values: Iterable[float], n_strata: int = 3) -> np.ndarray:
    """Split values into strata of equal count, smallest first.

    The values are ranked with a stable sort, so equal values keep the order in which
    they were given, and the ranks are cut into ``n_strata`` blocks of as equal a size
    as the count allows.

    Parameters
    ----------
    values : iterable of float
        One value per object, for instance an object area in um^2 or any other
        per-object covariate.
    n_strata : int, optional
        Number of strata; 3 gives tertiles.

    Returns
    -------
    ndarray of int
        Stratum index per value, from 0 (smallest values) to ``n_strata - 1``.
    """
    v = np.asarray(list(values), dtype=float)
    k = int(n_strata)
    if k < 1:
        raise ValueError("n_strata must be at least 1")
    if v.size == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(v, kind="stable")
    stratum = np.empty(v.size, dtype=np.int64)
    stratum[order] = np.arange(v.size) * k // v.size
    return stratum


def excess_summary(
    objects: pd.DataFrame,
    unit_col: str,
    *,
    by: Sequence[str] = (),
    distance_col: str = "mid_um",
    value_col: str = "excess_percent",
    exclude: Sequence = (),
    ddof: int = 1,
    require_all_units: bool = False,
) -> pd.DataFrame:
    """Excess coverage over the whole-field baseline, averaged per unit and summarised.

    Excess coverage is the band coverage minus the tested mask's whole-field occupancy,
    in percentage points; :func:`profile_objects` reports it per object and band. Here
    the objects of a unit are averaged first and the units are then summarised, so each
    unit counts once. Strata of a per-object covariate, for instance object area from
    :func:`stratify_equal_count`, can be kept apart through ``by``.

    Parameters
    ----------
    objects : DataFrame
        Per-object, per-band table with the unit and grouping columns.
    unit_col : str
        Column holding the unit of analysis; see :func:`unit_profiles` for the
        single-field case, where the object column stands in as the unit.
    by : sequence of str, optional
        Grouping columns kept separate, for instance a stratum or a region column.
    distance_col : str, optional
        Column identifying the band.
    value_col : str, optional
        Column holding the excess coverage in percentage points.
    exclude : sequence, optional
        Units left out entirely.
    ddof : int, optional
        Delta degrees of freedom of the standard deviation across units.
    require_all_units : bool, optional
        Raise when a group does not hold every unit of the table, which keeps a group
        from being summarised over a subset of the units.

    Returns
    -------
    DataFrame
        One row per grouping and band: the keys, the band-identity columns carried
        through by :func:`unit_profiles` and :func:`profile_summary`, ``n_units``,
        ``n_objects``, ``<value_col>_mean`` and ``<value_col>_sd``.
    """
    per_unit = unit_profiles(
        objects,
        unit_col,
        by=by,
        distance_col=distance_col,
        value_cols=(value_col,),
        exclude=exclude,
    )
    if require_all_units:
        expected = set(pd.unique(per_unit[unit_col]))
        keys = _by_list(by) + [distance_col]
        for key, block in per_unit.groupby(keys, dropna=False, sort=True):
            missing = sorted(expected - set(block[unit_col]), key=str)
            if missing:
                raise ValueError(f"group {key} is missing units {missing}")
    return profile_summary(
        per_unit, by=by, distance_col=distance_col, value_cols=(value_col,), ddof=ddof
    )
