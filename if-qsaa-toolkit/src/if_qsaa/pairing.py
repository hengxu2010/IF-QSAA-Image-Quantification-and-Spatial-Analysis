"""Object pairing between two co-registered binary masks of one field, and the
mask-translation control of the pair count.

Two rules of the method are implemented here.

Pairing rule
    An object of the first mask and an object of the second mask form a dual-positive
    pair when their masks share at least a given fraction of the area of the *smaller*
    of the two objects (10 % by the rule of the method). The relation is many-to-many:
    no one-to-one matching is attempted, so no tie-breaking is involved. Objects of
    either mask without such a partner are the two single-positive classes.

Translation control
    One mask is translated as a whole relative to the other by a random displacement
    whose length lies in a user-given range, and the pairs are recounted under the same
    pairing rule on every shift. Enrichment is the observed count divided by the mean
    of the controls, with a two-sided empirical P value from the control distribution.
    The label image itself is translated, so in the wrap-around scheme object number,
    area and shape are unchanged by construction.

    The number of shifts sets the smallest P value the control can report: the two-sided
    empirical P value of :func:`enrichment` cannot fall below ``2 / (n_draws + 1)``,
    whatever the data. The method's 1,000 shifts per field floor it just below 0.002;
    a count reduced to save time raises that floor, and a reported P value sitting
    exactly on it means only that no control draw reached the observed value -- it is
    not a smaller P value waiting to be read. Choose the number of shifts for the
    smallest P value the result has to be able to state.

All lengths in the public interface are in micrometres, areas in square micrometres and
densities per square millimetre; a displacement is an integer number of pixels, because
only a whole-pixel shift leaves the objects' shapes untouched.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from ._util import (
    as_bool_mask,
    check_same_shape,
    default_rng,
    label_mask,
    pixel_area_um2,
    um_to_px,
)

__all__ = [
    "DUAL",
    "A_ONLY",
    "B_ONLY",
    "PairResult",
    "label_objects",
    "pair_objects",
    "classify_objects",
    "field_summary",
    "translate_labels",
    "draw_displacements",
    "translation_control",
    "control_summary",
    "enrichment",
    "aggregate_counts",
    "aggregate_fractions",
]

#: Class of an object that has at least one partner in the other mask.
DUAL = "dual"
#: Class of an object of the first mask without a partner in the second mask.
A_ONLY = "a_only"
#: Class of an object of the second mask without a partner in the first mask.
B_ONLY = "b_only"

_UM_PER_MM = 1.0e3  # micrometres per millimetre, for areas in mm^2


# --------------------------------------------------------------------------- labelling

def label_objects(
    mask: np.ndarray, connectivity: int = 8
) -> tuple[np.ndarray, int, np.ndarray, tuple]:
    """Label the objects of one binary mask and measure their areas in pixels.

    Any non-zero pixel of ``mask`` is foreground. Labels are assigned by
    :func:`scipy.ndimage.label`, which numbers the components in raster order.

    Parameters
    ----------
    mask : ndarray
        2-D mask of one field. Non-zero means foreground.
    connectivity : {4, 8}, optional
        Pixel connectivity used to group foreground pixels into objects. The method's
        rule is 8-connectivity (a 3x3 structuring element).

    Returns
    -------
    labels : ndarray of int
        Label image, 0 = background, objects numbered from 1.
    n : int
        Number of objects.
    areas_px : ndarray of int
        Area of every label in pixels, indexed by label; element 0 is the number of
        background pixels.
    bboxes : tuple of (slice, slice)
        Bounding box of every object, in label order, with exclusive stops.
    """
    labels, n = label_mask(mask, connectivity)
    areas_px = np.bincount(labels.ravel(), minlength=n + 1)
    bboxes = tuple(ndi.find_objects(labels))
    return labels, n, areas_px, bboxes


# --------------------------------------------------------------------------- pairing

@dataclasses.dataclass(frozen=True, eq=False)
class PairResult:
    """Objects of two masks of one field and the pairs between them.

    Attributes
    ----------
    labels_a, labels_b : ndarray of int
        Label images of the two masks (0 = background).
    areas_a_px, areas_b_px : ndarray of int
        Object areas in pixels, indexed by label; element 0 is the background.
    pairs : ndarray of int, shape (n_pairs, 3)
        One row per pair: label in the first mask, label in the second mask, shared
        area in pixels. Rows are sorted by the two labels.
    partnered_a, partnered_b : ndarray of int
        Sorted labels that take part in at least one pair.
    bboxes_a, bboxes_b : tuple of (slice, slice)
        Bounding boxes of the objects, in label order, with exclusive stops.
    min_shared_frac : float
        Fraction of the smaller object's area that the shared area had to reach.
    connectivity : int
        Connectivity used for labelling.
    """

    labels_a: np.ndarray
    labels_b: np.ndarray
    areas_a_px: np.ndarray
    areas_b_px: np.ndarray
    pairs: np.ndarray
    partnered_a: np.ndarray
    partnered_b: np.ndarray
    bboxes_a: tuple
    bboxes_b: tuple
    min_shared_frac: float
    connectivity: int

    @property
    def n_a(self) -> int:
        """Number of objects in the first mask."""
        return int(self.areas_a_px.size - 1)

    @property
    def n_b(self) -> int:
        """Number of objects in the second mask."""
        return int(self.areas_b_px.size - 1)

    @property
    def n_pairs(self) -> int:
        """Number of pairs."""
        return int(self.pairs.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the field in pixels."""
        return tuple(self.labels_a.shape)


def _pairs_from_labels(
    labels_a: np.ndarray,
    areas_a_px: np.ndarray,
    labels_b: np.ndarray,
    areas_b_px: np.ndarray,
    min_shared_frac: float,
) -> np.ndarray:
    """Shared area of every overlapping label pair, kept when it reaches the rule."""
    both = (labels_a > 0) & (labels_b > 0)
    if not both.any():
        return np.zeros((0, 3), dtype=np.int64)
    rows, shared = np.unique(
        np.stack([labels_a[both], labels_b[both]], axis=1), axis=0, return_counts=True
    )
    smaller = np.minimum(areas_a_px[rows[:, 0]], areas_b_px[rows[:, 1]])
    keep = (shared / smaller) >= min_shared_frac
    return np.column_stack(
        [rows[keep].astype(np.int64), shared[keep].astype(np.int64)]
    )


def pair_objects(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    min_shared_frac: float = 0.10,
    connectivity: int = 8,
) -> PairResult:
    """Pair the objects of two co-registered binary masks of one field.

    Both masks are labelled, the shared area of every overlapping pair of objects is
    counted, and a pair is kept when the shared area reaches ``min_shared_frac`` of the
    area of the smaller of the two objects (the comparison is inclusive). The relation
    is many-to-many: one object may pair with several objects of the other mask.

    Parameters
    ----------
    mask_a, mask_b : ndarray
        Two 2-D masks of the same field, same shape, non-zero = foreground.
    min_shared_frac : float, optional
        Pairing rule: shared area / area of the smaller object, dimensionless, in
        (0, 1]. The method's rule is 0.10.
    connectivity : {4, 8}, optional
        Connectivity for labelling both masks; the method's rule is 8.

    Returns
    -------
    PairResult
        Labels, areas in pixels, bounding boxes, the pair list with shared areas in
        pixels, and the sorted labels that have at least one partner.
    """
    a = as_bool_mask(mask_a, "mask_a")
    b = as_bool_mask(mask_b, "mask_b")
    check_same_shape(a, b)
    if not 0.0 < float(min_shared_frac) <= 1.0:
        raise ValueError("min_shared_frac must lie in (0, 1]")

    labels_a, n_a, areas_a, bboxes_a = label_objects(a, connectivity)
    labels_b, n_b, areas_b, bboxes_b = label_objects(b, connectivity)
    pairs = _pairs_from_labels(labels_a, areas_a, labels_b, areas_b, float(min_shared_frac))
    return PairResult(
        labels_a=labels_a,
        labels_b=labels_b,
        areas_a_px=areas_a,
        areas_b_px=areas_b,
        pairs=pairs,
        partnered_a=np.unique(pairs[:, 0]).astype(np.int64),
        partnered_b=np.unique(pairs[:, 1]).astype(np.int64),
        bboxes_a=bboxes_a,
        bboxes_b=bboxes_b,
        min_shared_frac=float(min_shared_frac),
        connectivity=int(connectivity),
    )


def _pairs_frame(pairs: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label_a": pairs[:, 0].astype(np.int64),
            "label_b": pairs[:, 1].astype(np.int64),
            "shared_px": pairs[:, 2].astype(np.int64),
        }
    )


def classify_objects(
    result: PairResult, pixel_size_um: float, field_id: object = None
) -> pd.DataFrame:
    """Tidy per-object table with the pairing class of every object of both masks.

    One row per object of each mask. An object with at least one partner belongs to the
    dual-positive class; an object without a partner belongs to the single-positive
    class of its own mask. The named partner is the partner with the largest shared
    area; ties are broken by the smaller partner label.

    Parameters
    ----------
    result : PairResult
        Output of :func:`pair_objects`.
    pixel_size_um : float
        Pixel size in micrometres per pixel (square pixels), used to convert areas.
    field_id : object, optional
        Identifier of the field, copied into the ``field`` column.

    Returns
    -------
    DataFrame
        Columns: ``field``, ``mask`` ('a' or 'b'), ``label``, ``object_class``
        (``'dual'``, ``'a_only'`` or ``'b_only'``), ``area_um2`` (um^2),
        ``n_partners``, ``partner_label`` (nullable integer, the named partner),
        ``shared_area_um2`` (um^2, with the named partner) and
        ``shared_area_um2_total`` (um^2, over all partners).
    """
    px_area = pixel_area_um2(pixel_size_um)
    pairs = _pairs_frame(result.pairs)
    frames = []
    for mask, own, other, areas, single in (
        ("a", "label_a", "label_b", result.areas_a_px, A_ONLY),
        ("b", "label_b", "label_a", result.areas_b_px, B_ONLY),
    ):
        n = int(areas.size - 1)
        labels = np.arange(1, n + 1, dtype=np.int64)
        n_partners = np.zeros(n, dtype=np.int64)
        shared_total = np.zeros(n, dtype=np.int64)
        best_partner = np.full(n, -1, dtype=np.int64)
        best_shared = np.zeros(n, dtype=np.int64)
        if n and len(pairs):
            grouped = pairs.groupby(own)["shared_px"]
            counts = grouped.size()
            totals = grouped.sum()
            n_partners[counts.index.to_numpy() - 1] = counts.to_numpy()
            shared_total[totals.index.to_numpy() - 1] = totals.to_numpy()
            best = (
                pairs.sort_values([own, "shared_px", other], ascending=[True, False, True])
                .drop_duplicates(own)
            )
            idx = best[own].to_numpy() - 1
            best_partner[idx] = best[other].to_numpy()
            best_shared[idx] = best["shared_px"].to_numpy()
        has = n_partners > 0
        partner_col = pd.array(np.where(has, best_partner, 0), dtype="Int64")
        partner_col[~has] = pd.NA
        frames.append(
            pd.DataFrame(
                {
                    "field": [field_id] * n,
                    "mask": [mask] * n,
                    "label": labels,
                    "object_class": np.where(has, DUAL, single),
                    "area_um2": areas[1:].astype(float) * px_area,
                    "n_partners": n_partners,
                    "partner_label": partner_col,
                    "shared_area_um2": np.where(
                        has, best_shared.astype(float) * px_area, np.nan
                    ),
                    "shared_area_um2_total": shared_total.astype(float) * px_area,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _area_mm2(shape: tuple[int, int], pixel_size_um: float, valid: np.ndarray | None) -> float:
    """Area of the field (or of the valid region) in square millimetres."""
    if valid is None:
        n_px = float(int(shape[0]) * int(shape[1]))
    else:
        v = as_bool_mask(valid, "valid")
        if tuple(v.shape) != tuple(shape):
            raise ValueError("valid must have the shape of the field")
        n_px = float(v.sum())
    return n_px * pixel_area_um2(pixel_size_um) / (_UM_PER_MM ** 2)


def field_summary(
    result: PairResult,
    pixel_size_um: float,
    field_id: object = None,
    valid: np.ndarray | None = None,
) -> pd.DataFrame:
    """One-row summary of the pairing of one field.

    Parameters
    ----------
    result : PairResult
        Output of :func:`pair_objects`.
    pixel_size_um : float
        Pixel size in micrometres per pixel (square pixels).
    field_id : object, optional
        Identifier of the field, copied into the ``field`` column.
    valid : ndarray of bool, optional
        Region that counts as field area, same shape as the masks. Without it the whole
        frame is the area, which is the convention of the method.

    Returns
    -------
    DataFrame
        One row with ``field``, the object counts of both masks (``n_objects_a``,
        ``n_objects_b``), the numbers with a partner (``n_partnered_a``,
        ``n_partnered_b``), the single-positive counts (``n_a_only``, ``n_b_only``),
        ``n_pairs``, the fractions with a partner (``frac_with_partner_a``,
        ``frac_with_partner_b``, dimensionless, NaN when the mask has no object),
        ``area_mm2`` (mm^2) and ``pair_density_per_mm2`` (pairs mm^-2).
    """
    area_mm2 = _area_mm2(result.shape, pixel_size_um, valid)
    n_a, n_b = result.n_a, result.n_b
    part_a, part_b = int(result.partnered_a.size), int(result.partnered_b.size)
    return pd.DataFrame(
        {
            "field": [field_id],
            "n_objects_a": [n_a],
            "n_objects_b": [n_b],
            "n_partnered_a": [part_a],
            "n_partnered_b": [part_b],
            "n_a_only": [n_a - part_a],
            "n_b_only": [n_b - part_b],
            "n_pairs": [result.n_pairs],
            "frac_with_partner_a": [part_a / n_a if n_a else np.nan],
            "frac_with_partner_b": [part_b / n_b if n_b else np.nan],
            "area_mm2": [area_mm2],
            "pair_density_per_mm2": [result.n_pairs / area_mm2 if area_mm2 > 0 else np.nan],
        }
    )


# --------------------------------------------------------------- translation control

def translate_labels(
    labels: np.ndarray, dy_px: int, dx_px: int, rot90: int = 0, mode: str = "wrap"
) -> np.ndarray:
    """Translate a label image as a whole, optionally after a quarter-turn rotation.

    The label image, not the binary mask, is moved, so every object keeps its identity
    even when it is cut by the frame border.

    Parameters
    ----------
    labels : ndarray of int
        Label image to move (0 = background).
    dy_px, dx_px : int
        Displacement in pixels, downwards and to the right.
    rot90 : int, optional
        Number of 90 degree rotations applied before the shift. A quarter turn needs a
        square field, otherwise the shape would change.
    mode : {'wrap', 'crop'}, optional
        ``'wrap'`` moves the image on a torus: object number, area and shape are
        unchanged by construction. ``'crop'`` moves it without wrap-around, so the part
        of an object that leaves the frame is lost.

    Returns
    -------
    ndarray of int
        Translated label image, same shape and dtype as ``labels``.
    """
    arr = np.asarray(labels)
    if arr.ndim != 2:
        raise ValueError("labels must be 2-D")
    k = int(rot90) % 4
    if k % 2 and arr.shape[0] != arr.shape[1]:
        raise ValueError("a quarter-turn rotation needs a square field")
    if k:
        arr = np.rot90(arr, k)
    dy, dx = int(dy_px), int(dx_px)
    if mode == "wrap":
        return np.roll(arr, (dy, dx), axis=(0, 1))
    if mode != "crop":
        raise ValueError("mode must be 'wrap' or 'crop'")
    out = np.zeros_like(arr)
    h, w = arr.shape
    if abs(dy) >= h or abs(dx) >= w:
        return out
    out[
        slice(max(0, dy), h - max(0, -dy)), slice(max(0, dx), w - max(0, -dx))
    ] = arr[slice(max(0, -dy), h - max(0, dy)), slice(max(0, -dx), w - max(0, dx))]
    return out


def draw_displacements(
    n_draws: int = 1000,
    *,
    min_length_um: float,
    max_length_um: float,
    pixel_size_um: float,
    scheme: str = "polar",
    rotations: bool = False,
    seed: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    """Draw whole-pixel displacements whose length lies in a given range.

    Every draw is rejected and redrawn until the length of the rounded, whole-pixel
    displacement lies in ``[min_length_um, max_length_um]``, so the range holds for the
    displacement that is actually applied.

    Parameters
    ----------
    n_draws : int, optional
        Number of displacements. The method's rule for the translation control is
        1,000 shifts per field. The count also sets the smallest P value the control
        can report, because the two-sided empirical P value of :func:`enrichment` never
        falls below ``2 / (n_draws + 1)``: 1,000 shifts floor it just below 0.002, 200
        shifts at about 0.01.
    min_length_um, max_length_um : float
        Smallest and largest displacement length in micrometres, inclusive. These
        depend on the field size and on the spatial scale of the structures, so no
        default is offered.
    pixel_size_um : float
        Pixel size in micrometres per pixel (square pixels).
    scheme : {'polar', 'annulus_area', 'lattice'}, optional
        How a displacement is drawn. ``'polar'``: length uniform in the range, direction
        uniform on the circle. ``'annulus_area'``: uniform over the area of the annulus
        between the two lengths (density proportional to the length). ``'lattice'``:
        uniform over the whole-pixel displacements inside the annulus.
    rotations : bool, optional
        When true, each draw also gets a rotation of 0, 90, 180 or 270 degrees with
        equal probability.
    seed : int or Generator, optional
        Seed or generator; a fixed seed reproduces the draws exactly.

    Returns
    -------
    DataFrame
        One row per draw: ``draw``, ``dy_px``, ``dx_px`` (whole pixels), ``rot90``
        (number of quarter turns) and ``length_um`` (um).
    """
    n = int(n_draws)
    if n <= 0:
        raise ValueError("n_draws must be positive")
    min_px = um_to_px(min_length_um, pixel_size_um)
    max_px = um_to_px(max_length_um, pixel_size_um)
    if min_px < 0 or max_px < min_px:
        raise ValueError("need 0 <= min_length_um <= max_length_um")
    rng = default_rng(seed)

    def block(size: int) -> tuple[np.ndarray, np.ndarray]:
        if scheme == "lattice":
            m = int(np.floor(max_px))
            return (
                rng.integers(-m, m + 1, size).astype(np.int64),
                rng.integers(-m, m + 1, size).astype(np.int64),
            )
        if scheme == "polar":
            length = rng.uniform(min_px, max_px, size)
        elif scheme == "annulus_area":
            length = np.sqrt(rng.uniform(min_px ** 2, max_px ** 2, size))
        else:
            raise ValueError("scheme must be 'polar', 'annulus_area' or 'lattice'")
        angle = rng.uniform(0.0, 2.0 * np.pi, size)
        return (
            np.rint(length * np.sin(angle)).astype(np.int64),
            np.rint(length * np.cos(angle)).astype(np.int64),
        )

    dy_kept: list[np.ndarray] = []
    dx_kept: list[np.ndarray] = []
    found = 0
    size = max(n, 64)
    for _ in range(200):
        if found >= n:
            break
        dy, dx = block(size)
        length = np.hypot(dy, dx)
        keep = (length >= min_px) & (length <= max_px) & (length > 0)
        if keep.any():
            dy_kept.append(dy[keep])
            dx_kept.append(dx[keep])
            found += int(keep.sum())
    if found < n:
        raise ValueError(
            "could not draw displacements in this length range: no whole-pixel "
            "displacement of that length exists, widen the range"
        )
    dy = np.concatenate(dy_kept)[:n]
    dx = np.concatenate(dx_kept)[:n]
    rot = rng.integers(0, 4, n).astype(np.int64) if rotations else np.zeros(n, dtype=np.int64)
    return pd.DataFrame(
        {
            "draw": np.arange(n, dtype=np.int64),
            "dy_px": dy,
            "dx_px": dx,
            "rot90": rot,
            "length_um": np.hypot(dy, dx) * float(pixel_size_um),
        }
    )


def _displacement_rows(displacements) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read (dy, dx, rot90) from a DataFrame or an (n, 2) / (n, 3) array."""
    if isinstance(displacements, pd.DataFrame):
        dy = displacements["dy_px"].to_numpy(dtype=np.int64)
        dx = displacements["dx_px"].to_numpy(dtype=np.int64)
        rot = (
            displacements["rot90"].to_numpy(dtype=np.int64)
            if "rot90" in displacements
            else np.zeros(dy.size, dtype=np.int64)
        )
        return dy, dx, rot
    arr = np.atleast_2d(np.asarray(displacements, dtype=np.int64))
    if arr.shape[1] == 2:
        return arr[:, 0], arr[:, 1], np.zeros(arr.shape[0], dtype=np.int64)
    if arr.shape[1] == 3:
        return arr[:, 0], arr[:, 1], arr[:, 2]
    raise ValueError("displacements must have 2 (dy, dx) or 3 (dy, dx, rot90) columns")


def translation_control(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    displacements,
    pixel_size_um: float,
    *,
    min_shared_frac: float = 0.10,
    connectivity: int = 8,
    mode: str = "wrap",
    valid: np.ndarray | None = None,
) -> pd.DataFrame:
    """Recount the pairs after each translation of the first mask.

    The label image of the first mask is translated relative to the second mask and the
    pairs are recounted with the same pairing rule, which gives the distribution of
    pair counts expected from the object densities alone.

    Parameters
    ----------
    mask_a, mask_b : ndarray
        The two masks of one field, same shape; the first one is the one that moves.
    displacements : DataFrame or array-like
        The shifts to apply: a table from :func:`draw_displacements`, or an array with
        columns (dy_px, dx_px) or (dy_px, dx_px, rot90). Their number is the number of
        control draws, and so sets the smallest P value :func:`enrichment` can report
        from this control (``2 / (n_draws + 1)``).
    pixel_size_um : float
        Pixel size in micrometres per pixel (square pixels).
    min_shared_frac : float, optional
        Pairing rule, as in :func:`pair_objects`.
    connectivity : {4, 8}, optional
        Connectivity for labelling, as in :func:`pair_objects`.
    mode : {'wrap', 'crop'}, optional
        Displacement scheme, see :func:`translate_labels`. In ``'wrap'`` the moved
        objects keep their number, area and shape.
    valid : ndarray of bool, optional
        Region that counts as field area for the density column.

    Returns
    -------
    DataFrame
        One row per draw: ``draw``, ``dy_px``, ``dx_px``, ``rot90``,
        ``n_objects_a`` (objects of the moved mask still present in the frame),
        ``n_pairs``, ``n_partnered_a``, ``n_partnered_b``, ``frac_with_partner_a``,
        ``frac_with_partner_b``, ``pair_density_per_mm2`` (pairs mm^-2) and
        ``retained_frac_a`` (fraction of the moved mask's area still in the frame,
        1 in the wrap-around scheme).
    """
    a = as_bool_mask(mask_a, "mask_a")
    b = as_bool_mask(mask_b, "mask_b")
    check_same_shape(a, b)
    if not 0.0 < float(min_shared_frac) <= 1.0:
        raise ValueError("min_shared_frac must lie in (0, 1]")
    labels_a, n_a, areas_a, _ = label_objects(a, connectivity)
    labels_b, n_b, areas_b, _ = label_objects(b, connectivity)
    area_mm2 = _area_mm2(a.shape, pixel_size_um, valid)
    area_a_px = float(areas_a[1:].sum())
    dy_all, dx_all, rot_all = _displacement_rows(displacements)

    rows = []
    for draw, (dy, dx, rot) in enumerate(zip(dy_all, dx_all, rot_all)):
        moved = translate_labels(labels_a, int(dy), int(dx), int(rot), mode=mode)
        areas_moved = np.bincount(moved.ravel(), minlength=n_a + 1)
        pairs = _pairs_from_labels(moved, areas_moved, labels_b, areas_b, float(min_shared_frac))
        present_a = int((areas_moved[1:] > 0).sum())
        part_a = int(np.unique(pairs[:, 0]).size)
        part_b = int(np.unique(pairs[:, 1]).size)
        rows.append(
            {
                "draw": int(draw),
                "dy_px": int(dy),
                "dx_px": int(dx),
                "rot90": int(rot),
                "n_objects_a": present_a,
                "n_pairs": int(pairs.shape[0]),
                "n_partnered_a": part_a,
                "n_partnered_b": part_b,
                "frac_with_partner_a": part_a / present_a if present_a else np.nan,
                "frac_with_partner_b": part_b / n_b if n_b else np.nan,
                "pair_density_per_mm2": pairs.shape[0] / area_mm2 if area_mm2 > 0 else np.nan,
                "retained_frac_a": (
                    float(areas_moved[1:].sum()) / area_a_px if area_a_px > 0 else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


_BOOKKEEPING = ("draw", "dy_px", "dx_px", "rot90", "length_um")


def control_summary(
    control: pd.DataFrame,
    value_cols: Sequence[str] | None = None,
    percentiles: tuple[float, float] = (2.5, 97.5),
) -> pd.DataFrame:
    """Mean, standard deviation and percentile band of a control distribution.

    Parameters
    ----------
    control : DataFrame
        Per-draw table, for instance from :func:`translation_control`.
    value_cols : sequence of str, optional
        Columns to summarise. By default every numeric column that is not draw
        bookkeeping (``draw``, ``dy_px``, ``dx_px``, ``rot90``, ``length_um``).
    percentiles : (float, float), optional
        The two percentiles of the band, in per cent. The default (2.5, 97.5) is the
        central 95 % of the draws, with linear interpolation between order statistics.

    Returns
    -------
    DataFrame
        One row with ``n_draws``, the two band percentiles, and for every summarised
        column ``<col>_mean``, ``<col>_sd`` (sample s.d., ddof = 1), ``<col>_lo`` and
        ``<col>_hi``.
    """
    if len(percentiles) != 2 or percentiles[0] > percentiles[1]:
        raise ValueError("percentiles must be two increasing values in per cent")
    if value_cols is None:
        value_cols = [
            c
            for c in control.columns
            if c not in _BOOKKEEPING and pd.api.types.is_numeric_dtype(control[c])
        ]
    out: dict[str, list] = {
        "n_draws": [int(len(control))],
        "band_lo_percentile": [float(percentiles[0])],
        "band_hi_percentile": [float(percentiles[1])],
    }
    for col in value_cols:
        v = pd.to_numeric(control[col], errors="coerce").to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            lo = hi = mean = sd = np.nan
        else:
            mean = float(v.mean())
            sd = float(v.std(ddof=1)) if v.size > 1 else np.nan
            lo, hi = (float(x) for x in np.percentile(v, list(percentiles)))
        out[f"{col}_mean"] = [mean]
        out[f"{col}_sd"] = [sd]
        out[f"{col}_lo"] = [lo]
        out[f"{col}_hi"] = [hi]
    return pd.DataFrame(out)


def enrichment(
    observed: float,
    control: Iterable[float],
    percentiles: tuple[float, float] = (2.5, 97.5),
) -> dict:
    """Enrichment of an observed count over its translation controls.

    Enrichment is the observed value divided by the mean of the controls. The two-sided
    empirical P value doubles the smaller tail of the control distribution, each tail
    counted with the observed value added to it::

        p = min(1, 2 * min((1 + #{c >= observed}), (1 + #{c <= observed})) / (n + 1))

    so that P is never zero and never below 2 / (n + 1). That bound is the smallest P
    value this control can report, and it is a property of the number of draws, not of
    the data: a P value equal to it says that no control draw reached the observed
    value, and nothing more. Reporting a smaller one needs more draws
    (see :func:`draw_displacements`).

    Parameters
    ----------
    observed : float
        Observed count, density (mm^-2) or fraction.
    control : iterable of float
        The control values, one per draw, in the same unit as ``observed``.
    percentiles : (float, float), optional
        Percentiles of the reported band, in per cent; default central 95 %.

    Returns
    -------
    dict
        ``observed``, ``control_mean``, ``control_sd`` (ddof = 1), ``band_lo``,
        ``band_hi``, ``enrichment`` (dimensionless; infinite when the controls are all
        zero and the observed value is not), ``p_value``, ``n_draws``,
        ``n_control_ge`` and ``n_control_le``.
    """
    if len(percentiles) != 2 or percentiles[0] > percentiles[1]:
        raise ValueError("percentiles must be two increasing values in per cent")
    obs = float(observed)
    draws = np.asarray(list(control), dtype=float)
    draws = draws[np.isfinite(draws)]
    n = int(draws.size)
    if n == 0:
        raise ValueError("control holds no finite value")
    mean = float(draws.mean())
    n_ge = int((draws >= obs).sum())
    n_le = int((draws <= obs).sum())
    p = min(1.0, 2.0 * min(1 + n_ge, 1 + n_le) / (n + 1))
    if mean == 0.0:
        ratio = np.nan if obs == 0.0 else np.inf
    else:
        ratio = obs / mean
    lo, hi = (float(x) for x in np.percentile(draws, list(percentiles)))
    return {
        "observed": obs,
        "control_mean": mean,
        "control_sd": float(draws.std(ddof=1)) if n > 1 else np.nan,
        "band_lo": lo,
        "band_hi": hi,
        "enrichment": float(ratio),
        "p_value": float(p),
        "n_draws": n,
        "n_control_ge": n_ge,
        "n_control_le": n_le,
    }


# --------------------------------------------------------------------------- grouping

def _by_list(by) -> list[str]:
    return [by] if isinstance(by, str) else list(by)


def _need(frame: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")


def aggregate_counts(
    fields: pd.DataFrame,
    by,
    count_cols: Sequence[str],
    control_cols: Sequence[str] = (),
    area_col: str | None = None,
) -> pd.DataFrame:
    """Sum per-field counts to any grouping level and form densities and enrichments.

    Counts are summed inside the group before anything is divided, so every result is
    count-weighted and not an average of per-field values. A group's control
    expectation is the sum of the per-field control means, taken over the summed area.

    Parameters
    ----------
    fields : DataFrame
        One row per field, with the grouping columns, the counts, the per-field control
        means and, optionally, the field area.
    by : str or sequence of str
        Grouping columns, for instance the field, region, side or subject column.
    count_cols : sequence of str
        Observed count columns to sum.
    control_cols : sequence of str, optional
        Control-mean columns, one per observed column in the same order. When given,
        ``<count>_enrichment`` = sum(count) / sum(control) is added.
    area_col : str, optional
        Column holding the area of each field in mm^2. When given, the summed area and
        ``<col>_per_mm2`` for every count and control column are added.

    Returns
    -------
    DataFrame
        One row per group: the grouping columns, ``n_fields``, the summed counts and
        controls, the summed area, the densities (mm^-2) and the enrichments.
    """
    keys = _by_list(by)
    count_cols = list(count_cols)
    control_cols = list(control_cols)
    if control_cols and len(control_cols) != len(count_cols):
        raise ValueError("control_cols must be empty or as long as count_cols")
    sum_cols = count_cols + control_cols + ([area_col] if area_col else [])
    _need(fields, keys + sum_cols)

    grouped = fields.groupby(keys, dropna=False, sort=True)
    out = grouped[sum_cols].sum(min_count=1).reset_index()
    out.insert(len(keys), "n_fields", grouped.size().reset_index(drop=True).to_numpy())
    for count, ctl in zip(count_cols, control_cols):
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"{count}_enrichment"] = out[count].to_numpy(dtype=float) / out[ctl].to_numpy(
                dtype=float
            )
    if area_col:
        area = out[area_col].to_numpy(dtype=float)
        for col in count_cols + control_cols:
            with np.errstate(divide="ignore", invalid="ignore"):
                out[f"{col}_per_mm2"] = out[col].to_numpy(dtype=float) / area
    return out


def aggregate_fractions(
    fields: pd.DataFrame,
    by,
    fractions: Mapping[str, tuple[str, str]],
    percent: bool = False,
) -> pd.DataFrame:
    """Count-weighted fractions at any grouping level.

    Inside each group the numerator and the denominator counts are summed first and
    divided afterwards, which is the count-weighted fraction and not the mean of the
    per-field fractions.

    Parameters
    ----------
    fields : DataFrame
        One row per field.
    by : str or sequence of str
        Grouping columns.
    fractions : mapping
        ``{output column: (numerator column, denominator column)}``.
    percent : bool, optional
        Report 100 x the fraction instead of the fraction.

    Returns
    -------
    DataFrame
        One row per group: the grouping columns, ``n_fields``, the summed numerators and
        denominators, and one column per requested fraction (NaN where the summed
        denominator is zero).
    """
    keys = _by_list(by)
    cols: list[str] = []
    for num, den in fractions.values():
        for c in (num, den):
            if c not in cols:
                cols.append(c)
    _need(fields, keys + cols)

    grouped = fields.groupby(keys, dropna=False, sort=True)
    out = grouped[cols].sum(min_count=1).reset_index()
    out.insert(len(keys), "n_fields", grouped.size().reset_index(drop=True).to_numpy())
    scale = 100.0 if percent else 1.0
    for name, (num, den) in fractions.items():
        d = out[den].to_numpy(dtype=float)
        n = out[num].to_numpy(dtype=float)
        out[name] = np.where(d > 0, scale * n / np.where(d > 0, d, 1.0), np.nan)
    return out
