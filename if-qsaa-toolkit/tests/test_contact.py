"""Tests for if_qsaa.contact on synthetic fields with known ground truth.

Every image is built here from plain geometry, so the expected masks, object counts,
contact flags, distances and enrichments are known before the code runs. Randomness is
always seeded; nothing is read from disk.
"""

from __future__ import annotations


import numpy as np
import pandas as pd
import pytest


from if_qsaa import contact as ct
from if_qsaa._util import label_mask


# --------------------------------------------------------------------------- builders

def disc_mask(shape, centre, radius_px):
    """Boolean disc of the given radius in pixels."""
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    return (rows - centre[0]) ** 2 + (cols - centre[1]) ** 2 <= radius_px**2


def stripe_mask(shape, starts, width_px):
    """Boolean mask of vertical stripes starting at the given columns."""
    mask = np.zeros(shape, dtype=bool)
    for start in starts:
        mask[:, start : start + width_px] = True
    return mask


# Geometry of the end-to-end synthetic field. Lengths in pixels unless stated.
SIDE = 120
PIXEL_SIZE_UM = 0.5
STRIPE_STARTS = (10, 50, 90)
STRIPE_WIDTH = 3
OBJECT_RADIUS = 3
# Objects two pixels away from a stripe: the 2-um ring reaches the stripe.
CONTACT_CENTRES = ((20, 18), (40, 58), (60, 98), (80, 18), (30, 58), (70, 98))
# Objects in the middle of a gap: the ring reaches no stripe.
FAR_CENTRES = ((20, 30), (40, 70), (60, 110), (80, 30), (30, 70), (70, 110))
VOID_CENTRE = (108, 60)
VOID_RADIUS = 12


def synthetic_field(seed=11, with_void=True):
    """Three-channel field with marker stripes, readout discs and one dark void.

    Returns the field and the ground truth: the object centres in contact, the object
    centres out of contact and the void mask.
    """
    shape = (SIDE, SIDE)
    rng = np.random.default_rng(seed)
    marker_truth = stripe_mask(shape, STRIPE_STARTS, STRIPE_WIDTH)
    objects_truth = np.zeros(shape, dtype=bool)
    for centre in CONTACT_CENTRES + FAR_CENTRES:
        objects_truth |= disc_mask(shape, centre, OBJECT_RADIUS)
    void_truth = (
        disc_mask(shape, VOID_CENTRE, VOID_RADIUS)
        if with_void
        else np.zeros(shape, dtype=bool)
    )
    noise = rng.integers(-1, 2, size=(3,) + shape).astype(float)
    readout = 10.0 + 190.0 * objects_truth + noise[0]
    marker = 12.0 + 188.0 * marker_truth + noise[1]
    nuclear = 20.0 + noise[2]
    for plane in (readout, marker, nuclear):
        plane[void_truth] = 0.0
    field = ct.ContactField(
        readout=readout,
        marker=marker,
        nuclear=nuclear,
        pixel_size_um=PIXEL_SIZE_UM,
    )
    return field, void_truth


def analysis_parameters(**overrides):
    """Parameters for the synthetic field: the required ones plus a few test choices."""
    params = ct.default_contact_parameters()
    params.update(
        background_radius_um=10.0,
        readout_floor_grey=50.0,
        marker_k=3.0,
        marker_fill_hole_um2=5.0,
        marker_min_area_um2=1.0,
        saturation_level=255.0,
        bright_min_level=50.0,
        min_source_range=10.0,
        gate_min_contrast=1.0,
        n_randomisations=200,
        seed=5,
    )
    params.update(overrides)
    return params


# --------------------------------------------------------------------------- tissue

def test_tissue_void_mask_finds_a_dark_hole_and_ignores_specks():
    shape = (80, 80)
    rng = np.random.default_rng(0)
    total = 300.0 + rng.normal(0.0, 5.0, shape)    # textured tissue background
    hole = disc_mask(shape, (40, 40), 10)          # 317 px, well above the area floor
    speck = disc_mask(shape, (10, 70), 2)          # 13 px, below it
    total[hole | speck] = 10.0
    tissue, void = ct.tissue_void_mask(
        total, 0.5, smoothing_um=0.5, min_void_area_um2=30.0, dilate_um=0.0
    )
    assert void[hole].all()
    assert not void[speck].any()
    assert np.array_equal(tissue, ~void)
    # the void stays local: the rest of the field is tissue
    assert tissue.mean() > 0.9


def test_tissue_void_mask_dilates_and_respects_valid():
    shape = (60, 60)
    rng = np.random.default_rng(1)
    total = 300.0 + rng.normal(0.0, 5.0, shape)
    hole = disc_mask(shape, (30, 30), 9)
    total[hole] = 0.0
    _, undilated = ct.tissue_void_mask(
        total, 1.0, min_void_area_um2=10.0, dilate_um=0.0
    )
    _, dilated = ct.tissue_void_mask(
        total, 1.0, min_void_area_um2=10.0, dilate_um=2.0
    )
    assert dilated.sum() > undilated.sum()
    # a pixel outside the valid mask is never called void
    valid = np.ones(shape, dtype=bool)
    valid[30, 30] = False
    _, masked = ct.tissue_void_mask(
        total, 1.0, min_void_area_um2=10.0, dilate_um=0.0, valid=valid
    )
    assert not masked[30, 30]


# --------------------------------------------------------------------------- marker mask

def test_marker_territory_mask_closes_a_gap_and_drops_specks():
    plane = np.zeros((40, 40))
    plane[10:16, 10:16] = 100.0
    plane[10:16, 17:23] = 100.0                    # one-pixel gap between the blocks
    plane[30, 30] = 100.0                          # single-pixel speck
    closed = ct.marker_territory_mask(
        plane, 50.0, 1.0, fill_hole_um2=0.0, min_area_um2=2.0, closing_px=2.0
    )
    _, n_closed = label_mask(closed, connectivity=8)
    assert n_closed == 1                           # the two blocks became one territory
    assert not closed[30, 30]                      # the speck is below the area floor
    open_gap = ct.marker_territory_mask(
        plane, 50.0, 1.0, fill_hole_um2=0.0, min_area_um2=2.0, closing_px=0.0
    )
    _, n_open = label_mask(open_gap, connectivity=8)
    assert n_open == 2


def test_marker_territory_mask_fills_only_small_holes():
    plane = np.zeros((40, 40))
    plane[5:16, 5:16] = 100.0
    plane[10, 10] = 0.0                            # pinhole, 1 px
    plane[20:31, 20:31] = 100.0
    plane[24:27, 24:27] = 0.0                      # large hole, 9 px
    mask = ct.marker_territory_mask(
        plane, 50.0, 1.0, fill_hole_um2=4.0, min_area_um2=1.0, closing_px=0.0
    )
    assert mask[10, 10]                            # 1 px <= 4 px, filled
    assert not mask[25, 25]                        # 9 px > 4 px, kept open


def test_segment_marker_territory_recovers_a_known_blob():
    shape = (64, 64)
    rng = np.random.default_rng(1)
    blob = np.zeros(shape, dtype=bool)
    blob[22:42, 22:42] = True                      # 400 px, 100 um^2 at 0.5 um/px
    channel = 15.0 + 185.0 * blob + rng.normal(0.0, 2.0, shape)
    territory = ct.segment_marker_territory(
        channel,
        0.5,
        background_radius_um=20.0,                  # a ball larger than the structure
        noise_k=3.0,
        fill_hole_um2=1.0,
        min_area_um2=1.0,
        closing_px=0.0,
    )
    assert territory.threshold > territory.background_mean
    assert (territory.mask & blob).sum() == blob.sum()     # the whole blob is found
    # the mask also holds the smoothing skirt of the edge, but not much more
    assert territory.mask.sum() < 1.6 * blob.sum()
    assert territory.area_fraction == pytest.approx(territory.mask.mean())


# --------------------------------------------------------------------------- objects

def test_segment_readout_objects_recovers_three_discs():
    shape = (60, 60)
    plane = np.zeros(shape)
    centres = ((15, 15), (15, 45), (45, 30))
    truth = np.zeros(shape, dtype=bool)
    for centre in centres:
        truth |= disc_mask(shape, centre, 4)
    plane[truth] = 100.0
    labels = ct.segment_readout_objects(plane, 50.0, 1.0, min_area_um2=4.0)
    assert labels.max() == 3
    areas = np.bincount(labels.ravel())[1:]
    assert sorted(areas) == [disc_mask(shape, centres[0], 4).sum()] * 3


def test_segment_readout_objects_area_floor_uses_4_connected_components():
    plane = np.zeros((20, 20))
    plane[5, 5] = 100.0
    plane[6, 6] = 100.0                            # touches the first only at a corner
    plane[10:14, 10:14] = 100.0                    # 16 px, survives
    labels = ct.segment_readout_objects(plane, 50.0, 1.0, min_area_um2=2.0)
    assert labels.max() == 1
    assert labels[5, 5] == 0 and labels[6, 6] == 0
    assert labels[11, 11] == 1


def test_segment_readout_objects_fills_pinholes_but_keeps_large_holes():
    plane = np.zeros((40, 40))
    plane[5:16, 5:16] = 100.0
    plane[10, 10] = 0.0                            # pinhole
    annulus = disc_mask((40, 40), (28, 28), 7) & ~disc_mask((40, 40), (28, 28), 4)
    plane[annulus] = 100.0                         # ring with a 49-px lumen
    labels = ct.segment_readout_objects(
        plane, 50.0, 1.0, min_area_um2=4.0, pinhole_area_um2=1.0
    )
    assert labels[10, 10] > 0                      # 1 px hole filled
    assert labels[28, 28] == 0                     # the lumen of the ring stays empty


def test_segment_readout_objects_can_split_touching_objects():
    shape = (40, 40)
    plane = np.zeros(shape)
    pair = disc_mask(shape, (20, 16), 5) | disc_mask(shape, (20, 24), 5)
    plane[pair] = 100.0
    plain = ct.segment_readout_objects(plane, 50.0, 1.0, min_area_um2=4.0)
    assert plain.max() == 1                        # the two discs touch, so one object
    split = ct.segment_readout_objects(
        plane, 50.0, 1.0, min_area_um2=4.0, split=True, peak_distance_um=4.0
    )
    assert split.max() == 2
    assert sorted(np.unique(split)) == [0, 1, 2]   # the survivors are renumbered
    assert (split > 0).sum() == pair.sum()         # splitting loses no pixel


def test_segment_readout_objects_keeps_an_object_at_the_border():
    plane = np.zeros((30, 30))
    plane[0:4, 0:4] = 100.0
    labels = ct.segment_readout_objects(plane, 50.0, 1.0, min_area_um2=4.0)
    assert labels.max() == 1
    assert labels[0, 0] == 1


def test_segment_readout_objects_empty_plane_gives_no_labels():
    labels = ct.segment_readout_objects(np.zeros((20, 20)), 1.0, 1.0)
    assert labels.max() == 0
    pixels = ct.object_pixels(labels)
    assert pixels.n_objects == 0
    assert pixels.rows.size == 0
    rings = ct.object_rings(labels, 1.0)
    assert rings.n_objects == 0
    assert ct.ring_coverage(rings, np.zeros((20, 20), dtype=bool)).size == 0


def test_object_centroids_weighted_and_unweighted():
    labels = np.zeros((5, 5), dtype=np.int32)
    labels[0, 0] = 1
    labels[0, 2] = 1
    weights = np.zeros((5, 5))
    weights[0, 0] = 1.0
    weights[0, 2] = 3.0
    plain = ct.object_centroids(labels)
    weighted = ct.object_centroids(labels, weights=weights)
    assert plain[0] == pytest.approx([0.0, 1.0])
    assert weighted[0] == pytest.approx([0.0, 1.5])
    # an object without weight falls back to its unweighted centroid
    fallback = ct.object_centroids(labels, weights=np.zeros((5, 5)))
    assert fallback[0] == pytest.approx([0.0, 1.0])


# --------------------------------------------------------------------------- distance

def test_signed_distance_um_half_pixel_convention():
    mask = np.zeros((11, 11), dtype=bool)
    mask[4:7, 4:7] = True
    signed = ct.signed_distance_um(mask, 2.0)
    assert signed[5, 5] == pytest.approx(-1.5 * 2.0)   # one pixel inside the edge
    assert signed[4, 4] == pytest.approx(-0.5 * 2.0)   # outermost mask pixel
    assert signed[3, 4] == pytest.approx(0.5 * 2.0)    # first pixel outside
    assert signed.dtype == np.float32


def test_distance_classes_and_ecdf():
    classes = ct.distance_classes([-3.0, 0.0, 1.0, 2.0, 2.5], apposed_um=2.0)
    assert list(classes) == [0, 0, 1, 1, 2]
    grid = np.array([-1.0, 0.0, 1.0, 2.0])
    ecdf = ct.ecdf_on_grid([-0.5, 0.5, 1.5, 1.5], grid)
    assert ecdf == pytest.approx([0.0, 0.25, 0.5, 1.0])
    assert np.isnan(ct.ecdf_on_grid([], grid)).all()


def test_distance_ecdf_grid_includes_both_ends():
    grid = ct.distance_ecdf_grid(-10.0, 30.0, 0.25)
    assert grid.size == 161
    assert grid[0] == -10.0 and grid[-1] == 30.0
    # the window of the whole-curve test cuts the grid exactly
    assert ((grid >= -5.0) & (grid <= 30.0)).sum() == 141


def test_distance_at_points_rounds_and_clips():
    values = np.arange(25.0).reshape(5, 5)
    read = ct.distance_at_points(values, np.array([[0.4, 0.4], [-3.0, 99.0]]))
    assert read[0] == values[0, 0]
    assert read[1] == values[0, 4]


# --------------------------------------------------------------------------- contact

def test_contact_ring_coverage_on_known_geometry():
    shape = (21, 21)
    labels = np.zeros(shape, dtype=np.int32)
    labels[10, 10] = 1
    marker = np.zeros(shape, dtype=bool)
    marker[10, 12] = True                          # one ring pixel of twelve
    coverage, contact, rings = ct.contact_ring_coverage(
        labels, marker, 1.0, ring_um=2.0, min_cover=0.05
    )
    assert rings.counts[0] == 12                   # disk of radius 2 minus the centre
    assert coverage[0] == pytest.approx(1 / 12)
    assert contact[0]
    # the object's own footprint is never counted
    shadow = np.zeros(shape, dtype=bool)
    shadow[10, 10] = True
    coverage_shadow, contact_shadow, _ = ct.contact_ring_coverage(
        labels, shadow, 1.0, ring_um=2.0
    )
    assert coverage_shadow[0] == 0.0
    assert not contact_shadow[0]


def test_contact_rule_is_inclusive_at_the_threshold():
    shape = (21, 21)
    labels = np.zeros(shape, dtype=np.int32)
    labels[10, 10] = 1
    marker = np.zeros(shape, dtype=bool)
    marker[10, 12] = True
    coverage, _, _ = ct.contact_ring_coverage(labels, marker, 1.0, ring_um=2.0)
    exact = float(coverage[0])
    _, at_threshold, _ = ct.contact_ring_coverage(
        labels, marker, 1.0, ring_um=2.0, min_cover=exact
    )
    _, above_threshold, _ = ct.contact_ring_coverage(
        labels, marker, 1.0, ring_um=2.0, min_cover=exact + 1e-9
    )
    assert at_threshold[0]
    assert not above_threshold[0]


def test_object_rings_truncate_at_the_frame_edge():
    shape = (21, 21)
    labels = np.zeros(shape, dtype=np.int32)
    labels[0, 0] = 1
    labels[10, 10] = 2
    rings = ct.object_rings(labels, 1.0, ring_um=2.0)
    assert rings.counts[0] < rings.counts[1]
    assert rings.rows.min() >= 0 and rings.cols.min() >= 0


def test_pooled_coverage_pools_over_pixels_not_objects():
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[0, 0:4] = 1                             # four pixels, all on the mask
    labels[5, 5] = 2                               # one pixel, off the mask
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, 0:4] = True
    pixels = ct.object_pixels(labels)
    assert ct.pooled_coverage(pixels, mask) == pytest.approx(4 / 5)
    assert ct.ring_coverage(pixels, mask) == pytest.approx([1.0, 0.0])
    select = np.array([False, True])
    assert ct.pooled_coverage(pixels, mask, select=select) == 0.0


# --------------------------------------------------------------------------- transforms

def test_rigid_transform_four_turns_return_to_the_start():
    size = 9
    points = np.array([[1.0, 2.0], [7.0, 3.0], [4.0, 4.0]])
    moved = points
    for _ in range(4):
        moved = ct.rigid_transform(moved, 1, 0, 0, size)
    assert moved == pytest.approx(points)
    assert ct.rigid_transform(points, 2, 0, 0, size) == pytest.approx(
        ct.rigid_transform(ct.rigid_transform(points, 1, 0, 0, size), 1, 0, 0, size)
    )


def test_rigid_transform_preserves_shape_and_wraps():
    size = 20
    points = np.array([[2, 3], [2, 6], [5, 3]])
    moved = ct.rigid_transform(points, 3, 7, 13, size)
    def pairwise(arr):
        diff = arr[:, None, :] - arr[None, :, :]
        return np.sort(np.hypot(diff[..., 0], diff[..., 1]).ravel())
    assert pairwise(moved) == pytest.approx(pairwise(points))
    assert moved.min() >= 0 and moved.max() < size
    # a pure translation wraps around the frame
    wrapped = ct.rigid_transform(np.array([[19, 19]]), 0, 1, 1, size)
    assert list(wrapped[0]) == [0, 0]


def test_draw_rigid_transforms_is_seeded_and_respects_the_minimum_shift():
    size, count, limit = 60, 400, 20.0
    first = ct.draw_rigid_transforms(size, count, min_shift_px=limit, rng=4)
    again = ct.draw_rigid_transforms(size, count, min_shift_px=limit, rng=4)
    other = ct.draw_rigid_transforms(size, count, min_shift_px=limit, rng=5)
    for a, b in zip(first, again):
        assert np.array_equal(a, b)
    assert not np.array_equal(first[1], other[1])
    turns, rows, cols = first
    assert set(np.unique(turns)) <= {0, 1, 2, 3}
    unrotated = turns == 0
    shift = np.hypot(
        np.minimum(rows[unrotated], size - rows[unrotated]),
        np.minimum(cols[unrotated], size - cols[unrotated]),
    )
    assert (shift >= limit).all()
    # a rotated draw may sit anywhere, including near the identity in translation
    assert rows.max() < size and cols.max() < size


def test_draw_rigid_transforms_rejects_impossible_requests():
    with pytest.raises(ValueError):
        ct.draw_rigid_transforms(20, 10, min_shift_px=100.0)
    with pytest.raises(ValueError):
        ct.draw_rigid_transforms(20, 0)


# --------------------------------------------------------------------------- null model

# Scattered marker territories, each with one object lying against its edge.
NULL_SIDE = 96
NULL_CENTRES = (
    (10, 10), (12, 46), (40, 24), (30, 70), (58, 12), (66, 50),
    (46, 86), (80, 30), (22, 86), (84, 66), (70, 88), (90, 10),
)


def clustered_null_inputs():
    """Objects placed against scattered marker squares: every object is in contact."""
    shape = (NULL_SIDE, NULL_SIDE)
    marker = np.zeros(shape, dtype=bool)
    labels = np.zeros(shape, dtype=np.int32)
    for index, (row, col) in enumerate(NULL_CENTRES, start=1):
        marker[row - 3 : row + 4, col - 3 : col + 4] = True   # 7x7 territory
        labels[row - 1 : row + 2, col + 4 : col + 7] = index  # 3x3 object at its edge
    centroids = ct.object_centroids(labels)
    distance = ct.signed_distance_um(marker, 1.0)
    tissue = np.ones(shape, dtype=bool)
    return labels, centroids, marker, distance, tissue


def test_contact_null_model_finds_enrichment_for_a_clustered_pattern():
    labels, centroids, marker, distance, tissue = clustered_null_inputs()
    result = ct.contact_null_model(
        labels,
        centroids,
        marker,
        distance,
        tissue,
        1.0,
        n_randomisations=200,
        min_shift_um=20.0,
        seed=3,
    )
    assert result.n_objects == labels.max()
    assert result.contact_fraction == 1.0            # every object touches a stripe
    assert result.contact_count == result.n_objects
    assert result.contact_enrichment > 2.0
    assert result.contact_P == pytest.approx(1 / 201)      # the floor of 200 draws
    # the enrichment and the P value are exactly the definitions, on the stored null
    assert result.contact_enrichment == pytest.approx(
        result.contact_fraction / result.contact_null.mean()
    )
    centre = result.contact_null.mean()
    extreme = int(
        np.sum(
            np.abs(result.contact_null - centre)
            >= abs(result.contact_fraction - centre) - 1e-12
        )
    )
    assert result.contact_P == pytest.approx((extreme + 1) / (result.n_randomisations + 1))
    # every randomised contact fraction is a multiple of 1 / n
    assert np.allclose(
        result.contact_null * result.n_objects,
        np.round(result.contact_null * result.n_objects),
    )
    assert result.accepted_fraction_mean == 1.0      # the whole field is tissue
    assert result.ecdf_null.shape == (200, result.ecdf_grid_um.size)
    assert result.ecdf_null.dtype == np.float32
    assert result.envelope is not None and result.envelope.n == 200


def test_contact_null_model_is_reproducible_with_a_seed():
    inputs = clustered_null_inputs()
    first = ct.contact_null_model(*inputs, 1.0, n_randomisations=120, seed=17)
    again = ct.contact_null_model(*inputs, 1.0, n_randomisations=120, seed=17)
    other = ct.contact_null_model(*inputs, 1.0, n_randomisations=120, seed=18)
    assert np.array_equal(first.contact_null, again.contact_null)
    assert np.array_equal(first.ecdf_null, again.ecdf_null)
    assert first.contact_P == again.contact_P
    assert not np.array_equal(first.contact_null, other.contact_null)


def test_contact_null_model_marker_everywhere_and_nowhere():
    shape = (40, 40)
    labels = np.zeros(shape, dtype=np.int32)
    labels[10:13, 10:13] = 1
    labels[25:28, 25:28] = 2
    centroids = ct.object_centroids(labels)
    tissue = np.ones(shape, dtype=bool)
    full = np.ones(shape, dtype=bool)
    everywhere = ct.contact_null_model(
        labels,
        centroids,
        full,
        ct.signed_distance_um(full, 1.0),
        tissue,
        1.0,
        n_randomisations=50,
        min_shift_um=5.0,
        seed=1,
    )
    assert everywhere.contact_fraction == 1.0
    assert (everywhere.contact_null == 1.0).all()
    assert everywhere.contact_enrichment == pytest.approx(1.0)
    assert everywhere.contact_P == 1.0
    assert everywhere.contained_fraction == 1.0
    empty = np.zeros(shape, dtype=bool)
    nowhere = ct.contact_null_model(
        labels,
        centroids,
        empty,
        ct.signed_distance_um(empty, 1.0),
        tissue,
        1.0,
        n_randomisations=50,
        min_shift_um=5.0,
        seed=1,
    )
    assert nowhere.contact_fraction == 0.0
    assert (nowhere.contact_null == 0.0).all()
    assert np.isnan(nowhere.contact_enrichment)      # no ratio against a zero mean
    assert nowhere.contact_P == 1.0


def test_contact_null_model_single_object_and_classes():
    shape = (40, 40)
    marker = np.zeros(shape, dtype=bool)
    marker[15:25, 15:25] = True
    labels = np.zeros(shape, dtype=np.int32)
    labels[19:22, 19:22] = 1                         # inside the marker territory
    centroids = ct.object_centroids(labels)
    result = ct.contact_null_model(
        labels,
        centroids,
        marker,
        ct.signed_distance_um(marker, 1.0),
        np.ones(shape, dtype=bool),
        1.0,
        n_randomisations=40,
        min_shift_um=5.0,
        apposed_um=2.0,
        seed=2,
    )
    assert result.n_objects == 1
    assert result.distance_um[0] < 0                 # the centroid is inside
    assert result.contained_fraction == 1.0
    assert list(result.class_fractions) == [1.0, 0.0, 0.0]
    assert result.marker_overlap_fraction == 1.0     # all object pixels on the marker
    assert result.median_distance_um == pytest.approx(result.distance_um[0])
    assert set(np.unique(result.contact_null)) <= {0.0, 1.0}


def test_contact_null_model_keeps_objects_inside_tissue():
    labels, centroids, marker, distance, _ = clustered_null_inputs()
    tissue = np.ones(labels.shape, dtype=bool)
    tissue[:, NULL_SIDE // 2 :] = False              # half the field is void
    result = ct.contact_null_model(
        labels,
        centroids,
        marker,
        distance,
        tissue,
        1.0,
        n_randomisations=100,
        min_shift_um=20.0,
        seed=6,
    )
    assert 0.0 < result.accepted_fraction_mean < 1.0


def test_contact_null_model_takes_tissue_none_as_the_whole_field():
    # The same option segment_marker_territory offers: None means every pixel, so a
    # field without voids needs no hand-built mask of ones.
    labels, centroids, marker, distance, tissue = clustered_null_inputs()
    settings = dict(n_randomisations=60, min_shift_um=20.0, seed=5)
    spelled_out = ct.contact_null_model(
        labels, centroids, marker, distance, tissue, 1.0, **settings
    )
    implied = ct.contact_null_model(
        labels, centroids, marker, distance, None, 1.0, **settings
    )
    assert implied.accepted_fraction_mean == 1.0
    assert np.array_equal(implied.contact_null, spelled_out.contact_null)
    assert np.array_equal(implied.ecdf_null, spelled_out.ecdf_null)
    assert implied.contact_P == spelled_out.contact_P


def test_contact_null_model_can_drop_the_null_curves():
    inputs = clustered_null_inputs()
    result = ct.contact_null_model(
        *inputs, 1.0, n_randomisations=30, keep_null_ecdf=False, seed=1
    )
    assert result.ecdf_null.shape[0] == 0
    assert result.envelope is None
    assert result.contact_fraction == 1.0
    assert result.contact_null.size == 30


def test_contact_null_model_validates_its_inputs():
    shape = (20, 30)
    labels = np.zeros(shape, dtype=np.int32)
    labels[5, 5] = 1
    marker = np.zeros(shape, dtype=bool)
    args = (labels, np.array([[5.0, 5.0]]), marker, ct.signed_distance_um(marker, 1.0),
            np.ones(shape, dtype=bool), 1.0)
    with pytest.raises(ValueError):                  # the field is not square
        ct.contact_null_model(*args, n_randomisations=5)
    square = np.zeros((20, 20), dtype=np.int32)
    empty_marker = np.zeros((20, 20), dtype=bool)
    with pytest.raises(ValueError):                  # no object to analyse
        ct.contact_null_model(
            square,
            np.zeros((0, 2)),
            empty_marker,
            ct.signed_distance_um(empty_marker, 1.0),
            np.ones((20, 20), dtype=bool),
            1.0,
            n_randomisations=5,
        )
    square[5, 5] = 1
    with pytest.raises(ValueError):                  # one centroid per object
        ct.contact_null_model(
            square,
            np.zeros((2, 2)),
            empty_marker,
            ct.signed_distance_um(empty_marker, 1.0),
            np.ones((20, 20), dtype=bool),
            1.0,
            n_randomisations=5,
        )


# --------------------------------------------------------------------------- whole curve

def test_envelope_test_on_a_curve_at_the_randomisation_mean():
    grid = ct.distance_ecdf_grid(-10.0, 30.0, 0.25)
    rng = np.random.default_rng(0)
    null = np.clip(0.5 + 0.05 * rng.standard_normal((200, grid.size)), 0.0, 1.0)
    observed = null.mean(axis=0)
    result = ct.envelope_test(observed, null, grid)
    # the curves are compared in float32, so the departure is zero to that precision
    assert result.statistic == pytest.approx(0.0, abs=1e-4)
    assert result.P == 1.0                           # nothing can be less extreme
    assert result.n == 200
    assert result.n_points == 141
    assert result.window_um == (-5.0, 30.0)


def test_envelope_test_detects_a_departure_and_honours_the_window():
    grid = ct.distance_ecdf_grid(-10.0, 30.0, 0.25)
    rng = np.random.default_rng(1)
    null = 0.5 + 0.02 * rng.standard_normal((100, grid.size))
    inside = null.mean(axis=0).copy()
    inside[(grid >= 0) & (grid <= 5)] += 0.4         # a departure inside the window
    detected = ct.envelope_test(inside, null, grid)
    assert detected.statistic > 5.0
    assert detected.P == pytest.approx(1 / 101)
    outside = null.mean(axis=0).copy()
    outside[grid < -6] += 0.4                        # a departure outside the window
    ignored = ct.envelope_test(outside, null, grid)
    assert ignored.statistic == pytest.approx(0.0, abs=1e-4)
    assert ignored.P == 1.0


def test_envelope_test_ignores_grid_points_without_spread():
    grid = np.array([-5.0, 0.0, 5.0])
    null = np.array([[0.0, 0.2, 1.0], [0.0, 0.4, 1.0], [0.0, 0.6, 1.0]])
    observed = np.array([0.9, 0.4, 0.1])             # differs only where spread is zero
    result = ct.envelope_test(observed, null, grid)
    assert result.n_points == 1
    assert np.isfinite(result.statistic)
    flat = ct.envelope_test(
        np.array([0.9, 0.9, 0.9]), np.ones((3, 3)), grid
    )
    assert np.isnan(flat.statistic) and flat.P == 1.0
    with pytest.raises(ValueError):
        ct.envelope_test(observed, null, grid, window_um=(100.0, 200.0))


# --------------------------------------------------------------------------- whole field

def test_analyse_contact_field_recovers_the_known_contacts():
    field, void_truth = synthetic_field()
    analysis = ct.analyse_contact_field(field, analysis_parameters())
    assert analysis.n_objects_total == len(CONTACT_CENTRES) + len(FAR_CENTRES)
    assert analysis.n_analysed == analysis.n_objects_total
    assert analysis.n_gate_rejected == 0
    assert analysis.void.sum() > 0
    assert analysis.void[disc_mask((SIDE, SIDE), VOID_CENTRE, VOID_RADIUS - 4)].all()
    assert analysis.void[void_truth].mean() > 0.7
    assert analysis.marker_mask.sum() > 0
    table = analysis.objects
    assert len(table) == analysis.n_objects_total
    assert set(("label", "distance_um", "contact", "analysed")) <= set(table.columns)
    # every object is identified by its centre, and the contact flags are the geometry
    centres = list(zip(table["centroid_row_px"], table["centroid_col_px"]))
    expected = []
    for row, col in centres:
        nearest = min(abs(col - start) for start in STRIPE_STARTS)
        expected.append(nearest < 15)
    assert list(table["contact"].astype(bool)) == expected
    assert analysis.null is not None
    assert analysis.null.contact_fraction == pytest.approx(0.5)
    assert analysis.null.n_objects == 12
    # the readout objects lie outside the marker territory, so distances are positive
    assert (table["distance_um"] > 0).all()
    assert set(table["distance_class"]) <= set(ct.DISTANCE_CLASSES)


def test_analyse_contact_field_is_reproducible_and_honours_the_void_switch():
    field, _ = synthetic_field()
    params = analysis_parameters(n_randomisations=60)
    first = ct.analyse_contact_field(field, params)
    again = ct.analyse_contact_field(field, params)
    assert np.array_equal(first.null.contact_null, again.null.contact_null)
    assert first.null.contact_P == again.null.contact_P
    without_void = ct.analyse_contact_field(
        field, analysis_parameters(n_randomisations=60, use_void=False)
    )
    assert without_void.void.sum() == 0
    assert without_void.tissue.all()
    assert without_void.null.accepted_fraction_mean == 1.0


def test_analyse_contact_field_gate_switch_and_headline():
    field, _ = synthetic_field()
    analysis = ct.analyse_contact_field(
        field, analysis_parameters(n_randomisations=60, use_gate=False)
    )
    row = ct.contact_headline(analysis)
    assert row["n_objects"] == 12
    assert row["contact_fraction"] == pytest.approx(0.5)
    assert row["contact_enrichment"] == pytest.approx(analysis.null.contact_enrichment)
    assert row["envelope_P"] == analysis.null.envelope.P
    assert all(not isinstance(value, np.ndarray) for value in row.values())
    # no cross-talk was put into the synthetic field, so the coefficients stay at zero
    assert row["alpha"] == 0.0 and row["beta"] == 0.0 and row["gamma"] == 0.0


def test_analyse_contact_field_requires_the_dataset_specific_parameters():
    field, _ = synthetic_field()
    with pytest.raises(KeyError) as missing:
        ct.analyse_contact_field(field, ct.default_contact_parameters())
    for name in ct.required_contact_parameters():
        assert name in str(missing.value)
    with pytest.raises(KeyError):
        ct.analyse_contact_field(field, analysis_parameters(contact_ring_xm=2.0))


def test_analyse_contact_field_without_objects_gives_no_null():
    field, _ = synthetic_field()
    analysis = ct.analyse_contact_field(
        field, analysis_parameters(readout_floor_grey=250.0, n_randomisations=20)
    )
    assert analysis.n_objects_total == 0
    assert analysis.null is None
    row = ct.contact_headline(analysis)
    assert np.isnan(row["contact_fraction"])
    assert row["n_randomisations"] == 0


def test_run_contact_sensitivity_returns_one_row_per_setting():
    field, _ = synthetic_field()
    grid = {
        "contact ring 1 um": {"contact_ring_um": 1.0},
        "contact cover >= 10 %": {"contact_min_cover": 0.10},
        "no void mask": {"use_void": False},
    }
    table = ct.run_contact_sensitivity(
        field, analysis_parameters(), grid, n_randomisations=40
    )
    assert isinstance(table, pd.DataFrame)
    assert list(table["setting"]) == ["baseline"] + list(grid)
    assert table["n_randomisations"].eq(40).all()
    baseline = table.loc[0]
    narrow = table.loc[table["setting"] == "contact ring 1 um"].iloc[0]
    stricter = table.loc[table["setting"] == "contact cover >= 10 %"].iloc[0]
    # a narrower ring and a higher coverage requirement cannot add contacts
    assert narrow["contact_fraction"] <= baseline["contact_fraction"]
    assert stricter["contact_fraction"] <= baseline["contact_fraction"]
    assert table["n_objects"].eq(12).all()


def test_every_unmixing_variant_runs_in_the_sensitivity_grid():
    """The cross-talk variants of the unmixing module must run here as they are.

    A key the analysis does not know is rejected, and a variant that needs a parameter the
    defaults leave unset stops the whole grid after the earlier settings have run, so the
    whole mapping is run once, end to end, with the one parameter the footprint variants
    need supplied as a grey level.
    """
    from if_qsaa import unmixing as um

    field, _ = synthetic_field()
    variants = um.unmixing_variants()
    params = analysis_parameters(footprint_min_level=40.0)
    table = ct.run_contact_sensitivity(field, params, variants, n_randomisations=20)
    assert list(table["setting"]) == ["baseline"] + list(variants)
    assert table["n_randomisations"].eq(20).all()
