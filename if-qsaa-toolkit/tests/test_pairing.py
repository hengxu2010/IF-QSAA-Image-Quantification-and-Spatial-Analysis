"""Tests for if_qsaa.pairing on synthetic fields with known ground truth."""

import numpy as np
import pandas as pd
import pytest

from if_qsaa import pairing

PX = 0.5  # micrometres per pixel used throughout these tests


def box(mask, y0, y1, x0, x1):
    """Set a rectangle of a boolean field."""
    mask[y0:y1, x0:x1] = True
    return mask


def label_at(labels, y, x):
    """Label of the object that covers one pixel."""
    value = int(labels[y, x])
    assert value > 0, "no object at that pixel"
    return value


@pytest.fixture
def field():
    """Two masks of one field with every pairing case built in by hand.

    Mask A objects, each 10 x 10 px: one with a large overlap, one with an overlap of
    exactly 10 % of its area, one with an overlap just below 10 %, one alone, and one
    holding two small objects of mask B.
    """
    a = np.zeros((80, 80), dtype=bool)
    b = np.zeros((80, 80), dtype=bool)
    box(a, 5, 15, 5, 15)      # large overlap
    box(b, 5, 15, 10, 20)     # shares 5 x 10 = 50 px
    box(a, 25, 35, 5, 15)     # overlap exactly at the rule
    box(b, 25, 35, 14, 24)    # shares 1 x 10 = 10 px of 100 px
    box(a, 45, 55, 5, 15)     # overlap just below the rule
    box(b, 46, 56, 14, 24)    # shares 1 x 9 = 9 px, both objects 100 px
    box(a, 65, 75, 5, 15)     # no object of B anywhere near
    box(b, 65, 75, 30, 40)    # no object of A anywhere near
    box(a, 5, 15, 40, 50)     # holds two small objects of B
    box(b, 6, 9, 41, 44)      # 9 px, wholly inside
    box(b, 11, 14, 41, 44)    # 9 px, wholly inside
    return a, b


def test_pairing_rule_and_classes(field):
    a, b = field
    res = pairing.pair_objects(a, b)
    assert (res.n_a, res.n_b) == (5, 6)

    big_a = label_at(res.labels_a, 6, 6)
    edge_a = label_at(res.labels_a, 26, 6)
    under_a = label_at(res.labels_a, 46, 6)
    alone_a = label_at(res.labels_a, 66, 6)
    host_a = label_at(res.labels_a, 6, 41)
    big_b = label_at(res.labels_b, 6, 19)
    edge_b = label_at(res.labels_b, 26, 23)
    under_b = label_at(res.labels_b, 46, 23)
    alone_b = label_at(res.labels_b, 66, 31)
    small_b1 = label_at(res.labels_b, 7, 42)
    small_b2 = label_at(res.labels_b, 12, 42)

    assert {tuple(row) for row in res.pairs} == {
        (big_a, big_b, 50),
        (edge_a, edge_b, 10),
        (host_a, small_b1, 9),
        (host_a, small_b2, 9),
    }
    assert set(res.partnered_a) == {big_a, edge_a, host_a}
    assert set(res.partnered_b) == {big_b, edge_b, small_b1, small_b2}
    assert res.n_pairs == 4
    # the pair rows are sorted by the two labels
    order = res.pairs[:, 0] * (res.n_b + 1) + res.pairs[:, 1]
    assert np.all(np.diff(order) > 0)
    # the object just below the rule stays single-positive in both masks
    assert under_a not in set(res.partnered_a)
    assert under_b not in set(res.partnered_b)

    table = pairing.classify_objects(res, PX, field_id="f")
    assert len(table) == res.n_a + res.n_b
    assert set(table["field"]) == {"f"}
    cls = {
        (m, int(lab)): c
        for m, lab, c in zip(table["mask"], table["label"], table["object_class"])
    }
    assert cls[("a", big_a)] == pairing.DUAL
    assert cls[("a", under_a)] == pairing.A_ONLY
    assert cls[("a", alone_a)] == pairing.A_ONLY
    assert cls[("b", under_b)] == pairing.B_ONLY
    assert cls[("b", alone_b)] == pairing.B_ONLY
    assert cls[("b", small_b1)] == pairing.DUAL

    host = table[(table["mask"] == "a") & (table["label"] == host_a)].iloc[0]
    assert host["n_partners"] == 2
    # both partners share the same area, so the smaller label is named
    assert int(host["partner_label"]) == min(small_b1, small_b2)
    assert host["shared_area_um2"] == pytest.approx(9 * PX * PX)
    assert host["shared_area_um2_total"] == pytest.approx(18 * PX * PX)
    assert host["area_um2"] == pytest.approx(100 * PX * PX)

    lonely = table[(table["mask"] == "a") & (table["label"] == alone_a)].iloc[0]
    assert lonely["n_partners"] == 0
    assert pd.isna(lonely["partner_label"])
    assert pd.isna(lonely["shared_area_um2"])


def test_rule_is_inclusive_and_tunable(field):
    a, b = field
    strict = pairing.pair_objects(a, b, min_shared_frac=0.11)
    assert strict.n_pairs == 3  # the overlap of exactly 10 % drops out
    loose = pairing.pair_objects(a, b, min_shared_frac=0.09)
    assert loose.n_pairs == 5  # the overlap of 9 % joins in
    with pytest.raises(ValueError):
        pairing.pair_objects(a, b, min_shared_frac=0.0)


def test_field_summary(field):
    a, b = field
    res = pairing.pair_objects(a, b)
    row = pairing.field_summary(res, PX, field_id="f").iloc[0]
    assert row["n_objects_a"] == 5 and row["n_objects_b"] == 6
    assert row["n_partnered_a"] == 3 and row["n_partnered_b"] == 4
    assert row["n_a_only"] == 2 and row["n_b_only"] == 2
    assert row["n_pairs"] == 4
    assert row["frac_with_partner_a"] == pytest.approx(3 / 5)
    assert row["frac_with_partner_b"] == pytest.approx(4 / 6)
    area_mm2 = 80 * 80 * PX * PX / 1e6
    assert row["area_mm2"] == pytest.approx(area_mm2)
    assert row["pair_density_per_mm2"] == pytest.approx(4 / area_mm2)

    valid = np.zeros_like(a)
    valid[:40] = True
    half = pairing.field_summary(res, PX, valid=valid).iloc[0]
    assert half["area_mm2"] == pytest.approx(area_mm2 / 2)


def test_connectivity_rule():
    a = np.zeros((8, 8), dtype=bool)
    a[1, 1] = a[2, 2] = True
    b = np.ones((8, 8), dtype=bool)
    assert pairing.pair_objects(a, b, connectivity=8).n_a == 1
    assert pairing.pair_objects(a, b, connectivity=4).n_a == 2


def test_empty_masks():
    empty = np.zeros((10, 10), dtype=bool)
    res = pairing.pair_objects(empty, empty)
    assert res.n_a == 0 and res.n_b == 0
    assert res.pairs.shape == (0, 3)
    table = pairing.classify_objects(res, PX)
    assert len(table) == 0
    assert "object_class" in table.columns
    row = pairing.field_summary(res, PX).iloc[0]
    assert np.isnan(row["frac_with_partner_a"])
    assert np.isnan(row["frac_with_partner_b"])
    assert row["pair_density_per_mm2"] == 0.0


def test_single_object_on_the_border():
    a = np.zeros((10, 10), dtype=bool)
    b = np.zeros((10, 10), dtype=bool)
    box(a, 0, 3, 0, 3)
    box(b, 0, 3, 0, 3)
    res = pairing.pair_objects(a, b)
    assert res.n_a == res.n_b == 1
    assert res.pairs.tolist() == [[1, 1, 9]]
    assert res.bboxes_a[0][0].start == 0 and res.bboxes_a[0][1].start == 0
    assert res.areas_a_px[1] == 9


def test_mismatched_shapes():
    with pytest.raises(ValueError):
        pairing.pair_objects(np.zeros((4, 4), bool), np.zeros((4, 5), bool))


def test_translate_labels_keeps_objects_when_wrapping():
    labels = np.zeros((12, 12), dtype=np.int32)
    labels[1:4, 1:4] = 1
    labels[9:12, 9:12] = 2  # touches the border, so it is cut by a wrap
    for dy, dx in ((3, 0), (0, -5), (7, 7)):
        moved = pairing.translate_labels(labels, dy, dx, mode="wrap")
        assert np.array_equal(
            np.bincount(moved.ravel(), minlength=3), np.bincount(labels.ravel(), minlength=3)
        )
    shifted = pairing.translate_labels(labels, 2, 1, mode="wrap")
    assert shifted[3, 2] == 1
    # a full turn around the torus is the identity
    assert np.array_equal(pairing.translate_labels(labels, 12, 12, mode="wrap"), labels)


def test_translate_labels_crop_loses_what_leaves_the_frame():
    labels = np.zeros((12, 12), dtype=np.int32)
    labels[1:4, 1:4] = 1
    moved = pairing.translate_labels(labels, -2, 0, mode="crop")
    assert int((moved == 1).sum()) == 6  # one row of three pixels is gone
    gone = pairing.translate_labels(labels, 20, 0, mode="crop")
    assert not gone.any()


def test_translate_labels_rotation():
    labels = np.zeros((6, 6), dtype=np.int32)
    labels[0, 0] = 1
    turned = pairing.translate_labels(labels, 0, 0, rot90=1, mode="wrap")
    assert turned[5, 0] == 1
    with pytest.raises(ValueError):
        pairing.translate_labels(np.zeros((4, 6), dtype=np.int32), 0, 0, rot90=1)
    with pytest.raises(ValueError):
        pairing.translate_labels(labels, 1, 1, mode="nowhere")


@pytest.mark.parametrize("scheme", ["polar", "annulus_area", "lattice"])
def test_draw_displacements_respects_the_length_range(scheme):
    draws = pairing.draw_displacements(
        200,
        min_length_um=4.0,
        max_length_um=10.0,
        pixel_size_um=PX,
        scheme=scheme,
        rotations=True,
        seed=11,
    )
    assert len(draws) == 200
    length_um = np.hypot(draws["dy_px"], draws["dx_px"]) * PX
    assert np.allclose(length_um, draws["length_um"])
    assert length_um.min() >= 4.0 - 1e-12
    assert length_um.max() <= 10.0 + 1e-12
    assert set(draws["rot90"]) <= {0, 1, 2, 3}
    assert draws["dy_px"].dtype.kind == "i"


def test_draw_displacements_is_reproducible_and_checks_its_range():
    kw = dict(min_length_um=2.0, max_length_um=5.0, pixel_size_um=PX)
    first = pairing.draw_displacements(50, seed=3, **kw)
    again = pairing.draw_displacements(50, seed=3, **kw)
    other = pairing.draw_displacements(50, seed=4, **kw)
    pd.testing.assert_frame_equal(first, again)
    assert not first[["dy_px", "dx_px"]].equals(other[["dy_px", "dx_px"]])
    assert set(pairing.draw_displacements(10, seed=1, **kw)["rot90"]) == {0}
    with pytest.raises(ValueError):
        pairing.draw_displacements(10, min_length_um=0.1, max_length_um=0.2, pixel_size_um=1.0)
    with pytest.raises(ValueError):
        pairing.draw_displacements(10, min_length_um=5.0, max_length_um=1.0, pixel_size_um=1.0)


@pytest.fixture
def one_pair_field():
    """One object per mask, lying on top of each other: one observed pair."""
    a = np.zeros((40, 40), dtype=bool)
    b = np.zeros((40, 40), dtype=bool)
    box(a, 5, 15, 5, 15)
    box(b, 5, 15, 5, 15)
    return a, b


def test_translation_control_counts_known_shifts(one_pair_field):
    a, b = one_pair_field
    observed = pairing.pair_objects(a, b)
    assert observed.n_pairs == 1
    shifts = [[20, 0, 0], [0, 20, 0], [40, 0, 0]]  # the last one wraps back home
    control = pairing.translation_control(a, b, shifts, PX, mode="wrap")
    assert control["n_pairs"].tolist() == [0, 0, 1]
    assert control["n_objects_a"].tolist() == [1, 1, 1]
    assert np.allclose(control["retained_frac_a"], 1.0)
    assert control["n_partnered_b"].tolist() == [0, 0, 1]
    area_mm2 = 40 * 40 * PX * PX / 1e6
    assert control["pair_density_per_mm2"].iloc[2] == pytest.approx(1 / area_mm2)


def test_translation_control_crop_mode_drops_objects(one_pair_field):
    a, b = one_pair_field
    control = pairing.translation_control(a, b, [[35, 0]], PX, mode="crop")
    row = control.iloc[0]
    assert row["n_objects_a"] == 0
    assert row["n_pairs"] == 0
    assert row["retained_frac_a"] == pytest.approx(0.0)
    assert np.isnan(row["frac_with_partner_a"])


def test_translation_control_with_quarter_turns():
    centred = np.zeros((40, 40), dtype=bool)
    box(centred, 15, 25, 15, 25)  # a quarter turn maps this object onto itself
    control = pairing.translation_control(
        centred, centred, [[0, 0, k] for k in range(4)], PX
    )
    assert control["n_pairs"].tolist() == [1, 1, 1, 1]
    off_centre = np.zeros((40, 40), dtype=bool)
    box(off_centre, 2, 8, 2, 8)
    turned = pairing.translation_control(off_centre, off_centre, [[0, 0, 1]], PX)
    assert turned["n_pairs"].tolist() == [0]
    assert turned["n_objects_a"].tolist() == [1]  # the object itself is untouched
    with pytest.raises(ValueError):
        pairing.translation_control(
            np.zeros((10, 12), bool), np.zeros((10, 12), bool), [[1, 1, 1]], PX
        )


def test_translation_control_is_reproducible(one_pair_field):
    a, b = one_pair_field
    kw = dict(min_length_um=5.0, max_length_um=9.0, pixel_size_um=PX)
    draws = pairing.draw_displacements(25, seed=7, **kw)
    first = pairing.translation_control(a, b, draws, PX)
    again = pairing.translation_control(
        a, b, pairing.draw_displacements(25, seed=7, **kw), PX
    )
    pd.testing.assert_frame_equal(first, again)


def test_control_summary_and_enrichment():
    control = pd.DataFrame(
        {"draw": [0, 1, 2], "dy_px": [1, 2, 3], "dx_px": [0, 0, 0], "n_pairs": [0, 0, 1]}
    )
    summary = pairing.control_summary(control).iloc[0]
    assert summary["n_draws"] == 3
    assert summary["n_pairs_mean"] == pytest.approx(1 / 3)
    assert summary["n_pairs_sd"] == pytest.approx(np.std([0, 0, 1], ddof=1))
    lo, hi = np.percentile([0, 0, 1], [2.5, 97.5])
    assert summary["n_pairs_lo"] == pytest.approx(lo)
    assert summary["n_pairs_hi"] == pytest.approx(hi)
    assert "dy_px_mean" not in summary.index  # bookkeeping columns are left alone

    out = pairing.enrichment(1.0, [0, 0, 1])
    assert out["control_mean"] == pytest.approx(1 / 3)
    assert out["enrichment"] == pytest.approx(3.0)
    assert out["n_control_ge"] == 1 and out["n_control_le"] == 3
    assert out["p_value"] == pytest.approx(1.0)

    strong = pairing.enrichment(5.0, [0.0, 0.0, 0.0, 0.0])
    assert np.isinf(strong["enrichment"])
    assert strong["p_value"] == pytest.approx(2 / 5)
    assert np.isnan(pairing.enrichment(0.0, [0.0, 0.0])["enrichment"])
    with pytest.raises(ValueError):
        pairing.enrichment(1.0, [])


def test_the_number_of_draws_sets_the_smallest_reportable_p_value():
    """Documented property of the control: P can never fall below 2 / (n_draws + 1)."""
    for n_draws in (4, 50, 200):
        floored = pairing.enrichment(1.0, np.zeros(n_draws))   # no draw reaches 1
        assert floored["n_draws"] == n_draws
        assert floored["p_value"] == pytest.approx(2.0 / (n_draws + 1))
        assert floored["p_value"] > 0.0
    # and a count reduced to save time raises that floor
    assert (pairing.enrichment(1.0, np.zeros(200))["p_value"]
            > pairing.enrichment(1.0, np.zeros(1000))["p_value"])


@pytest.fixture
def per_field_table():
    return pd.DataFrame(
        {
            "subject": ["s1", "s1", "s2"],
            "region": ["r1", "r2", "r1"],
            "n_pairs": [4, 6, 10],
            "n_pairs_control_mean": [2.0, 1.0, 4.0],
            "area_mm2": [0.5, 0.5, 1.0],
            "n_objects_b": [10, 20, 30],
            "n_partnered_b": [2, 5, 9],
        }
    )


def test_aggregate_counts(per_field_table):
    out = pairing.aggregate_counts(
        per_field_table,
        "subject",
        count_cols=["n_pairs"],
        control_cols=["n_pairs_control_mean"],
        area_col="area_mm2",
    ).set_index("subject")
    assert out.loc["s1", "n_fields"] == 2
    assert out.loc["s1", "n_pairs"] == 10
    assert out.loc["s1", "n_pairs_control_mean"] == pytest.approx(3.0)
    assert out.loc["s1", "area_mm2"] == pytest.approx(1.0)
    assert out.loc["s1", "n_pairs_enrichment"] == pytest.approx(10 / 3)
    assert out.loc["s1", "n_pairs_per_mm2"] == pytest.approx(10.0)
    assert out.loc["s1", "n_pairs_control_mean_per_mm2"] == pytest.approx(3.0)
    assert out.loc["s2", "n_pairs_enrichment"] == pytest.approx(2.5)

    two_keys = pairing.aggregate_counts(
        per_field_table, ["subject", "region"], count_cols=["n_pairs"]
    )
    assert len(two_keys) == 3
    with pytest.raises(KeyError):
        pairing.aggregate_counts(per_field_table, "subject", count_cols=["nope"])
    with pytest.raises(ValueError):
        pairing.aggregate_counts(
            per_field_table, "subject", count_cols=["n_pairs"], control_cols=["a", "b"]
        )


def test_aggregate_fractions_is_count_weighted(per_field_table):
    out = pairing.aggregate_fractions(
        per_field_table,
        "subject",
        {"frac_with_partner_b": ("n_partnered_b", "n_objects_b")},
    ).set_index("subject")
    # 7 of 30 objects, not the mean of 2/10 and 5/20
    assert out.loc["s1", "frac_with_partner_b"] == pytest.approx(7 / 30)
    assert out.loc["s2", "frac_with_partner_b"] == pytest.approx(0.3)
    percent = pairing.aggregate_fractions(
        per_field_table,
        "subject",
        {"pct": ("n_partnered_b", "n_objects_b")},
        percent=True,
    ).set_index("subject")
    assert percent.loc["s1", "pct"] == pytest.approx(100 * 7 / 30)

    empty = pd.DataFrame({"subject": ["s1"], "num": [0], "den": [0]})
    out = pairing.aggregate_fractions(empty, "subject", {"f": ("num", "den")})
    assert np.isnan(out["f"].iloc[0])


def test_end_to_end_enrichment_of_a_structured_field():
    """Objects that truly sit on each other pair more often than any shifted control."""
    rng = np.random.default_rng(5)
    a = np.zeros((120, 120), dtype=bool)
    b = np.zeros((120, 120), dtype=bool)
    for _ in range(12):
        y, x = rng.integers(5, 110, 2)
        box(a, y, y + 6, x, x + 6)
        box(b, y + 1, y + 7, x + 1, x + 7)
    observed = pairing.pair_objects(a, b)
    draws = pairing.draw_displacements(
        60, min_length_um=10.0, max_length_um=25.0, pixel_size_um=PX, seed=2
    )
    control = pairing.translation_control(a, b, draws, PX)
    out = pairing.enrichment(observed.n_pairs, control["n_pairs"])
    assert out["enrichment"] > 1.0
    assert out["p_value"] <= 2 / 61 + 1e-12
