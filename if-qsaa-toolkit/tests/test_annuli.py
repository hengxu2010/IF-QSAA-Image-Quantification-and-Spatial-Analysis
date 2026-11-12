"""Tests for if_qsaa.annuli on synthetic fields and tables with known ground truth."""

import types

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage as ndi

from if_qsaa import annuli

PX = 1.0  # micrometres per pixel: a straight boundary then gives whole-pixel distances


@pytest.fixture
def slab():
    """A reference slab spanning the field, so every band is a flat row of pixels.

    The reference object fills rows 20-29 over the full width of a 60 x 40 field, so a
    pixel in row 19 or 30 is exactly one pixel from the object, a pixel in row 18 or 31
    exactly two, and the bands are whole rows whose pixel counts are known.
    """
    ref = np.zeros((60, 40), dtype=bool)
    ref[20:30, :] = True
    other = np.zeros((60, 40), dtype=bool)
    other[19, :] = True        # the whole first band outside
    other[30, :] = True
    other[31, :10] = True      # a quarter of the second band outside
    labels, n = ndi.label(ref, structure=np.ones((3, 3), int))
    assert n == 1
    return labels, other


def test_signed_distance_convention(slab):
    labels, _ = slab
    signed = annuli.signed_boundary_distance_um(labels == 1, PX)
    assert signed[19, 0] == pytest.approx(1.0)   # first pixel outside
    assert signed[18, 0] == pytest.approx(2.0)
    assert signed[20, 0] == pytest.approx(0.0)   # outermost pixel layer of the object
    assert signed[21, 0] == pytest.approx(-1.0)
    assert signed[29, 0] == pytest.approx(0.0)
    assert np.all(signed[20:30] <= 0)            # the footprint is the band <= 0
    assert np.all(signed[:20] > 0) and np.all(signed[30:] > 0)
    scaled = annuli.signed_boundary_distance_um(labels == 1, 0.25)
    assert scaled[18, 0] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        annuli.signed_boundary_distance_um(np.zeros((5, 5), bool), PX)
    with pytest.raises(ValueError):
        annuli.signed_boundary_distance_um(labels == 1, 0.0)


def test_band_geometry_and_coverage(slab):
    labels, other = slab
    bands = annuli.concentric_bands(labels == 1, (0.0, 1.0, 2.0), PX)
    assert [int(b.sum()) for b in bands] == [400, 80, 80]
    coverage = annuli.band_coverage_percent(bands, other)
    assert coverage == pytest.approx([0.0, 100.0, 12.5])


def test_whole_field_occupancy(slab):
    _, other = slab
    assert annuli.whole_field_occupancy_percent(other) == pytest.approx(
        100.0 * 90 / (60 * 40)
    )
    valid = np.zeros((60, 40), dtype=bool)
    valid[19, :] = True
    assert annuli.whole_field_occupancy_percent(other, valid) == pytest.approx(100.0)
    assert np.isnan(annuli.whole_field_occupancy_percent(other, np.zeros((60, 40), bool)))


def test_object_profile_recovers_the_known_coverage(slab):
    labels, other = slab
    profile = annuli.object_profile(labels, 1, other, (0.0, 1.0, 2.0), PX, field_id="f")
    assert list(profile["band"]) == [0, 1, 2]
    assert list(profile["inside"]) == [True, False, False]
    assert profile["inner_um"].tolist() == [-np.inf, 0.0, 1.0]
    assert profile["outer_um"].tolist() == [0.0, 1.0, 2.0]
    assert profile["mid_um"].tolist() == [0.0, 0.5, 1.5]
    # the footprint band has no finite inner edge, so it is named rather than written
    assert profile["band_label"].tolist() == ["footprint", "(0, 1]", "(1, 2]"]
    assert profile["band_px"].tolist() == [400, 80, 80]
    assert profile["band_area_um2"].tolist() == [400.0, 80.0, 80.0]
    assert profile["coverage_percent"].tolist() == pytest.approx([0.0, 100.0, 12.5])
    baseline = 100.0 * 90 / (60 * 40)
    assert np.allclose(profile["baseline_percent"], baseline)
    assert profile["excess_percent"].tolist() == pytest.approx(
        [0.0 - baseline, 100.0 - baseline, 12.5 - baseline]
    )
    assert np.allclose(profile["object_area_um2"], 400.0)
    assert set(profile["field"]) == {"f"}


def test_profile_matches_the_uncropped_band_computation(slab):
    """The profile is computed on a crop around the object; that must change nothing."""
    labels, other = slab
    edges = (0.0, 1.0, 2.0, 5.0)
    profile = annuli.object_profile(labels, 1, other, edges, PX)
    bands = annuli.concentric_bands(labels == 1, edges, PX)
    assert profile["band_px"].tolist() == [int(b.sum()) for b in bands]
    assert profile["coverage_percent"].tolist() == pytest.approx(
        annuli.band_coverage_percent(bands, other)
    )


def test_signed_bands_reach_inside_the_object(slab):
    labels, other = slab
    profile = annuli.object_profile(
        labels, 1, other, (-2.0, -1.0, 0.0, 1.0, 2.0), PX, include_inside=False
    )
    assert profile["band_px"].tolist() == [80, 80, 80, 80]
    assert profile["mid_um"].tolist() == pytest.approx([-1.5, -0.5, 0.5, 1.5])
    assert profile["band_label"].tolist() == ["(-2, -1]", "(-1, 0]", "(0, 1]", "(1, 2]"]
    # the two inner shells hold no pixel of the tested mask
    assert profile["coverage_percent"].tolist() == pytest.approx([0.0, 0.0, 100.0, 12.5])


def test_empty_band_gives_no_coverage(slab):
    labels, other = slab
    profile = annuli.object_profile(labels, 1, other, (0.0, 0.4), PX)
    outward = profile[~profile["inside"]].iloc[0]
    assert outward["band_px"] == 0
    assert np.isnan(outward["coverage_percent"])
    assert np.isnan(outward["excess_percent"])


def test_edge_argument_checks(slab):
    labels, other = slab
    with pytest.raises(ValueError):
        annuli.object_profile(labels, 1, other, (1.0, 2.0), PX)  # inside needs 0 first
    with pytest.raises(ValueError):
        annuli.object_profile(labels, 1, other, (0.0, 2.0, 1.0), PX)
    with pytest.raises(ValueError):
        annuli.object_profile(labels, 1, other, (0.0,), PX)
    with pytest.raises(ValueError):
        annuli.object_profile(labels, 7, other, (0.0, 1.0), PX)


def test_other_reference_objects_can_be_excluded():
    ref = np.zeros((60, 40), dtype=bool)
    ref[20:30, :] = True   # the reference object
    ref[31, :] = True      # a second object of the same mask, two pixels away
    other = np.zeros((60, 40), dtype=bool)
    other[18, :] = True
    labels, n = ndi.label(ref, structure=np.ones((3, 3), int))
    assert n == 2
    edges = (0.0, 1.0, 2.0)
    kept = annuli.object_profile(labels, 1, other, edges, PX)
    dropped = annuli.object_profile(
        labels, 1, other, edges, PX, exclude_other_reference_objects=True
    )
    # the outer band holds rows 18 and 31; row 31 is the other reference object
    assert kept["band_px"].tolist() == [400, 80, 80]
    assert dropped["band_px"].tolist() == [400, 80, 40]
    assert kept["coverage_percent"].iloc[2] == pytest.approx(50.0)
    assert dropped["coverage_percent"].iloc[2] == pytest.approx(100.0)


def test_valid_region_restricts_bands(slab):
    labels, other = slab
    valid = np.zeros((60, 40), dtype=bool)
    valid[:, :20] = True
    profile = annuli.object_profile(labels, 1, other, (0.0, 1.0), PX, valid=valid)
    assert profile["band_px"].tolist() == [200, 40]


def test_profile_objects_over_two_objects():
    ref = np.zeros((40, 40), dtype=bool)
    ref[5:9, 5:9] = True
    ref[25:29, 25:29] = True
    other = np.zeros((40, 40), dtype=bool)
    other[4, 5:9] = True
    labels, n = ndi.label(ref, structure=np.ones((3, 3), int))
    assert n == 2
    table = annuli.profile_objects(labels, other, (0.0, 1.0, 2.0), PX, field_id="f")
    assert len(table) == 6
    assert sorted(set(table["object_label"])) == [1, 2]
    assert np.allclose(table["baseline_percent"], 100.0 * 4 / 1600)
    assert np.allclose(table["object_area_um2"], 16.0)
    first = table[table["object_label"] == 1]
    second = table[table["object_label"] == 2]
    assert first["coverage_percent"].iloc[1] > 0
    assert second["coverage_percent"].iloc[1] == pytest.approx(0.0)
    one = annuli.profile_objects(labels, other, (0.0, 1.0), PX, labels=[2])
    assert set(one["object_label"]) == {2}
    empty = annuli.profile_objects(
        np.zeros((5, 5), dtype=int), other[:5, :5], (0.0, 1.0), PX
    )
    assert len(empty) == 0
    assert "coverage_percent" in empty.columns


@pytest.fixture
def object_table():
    """Per-object, per-band table with two units and two bands."""
    rows = []
    for unit, coverage in (("u1", (10.0, 20.0)), ("u2", (30.0, 30.0))):
        for obj, value in enumerate(coverage):
            for mid, scale in ((0.5, 1.0), (1.5, 0.5)):
                rows.append(
                    {
                        "unit": unit,
                        "object_label": obj + 1,
                        "mid_um": mid,
                        "coverage_percent": value * scale,
                        "baseline_percent": 5.0,
                        "excess_percent": value * scale - 5.0,
                    }
                )
    return pd.DataFrame(rows)


def test_unit_profiles_and_summary(object_table):
    units = annuli.unit_profiles(object_table, "unit")
    near = units[units["mid_um"] == 0.5].set_index("unit")
    assert near.loc["u1", "coverage_percent"] == pytest.approx(15.0)
    assert near.loc["u2", "coverage_percent"] == pytest.approx(30.0)
    assert near.loc["u1", "n_objects"] == 2

    summary = annuli.profile_summary(units).set_index("mid_um")
    assert summary.loc[0.5, "coverage_percent_mean"] == pytest.approx(22.5)
    assert summary.loc[0.5, "coverage_percent_sd"] == pytest.approx(
        np.std([15.0, 30.0], ddof=1)
    )
    assert summary.loc[0.5, "n_units"] == 2
    assert summary.loc[0.5, "n_objects"] == 4
    assert summary.loc[0.5, "baseline_percent_mean"] == pytest.approx(5.0)

    only_one = annuli.unit_profiles(object_table, "unit", exclude=["u2"])
    assert set(only_one["unit"]) == {"u1"}
    assert np.isnan(annuli.profile_summary(only_one).iloc[0]["coverage_percent_sd"])
    with pytest.raises(KeyError):
        annuli.unit_profiles(object_table, "unit", value_cols=("nope",))


def two_object_profile():
    """Per-object profile of one field with two reference objects and three bands."""
    ref = np.zeros((40, 40), dtype=bool)
    ref[5:9, 5:9] = True
    ref[25:29, 25:29] = True
    other = np.zeros((40, 40), dtype=bool)
    other[4, 5:9] = True
    labels, n = ndi.label(ref, structure=np.ones((3, 3), int))
    assert n == 2
    return annuli.profile_objects(labels, other, (0.0, 1.0, 2.0), PX, field_id="f")


def test_band_identity_survives_the_summaries():
    """A summarised band keeps its geometry, so the table can be labelled on its own."""
    profile = two_object_profile()
    # one field holds no subject, so each object stands in as the unit of analysis
    units = annuli.unit_profiles(profile, "object_label")
    summary = annuli.profile_summary(units)
    identity = ["inner_um", "outer_um", "inside", "band_label"]
    assert list(units.columns[:7]) == ["object_label", "mid_um", *identity, "n_objects"]
    assert list(summary.columns[:6]) == ["mid_um", *identity, "n_units"]
    bands = summary.set_index("mid_um")
    assert list(summary["band_label"]) == ["footprint", "(0, 1]", "(1, 2]"]
    assert bands.loc[0.0, "inner_um"] == -np.inf and bool(bands.loc[0.0, "inside"])
    assert bands.loc[0.5, "inner_um"] == pytest.approx(0.0)
    assert bands.loc[1.5, "outer_um"] == pytest.approx(2.0)
    assert not bool(bands.loc[1.5, "inside"])
    assert bands.loc[0.5, "n_units"] == 2 and bands.loc[0.5, "n_objects"] == 2
    # the same identity reaches the excess summary
    excess = annuli.excess_summary(profile, "object_label")
    assert list(excess["band_label"]) == ["footprint", "(0, 1]", "(1, 2]"]


def test_band_identity_columns_can_be_chosen_and_must_be_unambiguous():
    profile = two_object_profile()
    chosen = annuli.unit_profiles(profile, "object_label", band_cols=["inside"])
    assert "inside" in chosen.columns and "band_label" not in chosen.columns
    # band_cols=[] carries nothing, the behaviour before the identity was carried
    bare = annuli.unit_profiles(profile, "object_label", band_cols=[])
    assert list(bare.columns) == ["object_label", "mid_um", "n_objects",
                                  "coverage_percent", "baseline_percent",
                                  "excess_percent"]
    assert list(annuli.profile_summary(bare, band_cols=[]).columns[:3]) == [
        "mid_um", "n_units", "n_objects"]
    with pytest.raises(KeyError):
        annuli.unit_profiles(profile, "object_label", band_cols=["nope"])
    # a table without any identity column still summarises, as before
    plain = annuli.unit_profiles(
        profile.drop(columns=["inner_um", "outer_um", "inside", "band_label"]),
        "object_label",
    )
    assert "band_label" not in plain.columns
    assert annuli.profile_summary(plain)["n_units"].tolist() == [2, 2, 2]
    # two different bands cannot be pooled under one midpoint
    clashing = pd.concat(
        [profile, profile.assign(outer_um=profile["outer_um"] + 1.0)], ignore_index=True
    )
    with pytest.raises(ValueError, match="outer_um"):
        annuli.unit_profiles(clashing, "object_label")
    units = annuli.unit_profiles(profile, "object_label")
    pooled = pd.concat(
        [units, units.assign(inner_um=units["inner_um"] + 0.5)], ignore_index=True
    )
    with pytest.raises(ValueError, match="inner_um"):
        annuli.profile_summary(pooled)


def test_profile_tests_uses_the_given_test(object_table):
    units = annuli.unit_profiles(object_table, "unit")
    seen = {}

    def fake_test(x, y, **kwargs):
        seen[len(seen)] = (tuple(x), tuple(y), kwargs)
        return {"p_value": 0.0123, "statistic": 3.0}

    out = annuli.profile_tests(units, test=fake_test, test_kwargs={"flag": True})
    assert list(out["mid_um"]) == [0.5, 1.5]
    near = out[out["mid_um"] == 0.5].iloc[0]
    assert near["n"] == 2
    assert near["n_above"] == 2
    assert near["mean_diff"] == pytest.approx(((15.0 - 5.0) + (30.0 - 5.0)) / 2)
    assert near["sd_diff"] == pytest.approx(np.std([10.0, 25.0], ddof=1))
    assert near["p_value"] == pytest.approx(0.0123)
    assert near["statistic"] == pytest.approx(3.0)
    assert seen[0][0] == (15.0, 30.0)
    assert seen[0][1] == (5.0, 5.0)
    assert seen[0][2] == {"flag": True}

    def result_object(x, y):
        return types.SimpleNamespace(statistic=2.0, pvalue=0.5)

    out = annuli.profile_tests(units, test=result_object)
    assert out["p_value"].tolist() == pytest.approx([0.5, 0.5])
    assert out["statistic"].tolist() == pytest.approx([2.0, 2.0])


def test_profile_tests_against_a_real_signed_rank_test():
    rng = np.random.default_rng(1)
    units = pd.DataFrame(
        {
            "unit": [f"u{i}" for i in range(8)] * 2,
            "mid_um": [0.5] * 8 + [1.5] * 8,
            "coverage_percent": np.concatenate(
                [20.0 + rng.normal(0, 1, 8), 6.0 + rng.normal(0, 1, 8)]
            ),
            "baseline_percent": [5.0] * 16,
        }
    )
    stats = pytest.importorskip("if_qsaa.stats")
    assert hasattr(stats, "wilcoxon_signed_rank_exact")
    out = annuli.profile_tests(units)
    assert len(out) == 2
    assert ((out["p_value"] >= 0) & (out["p_value"] <= 1)).all()
    near = out[out["mid_um"] == 0.5].iloc[0]
    assert near["n"] == 8 and near["n_above"] == 8
    assert near["p_value"] < 0.05


def test_stratify_equal_count():
    strata = annuli.stratify_equal_count([5.0, 1.0, 4.0, 2.0, 3.0, 6.0], 3)
    assert strata.tolist() == [2, 0, 1, 0, 1, 2]
    uneven = annuli.stratify_equal_count([1.0, 2.0, 3.0, 4.0, 5.0], 3)
    assert np.bincount(uneven).tolist() == [2, 2, 1]
    # equal values keep the order they were given in
    assert annuli.stratify_equal_count([7.0, 7.0, 7.0, 7.0], 2).tolist() == [0, 0, 1, 1]
    assert annuli.stratify_equal_count([], 3).size == 0
    assert annuli.stratify_equal_count([1.0, 2.0], 1).tolist() == [0, 0]
    with pytest.raises(ValueError):
        annuli.stratify_equal_count([1.0], 0)


def test_excess_summary_by_strata(object_table):
    table = object_table.copy()
    # one covariate value per object, the same in every band
    sizes = {("u1", 1): 1.0, ("u1", 2): 4.0, ("u2", 1): 2.0, ("u2", 2): 3.0}
    table["area_um2"] = [sizes[(u, o)] for u, o in zip(table["unit"], table["object_label"])]
    one_row_per_object = table[table["mid_um"] == 0.5]
    strata = dict(
        zip(
            zip(one_row_per_object["unit"], one_row_per_object["object_label"]),
            annuli.stratify_equal_count(one_row_per_object["area_um2"], 2),
        )
    )
    table["stratum"] = [strata[(u, o)] for u, o in zip(table["unit"], table["object_label"])]

    out = annuli.excess_summary(table, "unit", by=["stratum"], distance_col="mid_um")
    small = out[(out["stratum"] == 0) & (out["mid_um"] == 0.5)].iloc[0]
    # the small stratum holds one object per unit: 10 and 30 per cent coverage
    assert small["n_units"] == 2
    assert small["n_objects"] == 2
    assert small["excess_percent_mean"] == pytest.approx(((10.0 - 5.0) + (30.0 - 5.0)) / 2)
    assert small["excess_percent_sd"] == pytest.approx(np.std([5.0, 25.0], ddof=1))

    plain = annuli.excess_summary(table, "unit").set_index("mid_um")
    assert plain.loc[0.5, "excess_percent_mean"] == pytest.approx(
        np.mean([15.0 - 5.0, 30.0 - 5.0])
    )

    lopsided = table[~((table["unit"] == "u2") & (table["stratum"] == 1))]
    annuli.excess_summary(lopsided, "unit", by=["stratum"])  # tolerated by default
    with pytest.raises(ValueError):
        annuli.excess_summary(lopsided, "unit", by=["stratum"], require_all_units=True)
