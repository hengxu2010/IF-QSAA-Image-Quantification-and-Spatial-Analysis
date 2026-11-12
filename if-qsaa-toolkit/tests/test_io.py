"""Tests for if_qsaa.io on synthetic files written inside tmp_path."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tifffile

from if_qsaa import io as qio


def grey_image(shape=(6, 8), seed=0):
    """Synthetic 16-bit grey image with known values."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 4096, size=shape, dtype=np.uint16)


def rgb_image(shape=(5, 7), seed=1):
    """Synthetic 8-bit three-channel image with a distinct level per channel."""
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 60, size=shape + (3,), dtype=np.uint8)
    image[:, :, 1] += 70
    image[:, :, 2] += 140
    return image


# --------------------------------------------------------------------------- reading

def test_read_tiff_keeps_grey_levels(tmp_path):
    data = grey_image()
    path = tmp_path / "plane.tif"
    tifffile.imwrite(path, data)
    out = qio.read_image(path)
    assert out.dtype == np.float64
    assert out.shape == data.shape
    assert np.array_equal(out, data.astype(np.float64))


def test_read_png_keeps_grey_levels(tmp_path):
    data = grey_image(seed=3)
    path = tmp_path / "plane.png"
    qio.write_image(path, data)
    out = qio.read_image(path)
    assert np.array_equal(out, data.astype(np.float64))


def test_read_multichannel_keeps_channel_order(tmp_path):
    data = rgb_image()
    path = tmp_path / "colour.tif"
    tifffile.imwrite(path, data)
    out = qio.read_image(path)
    assert out.shape == data.shape
    assert np.array_equal(out, data.astype(np.float64))


def test_single_channel_sources_read_as_2d(tmp_path):
    data = grey_image(shape=(4, 5), seed=4)
    for name in ("one.tif", "one.png"):
        path = tmp_path / name
        qio.write_image(path, data)
        out = qio.read_image(path)
        assert out.ndim == 2
        assert np.array_equal(out, data.astype(np.float64))


def test_read_rejects_multipage_tiff(tmp_path):
    data = grey_image()
    path = tmp_path / "stack.tif"
    with tifffile.TiffWriter(path) as writer:
        writer.write(data)
        writer.write(data)
    with pytest.raises(ValueError, match="single-page"):
        qio.read_image(path)


def test_read_rejects_unknown_suffix_and_missing_file(tmp_path):
    path = tmp_path / "plane.dat"
    path.write_bytes(b"0")
    with pytest.raises(ValueError, match="unsupported image format"):
        qio.read_image(path)
    with pytest.raises(FileNotFoundError):
        qio.read_image(tmp_path / "absent.tif")


def test_read_image_accepts_float_tiff_without_rescaling(tmp_path):
    data = (grey_image(seed=5) / 7.0).astype(np.float32)
    path = tmp_path / "float.tif"
    qio.write_image(path, data)
    out = qio.read_image(path)
    assert np.array_equal(out, data.astype(np.float64))


# --------------------------------------------------------------------------- channels

def test_extract_channel_by_index_and_by_name():
    image = rgb_image()
    for index in range(3):
        assert np.array_equal(qio.extract_channel(image, index), image[:, :, index])
    names = {"first": 0, "second": 1, "third": 2}
    assert np.array_equal(qio.extract_channel(image, "third", channel_names=names), image[:, :, 2])
    order = ["first", "second", "third"]
    by_order = qio.extract_channel(image, "second", channel_names=order)
    assert np.array_equal(by_order, image[:, :, 1])


def test_extract_channel_axis_first():
    stacked = np.moveaxis(rgb_image(), 2, 0)
    plane = qio.extract_channel(stacked, 1, channel_axis=0)
    assert plane.shape == stacked.shape[1:]
    assert np.array_equal(plane, stacked[1])


def test_extract_channel_returns_a_copy():
    image = rgb_image().astype(np.float64)
    plane = qio.extract_channel(image, 0)
    plane[0, 0] = 12345.0
    assert image[0, 0, 0] != 12345.0


def test_extract_channel_errors():
    image = rgb_image()
    with pytest.raises(ValueError, match="3-D"):
        qio.extract_channel(image[:, :, 0], 0)
    with pytest.raises(ValueError, match="out of range"):
        qio.extract_channel(image, 3)
    with pytest.raises(ValueError, match="channel_names is required"):
        qio.extract_channel(image, "first")
    with pytest.raises(ValueError, match="unknown channel name"):
        qio.extract_channel(image, "fourth", channel_names=["first", "second", "third"])


# --------------------------------------------------------------------------- masks

@pytest.mark.parametrize("foreground", [1, 255])
def test_mask_round_trip_tolerates_either_convention(tmp_path, foreground):
    mask = np.zeros((6, 6), dtype=bool)
    mask[1:4, 2:5] = True
    path = tmp_path / f"mask_{foreground}.tif"
    qio.write_mask(path, mask, foreground=foreground)
    stored = tifffile.imread(path)
    assert stored.dtype == np.uint8
    assert set(np.unique(stored)) == {0, foreground}
    back = qio.read_mask(path)
    assert back.dtype == bool
    assert np.array_equal(back, mask)


def test_empty_mask_round_trip(tmp_path):
    mask = np.zeros((4, 3), dtype=bool)
    path = tmp_path / "empty.tif"
    qio.write_mask(path, mask)
    back = qio.read_mask(path)
    assert back.shape == mask.shape
    assert not back.any()


def test_mask_touching_the_border_round_trip(tmp_path):
    mask = np.zeros((5, 5), dtype=bool)
    mask[0, :] = True
    mask[:, -1] = True
    path = tmp_path / "border.tif"
    qio.write_mask(path, mask)
    assert np.array_equal(qio.read_mask(path), mask)


def test_write_mask_requires_tiff_and_valid_foreground(tmp_path):
    mask = np.zeros((3, 3), dtype=bool)
    with pytest.raises(ValueError, match="written as TIFF"):
        qio.write_mask(tmp_path / "mask.png", mask)
    with pytest.raises(ValueError, match="foreground"):
        qio.write_mask(tmp_path / "mask.tif", mask, foreground=0)


def test_read_mask_rejects_multichannel(tmp_path):
    path = tmp_path / "colour.tif"
    tifffile.imwrite(path, rgb_image())
    with pytest.raises(ValueError, match="2-D"):
        qio.read_mask(path)


# --------------------------------------------------------------------------- writing

def test_no_overwrite_policy(tmp_path):
    data = grey_image(shape=(3, 3))
    path = tmp_path / "plane.tif"
    qio.write_image(path, data)
    with pytest.raises(FileExistsError):
        qio.write_image(path, data)
    other = data + 1
    qio.write_image(path, other, overwrite=True)
    assert np.array_equal(qio.read_image(path), other.astype(np.float64))


def test_write_image_creates_parent_directory(tmp_path):
    path = tmp_path / "deeper" / "still" / "plane.tif"
    qio.write_image(path, grey_image(shape=(2, 2)))
    assert path.is_file()


def test_write_image_refuses_float_for_integer_formats(tmp_path):
    with pytest.raises(ValueError, match="integer pixels"):
        qio.write_image(tmp_path / "plane.png", np.zeros((3, 3), dtype=np.float32))


def test_write_labels_picks_the_smallest_type(tmp_path):
    labels = np.zeros((4, 4), dtype=np.int32)
    labels[0, 0] = 1
    labels[1, 1] = 7
    small = tmp_path / "small.tif"
    qio.write_labels(small, labels)
    stored = tifffile.imread(small)
    assert stored.dtype == np.uint16
    assert np.array_equal(stored, labels)

    labels[2, 2] = 70000
    big = tmp_path / "big.tif"
    qio.write_labels(big, labels)
    stored_big = tifffile.imread(big)
    assert stored_big.dtype == np.uint32
    assert np.array_equal(stored_big, labels)


def test_write_labels_rejects_float_and_negative(tmp_path):
    with pytest.raises(ValueError, match="integer array"):
        qio.write_labels(tmp_path / "labels.tif", np.zeros((3, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="non-negative"):
        qio.write_labels(tmp_path / "labels2.tif", -np.ones((3, 3), dtype=np.int32))


def test_write_table_keeps_full_precision(tmp_path):
    values = np.array([np.pi / 7.0, 1.0 / 3.0, 1e-17, 123456789.123456789])
    table = pd.DataFrame({"value": values, "key": ["a", "b", "c", "d"]})
    path = tmp_path / "table.csv"
    qio.write_table(path, table)
    back = pd.read_csv(path, encoding="utf-8-sig", float_precision="round_trip")
    assert list(back.columns) == ["value", "key"]
    assert np.array_equal(back["value"].to_numpy(), values)
    assert list(back["key"]) == ["a", "b", "c", "d"]


def test_write_table_formula_escaping_is_opt_in(tmp_path):
    table = pd.DataFrame({"key": ["=plain", "ordinary"]})
    plain = tmp_path / "plain.csv"
    qio.write_table(plain, table)
    assert "=plain" in plain.read_text(encoding="utf-8-sig")
    escaped = tmp_path / "escaped.csv"
    qio.write_table(escaped, table, escape_formulas=True)
    assert "'=plain" in escaped.read_text(encoding="utf-8-sig")


def test_write_table_rejects_other_objects(tmp_path):
    with pytest.raises(TypeError):
        qio.write_table(tmp_path / "table.csv", {"value": [1, 2]})


# --------------------------------------------------------------------------- calibration

def test_length_unit_to_um_known_and_unknown():
    assert qio.length_unit_to_um("um") == 1.0
    assert qio.length_unit_to_um(" Micron ") == 1.0
    assert qio.length_unit_to_um("nm") == pytest.approx(1e-3)
    assert qio.length_unit_to_um("mm") == pytest.approx(1e3)
    assert qio.length_unit_to_um("cm") == pytest.approx(1e4)
    assert qio.length_unit_to_um("inch") == pytest.approx(25400.0)
    with pytest.raises(ValueError):
        qio.length_unit_to_um("pixel")
    with pytest.raises(ValueError):
        qio.length_unit_to_um("  ")


def test_length_unit_to_um_reads_both_micro_signs_as_micrometres():
    """A micrometre unit must not fall through to metres.

    Image writers spell the micro prefix with MICRO SIGN (U+00B5) or, less often, with GREEK
    SMALL LETTER MU (U+03BC). If either is not recognised, the name reduces to ``m`` and the
    unit is read as a metre: a silent factor of a million that leaves every micrometre
    parameter sub-pixel and every result empty. The characters are built with ``chr`` so that
    the assertion cannot be weakened by this file being re-saved in another encoding.
    """
    for code_point in (0xB5, 0x3BC):
        unit = chr(code_point) + "m"
        assert qio._unit_slug(unit) == "um"
        assert qio.length_unit_to_um(unit) == 1.0
        assert qio.length_unit_to_um(unit.upper()) == 1.0
    assert qio.length_unit_to_um("m") == pytest.approx(1e6)


def test_pixel_size_from_tiff_centimetre_tags(tmp_path):
    expected = 0.25
    pixels_per_cm = 1e4 / expected
    path = tmp_path / "calibrated.tif"
    tifffile.imwrite(
        path,
        grey_image(shape=(3, 3)),
        resolution=(pixels_per_cm, pixels_per_cm),
        resolutionunit="CENTIMETER",
    )
    assert qio.pixel_size_from_tiff(path) == pytest.approx(expected)


def test_pixel_size_from_tiff_inch_tags(tmp_path):
    expected = 2.0
    pixels_per_inch = 25400.0 / expected
    path = tmp_path / "inch.tif"
    tifffile.imwrite(
        path,
        grey_image(shape=(3, 3)),
        resolution=(pixels_per_inch, pixels_per_inch),
        resolutionunit="INCH",
    )
    assert qio.pixel_size_from_tiff(path) == pytest.approx(expected)


def test_pixel_size_from_tiff_unit_in_image_metadata(tmp_path):
    expected = 0.5
    path = tmp_path / "with_unit.tif"
    tifffile.imwrite(
        path,
        grey_image(shape=(4, 4)),
        imagej=True,
        resolution=(1.0 / expected, 1.0 / expected),
        metadata={"unit": "um"},
    )
    assert qio.pixel_size_from_tiff(path) == pytest.approx(expected)


def test_pixel_size_from_tiff_without_a_unit_raises(tmp_path):
    path = tmp_path / "bare.tif"
    tifffile.imwrite(path, grey_image(shape=(3, 3)))
    with pytest.raises(ValueError, match="unit of length"):
        qio.pixel_size_from_tiff(path)


def test_pixel_size_from_tiff_rejects_non_square_pixels(tmp_path):
    path = tmp_path / "anisotropic.tif"
    tifffile.imwrite(
        path,
        grey_image(shape=(3, 3)),
        resolution=(40000.0, 20000.0),
        resolutionunit="CENTIMETER",
    )
    with pytest.raises(ValueError, match="non-square"):
        qio.pixel_size_from_tiff(path)
    assert qio.pixel_size_from_tiff(path, rtol=2.0) == pytest.approx(0.25)


def test_pixel_size_from_tiff_rejects_other_formats(tmp_path):
    path = tmp_path / "plane.png"
    qio.write_image(path, grey_image(shape=(3, 3)))
    with pytest.raises(ValueError, match="unsupported image format"):
        qio.pixel_size_from_tiff(path)


def test_scale_bar_calibration():
    pixel_size = qio.pixel_size_from_scale_bar(20.0, 100.0)
    assert pixel_size == pytest.approx(0.2)
    with pytest.raises(ValueError):
        qio.pixel_size_from_scale_bar(20.0, 0.0)


def test_scale_bar_from_layout_matches_hand_calculation():
    # An image 1000 px wide placed 10 mm wide carries 0.01 mm per native pixel,
    # so a 2 mm bar on the layout is 200 native pixels long.
    bar_px = qio.scale_bar_px_from_layout(2.0, 10.0, 1000)
    assert bar_px == pytest.approx(200.0)
    assert qio.pixel_size_from_scale_bar(50.0, bar_px) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        qio.scale_bar_px_from_layout(2.0, 0.0, 1000)


def test_points_to_mm():
    assert qio.points_to_mm(72.0) == pytest.approx(25.4)
    assert qio.points_to_mm(0.0) == 0.0


# --------------------------------------------------------------------------- folders

def make_tree(root):
    """Write a small synthetic folder of images and one file to be ignored."""
    (root / "sub").mkdir()
    names = [
        root / "site02_conditionB.tif",
        root / "site01_conditionA.tif",
        root / "sub" / "site03_conditionA.png",
    ]
    for path in names:
        qio.write_image(path, grey_image(shape=(2, 2)))
    (root / "notes.txt").write_text("ignored", encoding="utf-8")
    return names


def test_list_images_is_sorted_and_filtered(tmp_path):
    make_tree(tmp_path)
    found = qio.list_images(tmp_path)
    assert [p.name for p in found] == [
        "site01_conditionA.tif",
        "site02_conditionB.tif",
        "site03_conditionA.png",
    ]
    assert found == sorted(found, key=lambda item: str(item).casefold())
    flat = qio.list_images(tmp_path, recursive=False)
    assert [p.name for p in flat] == ["site01_conditionA.tif", "site02_conditionB.tif"]
    only_tif = qio.list_images(tmp_path, suffixes=["TIF"])
    assert [p.name for p in only_tif] == ["site01_conditionA.tif", "site02_conditionB.tif"]


def test_list_images_on_an_empty_folder(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert qio.list_images(empty) == []
    with pytest.raises(NotADirectoryError):
        qio.list_images(tmp_path / "absent")


def test_image_table_parses_user_supplied_keys(tmp_path):
    make_tree(tmp_path)
    pattern = r"site(?P<site>\d+)_condition(?P<condition>[A-Z])"
    table = qio.image_table(tmp_path, pattern)
    assert list(table.columns) == ["path", "name", "site", "condition"]
    assert list(table["site"]) == ["01", "02", "03"]
    assert list(table["condition"]) == ["A", "B", "A"]
    assert table["path"].iloc[0].endswith("site01_conditionA.tif")
    # The rows follow the file order, so a second call gives the same table.
    pd.testing.assert_frame_equal(table, qio.image_table(tmp_path, pattern))


def test_image_table_targets_and_full_match(tmp_path):
    make_tree(tmp_path)
    pattern = r"site(?P<site>\d+)_condition(?P<condition>[A-Z])"
    stems = qio.image_table(tmp_path, pattern, target="stem", full_match=True)
    assert list(stems["site"]) == ["01", "02", "03"]
    with pytest.raises(ValueError, match="does not match"):
        qio.image_table(tmp_path, pattern, target="name", full_match=True)
    relative = qio.image_table(
        tmp_path, r"(?P<folder>[^/]*)/?site(?P<site>\d+)", target="relative"
    )
    assert list(relative["folder"]) == ["", "", "sub"]


def test_image_table_unmatched_handling(tmp_path):
    make_tree(tmp_path)
    qio.write_image(tmp_path / "other.tif", grey_image(shape=(2, 2)))
    pattern = r"site(?P<site>\d+)"
    with pytest.raises(ValueError, match="does not match"):
        qio.image_table(tmp_path, pattern)
    skipped = qio.image_table(tmp_path, pattern, on_unmatched="skip")
    assert list(skipped["site"]) == ["01", "02", "03"]


def test_image_table_requires_named_groups_and_free_column_names(tmp_path):
    make_tree(tmp_path)
    with pytest.raises(ValueError, match="named group"):
        qio.image_table(tmp_path, r"site(\d+)")
    with pytest.raises(ValueError, match="clash"):
        qio.image_table(tmp_path, r"(?P<name>site\d+)")
    with pytest.raises(ValueError, match="target"):
        qio.image_table(tmp_path, r"(?P<site>\d+)", target="elsewhere")
    with pytest.raises(ValueError, match="on_unmatched"):
        qio.image_table(tmp_path, r"(?P<site>\d+)", on_unmatched="ignore")


def test_image_table_on_an_empty_folder_keeps_the_columns(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    table = qio.image_table(empty, r"site(?P<site>\d+)")
    assert list(table.columns) == ["path", "name", "site"]
    assert len(table) == 0


def test_image_table_feeds_the_readers(tmp_path):
    make_tree(tmp_path)
    table = qio.image_table(tmp_path, r"site(?P<site>\d+)")
    shapes = [qio.read_image(path).shape for path in table["path"]]
    assert shapes == [(2, 2)] * 3
