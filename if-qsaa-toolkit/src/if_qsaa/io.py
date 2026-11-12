"""File input and output: images, binary masks, label images, tables, pixel-size calibration.

This is the only module of the package that touches the filesystem; the analysis
modules work on arrays and data frames that are read and written here.

Two rules hold everywhere in this module:

* Pixel values are never rescaled. An integer image keeps its grey levels when it is
  returned as a float array, so every intensity parameter of the package is expressed
  in the units of the input image.
* The package carries no naming convention and no spatial calibration of its own. A
  pixel size is either read from what a file itself states, or computed from a scale
  bar the caller measured, or passed in by the caller. File-name keys are parsed with a
  regular expression supplied by the caller.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import tifffile
from skimage import io as skimage_io

from ._util import as_bool_mask, as_float, check_2d

__all__ = [
    "SUPPORTED_IMAGE_SUFFIXES",
    "TIFF_SUFFIXES",
    "MM_PER_POINT",
    "read_image",
    "extract_channel",
    "read_mask",
    "write_image",
    "write_mask",
    "write_labels",
    "write_table",
    "length_unit_to_um",
    "pixel_size_from_tiff",
    "pixel_size_from_scale_bar",
    "scale_bar_px_from_layout",
    "points_to_mm",
    "list_images",
    "image_table",
]

#: Image formats this module reads and writes.
SUPPORTED_IMAGE_SUFFIXES = frozenset({".tif", ".tiff", ".png", ".bmp", ".jpg", ".jpeg", ".webp"})

#: Suffixes handled with ``tifffile``; everything else goes through ``skimage.io``.
TIFF_SUFFIXES = frozenset({".tif", ".tiff"})

#: Millimetres per typographic point, for scale bars measured on a layout in points.
MM_PER_POINT = 25.4 / 72.0

# Micrometres per unit of length. Keys are slugs produced by ``_unit_slug``.
_UM_PER_UNIT = {
    "nm": 1e-3,
    "nanometer": 1e-3,
    "nanometers": 1e-3,
    "nanometre": 1e-3,
    "nanometres": 1e-3,
    "um": 1.0,
    "micron": 1.0,
    "microns": 1.0,
    "micrometer": 1.0,
    "micrometers": 1.0,
    "micrometre": 1.0,
    "micrometres": 1.0,
    "mm": 1e3,
    "millimeter": 1e3,
    "millimeters": 1e3,
    "millimetre": 1e3,
    "millimetres": 1e3,
    "cm": 1e4,
    "centimeter": 1e4,
    "centimeters": 1e4,
    "centimetre": 1e4,
    "centimetres": 1e4,
    "m": 1e6,
    "meter": 1e6,
    "meters": 1e6,
    "metre": 1e6,
    "metres": 1e6,
    "in": 25400.0,
    "inch": 25400.0,
    "inches": 25400.0,
}

# TIFF ResolutionUnit tag: 1 = no unit, 2 = inch, 3 = centimetre.
_RESOLUTION_UNIT_UM = {2: 25400.0, 3: 1e4}

_FORMULA_LEAD = ("=", "+", "-", "@", "\t", "\r")

_RESERVED_TABLE_COLUMNS = ("path", "name")


# --------------------------------------------------------------------------- paths

def _suffix(path) -> str:
    """Lower-case suffix of a path, including the dot."""
    return Path(path).suffix.casefold()


def _normalise_suffixes(suffixes) -> frozenset:
    """Return a set of lower-case, dot-prefixed suffixes."""
    if suffixes is None:
        return SUPPORTED_IMAGE_SUFFIXES
    cleaned = set()
    for item in suffixes:
        text = str(item).strip().casefold()
        if not text:
            continue
        cleaned.add(text if text.startswith(".") else "." + text)
    if not cleaned:
        raise ValueError("suffixes must contain at least one file extension")
    return frozenset(cleaned)


def _check_readable(path, *, suffixes=None) -> Path:
    """Validate that a file exists and has a handled suffix."""
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f"image file not found: {file_path}")
    allowed = _normalise_suffixes(suffixes)
    if _suffix(file_path) not in allowed:
        raise ValueError(
            f"unsupported image format {_suffix(file_path)!r}; "
            f"handled formats are {sorted(allowed)}"
        )
    return file_path


def _prepare_output(path, overwrite: bool) -> Path:
    """Create the parent directory and apply the no-overwrite policy."""
    out_path = Path(path).expanduser()
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"output file already exists: {out_path} (pass overwrite=True to replace it)"
        )
    if str(out_path.parent):
        out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


# --------------------------------------------------------------------------- reading

def read_image(path, *, dtype: type = np.float64) -> np.ndarray:
    """Read a single-plane image from disk as a float array, without rescaling.

    TIFF files are read with ``tifffile``, all other formats with ``skimage.io``.
    Grey levels are preserved exactly: a 12-bit camera value of 2048 is returned as
    2048.0, so intensity thresholds stay in the units of the input image.

    Parameters
    ----------
    path : str or pathlib.Path
        File to read. The suffix must be one of :data:`SUPPORTED_IMAGE_SUFFIXES`.
    dtype : type, optional
        Floating-point type of the returned array (default ``numpy.float64``).

    Returns
    -------
    numpy.ndarray
        ``(rows, columns)`` for a single-channel image, ``(rows, columns, channels)``
        for a multichannel one. A trailing channel axis of length one is dropped.
        Channels are returned in the order in which they are stored in the file; no
        colour conversion is applied.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the suffix is not handled, if a TIFF holds more than one page (a z-stack
        or time series must be split into single planes first), or if the image is
        not two- or three-dimensional.
    """
    file_path = _check_readable(path)
    if _suffix(file_path) in TIFF_SUFFIXES:
        with tifffile.TiffFile(file_path) as handle:
            if len(handle.pages) != 1:
                raise ValueError(
                    f"only single-page images are read; this TIFF has "
                    f"{len(handle.pages)} pages: {file_path.name}"
                )
            array = handle.pages[0].asarray()
    else:
        array = skimage_io.imread(file_path)
    array = np.asarray(array)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    if array.ndim not in (2, 3):
        raise ValueError(
            f"expected a 2-D or multichannel image, got shape {array.shape}: {file_path.name}"
        )
    return as_float(array, dtype)


def extract_channel(
    image: np.ndarray,
    channel,
    *,
    channel_names=None,
    channel_axis: int = -1,
) -> np.ndarray:
    """Take one channel out of a multichannel image.

    The channel is addressed either by its integer position along ``channel_axis`` or
    by a name, in which case the caller supplies the mapping from names to positions.
    The package defines no channel names of its own.

    Parameters
    ----------
    image : numpy.ndarray
        Three-dimensional array, for example ``(rows, columns, channels)`` for an RGB
        image or ``(channels, rows, columns)`` for a channel-stacked one.
    channel : int or str
        Position of the channel along ``channel_axis``, or its name.
    channel_names : mapping of str to int, or sequence of str, optional
        Name of each channel. A mapping gives the position of each name directly; a
        sequence is read as the channel order, so its first entry names position 0.
        Required when ``channel`` is a name.
    channel_axis : int, optional
        Axis that holds the channels (default ``-1``, the last axis, which is how
        :func:`read_image` returns an interleaved image).

    Returns
    -------
    numpy.ndarray
        Two-dimensional float copy of the selected channel, values unchanged.

    Raises
    ------
    ValueError
        If the image is not three-dimensional, if a name is given without
        ``channel_names``, if the name is unknown, or if the position is out of range.
    """
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(
            f"a channel can only be taken from a 3-D image, got shape {array.shape}"
        )
    axis = int(channel_axis)
    if not -array.ndim <= axis < array.ndim:
        raise ValueError(f"channel_axis {axis} is out of range for shape {array.shape}")
    n_channels = array.shape[axis]

    if isinstance(channel, str):
        if channel_names is None:
            raise ValueError("channel_names is required when the channel is given by name")
        if isinstance(channel_names, Mapping):
            lookup = {str(key): int(value) for key, value in channel_names.items()}
        elif isinstance(channel_names, Sequence) and not isinstance(channel_names, str):
            lookup = {str(name): index for index, name in enumerate(channel_names)}
        else:
            raise TypeError("channel_names must be a mapping or a sequence of names")
        if channel not in lookup:
            raise ValueError(
                f"unknown channel name {channel!r}; names given are {sorted(lookup)}"
            )
        index = lookup[channel]
    else:
        index = int(channel)
    if not 0 <= index < n_channels:
        raise ValueError(
            f"channel index {index} is out of range for an image with {n_channels} channels"
        )
    plane = np.take(array, index, axis=axis)
    return as_float(np.ascontiguousarray(plane))


def read_mask(path) -> np.ndarray:
    """Read a binary mask and return it as a boolean array.

    Any non-zero pixel is foreground, so a mask stored as 0/1 and the same mask stored
    as 0/255 read identically.

    Parameters
    ----------
    path : str or pathlib.Path
        Mask file, usually a single-channel TIFF.

    Returns
    -------
    numpy.ndarray of bool
        ``(rows, columns)`` mask, ``True`` where the stored value is non-zero.

    Raises
    ------
    ValueError
        If the file does not hold a single-channel, two-dimensional image.
    """
    array = read_image(path)
    check_2d(array, "mask")
    return array != 0


# --------------------------------------------------------------------------- writing

def write_image(path, image: np.ndarray, *, overwrite: bool = False) -> Path:
    """Write an image, keeping its values and data type unchanged.

    TIFF files are written with ``tifffile`` (integer and floating-point pixels), all
    other formats with ``skimage.io``, which needs integer pixels. Nothing is rescaled,
    stretched or resampled. An existing file is not replaced unless ``overwrite`` is set.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination file; the suffix selects the format.
    image : numpy.ndarray
        Two-dimensional or ``(rows, columns, channels)`` array.
    overwrite : bool, optional
        Replace an existing file (default ``False``).

    Returns
    -------
    pathlib.Path
        The path written.

    Raises
    ------
    ValueError
        If the suffix is not handled, the array is not 2-D or 3-D, or a
        floating-point array is written to a format that stores integers only.
    FileExistsError
        If the file exists and ``overwrite`` is ``False``.
    """
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError(f"expected a 2-D or multichannel image, got shape {array.shape}")
    suffix = _suffix(path)
    if suffix not in SUPPORTED_IMAGE_SUFFIXES:
        raise ValueError(
            f"unsupported image format {suffix!r}; "
            f"handled formats are {sorted(SUPPORTED_IMAGE_SUFFIXES)}"
        )
    out_path = _prepare_output(path, overwrite)
    if suffix in TIFF_SUFFIXES:
        tifffile.imwrite(out_path, array)
        return out_path
    if array.dtype.kind not in ("u", "i"):
        raise ValueError(
            f"{suffix} stores integer pixels; convert the array before writing, "
            "or write a TIFF to keep floating-point values"
        )
    skimage_io.imsave(out_path, array, check_contrast=False)
    return out_path


def write_mask(path, mask: np.ndarray, *, foreground: int = 255, overwrite: bool = False) -> Path:
    """Write a binary mask as an 8-bit single-channel TIFF.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination file; the suffix must be ``.tif`` or ``.tiff``.
    mask : numpy.ndarray
        Two-dimensional mask; anything truthy is foreground.
    foreground : int, optional
        Value written for foreground pixels, 1 to 255 (default 255). Background is 0.
        :func:`read_mask` accepts either convention.
    overwrite : bool, optional
        Replace an existing file (default ``False``).

    Returns
    -------
    pathlib.Path
        The path written.
    """
    binary = as_bool_mask(mask)
    value = int(foreground)
    if not 1 <= value <= 255:
        raise ValueError("foreground must be between 1 and 255")
    suffix = _suffix(path)
    if suffix not in TIFF_SUFFIXES:
        raise ValueError(f"a mask is written as TIFF, got {suffix!r}")
    return write_image(path, binary.astype(np.uint8) * np.uint8(value), overwrite=overwrite)


def write_labels(path, labels: np.ndarray, *, overwrite: bool = False) -> Path:
    """Write a label image as an unsigned-integer TIFF.

    Label 0 is background. The smallest type that holds the largest label is used:
    16-bit up to 65535 labels, 32-bit above that.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination file; the suffix must be ``.tif`` or ``.tiff``.
    labels : numpy.ndarray
        Two-dimensional integer label image with non-negative values.
    overwrite : bool, optional
        Replace an existing file (default ``False``).

    Returns
    -------
    pathlib.Path
        The path written.
    """
    array = check_2d(labels, "labels")
    if array.dtype.kind not in ("u", "i"):
        raise ValueError("labels must be an integer array")
    if array.size and int(array.min()) < 0:
        raise ValueError("labels must be non-negative (0 = background)")
    suffix = _suffix(path)
    if suffix not in TIFF_SUFFIXES:
        raise ValueError(f"a label image is written as TIFF, got {suffix!r}")
    largest = int(array.max()) if array.size else 0
    dtype = np.uint16 if largest <= 65535 else np.uint32
    return write_image(path, array.astype(dtype), overwrite=overwrite)


def write_table(
    path,
    table: pd.DataFrame,
    *,
    overwrite: bool = False,
    encoding: str = "utf-8-sig",
    escape_formulas: bool = False,
) -> Path:
    """Write a data frame to CSV at full numerical precision.

    Floating-point columns are written with the shortest text that reads back as the
    same double, so a table written here and read back with
    ``pandas.read_csv(..., float_precision='round_trip')`` is value-identical.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination CSV file.
    table : pandas.DataFrame
        Table to write. The index is not written.
    overwrite : bool, optional
        Replace an existing file (default ``False``).
    encoding : str, optional
        Text encoding (default ``'utf-8-sig'``; the byte-order mark makes spreadsheet
        programs open the file as UTF-8).
    escape_formulas : bool, optional
        Prefix text cells that start with ``=``, ``+``, ``-``, ``@``, a tab or a
        carriage return with an apostrophe, so a spreadsheet does not read them as
        formulas (default ``False``, which writes the text unchanged).

    Returns
    -------
    pathlib.Path
        The path written.
    """
    if not isinstance(table, pd.DataFrame):
        raise TypeError("table must be a pandas DataFrame")
    out_path = _prepare_output(path, overwrite)
    frame = table
    if escape_formulas:
        frame = table.copy()
        for column in frame.columns:
            if frame[column].dtype == object:
                frame[column] = frame[column].map(_escape_formula)
    frame.to_csv(out_path, index=False, encoding=encoding)
    return out_path


def _escape_formula(value):
    """Neutralise a text cell that a spreadsheet would read as a formula."""
    if isinstance(value, str) and value[:1] in _FORMULA_LEAD:
        return "'" + value
    return value


# --------------------------------------------------------------------------- calibration

#: Characters a file may use for the micro prefix, all mapped to a plain ``u``: MICRO SIGN
#: (U+00B5), which is what image writers normally emit, and GREEK SMALL LETTER MU (U+03BC).
#: They are written as escapes so that this file stays pure ASCII and cannot be corrupted by
#: being re-saved in a single-byte encoding; dropping either one would make a micrometre unit
#: fall through to metres, a factor of a million, without raising.
_MICRO_SIGNS = ("\u00b5", "\u03bc")


def _unit_slug(unit: str) -> str:
    """Normalise a unit name: lower case, ``u`` for the micro sign, letters and digits only."""
    slug = str(unit).strip().lower()
    for sign in _MICRO_SIGNS:
        slug = slug.replace(sign, "u")
    return re.sub(r"[^a-z0-9]+", "", slug)


def length_unit_to_um(unit: str) -> float:
    """Convert a unit of length given by name into micrometres per unit.

    Parameters
    ----------
    unit : str
        Unit name, for example ``'um'``, the micro sign followed by ``m``, ``'nm'``,
        ``'mm'``, ``'cm'``, ``'m'``, ``'inch'``. Case, spacing and punctuation are
        ignored.

    Returns
    -------
    float
        Micrometres in one unit.

    Raises
    ------
    ValueError
        If the name is empty or is not a unit of length.
    """
    slug = _unit_slug(unit)
    if not slug:
        raise ValueError("the unit of length must not be empty")
    if slug not in _UM_PER_UNIT:
        raise ValueError(f"{unit!r} is not a known unit of length")
    return float(_UM_PER_UNIT[slug])


def _rational_to_float(value) -> float:
    """Read a TIFF rational tag value (numerator, denominator) as a float."""
    if isinstance(value, (tuple, list)) and len(value) == 2:
        numerator, denominator = float(value[0]), float(value[1])
        if denominator == 0:
            raise ValueError("a resolution tag has a zero denominator")
        return numerator / denominator
    return float(value)


def pixel_size_from_tiff(path, *, rtol: float = 1e-3) -> float:
    """Read the pixel size a TIFF states in its own resolution tags.

    The pixel size is ``um_per_unit / resolution``, where ``resolution`` is the
    ``XResolution`` tag (pixels per unit of length) and the unit comes from the
    ``ResolutionUnit`` tag (inch or centimetre; the TIFF default is inch when the tag
    is absent). When that tag says "no unit", the unit is taken from an explicit
    ``unit`` entry in ImageJ metadata if the file carries one. Nothing is guessed: a
    file that states no unit raises, and the caller then passes the pixel size
    explicitly.

    Parameters
    ----------
    path : str or pathlib.Path
        TIFF file to read the tags from. Pixel data are not decoded.
    rtol : float, optional
        Relative tolerance on the difference between the horizontal and vertical
        pixel size (default 1e-3). The package assumes square pixels.

    Returns
    -------
    float
        Pixel size in micrometres per pixel (um/px).

    Raises
    ------
    ValueError
        If the file states no resolution, no usable unit, a non-positive resolution,
        or pixels that are not square within ``rtol``.
    """
    file_path = _check_readable(path, suffixes=TIFF_SUFFIXES)
    with tifffile.TiffFile(file_path) as handle:
        page = handle.pages[0]
        x_tag = page.tags.get("XResolution")
        y_tag = page.tags.get("YResolution")
        unit_tag = page.tags.get("ResolutionUnit")
        imagej = handle.imagej_metadata

    if x_tag is None:
        raise ValueError(f"the file states no XResolution: {file_path.name}")
    unit_code = 2 if unit_tag is None else int(unit_tag.value)
    if unit_code in _RESOLUTION_UNIT_UM:
        um_per_unit = _RESOLUTION_UNIT_UM[unit_code]
    elif unit_code == 1:
        stated = None
        if isinstance(imagej, Mapping):
            stated = imagej.get("unit")
        if stated is None:
            raise ValueError(
                "the file states a resolution without a unit of length; "
                f"pass the pixel size explicitly: {file_path.name}"
            )
        um_per_unit = length_unit_to_um(str(stated))
    else:
        raise ValueError(f"unknown TIFF resolution unit {unit_code}: {file_path.name}")

    x_resolution = _rational_to_float(x_tag.value)
    y_resolution = x_resolution if y_tag is None else _rational_to_float(y_tag.value)
    if x_resolution <= 0 or y_resolution <= 0:
        raise ValueError(f"the file states a non-positive resolution: {file_path.name}")
    pixel_size_x = um_per_unit / x_resolution
    pixel_size_y = um_per_unit / y_resolution
    if not np.isclose(pixel_size_x, pixel_size_y, rtol=float(rtol), atol=0.0):
        raise ValueError(
            f"the file states non-square pixels ({pixel_size_x} vs {pixel_size_y} um/px): "
            f"{file_path.name}"
        )
    return float(pixel_size_x)


def pixel_size_from_scale_bar(bar_length_um: float, bar_length_px: float) -> float:
    """Pixel size from a scale bar measured on the image.

    The bar is measured end to end on the native pixel grid, and
    ``pixel_size_um = bar_length_um / bar_length_px``.

    Parameters
    ----------
    bar_length_um : float
        Physical length the bar stands for, in micrometres.
    bar_length_px : float
        Length of the bar measured on the image, in pixels.

    Returns
    -------
    float
        Pixel size in micrometres per pixel (um/px).
    """
    length_um = float(bar_length_um)
    length_px = float(bar_length_px)
    if length_um <= 0:
        raise ValueError("bar_length_um must be positive")
    if length_px <= 0:
        raise ValueError("bar_length_px must be positive")
    return length_um / length_px


def scale_bar_px_from_layout(
    bar_length_mm: float,
    displayed_extent_mm: float,
    native_extent_px: float,
) -> float:
    """Length of a scale bar in native pixels when the bar was drawn on a layout.

    An image placed on a layout at a known printed size carries
    ``mm_per_px = displayed_extent_mm / native_extent_px`` along that edge, so a bar
    drawn on the layout is ``bar_length_mm / mm_per_px`` native pixels long. Feed the
    result to :func:`pixel_size_from_scale_bar` together with the physical length the
    bar stands for. Lengths given in typographic points are converted with
    :func:`points_to_mm` first.

    Parameters
    ----------
    bar_length_mm : float
        Length of the bar on the layout, measured end to end, in millimetres.
    displayed_extent_mm : float
        Width or height of the placed image on the layout, in millimetres; use the
        same edge as the bar.
    native_extent_px : float
        Width or height of the image in its own pixels, along that same edge.

    Returns
    -------
    float
        Length of the bar in native pixels of the image.
    """
    bar_mm = float(bar_length_mm)
    extent_mm = float(displayed_extent_mm)
    extent_px = float(native_extent_px)
    if bar_mm <= 0:
        raise ValueError("bar_length_mm must be positive")
    if extent_mm <= 0:
        raise ValueError("displayed_extent_mm must be positive")
    if extent_px <= 0:
        raise ValueError("native_extent_px must be positive")
    mm_per_px = extent_mm / extent_px
    return bar_mm / mm_per_px


def points_to_mm(length_pt: float) -> float:
    """Convert a length in typographic points to millimetres (25.4 mm per 72 pt)."""
    return float(length_pt) * MM_PER_POINT


# --------------------------------------------------------------------------- folders

def list_images(folder, *, recursive: bool = True, suffixes=None) -> list:
    """List the image files of a folder in a stable order.

    Symbolic links are not followed, so the listing stays inside the folder.

    Parameters
    ----------
    folder : str or pathlib.Path
        Folder to scan.
    recursive : bool, optional
        Include sub-folders (default ``True``).
    suffixes : iterable of str, optional
        File extensions to keep, with or without the leading dot; case is ignored.
        Defaults to :data:`SUPPORTED_IMAGE_SUFFIXES`.

    Returns
    -------
    list of pathlib.Path
        Absolute paths, sorted case-insensitively by their full path, so the order
        does not depend on the filesystem.
    """
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"image folder not found: {root}")
    allowed = _normalise_suffixes(suffixes)
    found = []
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        if not recursive:
            directory_names[:] = []
        else:
            directory_names[:] = [
                name for name in directory_names if not (Path(current) / name).is_symlink()
            ]
        for file_name in file_names:
            file_path = Path(current) / file_name
            if _suffix(file_path) not in allowed:
                continue
            if file_path.is_symlink() or not file_path.is_file():
                continue
            found.append(file_path)
    return sorted(found, key=lambda item: str(item).casefold())


def image_table(
    folder,
    pattern: str,
    *,
    recursive: bool = True,
    suffixes=None,
    target: str = "name",
    full_match: bool = False,
    on_unmatched: str = "raise",
) -> pd.DataFrame:
    """Build a table of image paths with keys parsed out of the file names.

    The regular expression and its group names are supplied by the caller: each named
    group becomes one column, so the keys of a dataset (for example field, condition,
    channel or time point) are described by the caller and not by this package.

    Parameters
    ----------
    folder : str or pathlib.Path
        Folder to scan, as in :func:`list_images`.
    pattern : str
        Regular expression with at least one named group, ``(?P<name>...)``. No
        default: the naming convention belongs to the data, not to the package.
    recursive : bool, optional
        Include sub-folders (default ``True``).
    suffixes : iterable of str, optional
        File extensions to keep; defaults to :data:`SUPPORTED_IMAGE_SUFFIXES`.
    target : {'name', 'stem', 'relative'}, optional
        Text the expression is applied to: the file name with its suffix (default),
        the file name without its suffix, or the path relative to ``folder`` written
        with forward slashes.
    full_match : bool, optional
        Require the expression to match the whole text (default ``False``, which
        searches for it anywhere in the text).
    on_unmatched : {'raise', 'skip'}, optional
        What to do with a file the expression does not match (default ``'raise'``).

    Returns
    -------
    pandas.DataFrame
        One row per image, in the order of :func:`list_images`, with columns ``path``
        (absolute path as text), ``name`` (file name) and one column per named group.
        A group that did not take part in the match is ``None``.

    Raises
    ------
    ValueError
        If the expression has no named group, if a group is called ``path`` or
        ``name``, if ``target`` or ``on_unmatched`` is unknown, or if a file does not
        match while ``on_unmatched='raise'``.
    """
    if target not in ("name", "stem", "relative"):
        raise ValueError("target must be 'name', 'stem' or 'relative'")
    if on_unmatched not in ("raise", "skip"):
        raise ValueError("on_unmatched must be 'raise' or 'skip'")
    expression = re.compile(pattern)
    if not expression.groupindex:
        raise ValueError(
            "pattern must contain at least one named group '(?P<key>...)'; "
            "the group names become the columns of the table"
        )
    key_names = [
        name for name, _ in sorted(expression.groupindex.items(), key=lambda item: item[1])
    ]
    clashes = [name for name in key_names if name in _RESERVED_TABLE_COLUMNS]
    if clashes:
        raise ValueError(
            f"group names {clashes} clash with the fixed columns {list(_RESERVED_TABLE_COLUMNS)}"
        )

    root = Path(folder).expanduser().resolve()
    rows = []
    for file_path in list_images(root, recursive=recursive, suffixes=suffixes):
        if target == "name":
            text = file_path.name
        elif target == "stem":
            text = file_path.stem
        else:
            text = file_path.relative_to(root).as_posix()
        match = expression.fullmatch(text) if full_match else expression.search(text)
        if match is None:
            if on_unmatched == "raise":
                raise ValueError(f"the pattern does not match {text!r}")
            continue
        row = {"path": str(file_path), "name": file_path.name}
        row.update({name: match.group(name) for name in key_names})
        rows.append(row)
    columns = list(_RESERVED_TABLE_COLUMNS) + key_names
    return pd.DataFrame(rows, columns=columns)
