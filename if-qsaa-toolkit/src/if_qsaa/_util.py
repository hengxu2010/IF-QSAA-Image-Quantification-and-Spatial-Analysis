"""Small shared helpers: array checks, unit conversion, structuring elements, labelling, RNG.

Everything in this module is generic. No analysis logic and no default that depends on a
particular dataset lives here.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import ndimage as ndi

__all__ = [
    "as_float",
    "check_2d",
    "check_same_shape",
    "as_bool_mask",
    "um_to_px",
    "px_to_um",
    "area_um2_to_px",
    "area_px_to_um2",
    "pixel_area_um2",
    "disk",
    "structure",
    "label_mask",
    "default_rng",
    "mad",
]


# --------------------------------------------------------------------------- arrays

def as_float(image: np.ndarray, dtype: type = np.float64) -> np.ndarray:
    """Return ``image`` as a float array, without rescaling the values.

    Integer images keep their grey levels (a 12-bit camera value of 2048 stays 2048.0),
    so every intensity parameter of this package is expressed in the units of the input.
    """
    arr = np.asarray(image)
    if arr.dtype == np.bool_:
        return arr.astype(dtype)
    return arr.astype(dtype, copy=False)


def check_2d(image: np.ndarray, name: str = "image") -> np.ndarray:
    """Validate that ``image`` is two-dimensional and return it as an array."""
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {arr.shape}")
    return arr


def check_same_shape(*arrays: np.ndarray) -> None:
    """Validate that all arrays share one shape."""
    shapes = {np.asarray(a).shape for a in arrays}
    if len(shapes) > 1:
        raise ValueError(f"arrays must share one shape, got {sorted(shapes)}")


def as_bool_mask(mask: np.ndarray, name: str = "mask") -> np.ndarray:
    """Validate a 2-D mask and return it as ``bool``."""
    arr = check_2d(mask, name)
    return arr.astype(bool, copy=False)


# --------------------------------------------------------------------------- units

def um_to_px(length_um: float, pixel_size_um: float) -> float:
    """Convert a length in micrometres to pixels."""
    if pixel_size_um <= 0:
        raise ValueError("pixel_size_um must be positive")
    return float(length_um) / float(pixel_size_um)


def px_to_um(length_px: float, pixel_size_um: float) -> float:
    """Convert a length in pixels to micrometres."""
    return float(length_px) * float(pixel_size_um)


def pixel_area_um2(pixel_size_um: float) -> float:
    """Area of one pixel in square micrometres (square pixels are assumed)."""
    if pixel_size_um <= 0:
        raise ValueError("pixel_size_um must be positive")
    return float(pixel_size_um) ** 2


def area_um2_to_px(area_um2: float, pixel_size_um: float) -> float:
    """Convert an area in square micrometres to a number of pixels."""
    return float(area_um2) / pixel_area_um2(pixel_size_um)


def area_px_to_um2(area_px: float, pixel_size_um: float) -> float:
    """Convert a number of pixels to an area in square micrometres."""
    return float(area_px) * pixel_area_um2(pixel_size_um)


# --------------------------------------------------------------------------- shapes

def disk(radius_px: float) -> np.ndarray:
    """Flat disk-shaped structuring element of the given radius in pixels.

    A pixel belongs to the element when its centre lies within ``radius_px`` of the
    centre pixel. ``radius_px < 1`` gives the 4-connected cross used as the smallest
    isotropic element.
    """
    r = int(np.floor(float(radius_px)))
    if r < 1:
        return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return ((x * x + y * y) <= float(radius_px) ** 2).astype(np.uint8)


def structure(connectivity: int = 8) -> np.ndarray:
    """Structuring element for connected-component work: 4- or 8-connectivity."""
    if connectivity == 4:
        return ndi.generate_binary_structure(2, 1)
    if connectivity == 8:
        return ndi.generate_binary_structure(2, 2)
    raise ValueError("connectivity must be 4 or 8")


def label_mask(mask: np.ndarray, connectivity: int = 8) -> tuple[np.ndarray, int]:
    """Label the connected components of a binary mask.

    Returns the label image (0 = background, components numbered from 1) and the number
    of components.
    """
    labels, n = ndi.label(as_bool_mask(mask), structure=structure(connectivity))
    return labels, int(n)


# --------------------------------------------------------------------------- misc

def default_rng(seed: int | np.random.Generator | None = None) -> np.random.Generator:
    """Return a NumPy random generator; pass an int for a reproducible run."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def mad(values: Iterable[float], centre: float | None = None, scale: float = 1.4826) -> float:
    """Median absolute deviation, scaled to estimate a Gaussian standard deviation.

    ``scale=1.4826`` makes the result comparable with a standard deviation;
    pass ``scale=1.0`` for the raw MAD.
    """
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    c = float(np.median(arr)) if centre is None else float(centre)
    return float(scale * np.median(np.abs(arr - c)))
