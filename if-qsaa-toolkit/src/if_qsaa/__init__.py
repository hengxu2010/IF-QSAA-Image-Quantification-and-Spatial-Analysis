"""Image-analysis and statistics core of IF-QSAA.

IF-QSAA (immunofluorescence-integrated quiescent seed amplification assay) combines in situ seed
amplification with immunofluorescence on the same tissue section. The package holds the
image-analysis methods behind the same-section analyses of the accompanying publication:
segmentation of amplification products and of immunofluorescence readouts on one section,
correction of spectral cross-talk between them, contact between products and cell-marker
territories against a spatial null model, pairing of two readouts' objects with a translation
control, and concentric boundary-distance profiling around paired and unpaired objects. Supply
microscope, section and experiment parameters through the function arguments.

Modules
-------
io
    Reading images, channels and masks; writing masks, label images and tables; pixel size from
    image metadata or from a scale bar.
background
    Rolling-ball and Gaussian background estimation, background correction filters, censored noise
    estimation and the ``background + k * noise`` threshold rule.
segmentation
    Object segmentation: local signal-to-noise response, thresholding with hysteresis growth, mask
    clean-up, splitting of touching objects, per-object measurement and filtering.
learned_detector
    Optional learned detector over six interpretable features, with a fitting routine for
    training on labelled candidates.
unmixing
    Spectral cross-talk: coefficient calibration, fixed-coefficient subtraction, per-field empirical
    estimation with linear unmixing, bleed-through into the nuclear channel and the nuclear mask
    derived from it, residual-cross-talk object filters.
contact
    Contact between objects and cell-marker territories with a rigid-transformation null model,
    enrichment, empirical P values, the whole-curve envelope test and a parameter sensitivity grid.
pairing
    Pairing of two markers' objects in one section by shared area, the three-way classification of
    paired and single-positive objects, the translation control and enrichment at each level of
    aggregation.
annuli
    Concentric boundary-distance coverage profiles around objects, compared against a whole-field
    baseline, per distance and in strata of partner size.
stats
    Exact Wilcoxon signed-rank, Kruskal-Wallis, Friedman, percentile bootstrap intervals and
    two-sided empirical P values from a randomisation distribution.

Conventions
-----------
Lengths are given in micrometres and areas in square micrometres; a function that has to convert
between pixels and micrometres takes ``pixel_size_um`` explicitly. Intensity
parameters are in the grey levels of the input image, which is never rescaled on load. Randomised
procedures take a ``seed`` and reproduce their result for a given seed. Results come back as
dataclasses, dictionaries or :class:`pandas.DataFrame`; the statistical procedures name their P
value field ``P``, and the spatial procedures use the key or column ``p_value``.
"""

from __future__ import annotations

__version__ = "1.0.0"

from if_qsaa import (  # noqa: F401
    annuli,
    background,
    contact,
    io,
    learned_detector,
    pairing,
    segmentation,
    stats,
    unmixing,
)

__all__ = [
    "annuli",
    "background",
    "contact",
    "io",
    "learned_detector",
    "pairing",
    "segmentation",
    "stats",
    "unmixing",
    "__version__",
]
