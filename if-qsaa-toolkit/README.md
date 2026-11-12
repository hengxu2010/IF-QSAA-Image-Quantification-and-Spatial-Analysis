# if-qsaa-toolkit

Reference implementation of the image-analysis and statistical methods behind the **same-section**
analyses of **IF-QSAA** (immunofluorescence-integrated quiescent seed amplification assay), as
described in the accompanying publication, *Spatial mapping of α-synuclein seeding activity and
pathology using immunofluorescence-integrated quiescent seed amplification* (manuscript under
review). In IF-QSAA an in situ quiescent seed amplification assay, read out with thioflavin T on
tissue sections, is combined on the *same* section with immunofluorescence. This package quantifies
the spatial relation between the amplified products and (i) cell-marker territories and (ii) a
second protein-pathology readout: contact against a spatial null model, object-to-object pairing
against a translation control, and concentric boundary-distance profiling.

Supply the pixel size, spectral coupling coefficients, intensity thresholds and experimental
group labels for your data through the function arguments.

## Installation

From the repository root, enter the package directory before installing:

```bash
cd if-qsaa-toolkit
python -m pip install -e .
```

To install the test dependencies and run the test suite, stay in that directory:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

With the dependencies installed, the tests can also run without installing the package itself:
run `python -m pytest -q` from `if-qsaa-toolkit/`. Scripts can reach the package with
`PYTHONPATH=src` from that same directory. The example commands below also use
`if-qsaa-toolkit/` as their working directory.

Python ≥ 3.9. Dependencies: numpy, scipy, pandas, scikit-image, tifffile. The test suite and the
examples were last run on Python 3.13 with numpy 2.2, scipy 1.16, pandas 2.3, scikit-image 0.26 and
tifffile 2026.5.

## Modules

| Module | What it computes |
| --- | --- |
| `if_qsaa.io` | reading images, channels and masks; writing masks, label images and tables; pixel size from image metadata or from a scale bar |
| `if_qsaa.background` | rolling-ball and Gaussian background estimation, background correction filters, censored noise estimation and the threshold rule `background + k · noise` |
| `if_qsaa.segmentation` | the object-segmentation core: local signal-to-noise response, thresholding with hysteresis growth, mask clean-up, splitting of touching objects, per-object measurement and filtering |
| `if_qsaa.learned_detector` | optional learned detector: six interpretable per-candidate features, per-image robust normalisation, logistic scoring and fitting on labelled candidates |
| `if_qsaa.unmixing` | spectral cross-talk: coefficient calibration from single-source controls, fixed-coefficient subtraction, per-field empirical estimation with 2 × 2 linear unmixing, bleed-through into the nuclear channel and the nuclei mask read from it, residual cross-talk object filters |
| `if_qsaa.contact` | object-to-territory contact scoring and the rigid-transformation randomisation null model: contact enrichment, empirical P values, the studentised whole-curve envelope test, and a parameter-sensitivity runner |
| `if_qsaa.pairing` | same-section pairing of two markers' objects by overlap of the smaller object, class assignment of paired and single-positive objects, and the translation (mask-shift) control with enrichment at any grouping level |
| `if_qsaa.annuli` | concentric boundary-distance profiles: coverage of each distance band around an object by the other marker's mask, against the whole-field baseline, per distance and in strata of partner size |
| `if_qsaa.stats` | the statistical procedures of the same-section analyses: exact Wilcoxon signed-rank (exact up to 100 pairs by default, the normal approximation above), Kruskal-Wallis, Friedman, percentile bootstrap confidence intervals, two-sided empirical P values from a randomisation distribution |

## How the parameters work

The API uses method settings, imaging parameters and experimental labels:

1. **Rules of the method** carry defaults, because they define the method: for example a positivity
   pairing rule of ≥ 10 % overlap of the smaller object, 10,000 randomisations for the contact
   null model, 1,000 shifts for the translation control, a 2 µm contact ring with 5 % occupancy,
   8-connectivity for objects.
2. **Properties of the user's imaging** have **no defaults** and must be passed: `pixel_size_um`,
   spectral coupling coefficients, intensity floors, bit depth, saturation level. One exception:
   the learned detector's settings (`DetectorParams`, `measure_objects`) start from generic 16-bit
   defaults (full scale 65535 and saturation 65000 grey levels), which have to be replaced
   with the values of the user's own detector. Its reference channel is subtracted with unit
   weight, which is the detector's own rule; pass a coefficient calibrated on single-stained
   controls to subtract more or less.
3. **Properties of the user's experiment** are supplied as data: group membership, the labels by
   which fields are aggregated, which fields pass quality control.

Spatial parameters are given in micrometres and converted internally with `pixel_size_um`, so the
same settings transfer between objectives and cameras. The learned detector is the exception here
too: its `DetectorParams` are in pixels of its working grid, so build them for your own pixel size
with `DetectorParams.from_micrometres`; and the per-field cross-talk estimator takes its minimum
structure sizes as pixel counts (`marker_min_px`, `alpha_min_px`, `gamma_min_px`). Intensity
parameters are in the grey levels of the input image, which is never rescaled on load.

Every randomised procedure takes a `seed`, and the same seed reproduces the same result.

Results come back as dataclasses, dictionaries or `pandas.DataFrame` objects, never as formatted
text. A P value is named `P` on the dataclasses that carry a single test result (the procedures in
`if_qsaa.stats` and `if_qsaa.contact.EnvelopeTestResult`) and `p_value` where it is one key or one
column among many, as in the tables and dictionaries of `if_qsaa.pairing` and `if_qsaa.annuli`.

## Adapting it to your own data

1. Measure your pixel size (`if_qsaa.io.pixel_size_from_scale_bar` or the image metadata).
2. Calibrate your own spectral coupling coefficients on single-stained control sections
   (`if_qsaa.unmixing.calibrate_coupling_coefficient`) for your microscope and filter set.
3. Set the segmentation parameters on a few fields, then keep them fixed across conditions, and
   report the sensitivity of your conclusion with `if_qsaa.contact.run_contact_sensitivity`.
4. Fit the learned detector on your own labelled candidates if you use it
   (`if_qsaa.learned_detector.fit_learned_model`). Re-check the model and threshold when imaging
   or staining conditions change.

## Examples

`examples/` contains one runnable script per analysis. Each script generates synthetic data
with a fixed seed, runs the analysis and prints the recovered values alongside the known
input values.

```bash
python examples/contact_demo.py            # after pip install -e .
PYTHONPATH=src python examples/contact_demo.py   # without installing
```

## Tests

`tests/` contains unit tests on synthetic data with known ground truth, including the degenerate
cases (empty masks, single objects, objects at the border, thresholds no pixel reaches, ties in
rank tests).

## Citation

If you use this code, please cite the accompanying publication (manuscript under review; the full
reference will be added here on publication).

## License

MIT. See `LICENSE`.
