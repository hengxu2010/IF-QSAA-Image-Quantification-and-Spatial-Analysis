"""Spectral cross-talk removal between two detection channels, end to end.

All input data are SYNTHETIC: the script plants a known coupling between a bright
"source" channel and a second detection channel with a fixed-seed numpy Generator, then
recovers it with ``if_qsaa.unmixing``. No image is read, nothing is written. It shows
(1) calibration of the coupling coefficient on a single-source control field, with the
least-squares and with the robust Theil-Sen estimator; (2) fixed-coefficient subtraction
of the cross-talk from a co-stained field; (3) the per-field empirical route, which
estimates both coefficients from the co-stained field itself and inverts the 2 x 2
mixing; and (4) the robustness switch, one measurement with and without the correction.

What to look for: every estimated coefficient and count is printed next to the planted
value; the median absolute error of channel two inside the source footprint falls by
more than an order of magnitude once the cross-talk is removed, while pixels outside it
stay as they were; and the apparent channel-two area inside the footprint collapses to
zero, the planted truth there.
"""

from __future__ import annotations

import numpy as np

from if_qsaa.unmixing import (calibrate_coupling_coefficient, clipped_fraction,
                              compare_unmixing, estimate_crosstalk, subtract_crosstalk,
                              unmix_two_channel, unsaturated_mask)

# Quantities a user must measure on their own images, in the grey levels of their own
# camera; these suit only the synthetic field below, so REPLACE EVERY ONE of them.
SOURCE_MIN_LEVEL = 100.0    # source-positive pixels used for the calibration
SATURATION_LEVEL = 4095.0   # detector saturation of the raw channels
MARKER_THRESHOLD = 25.0     # own-signal threshold of channel two
BRIGHT_SOURCE_LEVEL = 90.0  # floor of the bright-source selection
MIN_SOURCE_RANGE = 8.0      # minimum intensity span inside a source structure

# Synthetic ground-truth parameters.
PLANTED_BETA = 0.12         # source channel into channel two
PLANTED_ALPHA = 0.10        # channel two into the source channel
BACKGROUND_LEVEL = 60.0     # pedestal of the raw channels (real data: if_qsaa.background)
CONTROL_OFFSET = 8.0        # own background of channel two in the control field
SOURCE_AMPLITUDE, MARKER_AMPLITUDE, NOISE_SD = 400.0, 120.0, 1.5
FIELD_SHAPE, CONTROL_SHAPE, SEED = (300, 300), (200, 200), 4321

# Both robust estimators cost O(n^2) in the number of pixels, so the demo caps them.
CALIBRATION_MAX_PIXELS = 2500  # the module itself applies no cap
STRUCTURE_SUBSAMPLE_PX = 150   # the module's own setting is 400 pixels per structure


# -- synthesise ---------------------------------------------------------------

def _disks(shape, centres, radius_px, amplitude):
    """Disks with a linear radial falloff, so intensity spans a range inside each."""
    rows, cols = np.ogrid[: shape[0], : shape[1]]
    out = np.zeros(shape, dtype=float)
    for centre in centres:
        distance = np.hypot(rows - centre[0], cols - centre[1])
        inside = distance <= radius_px
        out[inside] = amplitude * (1.0 - 0.5 * distance[inside] / radius_px)
    return out


def _cells(step, n, parity):
    """Centres of a square grid of ``n`` by ``n`` cells whose index sum has ``parity``."""
    return [(step + step * i, step + step * j)
            for i in range(n) for j in range(n) if (i + j) % 2 == parity]


def synthesise_control(rng):
    """Single-source control: bright source objects, no own signal in channel two."""
    centres = _cells(33, 5, 0) + _cells(33, 5, 1)
    source = BACKGROUND_LEVEL + _disks(CONTROL_SHAPE, centres, 7.0, SOURCE_AMPLITUDE)
    source += rng.normal(0.0, NOISE_SD, CONTROL_SHAPE)
    observed = (CONTROL_OFFSET + PLANTED_BETA * source
                + rng.normal(0.0, NOISE_SD, CONTROL_SHAPE))
    return {"source": source, "observed": observed, "n_objects": len(centres)}


def synthesise_costained(rng):
    """Co-stained field: small source objects, larger channel-two structures, mixed."""
    structures = _cells(37, 7, 0)
    source = _disks(FIELD_SHAPE, _cells(37, 7, 1), 3.0, SOURCE_AMPLITUDE)
    marker = _disks(FIELD_SHAPE, structures, 8.0, MARKER_AMPLITUDE)
    noise = rng.normal(0.0, NOISE_SD, (2,) + FIELD_SHAPE)
    raw_source = BACKGROUND_LEVEL + source + PLANTED_ALPHA * marker + noise[0]
    raw_marker = BACKGROUND_LEVEL + marker + PLANTED_BETA * source + noise[1]
    # The per-field route works on background residuals; the planted pedestal is removed
    # here, whereas on real images the residuals come from if_qsaa.background.
    return {"source_raw": raw_source, "marker_raw": raw_marker,
            "source_res": raw_source - BACKGROUND_LEVEL,
            "marker_res": raw_marker - BACKGROUND_LEVEL,
            "marker_truth": BACKGROUND_LEVEL + marker,  # what channel two should read
            "marker_res_truth": marker, "footprint": source > 0.0,
            "n_structures": len(structures)}


# -- analyse ------------------------------------------------------------------

def _error(estimate, truth, where):
    """Median absolute difference from the planted truth, in grey levels."""
    return float(np.median(np.abs(estimate[where] - truth[where])))


def _measure(params):
    """Headline numbers of one field, with or without the cross-talk correction."""
    field = params["field"]
    marker = field["marker_res"]
    if params["unmix"]:
        # nuclear_corr is None: this field has only the two channels, so no nuclear term
        _, marker = unmix_two_channel(field["source_res"], marker, None,
                                      params["alpha"], params["beta"])
    positive = marker > MARKER_THRESHOLD
    return {"channel_two_area_px": int(positive.sum()),
            "area_in_footprint_px": int((positive & field["footprint"]).sum()),
            "median_abs_error": round(
                _error(marker, field["marker_res_truth"], field["footprint"]), 2)}


def analyse(control, field):
    """Calibrate, subtract, estimate per field, and run the correction on and off."""
    out = {}
    for estimator in ("least_squares", "theil_sen"):
        out[estimator] = calibrate_coupling_coefficient(
            control["observed"], control["source"], estimator=estimator,
            source_min_level=SOURCE_MIN_LEVEL, saturation_level=SATURATION_LEVEL,
            max_pixels=CALIBRATION_MAX_PIXELS, seed=SEED)
    # The raw channels carry a pedestal, which must not be scaled by the coefficient: it
    # would be subtracted from the whole marker channel as a constant offset. It is given
    # here in grey levels; source_background_percentile=50.0 is the same correction read
    # off the source image itself, for a field whose pedestal is not known separately.
    args = (field["marker_raw"], field["source_raw"], out["theil_sen"]["coefficient"])
    pedestal = dict(source_background=BACKGROUND_LEVEL)
    corrected = subtract_crosstalk(*args, **pedestal)
    # The clipped fraction is read on raw images, which still sit above zero; on the
    # background residuals used below, half the background goes negative in any case.
    out["clipped"] = clipped_fraction(*args, **pedestal)
    out["errors"] = {name: (_error(field["marker_raw"], field["marker_truth"], where),
                            _error(corrected, field["marker_truth"], where))
                     for name, where in (("in", field["footprint"]),
                                         ("out", ~field["footprint"]))}
    # Two channels only: nuclear_corr and nuclei are left out, so gamma stays zero and
    # no placeholder image is needed; tissue defaults to the whole field.
    out["per_field"] = estimate_crosstalk(
        field["source_res"], field["marker_res"],
        usable=unsaturated_mask([field["source_raw"], field["marker_raw"]],
                                SATURATION_LEVEL),
        marker_thr=MARKER_THRESHOLD,
        beta_min_level=BRIGHT_SOURCE_LEVEL, min_source_range=MIN_SOURCE_RANGE,
        max_px=STRUCTURE_SUBSAMPLE_PX, seed=SEED)
    # unmixing_off_overrides() switches three steps off; the only one this measurement
    # has is the unmixing itself, and every overridden key must exist in base_params.
    out["switch"] = compare_unmixing(_measure, {
        "field": field, "unmix": True,
        "alpha": out["per_field"]["alpha"], "beta": out["per_field"]["beta"]},
        off_overrides={"unmix": False})
    out["planted_area_px"] = int((field["marker_res_truth"] > MARKER_THRESHOLD).sum())
    return out


# -- report -------------------------------------------------------------------

def _row(label, value, truth=None):
    print(f"  {label:<41}: {value}" + ("" if truth is None else f"   (planted {truth})"))


def report(control, field, out):
    ls, ts = out["least_squares"], out["theil_sen"]
    coef, err = out["per_field"], out["errors"]
    print("Spectral cross-talk removal, synthetic two-channel field")
    print(f"\nControl field: {control['n_objects']} source objects, no channel-two signal")
    _row("coupling coefficient, least squares", f"{ls['coefficient']:.4f}", PLANTED_BETA)
    _row("coupling coefficient, Theil-Sen", f"{ts['coefficient']:.4f}", PLANTED_BETA)
    _row("channel-two background (intercept)", f"{ls['intercept']:.2f}", CONTROL_OFFSET)
    _row("pixels fitted / passing selection", f"{ts['n_pixels']} / {ts['n_selected']}")
    print("\nFixed-coefficient subtraction, coefficient taken from the control field")
    _row("median |channel two - truth| in footprint",
         f"{err['in'][0]:.2f} before -> {err['in'][1]:.2f} after")
    _row("the same, outside the footprint",
         f"{err['out'][0]:.2f} before -> {err['out'][1]:.2f} after")
    _row("pixels clipped at zero", f"{100 * out['clipped']:.2f} %")
    print("\nPer-field empirical estimation, same field, no control section")
    _row("alpha, channel two into source", f"{coef['alpha']:.4f}", PLANTED_ALPHA)
    _row("beta, source into channel two", f"{coef['beta']:.4f}", PLANTED_BETA)
    _row("structures fitted for alpha", coef["n_marker_structures"], field["n_structures"])
    _row("bright pixels for beta",
         f"{coef['n_beta_pixels']} at >= {coef['beta_select_level']:.0f} grey levels")
    print("\nRobustness switch: one measurement with and without the correction")
    print(out["switch"].to_string(index=False))
    print("  channel two has no planted signal inside the source footprint, so pixels")
    print(f"  counted there are residual cross-talk; planted area elsewhere: "
          f"{out['planted_area_px']} px.")


def main():
    rng = np.random.default_rng(SEED)
    control = synthesise_control(rng)
    field = synthesise_costained(rng)
    report(control, field, analyse(control, field))


if __name__ == "__main__":
    main()
