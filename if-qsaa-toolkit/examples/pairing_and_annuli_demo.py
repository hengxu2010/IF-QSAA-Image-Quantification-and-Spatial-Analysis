"""Two-marker pairing in one field, and the distance profile around single-positive objects.

What it shows
    One synthetic field carries two binary masks: disks of marker A, and disks of marker B
    of which a known number sit on top of an A object, a second known number in a tight ring
    just outside an A object, and the rest anywhere else. The analysis then runs end to end:
    label and pair the masks by the overlap rule, classify every object, summarise the field,
    recount the pairs under random whole-mask translations for the enrichment over chance
    (if_qsaa.pairing), and profile the coverage of mask B in concentric boundary-distance
    bands around the single-positive A objects, against mask B's whole-field occupancy
    (if_qsaa.annuli).

What to look for in the output
    * every recovered count equals the planted one, and each mask's classes add up;
    * the observed pair count lies far above the translated controls, at the smallest
      empirical P value this number of shifts can give;
    * the single-positive objects' own footprints hold no mask B, which is what their class
      means, while the excess coverage peaks in the inner bands that hold the planted ring
      and fades to zero further out.

The example uses synthetic disk arrangements. No file is read or written, and nothing
is plotted.
"""

from __future__ import annotations

import numpy as np

from if_qsaa import annuli, pairing

# --- quantities the user must measure on their own images ---------------------------
PIXEL_SIZE_UM = 0.25        # REPLACE: micrometres per pixel of your objective and camera
# REPLACE: shift lengths of the translation control. They depend on the field size and on
# the spatial scale of the structures, so the library offers no default.
SHIFT_MIN_UM, SHIFT_MAX_UM = 5.0, 20.0

# --- rules of the method -----------------------------------------------------------
MIN_SHARED_FRAC = 0.10      # a pair shares >= 10 % of the smaller object's area
CONNECTIVITY = 8            # 8-connectivity groups pixels into objects

# --- settings of this example -------------------------------------------------------
# kept small so this example runs in seconds; the method's own setting for the
# translation control is 1,000 shifts per field
N_CONTROL_DRAWS = 200
SEED, CONTROL_SEED = 12345, 999
# Example band edges sized to the small synthetic objects below; choose the bands for the
# size of your own structures.
BAND_EDGES_UM = (0.0, 1.0, 2.0, 3.0, 5.0, 8.0)

# --- synthetic field geometry ----------------------------------------------------
FIELD_SHAPE, MARGIN_PX = (480, 480), 10
RADIUS_A_PX, RADIUS_B_PX = 6, 4
GRID, JITTER_PX = 6, 12     # A objects sit in jittered cells of a 6 x 6 grid
N_A_OBJECTS, N_PLANTED_PAIRS, N_PLANTED_RING, N_PLANTED_ELSEWHERE = 24, 10, 12, 40
RING_GAP_PX = (1, 5)        # nominal gap between the two disk boundaries, in pixels,
                            # drawn as a half-open range
N_B_OBJECTS = N_PLANTED_PAIRS + N_PLANTED_RING + N_PLANTED_ELSEWHERE
N_A_ONLY, N_B_ONLY = N_A_OBJECTS - N_PLANTED_PAIRS, N_B_OBJECTS - N_PLANTED_PAIRS


# ----------------------------------------------------------------------- synthesise

def _disk(radius_px: int) -> np.ndarray:
    y, x = np.ogrid[-radius_px : radius_px + 1, -radius_px : radius_px + 1]
    return (x * x + y * y) <= radius_px ** 2


def _stamp(mask: np.ndarray, cy: int, cx: int, disk: np.ndarray) -> None:
    r = disk.shape[0] // 2
    mask[cy - r : cy + r + 1, cx - r : cx + r + 1] |= disk


def _hits(mask: np.ndarray, cy: int, cx: int, probe: np.ndarray) -> bool:
    """True when the probe disk placed at (cy, cx) meets an object of ``mask``."""
    r = probe.shape[0] // 2
    return bool((mask[cy - r : cy + r + 1, cx - r : cx + r + 1] & probe).any())


def synthesise(rng):
    """Two masks of one synthetic field, with a known number of planted pairs."""
    disk_a, disk_b = _disk(RADIUS_A_PX), _disk(RADIUS_B_PX)
    apart = _disk(RADIUS_B_PX + 2)   # a 2 px gap keeps two B objects from merging into one
    mask_a, mask_b = (np.zeros(FIELD_SHAPE, dtype=bool) for _ in range(2))
    cell = FIELD_SHAPE[0] // GRID

    centres = []   # one A object per drawn cell, so neighbours cannot merge
    for index in rng.choice(GRID * GRID, N_A_OBJECTS, replace=False):
        jy, jx = rng.integers(-JITTER_PX, JITTER_PX + 1, 2)
        cy = int(index // GRID) * cell + cell // 2 + int(jy)
        cx = int(index % GRID) * cell + cell // 2 + int(jx)
        centres.append((cy, cx))
        _stamp(mask_a, cy, cx, disk_a)

    hosts = rng.permutation(N_A_OBJECTS)
    for k, host in enumerate(hosts[: N_PLANTED_PAIRS + N_PLANTED_RING]):
        if k < N_PLANTED_PAIRS:   # well inside its host: shared area far above the rule
            radius = float(rng.integers(0, RADIUS_A_PX - 2))
        else:                     # in a tight ring just outside its host
            radius = float(RADIUS_A_PX + RADIUS_B_PX + rng.integers(*RING_GAP_PX))
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        cy = int(round(centres[host][0] + radius * np.sin(angle)))
        cx = int(round(centres[host][1] + radius * np.cos(angle)))
        _stamp(mask_b, cy, cx, disk_b)

    placed = N_PLANTED_PAIRS + N_PLANTED_RING
    while placed < N_B_OBJECTS:   # anywhere else, sharing no pixel with an A object
        cy, cx = (int(rng.integers(MARGIN_PX, n - MARGIN_PX)) for n in FIELD_SHAPE)
        if not (_hits(mask_a, cy, cx, disk_b) or _hits(mask_b, cy, cx, apart)):
            _stamp(mask_b, cy, cx, disk_b)
            placed += 1
    return mask_a, mask_b


# -------------------------------------------------------------------------- analyse

def analyse(mask_a, mask_b):
    """Pair the two masks, run the translation control, then profile the distances."""
    result = pairing.pair_objects(
        mask_a, mask_b, min_shared_frac=MIN_SHARED_FRAC, connectivity=CONNECTIVITY
    )
    classes = pairing.classify_objects(result, PIXEL_SIZE_UM, field_id="synthetic")
    summary = pairing.field_summary(result, PIXEL_SIZE_UM, field_id="synthetic")
    shifts = pairing.draw_displacements(
        N_CONTROL_DRAWS, min_length_um=SHIFT_MIN_UM, max_length_um=SHIFT_MAX_UM,
        pixel_size_um=PIXEL_SIZE_UM, seed=CONTROL_SEED,
    )
    control = pairing.translation_control(
        mask_a, mask_b, shifts, PIXEL_SIZE_UM,
        min_shared_frac=MIN_SHARED_FRAC, connectivity=CONNECTIVITY,
    )
    enriched = pairing.enrichment(result.n_pairs, control["n_pairs"])

    single_a = classes.loc[
        (classes["mask"] == "a") & (classes["object_class"] == pairing.A_ONLY), "label"
    ].to_numpy()
    # The footprints of the other reference objects are taken out of every band, so a
    # neighbour's own load of mask B cannot leak into this object's profile.
    profile = annuli.profile_objects(
        result.labels_a, mask_b, BAND_EDGES_UM, PIXEL_SIZE_UM, labels=single_a,
        include_inside=True, exclude_other_reference_objects=True, field_id="synthetic",
    )
    # The unit of analysis of the method is the subject; this example has one field, so
    # each reference object stands in as a unit of the band summary.
    bands = annuli.profile_summary(annuli.unit_profiles(profile, "object_label"))
    return classes, summary, enriched, profile, bands


# --------------------------------------------------------------------------- report

def report(classes, summary, enriched, profile, bands) -> None:
    f, e = summary.iloc[0], enriched
    seen = classes.groupby(["mask", "object_class"]).size().to_dict()
    print(f"Synthetic field {FIELD_SHAPE[0]} x {FIELD_SHAPE[1]} px at {PIXEL_SIZE_UM} um/px"
          f" = {f['area_mm2']:.4f} mm^2; a pair shares >= {100 * MIN_SHARED_FRAC:.0f} % of"
          f" the smaller object\n\nRecovered (planted in brackets)")
    for name, found, truth in (
        ("marker A objects", f["n_objects_a"], N_A_OBJECTS),
        ("marker B objects", f["n_objects_b"], N_B_OBJECTS),
        ("dual-positive A", seen.get(("a", pairing.DUAL), 0), N_PLANTED_PAIRS),
        ("single-positive A", seen.get(("a", pairing.A_ONLY), 0), N_A_ONLY),
        ("dual-positive B", seen.get(("b", pairing.DUAL), 0), N_PLANTED_PAIRS),
        ("single-positive B", seen.get(("b", pairing.B_ONLY), 0), N_B_ONLY),
        ("pairs", f["n_pairs"], N_PLANTED_PAIRS),
    ):
        print(f"  {name:<18}{int(found):4d}   (planted: {truth})")
    print(f"  {'pair density':<18}{f['pair_density_per_mm2']:4.0f}   per mm^2\n")

    # the draw count sets the smallest P value the control can report
    floor = 2.0 / (e["n_draws"] + 1)
    floor_note = (f"no control draw reached the observed count, so P sits on the smallest"
                  f" value {e['n_draws']} shifts can give"
                  if e["n_control_ge"] == 0 else
                  f"the smallest P value {e['n_draws']} shifts can give")
    print(f"Translation control: {e['n_draws']} shifts of {SHIFT_MIN_UM:.0f}-"
          f"{SHIFT_MAX_UM:.0f} um (the method's own setting is 1,000)\n"
          f"  observed {e['observed']:.0f} pairs; controls {e['control_mean']:.2f} +-"
          f" {e['control_sd']:.2f} (central 95 %: {e['band_lo']:.0f} to {e['band_hi']:.0f})\n"
          f"  enrichment {e['enrichment']:.1f}-fold over chance; P = {e['p_value']:.4f}"
          f" (two-sided, empirical)\n"
          f"  {floor_note}: 2 / (n + 1) = {floor:.4f}\n")

    print(f"Mask B around the {int(bands['n_units'].max())} single-positive A objects,"
          f" {N_PLANTED_RING} of which carry a planted neighbour (whole-field occupancy of"
          f" B {profile['baseline_percent'].iloc[0]:.2f} %)")
    print(f"  {'band (um)':<12}{'band px':>9}{'coverage %':>12}{'excess pp':>11}{'sd pp':>8}")
    # The bands come back by increasing distance, the object's own footprint first, and
    # each one carries its own identity: band_label names the footprint band, whose inner
    # edge is -inf, and gives the edges of every other band.
    for row in bands.itertuples():
        band_px = int(profile.loc[profile["mid_um"] == row.mid_um, "band_px"].sum())
        print(f"  {row.band_label:<12}{band_px:>9}{row.coverage_percent_mean:>12.2f}"
              f"{row.excess_percent_mean:>11.2f}{row.excess_percent_sd:>8.2f}")


def main() -> None:
    mask_a, mask_b = synthesise(np.random.default_rng(SEED))
    report(*analyse(mask_a, mask_b))


if __name__ == "__main__":
    main()
