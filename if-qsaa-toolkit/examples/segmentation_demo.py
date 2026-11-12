"""Object segmentation and measurement, end to end, on a synthetic fluorescence field.

The script plants a field whose truth it knows -- a smooth bright background gradient,
Poisson-like noise, and a known number of round objects of two radii blurred with a Gaussian
point-spread function, three of them as touching pairs and one across the frame border --
then runs the method on it without telling the analysis what was planted: a rolling-ball
background estimate, a censored noise estimate, the threshold rule ``background + k * noise``,
mask clean-up, removal of border objects, splitting of touching objects, per-object
measurement and an area filter.

What to look for: every recovered value is printed next to the planted truth, so the border
step should drop exactly the object planted across the edge, splitting should gain one object
per planted touching pair, the final count should equal the number planted inside the frame,
and each size class should recover close to its planted area. All data are synthetic and come
from a fixed seed; no file is read or written.

The background and noise estimates reflect two properties of the rolling-ball model
documented in ``if_qsaa.background``. The background estimate sits *below* the
planted background, by a share of the noise that grows with the noise, because the method
reduces the image by the block minimum before rolling the ball; the printed offset is
negative for that reason. And the noise scale read from the ball residual lands a little
*above* the planted Poisson sigma, because the ball background is reduced and bilinearly
enlarged and so cannot follow the curvature of the planted background, leaving that mismatch
in the residual. Neither hurts the threshold used here: ``censored_noise_threshold`` reads the
background mean from the same residual, so the offset cancels. An absolute level would not
cancel, which is why ``INTENSITY_FLOOR`` below is off and why a floor calibrated on one batch
does not transfer to images of a different noise level. Where a Gaussian residual is
available, ``background.estimate_shared_noise_scale`` is the estimator to prefer for a noise
scale, and the ``k`` below belongs to whichever residual the scale was measured on.
"""

import numpy as np
from scipy import ndimage as ndi

from if_qsaa import background as bg
from if_qsaa import segmentation as seg

# Properties of YOUR imaging. Measure both on your own data and replace them: a pixel size
# and an intensity floor from someone else's microscope and batch do not transfer.
PIXEL_SIZE_UM = 0.25      # micrometres per pixel
INTENSITY_FLOOR = 0.0     # absolute floor of the threshold in grey levels, 0 switches it off

# Rule of the method.
NOISE_K = 4.0             # the k of background + k * noise

# Settings of this example, sized to the synthetic objects below. They are not rules of
# the method: choose them for the size of your own structures.
BG_RADIUS_UM = 8.0        # rolling-ball radius, larger than any object
OPEN_UM = 0.3             # opening radius that removes single-pixel specks
HOLE_UM2 = 0.5            # largest enclosed hole filled
PEAK_DISTANCE_UM = 1.0    # smallest separation of two splittable object centres
MIN_AREA_UM2 = 0.9        # smallest object kept
MATCH_TOLERANCE_UM = 0.7  # scoring only: tolerance for calling a centroid a planted object

# Synthetic field settings for a short, deterministic example run.
SEED = 7391
SHAPE = (288, 288)
N_SMALL, N_LARGE, TOUCHING_PAIRS, BORDER_OBJECTS = 16, 8, 3, 1
SMALL_RADIUS_PX, LARGE_RADIUS_PX = 3.0, 6.0
PSF_SIGMA_PX, OBJECT_AMPLITUDE = 0.8, 240.0
BACKGROUND_LOW, BACKGROUND_HIGH = 430.0, 780.0


# -------------------------------------------------------------------------- synthesise

def synthesise(rng):
    """One synthetic field, and the truth that was planted into it."""
    height, width = SHAPE
    yy, xx = np.mgrid[0:height, 0:width].astype(float)
    ramp = 0.6 * xx / (width - 1) + 0.4 * yy / (height - 1)
    patch = np.exp(-((xx - 0.3 * width) ** 2 + (yy - 0.7 * height) ** 2) / (2.0 * (0.25 * width) ** 2))
    planted_bg = BACKGROUND_LOW + (BACKGROUND_HIGH - BACKGROUND_LOW) * (0.6 * ramp + 0.4 * patch)
    # Centres: a jittered grid, large objects first, then a touching partner for the first
    # few of them, then one object sitting across the frame border.
    slots = [(y, x) for y in range(24, height - 24, 36) for x in range(24, width - 24, 36)]
    centres = [(slots[i][0] + rng.uniform(-4.0, 4.0), slots[i][1] + rng.uniform(-4.0, 4.0))
               for i in rng.choice(len(slots), N_LARGE + N_SMALL, replace=False)]
    radii = [LARGE_RADIUS_PX] * N_LARGE + [SMALL_RADIUS_PX] * N_SMALL
    for rank in range(TOUCHING_PAIRS):
        angle, gap = rng.uniform(0.0, 2.0 * np.pi), 1.55 * LARGE_RADIUS_PX
        centres.append((centres[rank][0] + gap * np.sin(angle), centres[rank][1] + gap * np.cos(angle)))
        radii.append(LARGE_RADIUS_PX)
    centres.append((36.0, 2.0))
    radii.append(LARGE_RADIUS_PX)
    centres, radii = np.asarray(centres), np.asarray(radii)

    disks = [((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2 for (cy, cx), r in zip(centres, radii)]
    profile = ndi.gaussian_filter(np.logical_or.reduce(disks).astype(float), PSF_SIGMA_PX)
    image = rng.poisson(planted_bg + OBJECT_AMPLITUDE * profile).astype(float)
    return image, {"background": planted_bg, "centres": centres,
                   "kinds": np.where(radii >= LARGE_RADIUS_PX, "large", "small"),
                   "areas": np.asarray([np.count_nonzero(d) for d in disks]) * PIXEL_SIZE_UM ** 2}


# ----------------------------------------------------------------------------- analyse

def analyse(image):
    """Run the method on one field, knowing nothing about what was planted."""
    estimated_bg = bg.rolling_ball_background(image, BG_RADIUS_UM, PIXEL_SIZE_UM)
    residual = image - estimated_bg
    threshold, mu, sigma, clipped = bg.censored_noise_threshold(residual, k=NOISE_K, floor=INTENSITY_FLOOR)
    # method="value" applies the level literally; pass grow_ratio below 1 to grow below it.
    mask, _, _, _ = seg.threshold_mask(residual, method="value", value=threshold)
    cleaned = seg.clean_mask(mask, PIXEL_SIZE_UM, open_um=OPEN_UM, max_hole_area_um2=HOLE_UM2)
    inside = seg.remove_border_objects(cleaned)
    split = seg.label_objects(inside, PIXEL_SIZE_UM, split=True, peak_distance_um=PEAK_DISTANCE_UM)
    table = seg.measure_objects(split, residual, PIXEL_SIZE_UM, noise_sigma=sigma)
    _, kept = seg.filter_objects(split, table, PIXEL_SIZE_UM, min_area_um2=MIN_AREA_UM2)
    return {"bg": estimated_bg, "threshold": threshold, "mu": mu, "sigma": sigma,
            "clipped": clipped, "table": table, "kept": kept,
            "n_clean": int(seg.label_objects(cleaned, PIXEL_SIZE_UM).max()),
            "n_inside": int(seg.label_objects(inside, PIXEL_SIZE_UM).max()),
            "n_split": int(split.max())}


# ------------------------------------------------------------------------------ report

def report(truth, out):
    """Print each recovered value next to the planted truth."""
    height, width = SHAPE
    planted_bg, centres, kinds, kept = truth["background"], truth["centres"], truth["kinds"], out["kept"]
    expected = len(centres) - BORDER_OBJECTS   # the border object cannot be measured
    gaps = np.hypot(centres[:, 0] - kept["centroid_y_px"].to_numpy()[:, None],
                    centres[:, 1] - kept["centroid_x_px"].to_numpy()[:, None])
    nearest, offset = gaps.argmin(axis=1), gaps.min(axis=1)
    good = offset <= MATCH_TOLERANCE_UM / PIXEL_SIZE_UM
    hit = set(nearest[good].tolist())          # planted objects a measured centroid found
    resolved = sum(rank in hit and N_LARGE + N_SMALL + rank in hit for rank in range(TOUCHING_PAIRS))

    def line(label, found, planted=None):
        print(f"{label:<18}: {found}" + ("" if planted is None else f"   (planted {planted})"))

    line("synthetic field", f"{width} x {height} px at {PIXEL_SIZE_UM} um/px, {width * PIXEL_SIZE_UM:.0f} um square")
    line("objects planted", f"{len(centres)}: {int((kinds == 'small').sum())} small, {int((kinds == 'large').sum())}"
         f" large, {TOUCHING_PAIRS} touching pairs, {BORDER_OBJECTS} across the border")

    print("\n-- background and noise, in grey levels --")
    line("background range", f"{out['bg'].min():6.1f} to {out['bg'].max():6.1f}, offset "
         f"{np.mean(out['bg'] - planted_bg):+.1f} (the ball rides under the noise)",
         f"{planted_bg.min():.1f} to {planted_bg.max():.1f}")
    line("noise sigma", f"{out['sigma']:6.2f} from the ball residual, which reads a little high; "
         f"{100 * out['clipped']:.1f} % of its values sit below the censoring level",
         f"Poisson sigma {np.sqrt(planted_bg.mean()):.2f}")
    line("threshold", f"{out['mu']:.2f} + {NOISE_K:g} x {out['sigma']:.2f} = "
         f"{out['threshold']:.1f}, floor {INTENSITY_FLOOR:g}")

    print("\n-- what each step changed --")
    line("border removal", f"{out['n_clean']} -> {out['n_inside']} cleaned components",
         f"{BORDER_OBJECTS} on the border")
    line("splitting", f"{out['n_inside']} -> {out['n_split']} objects", f"{TOUCHING_PAIRS} touching pairs")
    line("area filter", f"{len(out['table'])} -> {len(kept)} objects at >= {MIN_AREA_UM2:g} um2"
         f", smallest measured {out['table']['area_um2'].min():.2f} um2")

    print("\n-- objects --")
    line("objects found", len(kept), f"{expected} inside the frame")
    line("matching", f"{len(hit)} matched, {int((~good).sum())} spurious, {expected - len(hit)} missed, "
         f"centroid error median {np.median(offset) * PIXEL_SIZE_UM:.3f} um")
    line("pairs resolved", f"{resolved} of {TOUCHING_PAIRS} split into two objects")
    for kind in ("small", "large"):
        pick = good & (kinds[nearest] == kind)
        if pick.any():
            line(f"area, {kind} (n={int(pick.sum())})", f"median {np.median(kept['area_um2'].to_numpy()[pick]):6.2f} um2",
                 f"{np.median(truth['areas'][nearest[pick]]):.2f} um2")


def main():
    image, truth = synthesise(np.random.default_rng(SEED))
    report(truth, analyse(image))


if __name__ == "__main__":
    main()
