"""Fit the learned detector on synthetic candidates and test it on a held-out field.

The library proposes candidates and measures its six features, every candidate is labelled
automatically from the synthetic ground truth (a candidate that overlaps a planted object
is a positive), a logistic model is fitted on those labels, its threshold is set at the
maximum F1 score on the training candidates, and the model is applied to a second,
independent synthetic field.
Fitting and scoring are given the same thing: the raw feature table of each field, which
both of them normalise field by field with the settings the model carries.

Look in the output for the objects recovered on the held-out field next to the number
planted, for how few of the planted bleed-through artefacts survive, for the precision and
recall of the accepted candidates, and for the coefficients, which say which features
carried the decision: the contrast features should come out positive and the
reference-channel feature negative, since the artefacts are bright in both channels.

All data here are synthetic and are generated with a fixed seed. Nothing is read from or
written to disk, and nothing is plotted.
"""

import numpy as np

from if_qsaa import learned_detector as ld

# ----------------------------------------------------------- values the user must measure
# Replace all four with your own measurements; they are here only to let the example run.
PIXEL_SIZE_UM = 0.25         # micrometres per pixel of your own images
REFERENCE_COEFFICIENT = 0.5  # spectral coupling of the reference channel into the target
                             # channel, calibrated on single-stained control sections
NOISE_FLOOR = 10.0           # lower bound of the noise estimate, in grey levels: the read
                             # noise of your own detector (None for no bound)
SATURATION_LEVEL = 60000.0   # grey level at which your detector's range runs out

# Synthetic example settings. Three small fields are labelled here so that
# the run stays short; the method asks for labelled candidates from enough fields to cover
# the user's own staining batches.
L2_PENALTY, REVIEW_MARGIN = 1.0, 1.5
TRAIN_SEEDS, TEST_SEED = (101, 102, 103), 204
PARAMS = ld.DetectorParams(noise_floor=NOISE_FLOOR,
                           reference_coefficient=REFERENCE_COEFFICIENT)

# ------------------------------------------------------------------------------ synthesise
FIELD_PX = 200
PER_HALF = 9            # bright compact objects per half of the field: the ground truth
N_DISTRACTORS = 20      # dim, broader spots, all of them in the noisy half
PER_HALF_ARTEFACTS = 4  # spots that are bright in the reference channel as well
TOLERANCE_PX = 2        # how close to a planted centre a candidate counts as a hit
# Spots sit on a jittered 20 px grid; the right half of the field is the noisy one.
QUIET_CELLS = [(row, col) for row in range(1, 10) for col in range(1, 5)]
NOISY_CELLS = [(row, col) for row in range(1, 10) for col in range(6, 10)]


def _spot(shape, y, x, amplitude, sigma_px):
    """One Gaussian spot of the given amplitude, in grey levels."""
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return amplitude * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2.0 * sigma_px**2))


def synthesise_field(seed):
    """Target channel, reference channel, planted object centres and artefact centres."""
    rng = np.random.default_rng(seed)
    shape, half = (FIELD_PX, FIELD_PX), FIELD_PX // 2
    target = rng.normal(420.0, 18.0, shape)
    target[:, half:] += rng.normal(0.0, 55.0, (FIELD_PX, half))
    reference = rng.normal(260.0, 12.0, shape)

    def places(pool, count):
        cells = (pool[index] for index in rng.permutation(len(pool))[:count])
        return [(row * 20 + int(rng.integers(-3, 4)), col * 20 + int(rng.integers(-3, 4)))
                for row, col in cells]

    quiet = places(QUIET_CELLS, PER_HALF + PER_HALF_ARTEFACTS)
    noisy = places(NOISY_CELLS, PER_HALF + PER_HALF_ARTEFACTS + N_DISTRACTORS)
    centres = quiet[:PER_HALF] + noisy[:PER_HALF]
    artefacts = quiet[PER_HALF:] + noisy[PER_HALF : PER_HALF + PER_HALF_ARTEFACTS]
    for y, x in centres:
        target += _spot(shape, y, x, rng.uniform(2400.0, 3400.0), rng.uniform(1.4, 2.0))
    for y, x in noisy[PER_HALF + PER_HALF_ARTEFACTS :]:
        target += _spot(shape, y, x, rng.uniform(280.0, 460.0), 2.6)
    for y, x in artefacts:
        # A particle seen in both channels: a residual survives in the target channel once
        # the reference has been subtracted with the coupling coefficient set above.
        amplitude = rng.uniform(1800.0, 2600.0)
        target += _spot(shape, y, x, amplitude, 1.8)
        reference += _spot(shape, y, x, amplitude, 1.8)
    return target.astype(np.float32), reference.astype(np.float32), centres, artefacts


# --------------------------------------------------------------------------------- analyse
def labels_from_truth(candidate_set, places):
    """Label 1 every candidate whose pixels overlap one of the planted positions."""
    def hit(candidate, y, x):
        return np.any((np.abs(candidate.ys - y) <= TOLERANCE_PX)
                      & (np.abs(candidate.xs - x) <= TOLERANCE_PX))

    return np.array([int(any(hit(c, y, x) for y, x in places))
                     for c in candidate_set.candidates], dtype=int)


def covered(mask, places):
    """How many of the planted positions the mask covers."""
    r = TOLERANCE_PX
    return sum(1 for y, x in places if mask[y - r : y + r + 1, x - r : x + r + 1].any())


def analyse():
    """Fit on the training fields, then score a held-out field against its own truth."""
    tables, labels, n_train = [], [], 0
    for seed in TRAIN_SEEDS:
        target, reference, centres, _ = synthesise_field(seed)
        found = ld.detect_candidates(target, reference, params=PARAMS)
        # The raw table of every field, in the form scoring reads it: the fit normalises
        # each field on its own with the settings it writes into the model.
        tables.append(found.features)
        labels.append(labels_from_truth(found, centres))
        n_train += len(found.candidates)
    label = np.concatenate(labels)
    # Objects are rare among the candidates, so the positives are weighted up.
    n_positive = int(label.sum())
    weight = np.where(label == 1, (label.size - n_positive) / max(n_positive, 1), 1.0)
    model = ld.fit_learned_model(tables, label, l2_penalty=L2_PENALTY,
                                 sample_weight=weight)
    scores = np.concatenate([ld.score_candidates(one, model) for one in tables])
    cut = ld.choose_threshold_max_f1(scores, label)
    model = model.with_threshold(cut)

    target, reference, centres, artefacts = synthesise_field(TEST_SEED)
    result = ld.segment_learned(target, model, reference, params=PARAMS,
                                review_margin=REVIEW_MARGIN)
    truth = labels_from_truth(result.candidate_set, centres)
    accepted = result.scores >= result.threshold
    hit = int(np.count_nonzero(accepted & (truth == 1)))
    wrong = int(np.count_nonzero(accepted & (truth == 0)))
    missed = int(np.count_nonzero(~accepted & (truth == 1)))
    objects = ld.measure_objects(result.main_mask, result.candidate_set.plane,
                                 result.candidate_set.response, PIXEL_SIZE_UM,
                                 response_reference=PARAMS.seed_threshold,
                                 saturation_level=SATURATION_LEVEL)
    areas = objects["area_um2"].to_numpy(dtype=float) if len(objects) else np.zeros(1)
    fit = {"candidates": n_train, "objects": n_positive, "cut": cut}
    test = {"planted": len(centres), "found": covered(result.main_mask, centres),
            "artefacts": len(artefacts), "kept": covered(result.main_mask, artefacts),
            "candidates": result.n_candidates, "accepted": result.n_accepted,
            "review": result.n_review, "hit": hit, "wrong": wrong, "missed": missed,
            "precision": hit / max(hit + wrong, 1), "recall": hit / max(hit + missed, 1),
            "masked": len(objects), "area_min": float(areas.min()),
            "area_max": float(areas.max())}
    return model, fit, test


# ---------------------------------------------------------------------------------- report
def report(model, fit, test):
    """Print the fit, the held-out performance and what the coefficients say."""
    print("Learned detector: fitted and tested on synthetic data")
    print("=" * 53)
    print(f"training: {len(TRAIN_SEEDS)} fields of {FIELD_PX}x{FIELD_PX} px, "
          f"{fit['candidates']} candidates, {fit['objects']} of them objects")
    print(f"threshold at the maximum F1 score on these candidates: {fit['cut']:+.2f}")
    print()
    print(f"held-out field: {test['candidates']} candidates proposed, "
          f"{test['accepted']} accepted, {test['review']} in the review band")
    print(f"  objects found      : {test['found']} (planted: {test['planted']})")
    print(f"  bleed-through kept : {test['kept']} (planted: {test['artefacts']})")
    print(f"  accepted           : {test['hit']} on objects, {test['wrong']} not, "
          f"{test['missed']} positives missed")
    print(f"  precision / recall : {test['precision']:.3f} / {test['recall']:.3f}")
    print(f"  objects measured   : {test['masked']} in the mask, "
          f"{test['area_min']:.2f} to {test['area_max']:.2f} um2 each")
    print()
    print("what carried the decision (logistic coefficients, largest first)")
    for position in np.argsort(-np.abs(model.coef)):
        print(f"  {model.feature_names[position]:<9} {model.coef[position]:+.3f}")
    print()
    print("The coefficients above were fitted to this synthetic example.")
    print("Fit the model on your own labelled candidates.")
    print("Re-check the model and threshold when imaging")
    print("or staining conditions change.")


def main():
    report(*analyse())


if __name__ == "__main__":
    main()
