"""Contact of readout objects with cell-marker territories, against a rigid null model.

What the example shows: on one synthetic square field, the marker territory is segmented
from a marker channel and the readout objects from a readout channel, operational contact
is scored on the ring just outside every object, the signed centroid-to-territory
distance is measured, and both are tested against the rigid-transformation null model,
which turns and shifts the whole object pattern while the territory stays fixed; the
distance distribution is also tested over its whole curve, and the contact rule is varied
in a sensitivity scan. Two populations of objects are planted in the same field, one in
the band just outside the territory edges and one at random, and tested separately.

What to look for: every count stands next to the planted one; the touching population
should show a contact enrichment clearly above one, a small empirical P value and a
rejecting envelope test, and the randomly placed one an enrichment near one with P values
that do not reject, which is what a correct null model has to give; the touching
population should also stay enriched across the sensitivity scan.

Run with ``PYTHONPATH=src python examples/contact_demo.py``.
"""

import numpy as np
from scipy import ndimage as ndi

from if_qsaa import contact, stats

# Quantities that belong to the user's own imaging: REPLACE each one with your own
# measurement, none of them has a generic value.
PIXEL_SIZE_UM = 0.25           # metadata, or if_qsaa.io.pixel_size_from_scale_bar
READOUT_THRESHOLD_GREY = 30.0  # grey-level floor of the readout threshold (a real
#                                channel: background + k * noise, if_qsaa.background)
MARKER_BACKGROUND_RADIUS_UM = 15.0  # rolling ball, larger than the marker structures
MARKER_NOISE_K = 5.0           # noise multiple the diffuse marker territory is cut at

# Settings of this example: obviously synthetic choices.
SEED, NULL_SEED = 7, 11        # fixed seeds, so field and randomisations both reproduce
FIELD_PX = 360                 # square: the null model turns the field by 90 degrees
N_TERRITORIES, N_PER_POPULATION = 7, 30
SITE_SPACING_UM, SPOT_RADIUS_UM = 2.5, 0.6
TOUCHING_BAND_UM = (0.8, 2.0)  # planted distance outside the territory edge
N_RANDOMISATIONS = 200         # reduced so the example runs in seconds; the method's
SCAN_RANDOMISATIONS = 100      # own setting is 10,000 randomisations
CONTACT_RULES = ((1.0, 0.05), (2.0, 0.05), (3.0, 0.05), (2.0, 0.15))  # ring um, cover


# --------------------------------------------------------------------------- synthesise

def synthesise(rng):
    """Two synthetic channels, the planted territory mask and the planted populations."""
    rows, cols = np.ogrid[:FIELD_PX, :FIELD_PX]
    truth = np.zeros((FIELD_PX, FIELD_PX), dtype=bool)
    for _ in range(N_TERRITORIES):
        row0, col0 = rng.integers(50, FIELD_PX - 50, 2)
        truth |= (rows - row0) ** 2 + (cols - col0) ** 2 <= (
            rng.uniform(5.0, 8.0) / PIXEL_SIZE_UM) ** 2
    marker = ndi.gaussian_filter(40.0 + 120.0 * truth, 0.8)
    marker = marker + rng.normal(0.0, 3.0, marker.shape)

    # candidate sites on a grid, so that two planted objects can never merge into one
    step = int(round(SITE_SPACING_UM / PIXEL_SIZE_UM))
    axis = np.arange(step, FIELD_PX - step, step)
    sites = np.stack(np.meshgrid(axis, axis, indexing="ij"), -1).reshape(-1, 2)
    at_site = ndi.distance_transform_edt(~truth)[sites[:, 0], sites[:, 1]] * PIXEL_SIZE_UM
    near, far = TOUCHING_BAND_UM
    band = np.flatnonzero((at_site >= near) & (at_site <= far))
    touching = rng.choice(band, N_PER_POPULATION, replace=False)
    elsewhere = rng.choice(np.setdiff1d(np.arange(len(sites)), touching),
                           N_PER_POPULATION, replace=False)
    planted = {"touching the edges": sites[touching], "randomly placed": sites[elsewhere]}

    spots = np.zeros((FIELD_PX, FIELD_PX), dtype=bool)
    spot_px2 = (SPOT_RADIUS_UM / PIXEL_SIZE_UM) ** 2
    for row0, col0 in np.concatenate(list(planted.values())):
        spots |= (rows - row0) ** 2 + (cols - col0) ** 2 <= spot_px2
    readout = ndi.gaussian_filter(150.0 * spots, 1.0)
    return marker, readout + rng.normal(0.0, 3.0, readout.shape), truth, planted


# --------------------------------------------------------------------------- analyse

def analyse(marker, readout, planted):
    """Segment both channels, score contact, test every population against the null."""
    territory = contact.segment_marker_territory(
        marker, PIXEL_SIZE_UM, background_radius_um=MARKER_BACKGROUND_RADIUS_UM,
        noise_k=MARKER_NOISE_K, fill_hole_um2=4.0, min_area_um2=20.0)
    # the synthetic readout plane has a flat background and is thresholded as it is; a
    # real channel is background-corrected first (if_qsaa.background)
    labels = contact.segment_readout_objects(readout, READOUT_THRESHOLD_GREY, PIXEL_SIZE_UM)
    ids = np.unique(labels)[1:]  # label 0 is the background
    centroids = contact.object_centroids(labels, weights=np.clip(readout, 0.0, None))
    _coverage, flags, _rings = contact.contact_ring_coverage(
        labels, territory.mask, PIXEL_SIZE_UM)
    distance = contact.signed_distance_um(territory.mask, PIXEL_SIZE_UM)
    tissue = None  # every pixel is tissue here; a real field: contact.tissue_void_mask

    def null(select, n_randomisations, **rule):
        """The null model of one group of the field's objects, on common random numbers."""
        return contact.contact_null_model(
            labels, centroids[select], territory.mask, distance, tissue, PIXEL_SIZE_UM,
            object_ids=ids[select], n_randomisations=n_randomisations, seed=NULL_SEED,
            **rule)

    groups = {}
    for name, centres in planted.items():
        select = np.isin(ids, labels[centres[:, 0], centres[:, 1]])
        groups[name] = dict(
            select=select, null=null(select, N_RANDOMISATIONS),
            ci=stats.bootstrap_ci(flags[select].astype(float), np.mean,
                                  n_resamples=2000, seed=SEED))
    scanned = next(iter(groups))  # the planted-touching population
    scan = [(ring_um, cover, null(
                groups[scanned]["select"], SCAN_RANDOMISATIONS, contact_ring_um=ring_um,
                contact_min_cover=cover, keep_null_ecdf=False))
            for ring_um, cover in CONTACT_RULES]
    return dict(territory=territory, n_found=ids.size, n_contact=int(flags.sum()),
                groups=groups, scanned=scanned, scan=scan)


# --------------------------------------------------------------------------- report

def report(truth, planted, result):
    """Print what the analysis recovered next to the synthetic truth."""
    print("Contact with cell-marker territories, tested against a rigid null model")
    print(f"synthetic field {FIELD_PX} x {FIELD_PX} px at {PIXEL_SIZE_UM} um/px; "
          f"territory area fraction {result['territory'].mask.mean():.3f} (planted "
          f"{truth.mean():.3f}: a blurred edge segments slightly wide); objects found "
          f"{result['n_found']} (planted {sum(len(c) for c in planted.values())}), "
          f"{result['n_contact']} of them in contact")
    for name, group in result["groups"].items():
        null, ci = group["null"], group["ci"]
        print(f"\n'{name}': {null.n_objects} objects (planted {len(planted[name])}), "
              f"{null.n_randomisations} randomisations")
        print(f"  contact fraction {null.contact_fraction:.3f} (null mean "
              f"{null.contact_null.mean():.3f}, bootstrap CI over the objects "
              f"{ci.low:.3f} to {ci.high:.3f}) -> enrichment "
              f"{null.contact_enrichment:.2f}, P = {null.contact_P:.4f}")
        print(f"  median distance {null.median_distance_um:.2f} um (null mean "
              f"{null.median_distance_null.mean():.2f} um), P = "
              f"{null.median_distance_P:.4f}; whole-curve envelope statistic "
              f"{null.envelope.statistic:.2f}, P = {null.envelope.P:.4f}")
    print(f"\nsensitivity of '{result['scanned']}' to the contact rule, "
          f"{SCAN_RANDOMISATIONS} randomisations each")
    for ring_um, cover, null in result["scan"]:
        print(f"  ring {ring_um:.1f} um, min cover {cover:.2f}: contact "
              f"{null.contact_fraction:.3f}, enrichment {null.contact_enrichment:.2f}, "
              f"P = {null.contact_P:.4f}")


def main():
    marker, readout, truth, planted = synthesise(np.random.default_rng(SEED))
    report(truth, planted, analyse(marker, readout, planted))


if __name__ == "__main__":
    main()
