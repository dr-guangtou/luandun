# Note for the drafting agent: the JWST index-plane figure now uses D4000 at R = 50

Written 2026-09-26. `d4000_hminus_plane_with_jwst_data.{pdf,png}` and its caption were
regenerated with the model D4000 measured on spectra smoothed to R = 50 (FWHM) combined
with the 300 km/s dispersion, instead of the sigma = 300 km/s product used for D4000 in
every other figure. The H-minus bump keeps the R = 100 product. This resolves the
draft's pending item "replace Figure 7 with mock D4000 measured at R ~ 50, retaining
R ~ 100 near the H-minus bump" and the sentence in Section 4.2 saying the replacement is
pending.

Why R = 50: the observed PRISM pixels of the 19 galaxies sample the 4000 A break at an
observed-frame lambda / delta lambda of 70 to 90 (rest-frame pixels of 45 to 60 A, 3.4
to 4.5 pixels per 200 A D4000 band), and the PRISM resolving power there is about 50.

What changed numerically: nothing at the quoted precision. Over the whole population
the R = 50 D4000 differs from the sigma300 value by a median of -0.0006 (5th to 95th
percentile -0.024 to +0.021). Over the observed D4000 range (1.35 to 1.94) the deepest
TP-AGB-off epoch is still -0.023 mag and the TP-AGB-on population still spans -0.081 to
-0.041 mag (1st to 99th percentile). The caption now states the product and the size of
the shift. Update Section 4.2 and the Figure 7 caption in the draft accordingly, and
remove the pending-replacement comment.

Implementation: `broadening.py --products r50` builds the product (3400 to 5000 A) for
each SSP grid; `d4000_r50.py` recomputes D4000 for the exponential population of both
configurations and the metallicity tracks and caches
`output/population[_lw02]/d4000_r50.npz`; `publication_figures.py --final` substitutes
those columns for this figure only. No other figure or number was touched.
