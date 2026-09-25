# Note for the drafting agent: the new observed-stack figure

Written 2026-09-25. This note explains `observed_stack_vs_fsps_mocks.{pdf,png}` in this
folder, which replaces the draft's Figure 8 (`figures/NIRSpec_obs_mock_stack.pdf`, the
XSL-based matched stack labelled `obs-stack`). The caption is
`observed_stack_vs_fsps_mocks.tex`; every number is in
`observed_stack_vs_fsps_mocks.json`; the full search table is
`observed_stack_vs_fsps_mocks_search_agb_on.npy` (columns log Z, t_q, tau_q, mismatch).
The script is `stellar_pop/fsps_agb/jwst_spectrum_comparison.py` (2 minutes to run).
Background on the models is in `fsps_model_reference_note.md`, Sections 2 to 6 and 8.5.

## What the figure replaces and why

The old Figure 8 compared the observed stack with a stack of XSL mocks selected by
proximity to each galaxy in the bump versus D4000 plane, and its caption said that
comparison did not test the two FSPS configurations. The new figure makes exactly that
test: the same observed stack is drawn against FSPS mock spectra with and without
TP-AGB light, in the same normalisation, at the epoch of the sample, and the models are
not matched to the galaxies. The message is the one of Section 4.2: without TP-AGB
light no quenching history reproduces the observed bump, and with the empirical
TP-AGB templates the models bracket it.

## Observed side

- **Input.** The 19 NIRSpec PRISM spectra of Lu+2026 in `qg_spec/` whose IDs appear in
  `JWST_QG_indices.npz` (that table supplies the redshifts; the 8 other files in the
  folder have no redshift there and are not used). Each file is observed-frame
  wavelength in A, F_lambda and its error; galaxy 24686 has non-finite pixels that are
  dropped.
- **Rest frame and normalisation.** Wavelengths are divided by 1 + z and F_lambda
  multiplied by 1 + z. Each spectrum is divided by a straight line fitted (least
  squares, degree 1) to all pixels inside the two pseudo-continuum windows of the
  Hbump_opt definition, 1.494 to 1.539 and 1.746 to 1.791 micron. This is the pyphot
  degree-1 convention the manuscript adopts, applied identically to models.
- **Stack.** S/N-weighted mean on a uniform rest-frame grid of 45 A, the median pixel
  width of the 19 spectra in this region (range 27 to 60 A). Weights are (F / sigma)^2
  per galaxy and pixel. The grey band is the weighted standard deviation between
  galaxies at each pixel, so it is the galaxy-to-galaxy scatter, not the error of the
  mean.
- **Sanity check.** Recomputing the bump index from these normalised spectra
  (-2.5 log10 of the pixel mean over 1.570 to 1.734 micron) reproduces the table
  values to 0.001 to 0.002 mag for 18 galaxies. Galaxy 20150 gives -0.050 mag against
  the stored -0.059 mag, so its file may not be the one used for the table; flag it to
  the author rather than resolve it in the text.
- **Numbers.** Median redshift 1.357; the stack peaks at 1.080 in the feature band and
  has a bump index of -0.057 mag.

## Model side

- **Epoch.** One epoch only: the cosmic age at the median redshift, 4.57 Gyr for a flat
  LCDM cosmology with H0 = 70 km/s/Mpc and Omega_m = 0.3, with star formation starting
  at t = 0 (no formation-redshift offset). The nearest point of the 0.05 Gyr model grid
  is 4.55 Gyr. This replaces the earlier draft of the figure, which spanned all epochs
  up to 6 Gyr after quenching.
- **Spectra.** Composite spectra assembled from the cached FSPS SSP grids (MIST
  isochrones, C3K high-resolution library, Chabrier IMF) on the R = 100 product, with
  the paper's delayed-tau rise (tau = t_q) and exponential decline after t_q, then
  divided by the same straight-line pseudo-continuum and averaged into the same 45 A
  bins as the data. AGB off is the C3K configuration with `agb = 0`; AGB on is the
  empirical Lancon and Mouhcine (2002) O-rich templates (`use_lw_tpagb = 1`) with
  `agb = 2`, as everywhere else in the paper.
- **The coloured histograms (both panels).** Solar metallicity, time since quenching
  fixed at t - t_q = 1 Gyr (so t_q = 3.55 Gyr), quenching timescale tau_q = 0.1, 0.3, 1
  and 3 Gyr. They show how much the quenching timescale alone can move the model at
  this epoch: very little for AGB off, a few hundredths in normalised flux for AGB on.
- **The dashed histogram (AGB on only).** The closest model in a grid search at the same
  epoch over log Z/Zsun from -0.5 to +0.25 in 0.05 dex steps (interpolated in log Z
  between the four grid metallicities), t_q from 1.0 Gyr to the epoch in 0.1 Gyr steps
  (36 values) and 13 log-spaced tau_q values from 0.1 to 3 Gyr: 7280 models. The
  ranking statistic ("mismatch") is the RMS over 1.494 to 1.791 micron of (model minus
  stack) divided by the galaxy-to-galaxy scatter, so 1 means the model deviates by one
  scatter on average. No AGB-off search is drawn: it is not meaningful, because no
  AGB-off model comes close (see below).
- **Ratio panels (c, d).** The observed stack divided by each model, with the scatter
  band divided by the stack.

## Results to quote

| | AGB off | AGB on |
| --- | ---: | ---: |
| Bump index of the tau_q = 0.1 / 0.3 / 1 / 3 Gyr models [mag] | -0.007 / -0.005 / -0.001 / +0.001 | -0.072 / -0.072 / -0.064 / -0.058 |
| Peak normalised flux in the feature band | at most 1.03 (all four) | 1.07 to 1.09 |
| Mismatch of the tau_q = 0.1 / 0.3 / 1 / 3 Gyr models | 2.41 / 2.49 / 2.65 / 2.73 | 0.74 / 0.74 / 0.45 / 0.39 |
| Median observed / model over the feature band | 1.046 to 1.055 | 0.986 to 0.999 |
| Closest model in the search | not drawn (a one-off search gave log Z = -0.25, t_q = 1.0, tau_q = 0.10 Gyr at mismatch 2.03; nothing in the AGB-off grid is below 2.03) | log Z = +0.15, t_q = 1.2 Gyr, tau_q = 0.41 Gyr (t - t_q = 3.35 Gyr), bump index -0.059 mag, mismatch 0.36 |

The observed stack: bump index -0.057 mag, peak normalised flux 1.080.

Interpretation for the text:

- **AGB off cannot reproduce the observed bump for any quenching history.** At the
  sample epoch the four quenching timescales span 0.008 mag in index and all stay
  below 1.03 in the feature band, 5 per cent under the data across the whole band
  (panel c). Even the extreme corner of the search (oldest, most abruptly quenched,
  sub-solar) remains two scatters away. This is the key message of the left column.
- **AGB on brackets the observation.** The tau_q = 1 and 3 Gyr models at solar
  metallicity are already within the scatter across the feature band, and the closest
  model matches to 0.36 scatters with a residual that stays inside the grey band
  everywhere between the side windows (panel d).
- **The match does not constrain the quenching history.** The eight best AGB-on models
  all have mismatch 0.36 and span t_q = 1.1 to 1.8 Gyr with tau_q from 0.10 to 0.41
  Gyr at log Z = +0.15: the stack constrains the template and, loosely, the
  metallicity, not (t_q, tau_q). The caption states the degeneracy; the text should
  not quote the closest model as a fit.
- **Metallicity.** The preferred AGB-on metallicity is mildly super-solar (+0.15 dex).
  This is consistent with the AGB-on bump deepening with metallicity (Section 9 of the
  reference note), but it is a one-parameter preference from a stack, not a
  measurement.

## What the figure does not test, for the caveats

- The straight-line normalisation removes the continuum slope and the side-window
  levels, so the spectral shape outside 1.494 to 1.791 micron is not compared. In the
  earlier wide-range version the AGB-on models sat 5 to 8 per cent above the data
  below 1.45 and above 1.8 micron, where the empirical templates carry H2O absorption;
  that region is now outside the displayed range and outside the search metric. Say so
  if the text discusses the templates' H2O bands.
- The models are at a single epoch and metallicity per curve, with no dust, nebular
  emission or per-galaxy resolution matching; the PRISM resolution varies along the
  spectrum and R = 100 is the representative value near 1.6 micron in the observed
  frame at z ~ 1.4.
- The stack is uniform in weight only through S/N, so the high-S/N galaxies dominate;
  the three low-S/N galaxies (6283, 4458, 22897) contribute little.
- The comparison is a spectral-shape check, not a likelihood test: the scatter band is
  the spread between galaxies, not the uncertainty of the stack, and the mismatch
  statistic has no calibrated probability.

## Figure design

Two columns, AGB off left and AGB on right, each with a spectrum panel and a ratio
panel (height ratio 2.4 to 1), 7.1 inches wide by 3.9 inches tall for a full page
width. Display range 1.46 to 1.83 micron. All spectra drawn as steps on the 45 A grid.
Blue, yellow and red shading mark the blue pseudo-continuum, feature and red
pseudo-continuum windows. The configuration name sits in the top-right corner of each
spectrum panel; the closest-model parameters sit in the bottom-right of panel (b). One
legend below the panels; labels capitalised as sentences.

## Suggested text changes in the draft

- Section 4.2, the paragraph beginning "As a separate comparison of spectral shape":
  replace the description of the XSL matched stack with the FSPS comparison above, and
  drop "This XSL comparison is distinct from the new FSPS on/off test", since the new
  figure is that test.
- The caption of `obs-stack`: replace with `observed_stack_vs_fsps_mocks.tex` and point
  `\includegraphics` at `observed_stack_vs_fsps_mocks.pdf`.
- The comment line "Figure 8 is confirmed to use XSL mocks and is retained pending the
  author's later replacement" can be removed.
- Section 4.2 caveats and Section 5.1: the sentence about matching the instrument
  response still applies; add that the stack comparison is made at the single epoch of
  the median redshift and uses a straight-line normalisation that does not test the
  side-window levels.
