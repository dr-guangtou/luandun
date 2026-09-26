# Note for the drafting agent: the configurations are "TP-AGB on" and "TP-AGB off"

Written 2026-09-25. Every final figure, caption and note in this folder was regenerated
on this date with the labels "TP-AGB on" and "TP-AGB off". Earlier versions, and the
manuscript text written from them, used "AGB on" and "AGB off". The old wording must be
replaced throughout the draft (Sections 3.3 and 4.2, Appendix D, the abstract and
conclusions if they use it, and any figure reference). The figure file names
(`index_planes_agb_on_off`) are unchanged.

## Why the labels changed

Both FSPS parameters that define the two configurations act on the thermally pulsing
AGB phase only, MIST `phase == 5`:

- `agb` multiplies the IMF weight of phase-5 stars (`mod_gb.f90`, lines 24 and 54).
  Phase 6 (post-AGB) has its own weight `pagb`, left at 1, and phases 2 to 5 together
  have `redgb`, left at 1.
- `use_lw_tpagb = 1` reassigns spectra only to phase-5 stars with log Teff < 3.6 and
  C/O <= 1 (`getspec.f90`, line 90). All other stars, including the early AGB, keep
  their C3K model-atmosphere spectra.

The early asymptotic giant branch (MIST `phase == 4`) is never varied. It carries 5 to
7 per cent of the bolometric light of a solar-metallicity single population at every
age, and more than the TP-AGB phase once the population is older than about 3 Gyr
(IMF-weighted, Chabrier IMF, from the MIST isochrone table):

| log age | early AGB / total | TP-AGB / total |
| --- | ---: | ---: |
| 8.8 | 0.050 | 0.125 |
| 9.0 | 0.070 | 0.150 |
| 9.5 | 0.066 | 0.072 |
| 10.0 | 0.069 | 0.047 |

"AGB off" therefore overstated what was removed: the model without TP-AGB stars still
contains an AGB population. The circumstellar dust prescription is attached to phases 4
and 5 and remains on the early-AGB stars in both configurations.

## Definitions to use in the text

- **TP-AGB off**: MIST isochrones with the C3K library; TP-AGB stars removed
  (`agb = 0`); all other phases, including the early AGB, retained with C3K spectra.
- **TP-AGB on**: the same model with the TP-AGB stars assigned the empirical O-rich
  spectra of Lancon and Mouhcine (2002) (`use_lw_tpagb = 1`) at twice the default
  weight (`agb = 2`). In MIST every cool TP-AGB star (log Teff < 3.6) receives these
  templates; hotter TP-AGB stars and the carbon-star branch do not occur in the
  metallicity range used.

The comparison tests the TP-AGB contribution and its spectral implementation. It does
not test the early AGB, the RGB or the post-AGB, which are identical in both
configurations. When the text says that a feature "diagnoses the AGB contribution",
write "TP-AGB contribution".

## Where the old wording lived

The labels were set in one place (`AGB_CONFIGS` in `publication_figures.py`) and
propagated to the row titles and legends of all five final figures, to the four
captions, to `fsps_model_reference_note.md`, `observed_stack_vs_fsps_mocks_note.md` and
`docs/PUBLICATION.md`. All of these now read "TP-AGB on" and "TP-AGB off". The internal
analysis notes (`docs/ANALYSIS.md`, `docs/lessons.md`) always described `agb` as the
TP-AGB weight and are unaffected.
