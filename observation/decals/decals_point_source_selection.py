#!/usr/bin/env python

import os
import glob
import copy
import argparse
import warnings

import numpy as np

from multiprocessing import Pool
from joblib import Parallel, delayed

from astropy.table import Table, Column
from astropy.utils.exceptions import AstropyWarning

FITS_DIR = '/tigress/MERIAN/decals/sweep/9.0/'
G_MAG_CUT = 23.5 # mag

warnings.simplefilter('ignore', RuntimeWarning)
warnings.simplefilter('ignore', category=AstropyWarning)

COL_USE = [
    'BRICKNAME', 'OBJID', 'TYPE', 'RA', 'DEC', 'DCHISQ',
    'NOBS_G', 'NOBS_R', 'NOBS_Z', 'ANYMASK_G', 'ANYMASK_R',
    'ALLMASK_G', 'ALLMASK_R', 'PSFSIZE_G', 'PSFSIZE_R',
    'PSF_MAGLIM_G', 'PSF_MAGLIM_R', 'SHAPE_R',
    'FLAG', 'MAG_G_DERED', 'MAG_R_DERED', 'MAG_Z_DERED',
    'MASKBITS', 'FITBITS'
]

def psf_depth_to_mag(catalog, band):
    """Convert the 5-sigma PSF detection limits to magnitude."""
    return -2.5 * (np.log10(5.0 / np.sqrt(catalog['PSFDEPTH_' + band.upper().strip()])) - 9)

def sweep_flux_to_mag(catalog, band):
    """Convert flux into magnitude for DECaLS Sweep catalog."""
    flux_dered = catalog['FLUX_' + band.strip()] / catalog['MW_TRANSMISSION_' + band.strip()]
    return -2.5 * np.log10(flux_dered) + 22.5

def sweep_mag_err(catalog, band):
    """Estimate error of magnitude in certain band."""
    sn_ratio = catalog['FLUX_' + band.strip()] * np.sqrt(catalog['FLUX_IVAR_' + band.strip()])
    return 2.5 * np.log10(1. + 1. / sn_ratio)

def select_point_sources(sweep_file, verbose=False):
    """Select point sources from a SWEEP catalog."""
    if verbose:
        print("\n# Dealing with catalog: {:s}".format(sweep_file))
    sweep_name = os.path.split(sweep_file)[-1]

    sweep = Table.read(sweep_file)
    if verbose:
        print("# There are {:d} objects in the catalog".format(len(sweep)))

    # Flag for point sources with detections in all GRZ bands
    # ALLMASK_G/R = 2:     Saturated pixels
    # ALLMASK_G/R = 64:    Bleeding trails
    use_mask = (
        np.isfinite(sweep['FLUX_G']) & np.isfinite(sweep['FLUX_R']) &
        np.isfinite(sweep['FLUX_Z']) &
        np.isfinite(sweep['FLUX_IVAR_G']) & np.isfinite(sweep['FLUX_IVAR_R']) &
        np.isfinite(sweep['FLUX_IVAR_Z']) &
        ((sweep['ALLMASK_G'] & 2) != 2) & ((sweep['ALLMASK_R'] & 2) != 2) &
        ((sweep['ALLMASK_G'] & 64) != 64) & ((sweep['ALLMASK_R'] & 64) != 64) &
        (sweep['PSFDEPTH_G'] > 0) & (sweep['PSFDEPTH_R'] > 0) & 
        (sweep['NOBS_G'] > 1) & (sweep['NOBS_R'] > 1) &
        (sweep['FLUX_G'] > 0) & (sweep['FLUX_R'] > 0) & (sweep['FLUX_Z'] > 0))
    if verbose:
        print("# There are {:d} objects with useful flux in GRZ bands".format(use_mask.sum()))
    sweep_use = sweep[use_mask]

    # g-band magnitude
    gmag = sweep_flux_to_mag(sweep_use, 'G')

    # 5-sigma PSF magnitude limits
    sweep_use.add_column(
        Column(data=psf_depth_to_mag(sweep_use, 'G'), name='PSF_MAGLIM_G', unit='mag'))
    sweep_use.add_column(
        Column(data=psf_depth_to_mag(sweep_use, 'R'), name='PSF_MAGLIM_R', unit='mag'))
    # sweep_use.add_column(Column(data=psf_depth_to_mag(sweep_use, 'Z'), name='PSF_MAGLIM_Z'))
    # Designed depth for DECaLS:
    #   Required 5σ depths of g=24.0, r=23.4 and z=22.5 for an ELG galaxy with
    #   half-light radius of 0.45 arcsec.

    # Flag for point sources with S/N > 5 detections in all GRZ bands
    good_mask = ((sweep_use['FLUX_G'] * np.sqrt(sweep_use['FLUX_IVAR_G']) > 3.) &
                 (sweep_use['FLUX_R'] * np.sqrt(sweep_use['FLUX_IVAR_R']) > 3.) &
                 (sweep_use['PSF_MAGLIM_G'] >= 24.0) & (sweep_use['PSF_MAGLIM_R'] >= 23.4) &
                 (gmag <= G_MAG_CUT))
    if verbose:
        print("# There are {:d} objects with S/N>5 detections in GRZ bands".format(good_mask.sum()))
    sweep_good = sweep_use[good_mask]

    # In DR9, the issue is fixed, so just 'PSF' and 'DUP'
    psf_mask = sweep_good['TYPE'] == u'PSF'
    dup_mask = sweep_good['TYPE'] == u'DUP'
    # Convert FWHM into half-light radius
    psf_r50 = (sweep_good['PSFSIZE_R'] / 2.35482) * 1.17741
    # Round exponential model
    rex_mask = (sweep_good['TYPE'] == u'REX') & (sweep_good['SHAPE_R'] <= psf_r50)
    if verbose:
        print("# There are {:d} PSF objects".format(psf_mask.sum()))
        print("# There are {:d} DUP objects".format(dup_mask.sum()))
        print("# There are {:d} Small REX objects".format(rex_mask.sum()))

    sweep_good.add_column(Column(data=np.full(len(sweep_good), 0, dtype=np.int), name='FLAG'))
    sweep_good['FLAG'][psf_mask | dup_mask] = 1
    sweep_good['FLAG'][rex_mask] = 2

    psrc = copy.deepcopy(
        sweep_good[psf_mask | dup_mask | rex_mask])

    if len(psrc) < 1 and verbose:
        print("# No useful point source in {:s}".format(sweep_file))
        return

    # Convert flux into magnitude after MW extinction correction
    psrc.add_column(Column(data=sweep_flux_to_mag(psrc, 'G'), name='MAG_G_DERED', unit='mag'))
    psrc.add_column(Column(data=sweep_flux_to_mag(psrc, 'R'), name='MAG_R_DERED', unit='mag'))
    psrc.add_column(Column(data=sweep_flux_to_mag(psrc, 'Z'), name='MAG_Z_DERED', unit='mag'))

    # Convert flux invar into magnitude error
    psrc.add_column(Column(data=sweep_mag_err(psrc, 'G'), name='MAG_G_ERR', unit='mag'))
    psrc.add_column(Column(data=sweep_mag_err(psrc, 'R'), name='MAG_R_ERR', unit='mag'))
    psrc.add_column(Column(data=sweep_mag_err(psrc, 'Z'), name='MAG_Z_ERR', unit='mag'))

    # Only keep the useful columns
    psrc_use = psrc[COL_USE]

    psrc_file = os.path.join(FITS_DIR, 'psrc', sweep_name.replace('.fits', '_psrc.fits'))
    psrc_use.write(psrc_file, overwrite=True)

    return psrc_use

def run(args):
    """Run the point source selection."""
    sweep_list = glob.glob(FITS_DIR + '*.fits')
    print("There are {:d} FITS catalogs".format(len(sweep_list)))

    psrc_dir = os.path.join(FITS_DIR, 'psrc/')
    psrc_list = glob.glob(psrc_dir + '*.fits')
    print("{:d} catalogs have been processed".format(len(psrc_list)))

    psrc_list = [os.path.split(p.replace('_psrc', ''))[-1] for p in psrc_list]
    sweep_left = [s for s in sweep_list if os.path.split(s)[-1] not in psrc_list]
    print("{:d} sweep catalogs left to be processed".format(len(sweep_left)))

    # Test
    if args.test:
        _ = select_point_sources(sweep_left[0], verbose=True)
        return

    if args.njobs == 1:
        for sweep in sweep_left:
            _ = select_point_sources(sweep, verbose=True)
    else:
        _ = Parallel(n_jobs=args.njobs)(
            delayed(select_point_sources)(sweep) for sweep in sweep_list)

        # Using multiprocessing
        #with Pool(args.njobs) as p:
        #    _ = p.map(select_point_sources, sweep_left[0:3])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n', '--njobs', dest='njobs', help="Number of processors to run",
        default=4, type=int)
    parser.add_argument(
        '-t', '--test', action="store_true", dest='test', default=False)

    run_args = parser.parse_args()

    run(run_args)
