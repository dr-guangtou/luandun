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
#from joblib import Parallel, delayed

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
    use_mask = ((sweep['FLUX_G'] > 0) & (sweep['FLUX_R'] > 0) & (sweep['FLUX_Z'] > 0))
    if verbose:
        print("# There are {:d} objects with useful flux in GRZ bands".format(use_mask.sum()))

    # g-band magnitude
    gmag = sweep_flux_to_mag(sweep, 'G')

    # 5-sigma PSF magnitude limits
    sweep.add_column(Column(data=psf_depth_to_mag(sweep, 'G'), name='PSF_MAGLIM_G'))
    sweep.add_column(Column(data=psf_depth_to_mag(sweep, 'R'), name='PSF_MAGLIM_R'))
    sweep.add_column(Column(data=psf_depth_to_mag(sweep, 'Z'), name='PSF_MAGLIM_Z'))
    # Designed depth for DECaLS:
    # Required 5σ depths of g=24.0, r=23.4 and z=22.5 for an ELG galaxy with half-light radius of 0.45 arcsec.

    # Flag for point sources with S/N > 5 detections in all GRZ bands
    good_mask = ((sweep['FLUX_G'] * np.sqrt(sweep['FLUX_IVAR_G']) > 5.) &
                 (sweep['FLUX_R'] * np.sqrt(sweep['FLUX_IVAR_R']) > 5.) &
                 (sweep['PSF_MAGLIM_G'] >= 24.0) & (sweep['NOBS_G'] > 1) &
                 (sweep['PSF_MAGLIM_R'] >= 23.4) & (sweep['NOBS_R'] > 1) &
                 (gmag <= G_MAG_CUT))
    if verbose:
        print("# There are {:d} objects with S/N>5 detections in GRZ bands".format(good_mask.sum()))
        print("# There are {:d} useful objects".format((use_mask & good_mask).sum()))

    sweep_use = sweep[use_mask & good_mask]

    # In DR9, the issue is fixed, so just 'PSF' and 'DUP'
    psf_mask = sweep_use['TYPE'] == u'PSF'
    dup_mask = sweep_use['TYPE'] == u'DUP'
    # Convert FWHM into half-light radius
    psf_r50 = (sweep_use['PSFSIZE_R'] / 2.35482) * 1.17741
    # Round exponential model
    rex_mask = (sweep_use['TYPE'] == u'REX') & (sweep_use['SHAPE_R'] <= psf_r50)
    # Exponential model:
    #exp_mask = (sweep_use['TYPE'] == u'EXP') & (sweep_use['SHAPE_R'] <= psf_r50)
    # Sersic model
    #ser_mask = (sweep_use['TYPE'] == u'SER') & (sweep_use['SHAPE_R'] <= psf_r50)
    if verbose:
        print("# There are {:d} PSF objects".format(psf_mask.sum()))
        print("# There are {:d} DUP objects".format(dup_mask.sum()))
        print("# There are {:d} Small REX objects".format(rex_mask.sum()))
        #print("# There are {:d} Small EXP objects".format(exp_mask.sum()))
        #print("# There are {:d} Small SER objects".format(ser_mask.sum()))

    sweep_use.add_column(Column(data=np.full(len(sweep_use), 0, dtype=np.int), name='FLAG'))
    sweep_use['FLAG'][psf_mask | dup_mask] = 1
    sweep_use['FLAG'][rex_mask] = 2
    #sweep_use['FLAG'][exp_mask] = 3
    #sweep_use['FLAG'][ser_mask] = 4

    #psrc = copy.deepcopy(
    #    sweep_use[psf_mask | dup_mask | rex_mask | exp_mask | ser_mask])
    psrc = copy.deepcopy(
        sweep_use[psf_mask | dup_mask | rex_mask])

    if len(psrc) < 1 and verbose:
        print("# No useful point source in {:s}".format(sweep_file))
        return

    # Convert flux into magnitude after MW extinction correction
    psrc.add_column(Column(data=sweep_flux_to_mag(psrc, 'G'), name='MAG_G_DERED'))
    psrc.add_column(Column(data=sweep_flux_to_mag(psrc, 'R'), name='MAG_R_DERED'))
    psrc.add_column(Column(data=sweep_flux_to_mag(psrc, 'Z'), name='MAG_Z_DERED'))
    #psrc.add_column(Column(data=sweep_flux_to_mag(psrc, 'W1'), name='MAG_W1_DERED'))
    #psrc.add_column(Column(data=sweep_flux_to_mag(psrc, 'W2'), name='MAG_W2_DERED'))
    #psrc.add_column(Column(data=sweep_flux_to_mag(psrc, 'W3'), name='MAG_W3_DERED'))
    #psrc.add_column(Column(data=sweep_flux_to_mag(psrc, 'W4'), name='MAG_W4_DERED'))

    # Convert flux invar into magnitude error
    psrc.add_column(Column(data=sweep_mag_err(psrc, 'G'), name='MAG_G_ERR'))
    psrc.add_column(Column(data=sweep_mag_err(psrc, 'R'), name='MAG_R_ERR'))
    psrc.add_column(Column(data=sweep_mag_err(psrc, 'Z'), name='MAG_Z_ERR'))
    #psrc.add_column(Column(data=sweep_mag_err(psrc, 'W1'), name='MAG_W1_ERR'))
    #psrc.add_column(Column(data=sweep_mag_err(psrc, 'W2'), name='MAG_W2_ERR'))
    #psrc.add_column(Column(data=sweep_mag_err(psrc, 'W3'), name='MAG_W3_ERR'))
    #psrc.add_column(Column(data=sweep_mag_err(psrc, 'W4'), name='MAG_W4_ERR'))

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
