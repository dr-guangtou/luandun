#!/usr/bin/env python

import os
import glob
import copy
import argparse
import warnings

import numpy as np

from multiprocessing import Pool

from astropy.table import Table, Column
from astropy.utils.exceptions import AstropyWarning

FITS_DIR = '/tigress/MERIAN/decals/sweep/9.0/'
G_MAG_CUT = 23.5 # mag

warnings.simplefilter('ignore', RuntimeWarning)
warnings.simplefilter('ignore', category=AstropyWarning)
#from joblib import Parallel, delayed

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

    # In DR8, it should be 'PSF ', and 'DUP '
    # In DR9, the issue is fixed, so just 'PSF' and 'DUP'
    psf_mask = sweep['TYPE'] == u'PSF'
    dup_mask = sweep['TYPE'] == u'DUP'
    if verbose:
        print("# There are {:d} PSF objects".format(psf_mask.sum()))
        print("# There are {:d} DUP objects".format(dup_mask.sum()))

    psrc = copy.deepcopy(sweep[psf_mask | dup_mask])

    # Flag for point sources with detections in all GRZ bands
    use_mask = ((psrc['FLUX_G'] > 0) & (psrc['FLUX_R'] > 0) & (psrc['FLUX_Z'] > 0))
    if verbose:
        print("# There are {:d} objects with useful flux in GRZ bands".format(use_mask.sum()))
    #psrc.add_column(Column(data=use_mask, name='GRZ_DETECT'))

    # g-band magnitude
    gmag = sweep_flux_to_mag(psrc, 'G')

    # Flag for point sources with S/N > 5 detections in all GRZ bands
    good_mask = ((psrc['FLUX_G'] * np.sqrt(psrc['FLUX_IVAR_G']) > 5.) &
                 (psrc['FLUX_R'] * np.sqrt(psrc['FLUX_IVAR_R']) > 5.) &
                 (psrc['FLUX_Z'] * np.sqrt(psrc['FLUX_IVAR_Z']) > 5.) &
                 (gmag <= G_MAG_CUT))
    if verbose:
        print("# There are {:d} objects with S/N>5 detections in GRZ bands".format(good_mask.sum()))
    #psrc.add_column(Column(data=good_mask, name='GRZ_GOOD'))

    psrc_use = psrc[use_mask & good_mask]
    gmag_use = gmag[use_mask & good_mask]

    if len(psrc_use) < 1 and verbose:
        print("# No useful point source in {:s}".format(sweep_file))
        return

    # Convert flux into magnitude after MW extinction correction
    psrc_use.add_column(Column(data=gmag_use, name='MAG_G_DERED'))
    psrc_use.add_column(Column(data=sweep_flux_to_mag(psrc_use, 'R'), name='MAG_R_DERED'))
    psrc_use.add_column(Column(data=sweep_flux_to_mag(psrc_use, 'Z'), name='MAG_Z_DERED'))
    psrc_use.add_column(Column(data=sweep_flux_to_mag(psrc_use, 'W1'), name='MAG_W1_DERED'))
    psrc_use.add_column(Column(data=sweep_flux_to_mag(psrc_use, 'W2'), name='MAG_W2_DERED'))
    psrc_use.add_column(Column(data=sweep_flux_to_mag(psrc_use, 'W3'), name='MAG_W3_DERED'))
    psrc_use.add_column(Column(data=sweep_flux_to_mag(psrc_use, 'W4'), name='MAG_W4_DERED'))

    # Convert flux invar into magnitude error
    psrc_use.add_column(Column(data=sweep_mag_err(psrc_use, 'G'), name='MAG_G_ERR'))
    psrc_use.add_column(Column(data=sweep_mag_err(psrc_use, 'R'), name='MAG_R_ERR'))
    psrc_use.add_column(Column(data=sweep_mag_err(psrc_use, 'Z'), name='MAG_Z_ERR'))
    psrc_use.add_column(Column(data=sweep_mag_err(psrc_use, 'W1'), name='MAG_W1_ERR'))
    psrc_use.add_column(Column(data=sweep_mag_err(psrc_use, 'W2'), name='MAG_W2_ERR'))
    psrc_use.add_column(Column(data=sweep_mag_err(psrc_use, 'W3'), name='MAG_W3_ERR'))
    psrc_use.add_column(Column(data=sweep_mag_err(psrc_use, 'W4'), name='MAG_W4_ERR'))

    psrc_file = os.path.join(FITS_DIR, 'psrc', sweep_name.replace('.fits', '_psrc.fits'))
    psrc_use.write(psrc_file, overwrite=True)

    return psrc

def run(args):
    """Run the point source selection."""
    sweep_list = glob.glob(FITS_DIR + '*.fits')
    print("There are {:d} FITS catalogs".format(len(sweep_list)))

    # Test
    #_ = select_point_sources(sweep_list[0], verbose=True)

    #_ = Parallel(n_jobs=args.njobs)(
    #    delayed(select_point_sources)(sweep) for sweep in sweep_list[0:2])

    with Pool(args.njobs) as p:
        _ = p.map(select_point_sources, sweep_list[0:3])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--njobs", dest="njobs", help="Number of processors to run",
                        default=4, type=int)

    args = parser.parse_args()

    run(args)
