#!/usr/bin/env python

import os
import glob
import copy
import argparse
import warnings

import numpy as np

from multiprocessing import Pool
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import gridspec

from astropy.table import Table, Column
from astropy.utils.exceptions import AstropyWarning

FITS_DIR = '/tigress/MERIAN/decals/sweep/9.0/psrc/'

warnings.simplefilter('ignore', RuntimeWarning)
warnings.simplefilter('ignore', category=AstropyWarning)


def sweep_to_ra_dec_range(cat_name):
    '''Decode the Sweep catalog name into (RA, Dec) range.

    Parameters
    ----------
    sweep_name: `string`
        Path or the file name of the Sweep catalog.

    Returns
    -------
    box: `np.array`
        An array of the coordinates of the four corners of the box.

    '''
    sweep_name = cat_name.replace('_psrc', '')

    # Extract the RA, Dec ranges of the Sweep file
    radec_str = os.path.splitext(
        os.path.split(sweep_name)[-1])[0].replace('sweep-', '').split('-')

    # Get the minimum and maximum RA & Dec
    ra_min = float(radec_str[0][0:3])
    dec_min = float(radec_str[0][-3:]) * (-1 if radec_str[0][3] == 'm' else 1)

    ra_max = float(radec_str[1][0:3])
    dec_max = float(radec_str[1][-3:]) * (-1 if radec_str[1][3] == 'm' else 1)

    radec_range = [[ra_min, ra_max], [dec_min, dec_max]]
    radec_extent = [ra_min, ra_max, dec_min, dec_max]

    return radec_range, radec_extent

def check_psrc_cat(cat_name):
    """Check the number density of PSF and REX sources in each catalog."""
    # Get the (RA, Dec) ranges
    radec_range, radec_extent = sweep_to_ra_dec_range(cat_name)

    # Read in the data
    psrc_data = Table.read(cat_name)

    # Separate the PSF and REX objects
    psf_data = psrc_data[
        (psrc_data['FLAG'] == 1) & (psrc_data['MAG_G_DERED'] >= 20.0) &
        (psrc_data['MAG_G_DERED'] - psrc_data['MAG_R_DERED'] <= 0.9) &
        (psrc_data['MAG_G_DERED'] - psrc_data['MAG_R_DERED'] >= -0.1)]
    rex_data = psrc_data[
        (psrc_data['FLAG'] == 2) & (psrc_data['MAG_G_DERED'] >= 20.0) &
        (psrc_data['MAG_G_DERED'] - psrc_data['MAG_R_DERED'] <= 0.9) &
        (psrc_data['MAG_G_DERED'] - psrc_data['MAG_R_DERED'] >= -0.1)]

    # Get the histogram used for visualization
    psf_hist, ra_edges, dec_edges = np.histogram2d(
        psf_data['RA'], psf_data['DEC'], range=radec_range, bins=[120, 120])

    rex_hist, _, _ = np.histogram2d(
        rex_data['RA'], rex_data['DEC'], range=radec_range, bins=[120, 120])

    ra_cen = (ra_edges[:-1] + ra_edges[1:]) / 2.
    dec_cen = (dec_edges[:-1] + dec_edges[1:]) / 2.

    fig = plt.figure(figsize=(20, 8))
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.02, top=0.94)

    gs = gridspec.GridSpec(1, 3)
    gs.update(wspace=0.2, hspace=0.10)

    ax1 = plt.subplot(gs[0])
    ax1.set_title('PSF', fontsize=25)

    psf = ax1.imshow(
        psf_hist, origin='lower', extent=radec_extent, aspect='auto', interpolation=None)
    fig.colorbar(psf.T, ax=ax1, orientation="horizontal", pad=0.1)

    ax2 = plt.subplot(gs[1])
    ax2.set_title('REX', fontsize=25)

    rex = ax2.imshow(
        rex_hist, origin='lower', extent=radec_extent, aspect='auto', interpolation=None)
    fig.colorbar(rex.T, ax=ax2, orientation="horizontal", pad=0.1)

    ax3 = plt.subplot(gs[2])
    ax3.set_title('Ratio', fontsize=25)

    ratio = ax3.imshow(
        (rex_hist / psf_hist).T, origin='lower', extent=radec_extent, aspect='auto',
        interpolation=None)
    fig.colorbar(ratio, ax=ax3, orientation="horizontal", pad=0.1)
    fig.savefig(cat_name.replace('.fits', '.png'), dpi=150)
    plt.close(fig)

    return {
        'cat': cat_name,
        'ra_min': radec_extent[0], 'ra_max': radec_extent[1],
        'dec_min': radec_extent[2], 'dec_max': radec_extent[3],
        'psf': psf_hist, 'rex': rex_hist,
        'ra_edges': ra_edges, 'dec_edges': dec_edges,
        'ra_cen': ra_cen, 'dec_cen': dec_cen}


def run(args):
    """Run the point source selection."""
    psrc_list = glob.glob(FITS_DIR + '*.fits')
    print("There are {:d} point sources catalogs".format(len(psrc_list)))

    # Test
    if args.test:
        _ = check_psrc_cat(psrc_list[0])
        return

    if args.njobs == 1:
        for sweep in psrc_list:
            hist_list = check_psrc_cat(sweep)
    else:
        hist_list = Parallel(n_jobs=args.njobs)(
            delayed(check_psrc_cat)(sweep) for sweep in psrc_list)

    hist_table = Table(hist_list)
    hist_table.write("decals_dr9_psrc_hist.fits", overwrite=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n', '--njobs', dest='njobs', help="Number of processors to run",
        default=4, type=int)
    parser.add_argument(
        '-t', '--test', action="store_true", dest='test', default=False)

    run_args = parser.parse_args()

    run(run_args)
