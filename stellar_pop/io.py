"""Data input for A.S.A.P model."""
from __future__ import print_function, division, unicode_literals

import os
import pickle

import numpy as np

import emcee

from astropy.table import Table, Column
from astropy.cosmology import FlatLambdaCDM

from . import ensemble


__all__ = ["load_obs", "load_um", "save_pickle", "load_pickle", "load_npz_results", 
           "save_results_to_npz"]


def load_dsigma(cfg, verbose=False):
    """Read in the observed weak lensing sigma profiles.

    Parameters
    ----------
    cfg : dict
        Configuration parameters for the observations.
    verbose : boolen
        Blah, blah, blah

    Return
    ------
    """
    # Now the DeltaSigma data are stored in a numpy array
    # Mass bin is defined by: min_logm1, min_logm2, max_logm1, and max_logm2
    wl_dsigma = np.load(cfg['dsigma'])

    cfg['wl_n_bin'] = len(wl_dsigma)
    if verbose:
        if cfg['wl_n_bin'] > 1:
            print("# There are %d DSigma profiles in this sample" %
                  cfg['wl_n_bin'])
        else:
            print("# There is 1 DSigma profile in this sample")

    return wl_dsigma


def load_obs(cfg, verbose=True):
    """Load the observed data.

    Parameters
    ----------
    cfg : dict
        Configuration parameters for the observations.
    verbose : boolen
        Blah, blah, blah

    Return
    ------
        Dictionary that contains all the observations.

    """
    # Galaxy catalog.
    mass = Table.read(cfg['galaxy'])
    minn = np.array(mass[cfg['minn_col']])
    mtot = np.array(mass[cfg['mtot_col']])

    # Observed DeltaSigma profiles
    wl_dsigma = load_dsigma(cfg, verbose=verbose)
    cfg['dsigma_n_data'] = len(wl_dsigma[0]['dsigma']) * cfg['wl_n_bin']

    # Stellar mass functions
    if os.path.splitext(cfg['smf_inn'])[-1] == '.npy':
        smf_inn = np.load(cfg['smf_inn'])
    else:
        smf_inn = Table.read(cfg['smf_inn'])

    if os.path.splitext(cfg['smf_tot'])[-1] == '.npy':
        smf_tot = np.load(cfg['smf_tot'])
    else:
        smf_tot = Table.read(cfg['smf_tot'])

    # This is for a specific format of SMF
    cfg['smf_inn_min'] = np.min(smf_inn['logm_0'])
    cfg['smf_inn_max'] = np.max(smf_inn['logm_1'])
    cfg['smf_inn_nbin'] = len(smf_inn)

    cfg['smf_tot_min'] = np.min(smf_tot['logm_0'])
    cfg['smf_tot_max'] = np.max(smf_tot['logm_1'])
    cfg['smf_tot_nbin'] = len(smf_tot)

    cfg['ngal_use'] = ((mtot >= cfg['smf_tot_min']) &
                       (minn >= cfg['smf_inn_min'])).sum()

    cfg['min_mtot'] = cfg['smf_tot_min'] - 0.1

    cfg['smf_n_data'] = cfg['smf_tot_nbin'] + cfg['smf_inn_nbin']

    # Covariance of the SMF
    if cfg['smf_cov'] is not None:
        smf_cov = np.load(cfg['smf_cov'])
        assert cfg['smf_n_data'] == len(smf_cov)
    else:
        smf_cov = None

    if verbose:
        print("# SMF for total stellar mass: ")
        print("  %7.4f -- %7.4f in %d bins" % (cfg['smf_tot_min'],
                                               cfg['smf_tot_max'],
                                               cfg['smf_tot_nbin']))
        print("# SMF for inner stellar mass: ")
        print("  %7.4f -- %7.4f in %d bins" % (cfg['smf_inn_min'],
                                               cfg['smf_inn_max'],
                                               cfg['smf_inn_nbin']))

    logms_inn = minn[mtot >= cfg['smf_tot_min']]
    logms_tot = mtot[mtot >= cfg['smf_tot_min']]

    if os.path.isfile(cfg['smf_full']):
        smf_full = Table.read(cfg['smf_full'])
        smf_full[smf_full['smf'] <= 0]['smf'] = 1E-8
        smf_full[smf_full['smf_low'] <= 0]['smf_low'] = 1E-9
        smf_full[smf_full['smf_upp'] <= 0]['smf_upp'] = 1E-7
        smf_full = smf_full
    else:
        smf_full = None

    if verbose:
        print("# For inner stellar mass: ")
        print("    %d bins at %5.2f < logMinn < %5.2f" %
              (cfg['smf_inn_nbin'], cfg['smf_inn_min'],
               cfg['smf_inn_max']))
        print("# For total stellar mass: ")
        print("    %d bins at %5.2f < logMtot < %5.2f" %
              (cfg['smf_tot_nbin'], cfg['smf_tot_min'],
               cfg['smf_tot_max']))

    # Redshift range and observed volume
    cosmo = FlatLambdaCDM(H0=cfg['h0'] * 100, Om0=cfg['omega_m'])
    cfg['volume'] = (
        (cosmo.comoving_volume(np.nanmax(mass[cfg['z_col']])) - 
         cosmo.comoving_volume(np.nanmin(mass[cfg['z_col']]))) *
        (cfg['area'] / 41254.0)).value

    if verbose:
        print("# The volume of the HSC data is %15.2f Mpc^3" % cfg['volume'])

    return {'mass': mass, 'minn': minn, 'mtot': mtot,
            'logms_inn': logms_inn, 'logms_tot': logms_tot,
            'wl_dsigma': wl_dsigma,
            'smf_inn': smf_inn, 'smf_tot': smf_tot,
            'smf_full': smf_full, 'smf_cov': smf_cov}, cfg


def load_um(cfg, verbose=True):
    """Load the UniverseMachine data.

    Parameters
    ----------
    cfg : dict
        Configuration parameters for the UniverseMachine model.
    verbose : boolen
        Blah, blah, blah

    Return
    ------
        Dictionary that contains all the UniverseMachine data.

    """
    # Mock galaxy catalog
    um_mock = Table(np.load(cfg['galaxy']))

    # Only select the useful columns
    cols_use = ['halo_id', 'upid', 'sm', 'icl', 'x', 'y', 'z',
                'mtot_galaxy', 'mstar_mhalo', 'logms_gal',
                'logms_icl', 'logms_tot', 'logms_halo',
                'logmh_vir', 'logmh_peak', 'logmh_host']
    um_mock_use = um_mock[cols_use]

    # Value added a few useful columns
    um_mock_use.add_column(Column(data=(um_mock_use['mtot_galaxy'] /
                                        um_mock_use['mstar_mhalo']),
                                  name='frac_cen_tot'))
    um_mock_use.add_column(Column(data=(um_mock_use['sm'] /
                                        um_mock_use['mtot_galaxy']),
                                  name='frac_ins_cen'))
    um_mock_use.add_column(Column(data=(um_mock_use['icl'] /
                                        um_mock_use['mtot_galaxy']),
                                  name='frac_exs_cen'))
    um_mock_use = um_mock_use.as_array()

    # Load the pre-compute lensing pairs
    um_mass_encl = np.load(cfg['dsigma'])
    assert len(um_mock_use) == len(um_mass_encl)

    # Mask for central galaxies
    mask_central = (um_mock_use['upid'] == -1)
    if verbose:
        print("# %d out of %d galaxies are central" % (
            mask_central.sum(), len(um_mock_use)))

    # Mask for massive enough halo
    mask_mass = (um_mock_use[cfg['halo_col']] >= cfg['min_mvir'])

    return {'um_mock': um_mock_use[mask_mass],
            'um_mass_encl': um_mass_encl[mask_mass, :],
            'mask_central': mask_central[mask_mass]}


def save_pickle(pickle_file, data):
    """Save some data in pickle format.

    Parameters
    ----------
    pickle_file : string
        Name of the output Pickle file.
    data : array or dict
        Some data
    """
    pickle.dump(data, open(pickle_file, 'wb'))
    pickle_file.close()

    return


def load_pickle(pickle_file):
    """Load the pickled pickle.

    Parameters
    ----------
    pickle_file : string
        Name of the output Pickle file.
    """
    data = pickle.load(open(pickle_file, 'rb'))
    pickle_file.close()

    return data


def load_npz_results(mcmc_file):
    """Retrieve the MCMC results from .npz file."""
    mcmc_data = np.load(mcmc_file)

    return (mcmc_data['samples'], mcmc_data['chains'],
            mcmc_data['lnprob'], mcmc_data['best'],
            mcmc_data['position'], mcmc_data['acceptance'])


def save_results_to_npz(mcmc_results, mcmc_sampler, mcmc_file,
                        mcmc_ndims, verbose=True, frac=0.1, tol=20, c=5):
    """Save the MCMC run results."""
    mcmc_position, mcmc_lnprob, _ = mcmc_results

    mcmc_samples = mcmc_sampler.chain[:, :, :].reshape((-1, mcmc_ndims))
    mcmc_chains = mcmc_sampler.chain
    mcmc_lnprob = mcmc_sampler.lnprobability

    mcmc_params_stats = ensemble.mcmc_samples_stats(mcmc_samples)

    # Best parameter using the best log(prob)
    mcmc_best = mcmc_sampler.flatchain[mcmc_sampler.flatlnprobability.argmax()]

    # Best parameters using the mean of the last few samples
    _, n_step, n_dim = mcmc_chains.shape
    mcmc_mean = np.nanmean(
        mcmc_chains[:, -int(n_step * frac):, :].reshape([-1, n_dim]), axis=0)

    # Auto-correlation time
    try:
        tau = mcmc_sampler.get_autocorr_time(quiet=False, tol=tol, c=c)
        print("# Current autocorrelation time is", tau)
    except emcee.autocorr.AutocorrError:
        print("# The chain is shorter than {} x tau right now...".format(tol))
        tau = None

    np.savez(mcmc_file,
             samples=mcmc_samples, lnprob=np.array(mcmc_lnprob),
             best=np.array(mcmc_best), mean=np.asarray(mcmc_mean),
             chains=mcmc_chains, tau=tau,
             position=np.asarray(mcmc_position),
             acceptance=np.array(mcmc_sampler.acceptance_fraction))

    if verbose:
        print("#------------------------------------------------------")
        print("#  Mean acceptance fraction",
              np.mean(mcmc_sampler.acceptance_fraction))
        print("#------------------------------------------------------")
        print("#  Best ln(Probability): %11.5f" % np.max(mcmc_lnprob))
        print(mcmc_best)
        print("#------------------------------------------------------")
        print("#  Best parameters (mean):")
        print(mcmc_mean)
        print("#------------------------------------------------------")
        for param_stats in mcmc_params_stats:
            print(param_stats)
        print("#------------------------------------------------------")
