"""Model fitting using emcee sampler."""
from __future__ import print_function, division, unicode_literals

import numpy as np
from numpy.random import normal, multivariate_normal

import emcee

import io

__all__ = ['emcee_burnin_batch', 'setup_moves', 'setup_walkers', 'run_emcee_sampler',
           'mcmc_samples_stats']


def setup_moves(cfg_emcee, burnin=False):
    """Choose the Move object for emcee.

    Parameters
    ----------
    cfg_emcee : dict
        Configuration parameters for emcee
    burnin : bool, optional
        Whether this is for burnin

    Return
    ------
    emcee_moves : emcee.moves object
        Move object for emcee walkers.

    """
    move_col = 'sample_move' if not burnin else 'burnin_move'

    if cfg_emcee[move_col] == 'snooker':
        emcee_moves = emcee.moves.DESnookerMove()
    elif cfg_emcee[move_col] == 'stretch':
        emcee_moves = emcee.moves.StretchMove(a=cfg_emcee['stretch_a'])
    elif cfg_emcee[move_col] == 'walk':
        emcee_moves = emcee.moves.WalkMove(s=cfg_emcee['walk_s'])
    elif cfg_emcee[move_col] == 'kde':
        emcee_moves = emcee.moves.KDEMove()
    elif cfg_emcee[move_col] == 'de':
        emcee_moves = emcee.moves.DEMove(cfg_emcee['de_sigma'])
    else:
        raise Exception("Wrong option: stretch, walk, kde, de, snooker")

    return emcee_moves


def setup_walkers(cfg_emcee, params, level=0.1):
    """Initialize walkers for emcee.

    Parameters
    ----------
    cfg_emcee: dict
        Configuration parameters for emcee.
    params: asap.Parameter object
        Object for model parameters.
    level: float, optional

    Returns
    -------
    ini_positions: numpy array with (N_walker, N_param) shape
        Initial positions of all walkers.

    """
    # Initialize the walkers
    if cfg_emcee['ini_prior']:
        # Use the prior distributions for initial positions of walkers.
        return params.sample(nsamples=cfg_emcee['burnin_n_walker'])

    return params.perturb(nsamples=cfg_emcee['burnin_n_walker'], level=level)


def mcmc_samples_stats(mcmc_samples):
    """1D marginalized parameter constraints."""
    return map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
               zip(*np.percentile(mcmc_samples,
                                  [16, 50, 84], axis=0)))


def emcee_burnin_batch(sampler_burnin, ini_positions, params, n_step,
                       prefix='asap', best_position=True, verbose=True):
    """Run the burn-in process in batch mode."""
    if verbose:
        print("\n# Phase: Burn-in ...")

    # Number of parameters
    n_dim = ini_positions.shape[1]

    # Parameter limits
    params_limits = np.array([params.low, params.upp])

    # Running the burn-in process
    burnin_results = sampler_burnin.run_mcmc(
        ini_positions, n_step, progress=True, store=True)
    burnin_pos, burnin_prob, _ = burnin_results

    # Save the burn-in results
    io.save_results_to_npz(burnin_results, sampler_burnin,
                           prefix + '_burnin.npz', n_dim, verbose=verbose,
                           tol=20, c=5)

    # Find best walker position
    burnin_best = sampler_burnin.flatlnprobability.argmax()

    # Find new initial positions for all walkers
    if best_position:
        initial_center = sampler_burnin.flatchain[burnin_best, :]
    else:
        initial_center = None

    if burnin_pos.shape[0] < (n_dim * 2):
        ini_positions = reinitialize_ball(
            burnin_pos, burnin_prob, center=initial_center,
            limits=params_limits, disp_floor=0.01,
            prior_check=params)
    else:
        ini_positions = reinitialize_ball_covar(
            burnin_pos, burnin_prob, center=initial_center,
            limits=params_limits, disp_floor=0.01,
            prior_check=params, threshold=30)

    if verbose:
        print("\n# Done burn-in process! ")

    return burnin_results, ini_positions


def emcee_burnin_repeat(sampler_burnin, ini_positions, params, n_step, n_repeat,
                        prefix='asap', best_position=True, prob0=None, verbose=True):
    """Run the burn-in process in batch mode."""
    if verbose:
        print("\n# Phase: Burn-in ...")

    # Number of parameters
    n_dim = ini_positions.shape[1]

    # Parameter limits
    params_limits = np.array([params.low, params.upp])

    for ii in range(n_repeat):
        print("#\n#   Burn-in step %d/%d " % (ii + 1, n_repeat))

        burnin_results = sampler_burnin.run_mcmc(
            ini_positions, n_step, progress=True, store=True)
        burnin_pos, burnin_prob, _ = burnin_results

        # Save the burn-in results
        io.save_results_to_npz(
            burnin_results, sampler_burnin,
            prefix + '_burnin_%d.npz' % (ii + 1), n_dim, verbose=verbose)

        # Find best walker position
        burnin_best = sampler_burnin.flatlnprobability.argmax()

        # Is new position better than old position?
        if prob0 is None or sampler_burnin.flatlnprobability[burnin_best] > prob0:
            prob0 = sampler_burnin.flatlnprobability[burnin_best]
            if best_position:
                initial_center = sampler_burnin.flatchain[burnin_best, :]
            else:
                initial_center = None

        if ii == n_repeat:
            # Done burning.
            if verbose:
                print("\n# Done all burn-in processes!")

            # Don't construct new sampler ball after last burn-in.
            sampler_burnin.reset()
            continue

        if burnin_pos.shape[0] < (n_dim * 2):
            ini_positions = reinitialize_ball(
                burnin_pos, burnin_prob, center=initial_center,
                limits=params_limits, disp_floor=0.0,
                prior_check=params)
        else:
            ini_positions = reinitialize_ball_covar(
                burnin_pos, burnin_prob, center=initial_center,
                limits=params_limits, disp_floor=0.0,
                prior_check=params, threshold=30)

        # Rest the chain
        sampler_burnin.reset()

        if verbose:
            print("\n#   Done burn #{}".format(ii + 1))

    return burnin_results, ini_positions


def emcee_sample_batch(sampler, ini_positions, n_step, prefix='asap', verbose=True):
    """Run the sampling process in batch mode."""
    if verbose:
        print("\n# Phase: Sampling Run ...")

    # Number of parameters
    n_dim = ini_positions.shape[1]

    sample_results = sampler.run_mcmc(
        ini_positions, n_step, store=True, progress=True)

    # Save the burn-in results
    io.save_results_to_npz(sample_results, sampler,
                           prefix + '_sample.npz', n_dim, verbose=verbose,
                           tol=50, c=5)

    return sample_results


def sampler_ball(center, disp, size=1):
    """Produce a ball around a given position.

    Notes
    -----
    This is from `prospect.fitting.ensemble` by Ben Johnson:
        https://github.com/bd-j/prospector/blob/master/prospect/fitting/ensemble.py
    """
    ndim = center.shape[0]
    if np.size(disp) == 1:
        disp = np.zeros(ndim) + disp
    pos = normal(size=[size, ndim]) * disp[None, :] + center[None, :]
    return pos


def reinitialize_ball_covar(pos, prob, threshold=50.0, center=None,
                            disp_floor=0.0, **extras):
    """Estimate the parameter covariance matrix from the positions of a
    fraction of the current ensemble and sample positions from the multivariate
    gaussian corresponding to that covariance matrix.  If ``center`` is not
    given the center will be the mean of the (fraction of) the ensemble.
    :param pos:
        The current positions of the ensemble, ndarray of shape (nwalkers, ndim)
    :param prob:
        The current probabilities of the ensemble, used to reject some fraction
        of walkers with lower probability (presumably stuck walkers).  ndarray
        of shape (nwalkers,)
    :param threshold: default 50.0
        Float in the range [0,100] giving the fraction of walkers to throw away
        based on their ``prob`` before estimating the covariance matrix.
    :param center: optional
        The center of the multivariate gaussian. If not given or ``None``, then
        the center will be estimated from the mean of the postions of the
        acceptable walkers.  ndarray of shape (ndim,)
    :param limits: optional
        An ndarray of shape (2, ndim) giving lower and upper limits for each
        parameter.  The newly generated values will be clipped to these limits.
        If the result consists only of the limit then a vector of small random
        numbers will be added to the result.
    :returns pnew:
        New positions for the sampler, ndarray of shape (nwalker, ndim)

    Notes
    -----
    This is from `prospect.fitting.ensemble` by Ben Johnson:
        https://github.com/bd-j/prospector/blob/master/prospect/fitting/ensemble.py
    """
    pos = np.atleast_2d(pos)
    nwalkers = prob.shape[0]
    good = prob > np.percentile(prob, threshold)

    if center is None:
        center = pos[good, :].mean(axis=0)

    Sigma = np.cov(pos[good, :].T)
    Sigma[np.diag_indices_from(Sigma)] += disp_floor**2
    pnew = resample_until_valid(multivariate_normal, center, Sigma,
                                nwalkers, **extras)

    return pnew


def reinitialize_ball(pos, prob, center=None, ptiles=[25, 50, 75],
                      disp_floor=0., **extras):
    """Choose the best walker and build a ball around it based on the other
    walkers.  The scatter in the new ball is based on the interquartile range
    for the walkers in their current positions

    Notes
    -----
    This is from `prospect.fitting.ensemble` by Ben Johnson:
        https://github.com/bd-j/prospector/blob/master/prospect/fitting/ensemble.py
    """
    pos = np.atleast_2d(pos)
    nwalkers = pos.shape[0]

    if center is None:
        center = pos[prob.argmax(), :]
    tmp = np.percentile(pos, ptiles, axis=0)

    # 1.35 is the ratio between the 25-75% interquartile range and 1
    # sigma (for a normal distribution)
    scatter = np.abs((tmp[2] - tmp[0]) / 1.35)
    scatter = np.sqrt(scatter**2 + disp_floor**2)

    pnew = resample_until_valid(sampler_ball, center, scatter,
                                nwalkers, **extras)

    return pnew


def resample_until_valid(sampling_function, center, sigma, nwalkers,
                         limits=None, maxiter=1e3, prior_check=None, **extras):
    """Sample from the sampling function, with optional clipping to prior
    bounds and resampling in the case of parameter positions that are outside
    complicated custom priors.
    :param sampling_function:
        The sampling function to use, it must have the calling sequence
        ``sampling_function(center, sigma, size=size)``
    :param center:
        The center of the distribution
    :param sigma:
        Array describing the scatter of the distribution in each dimension.
        Can be two-dimensional, e.g. to describe a covariant multivariate
        normal (if the sampling function takes such a thing).
    :param nwalkers:
        The number of valid samples to produce.
    :param limits: (optional)
        Simple limits on the parameters, passed to ``clip_ball``.
    :param prior_check: (optional)
        An object that has a ``prior_product()`` method which returns the prior
        ln(probability) for a given parameter position.
    :param maxiter:
        Maximum number of iterations to try resampling before giving up and
        returning a set of parameter positions at least one of which is not
        within the prior.
    :returns pnew:
        New parameter positions, ndarray of shape (nwalkers, ndim)

    Notes
    -----
    This is from `prospect.fitting.ensemble` by Ben Johnson:
        https://github.com/bd-j/prospector/blob/master/prospect/fitting/ensemble.py
    """
    invalid = np.ones(nwalkers, dtype=bool)
    pnew = np.zeros([nwalkers, len(center)])

    for i in range(int(maxiter)):
        # replace invalid elements with new samples
        tmp = sampling_function(center, sigma, size=invalid.sum())
        pnew[invalid, :] = tmp
        if limits is not None:
            # clip to simple limits
            if sigma.ndim > 1:
                diag = np.diag(sigma)
            else:
                diag = sigma
            pnew = clip_ball(pnew, limits, diag)

        if prior_check is not None:
            # check the prior
            lnp = np.array([prior_check.lnprior(pos, nested=False) for pos in pnew])
            invalid = ~np.isfinite(lnp)
            if invalid.sum() == 0:
                # everything is valid, return
                return pnew
        else:
            # No prior check, return on first iteration
            return pnew
    # reached maxiter, return whatever exists so far
    print("initial position resampler hit ``maxiter``")

    return pnew


def clip_ball(pos, limits, disp):
    """Clip to limits.  If all samples below (above) limit, add (subtract) a
    uniform random number (scaled by ``disp``) to the limit.
    """
    npos = pos.shape[0]
    pos = np.clip(pos, limits[0], limits[1])

    for i, p in enumerate(pos.T):
        u = np.unique(p)
        if len(u) == 1:
            tiny = disp[i] * np.random.uniform(0, disp[i], npos)
            if u == limits[0, i]:
                pos[:, i] += tiny
            if u == limits[1, i]:
                pos[:, i] -= tiny

    return pos


def run_emcee_sampler(cfg, params, ln_probability, postargs=[], postkwargs={}, pool=None,
                      verbose=True, debug=False):
    """Fit A.S.A.P model using emcee sampler.

    Parameters
    ----------
    config_file: str
        Configuration file.
    verbose: boolen, optional
        Blah, blah, blah.  Default: True

    """
    # Initialize the walkers
    ini_positions = setup_walkers(cfg['model']['emcee'], params, level=0.1)

    # Number of parameters and walkers for burn-in process
    n_dim = params.n_param
    n_walkers = ini_positions.shape[0]
    n_step_burnin = cfg['model']['emcee']['burnin_n_sample']
    prefix = cfg['model']['prefix']

    # --------------------------- Burn-in Process ---------------------------- #
    # Setup the `Move` for burn-in walkers
    burnin_move = setup_moves(cfg['model']['emcee'], 'burnin_move')

    # Initialize sampler
    sampler_burnin = emcee.EnsembleSampler(
        n_walkers, n_dim, ln_probability, args=postargs, kwargs=postkwargs,
        pool=pool, moves=burnin_move)

    # Burn-in process
    n_repeat = cfg['model']['emcee']['burnin_n_repeat']
    # Repeat a few times with new initial positions if necessary
    if n_repeat == 1:
        burnin_results, new_ini_positions = emcee_burnin_batch(
            sampler_burnin, ini_positions, params, n_step_burnin,
            prefix=prefix, verbose=verbose,
            best_position=cfg['model']['emcee']['best_positions'])
    else:
        burnin_results, new_ini_positions = emcee_burnin_repeat(
            sampler_burnin, ini_positions, params, n_step_burnin, n_repeat,
            prefix=prefix, verbose=verbose,
            best_position=cfg['model']['emcee']['best_positions'])

    if debug:
        return burnin_results, sampler_burnin

    # --------------------------- Sampling Process --------------------------- #
    # Number of walkers and steps for the final sampling run
    # n_walkers_sample = cfg['model']['emcee']['sample_n_walker']
    n_step_sample = cfg['model']['emcee']['sample_n_sample']

    # Decide the Ensemble moves for walkers during the official run
    sample_move = setup_moves(cfg['model']['emcee'], 'sample_move')

    # Initialize sampler
    sampler_final = emcee.EnsembleSampler(
        n_walkers, n_dim, ln_probability, args=postargs, kwargs=postkwargs,
        pool=pool, moves=sample_move)

    # Final sampling run
    sample_results = emcee_sample_batch(
        sampler_final, new_ini_positions, n_step_sample, prefix=prefix, verbose=True)

    return sample_results, sampler_final
