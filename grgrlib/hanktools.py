
"""Almost exclusively taken from the replication codes of "Using the Sequence-Space Jacobian to Solve and Estimate Heterogeneous-Agent Models" (Auclert, Bardóczy, Rognlie, Straub)"""

import numpy as np
import pandas as pd
from numba import njit
from sequence_jacobian.estimation import all_covariances, log_likelihood
from grgrlib import load_as_module as load_model
from inspect import signature


@njit(nogil=True, cache=True)
def arma_irf(ar_coeff, ma_coeff, T):
    """Generates shock IRF for any ARMA process """

    x = np.empty((T,))
    n_ar = ar_coeff.size
    n_ma = ma_coeff.size
    # this means all MA coefficients are multiplied by -1 (this is what SW etc all have)
    sign_ma = -1
    for t in range(T):
        if t == 0:
            x[t] = 1
        else:
            ar_sum = 0
            for i in range(min(n_ar, t)):
                ar_sum += ar_coeff[i] * x[t - 1 - i]
            ma_term = 0
            if 0 < t <= n_ma:
                ma_term = ma_coeff[t - 1]
            x[t] = ar_sum + ma_term * sign_ma
    return x


def get_ma_inf(x, G, pars, shock_specs, observables, T, T_irf):
    """Compute the MA representation given G"""

    n_se, n_sh = len(observables), len(shock_specs)
    As = np.empty((T_irf, n_se, n_sh))

    for i_sh, sh in enumerate(shock_specs):

        arma_shock = arma_irf(np.array([x[pars.index(sh+'_AR_COEF')] if sh+'_AR_COEF' in pars else 0]),
                              np.array([x[pars.index(sh+'_MA_COEF')]
                                       if sh+'_MA_COEF' in pars else 0]),
                              T)

        if np.abs(arma_shock[-1]) > 1e20:
            raise ValueError(
                'ARMA shock misspecified, leading to explosive shock path!')

        # store for each series
        shockname = sh
        for i_se in range(n_se):
            As[:, i_se, i_sh] = (G[observables[i_se]]
                                 [shockname] @ arma_shock)[:T_irf]

    return As


def get_ll(x, model, data, data_func, sigma_measurement, jac_info, pars, observables, shock_specs, fixed_jacobians={}, ss_func=None, debug=False):

    try:
        # Update parameters and write new parameters into ss
        ss = jac_info['ss'].copy()
        T = jac_info['T']
        ss.update({p: x[pars.index(p)] for p in pars if '_COEF' not in p})

        if ss_func is not None:
            if len(signature(ss_func).parameters) == 1:
                print('hier')
                ss.update(ss_func(ss))
            else:
                ss.update(ss_func(ss, x))

        if data_func is False:
            data_adj = data.copy()
        else:
            data_adj = data_func(x, ss, data)

        # Compute model jacobian G
        G = model.solve_jacobian(
            ss, jac_info['unknowns'], jac_info['targets'], jac_info['exogenous'], T=T, Js=fixed_jacobians)

        # Compute log likelihood
        ll = loglik_f(x, pars, data_adj, sigma_measurement,
                      observables, shock_specs, T, G)

    except (ValueError, TypeError, np.linalg.LinAlgError):
        if debug:
            raise
        return -np.inf, None

    return ll, ss


def loglik_f(x, pars, data, sigma_measurement, observables, shock_specs, T, G):

    T_irf = T - 20
    n_se, n_sh = len(observables), len(shock_specs)

    # extract shock parameters from x: sigmas
    sigmas = np.array([x[pars.index(s+'_SIG_COEF')] for s in shock_specs])

    # Step 1
    As = get_ma_inf(x, G, pars, shock_specs, observables, T, T_irf)

    # Step 2
    Sigma = all_covariances(As, sigmas)  # burn-in for jit

    # Step 3
    llh = log_likelihood(data, Sigma, sigma_measurement)

    return llh


def get_normalized_data(ss, file, series):
    # load data. Note: data is *annualized* and *in percentages*
    df = pd.read_csv(file, index_col=0)
    df.index = pd.to_datetime(df.index)
    df.index = df.index.to_period("Q")

    crosswalk = {'y': 'Y', 'c': 'C', 'I': 'I',
                 'n': 'N', 'w': 'w', 'pi': 'pi', 'i': 'i'}

    # make quarterly
    for var in ['pi', 'i', 'y', 'c', 'I', 'n']:
        df[var] = df[var] / 4

    # convert quantities into percentage deviations from ss output
    df_out = df.copy()
    for var in ['y', 'c', 'I', 'n', 'w']:
        if var in ss:
            df_out[var] *= ss[crosswalk[var]] / ss['Y']

    return df_out[series].values


def get_prior(prior, shocks_dict=None, verbose=False):

    from emcwrap import get_prior as ew_get_prior

    extended_prior = prior.copy()

    # add priors for all AR or MA coefficients
    for action in ('AR_COEF', 'MA_COEF', 'SIG_COEF'):
        if action in extended_prior:
            done_sth = False
            if shocks_dict is None:
                raise ValueError(
                    f"`shocks_dict` must be provided if using `{action}` in prior specification.")
            for shock in shocks_dict:
                # only if it is not already set manually
                do = not (action == 'AR_COEF' and shocks_dict[shock] < 1)
                do &= not (action == 'MA_COEF' and shocks_dict[shock] < 2)
                do &= shock+'_'+action not in extended_prior
                if do:
                    if verbose:
                        print('   adding %s...' % (shock+'_'+action))
                    extended_prior[shock+'_'+action] = extended_prior[action]
                    done_sth = True
            # remove placeholder
            del extended_prior[action]
            if not done_sth:
                # warn if all were already set manually
                print(f'   Warning: Nothing to do for `{action}`')

    return *ew_get_prior(extended_prior, verbose), extended_prior


def irfs(shocklist, G, rho=.9):

    T = list(G[G.outputs[0]].values())[0].shape[0]

    if isinstance(shocklist, str) or len(shocklist) == 1:
        shocklist = (1, shocklist, 0)
    if len(shocklist) == 2:
        shocklist = (*shocklist, 0)

    ssize, stype, stime = shocklist

    drstar = ssize * rho ** np.arange(T)
    evars = [v for v in G.outputs if G[v]]
    X = np.empty((T, len(evars)))

    for i, v in enumerate(evars):
        X[:, i] = G[v][stype] @ drstar

    return pd.DataFrame(X, columns=evars)


def vix(elist, G):

    evars = [v for v in G.outputs if G[v]]

    return np.nonzero([v in elist for v in evars])
