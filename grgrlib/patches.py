#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np


def kombine_run_mcmc(self, N, p0=None, lnpost0=None, lnprop0=None, blob0=None, **kwargs):

    from kombine.sampler import _GetLnProbWrapper

    """
    This patches kombines run_mcmc to check if a progress bar is provided and if so, update it each step.

    Iterate :meth:`sample` for `N` iterations and return the result.

    :param N:
        The number of steps to take.

    :param p0: (optional)
        A list of the initial walker positions.  It should have the shape `(nwalkers, ndim)`.
        If ``None`` and the sampler has been run previously, it'll pick up where it left off.

    :param lnpost0: (optional)
        The list of log posterior probabilities for the walkers at positions `p0`. If ``lnpost0
        is None``, the initial values are calculated. It should have the shape `(nwalkers,
        ndim)`.

    :param lnprop0: (optional)
        List of log proposal densities for walkers at positions `p0`. If ``lnprop0 is None``,
        the initial values are calculated. It should have the shape `(nwalkers, ndim)`.

    :param blob0: (optional)
        The list of blob data for walkers at positions `p0`.

    :param kwargs: (optional)
        The rest is passed to :meth:`sample`.

    After `N` steps...

    :returns:
        * ``p`` - An array of current walker positions with shape `(nwalkers, ndim)`.

        * ``lnpost`` - The list of log posterior probabilities for the walkers at positions
          ``p``, with shape `(nwalkers, ndim)`.

        * ``lnprop`` - The list of log proposal densities for the walkers at positions `p`, with
          shape `(nwalkers, ndim)`.

        * ``blob`` - (if `lnprobfn` returns blobs) The list of blob data for the walkers at
          positions `p`.
    """

    m = self.pool.map

    if p0 is None:
        if self._last_run_mcmc_result is None:
            try:
                p0 = self.chain[-1]
                if lnpost0 is None:
                    lnpost0 = self.lnpost[-1]
                if lnprop0 is None:
                    lnprop0 = self.lnprop[-1]
            except IndexError:
                raise ValueError(
                    "Cannot have p0=None if the sampler hasn't been called.")
        else:
            p0 = self._last_run_mcmc_result[0]
            if lnpost0 is None:
                lnpost0 = self._last_run_mcmc_result[1]
            if lnprop0 is None:
                lnprop0 = self._last_run_mcmc_result[2]

    if self._kde is not None:
        if self._last_run_mcmc_result is None and (lnpost0 is None or lnprop0 is None):
            results = list(m(_GetLnProbWrapper(self._get_lnpost,
                                               self._kde, *self._lnpost_args), p0))

            if lnpost0 is None:
                lnpost0 = np.array([r[0] for r in results])
            if lnprop0 is None:
                lnprop0 = np.array([r[1] for r in results])

    pbar = kwargs.pop('pbar', None)

    if pbar is not None:
        pbar.write('[kombine:]'.ljust(15, ' ') + 'Updating KDE... (mean acceptance_fraction: ' + str(np.mean(
            self.acceptance_fraction).round(3)) + ', mean ACT: ' + str(np.mean(self.autocorrelation_times).round(3)) + ')')

    for results in self.sample(p0, lnpost0, lnprop0, blob0, N, **kwargs):

        if pbar is not None:
            pbar.update()
            pbar.set_description(
                'mean ll: '+str(np.mean(self._lnpost).round(5)))

    # Store the results for later continuation and toss out the blob
    self._last_run_mcmc_result = results[:3]

    return results
