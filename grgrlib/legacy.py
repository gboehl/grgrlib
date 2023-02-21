# -*- coding: utf-8 -*-
"""
this file contains functions that are not activly developed anymore and only provided for reproducibility purposes
"""


def tmcmc(
    self,
    nwalks,
    nsteps=200,
    ntemps=0,
    target=None,
    update_freq=False,
    check_likelihood=False,
    verbose=True,
    **mcmc_args
):
    """Run Tempered Ensemble MCMC

    Parameters
    ----------
    ntemps : int
    target : float
    nsteps : float
    """

    from grgrlib import map2arr
    from pydsge.mpile import prior_sampler

    update_freq = update_freq if update_freq <= nsteps else False

    # sample pars from prior
    pars = prior_sampler(
        self, nwalks, check_likelihood=check_likelihood, verbose=verbose)

    x = get_par(self, "prior_mean", asdict=False,
                full=False, verbose=verbose > 1)

    pbar = tqdm.tqdm(total=ntemps, unit="temp(s)", dynamic_ncols=True)
    tmp = 0

    for i in range(ntemps):

        # update tmp
        ll = self.lprob(x)
        lp = self.lprior(x)

        tmp = tmp * (ntemps - i - 1) / (ntemps - i) + (target - lp) / (ntemps - i) / (
            ll - lp
        )
        aim = lp + (ll - lp) * tmp

        if tmp >= 1:
            # print only once
            pbar.write(
                "[tmcmc:]".ljust(15, " ")
                + "Increasing temperature to %s°. Too hot! I'm out..."
                % np.round(100 * tmp, 3)
            )
            pbar.update()
            self.temp = 1
            # skip for-loop to exit
            continue

        pbar.write(
            "[tmcmc:]".ljust(15, " ")
            + "Increasing temperature to %2.5f°, aiming @ %4.3f." % (100 * tmp, aim)
        )
        pbar.set_description("[tmcmc: %2.3f°" % (100 * tmp))

        self.mcmc(
            p0=pars,
            nsteps=nsteps,
            temp=tmp,
            update_freq=update_freq,
            verbose=verbose > 1,
            append=i,
            report=pbar.write,
            **mcmc_args
        )

        self.temp = tmp
        self.mcmc_summary(tune=int(nsteps / 10),
                          calc_mdd=False, calc_ll_stats=True)

        pbar.update()

        pars = self.get_chain()[-1]
        lprobs_adj = self.get_log_prob()[-1]
        x = pars[lprobs_adj.argmax()]

    pbar.close()
    self.fdict["datetime"] = str(datetime.now())

    return pars
