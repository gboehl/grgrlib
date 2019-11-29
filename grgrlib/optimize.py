#!/bin/python
# -*- coding: utf-8 -*-

import tqdm
import time
import numpy as np
from grgrlib import map2arr


def cmaes(objective_fct, xstart, sigma, popsize=None, pool=None, biject=False, verb_disp=100, **args):
    """UI access point to `CMAES`
    """

    es = CMAES(xstart, sigma, popsize=popsize, biject=biject, **args)

    es.bfunc = (lambda x: 1/(1 + np.exp(x))) if biject else (lambda x: x)
    es.objective_fct = lambda x: objective_fct(es.bfunc(x))

    es.pool = pool
    es.stime = time.time()

    while not es.stop():
        es.run()
        es.disp(verb_disp)

    # do not print by default to allow silent verbosity
    if verb_disp:

        es.disp(1)
        print('[cma-es:]'.ljust(15, ' ') + 'termination by ' + es.stop())

    return es.result


class CMAESParameters(object):
    """static "internal" parameter setting for `CMAES`
    """

    def __init__(self, ndim, popsize, mu=None, cc=None, cs=None, c1=None, cmu=None, fatol=None, frtol=None, xtol=None, maxfev=None, active=None, scaled=False, elitist=False, ld_rule=None):
        """Set static, fixed "strategy" parameters.

        Parameters
        ---------- 
        ndim : int
            Dimensionality of the problem.
        popsize : int
            Population size.
        cc : float, optional
            Backward time horizon for the evolution path (automatically assigned by default) 
        cs : float, optional
            Makes partly up for the small variance loss in case the indicator is zero (automatically assigned by default) 
        c1 : float, optional
            Learning rate for the rank-one update of the covariance matrix (automatically assigned by default)
        cmu : float, optional
            Learning rate for the rank-μ update of the covariance matrix (automatically assigned by default)
        fatol : float, optional
            Absolute tolerance of function value (defaults to 1e-8)
        frtol : float, optional
            Relative tolerance of function value (defaults to 1e-8)
        xtol : float, optional
            Absolute tolerance of solution values (defaults to 1e-8)
        active : bool, optional
            Whether to use aCMA-ES, a modern variant (True by default, unless `scaled` CMA is used)
        scaled : bool, optional
            Whether to scale CMA-ES for large populatoins (False by default)
        """

        self.fatol = fatol or 1e-8
        self.frtol = frtol or 1e-8
        self.xtol = xtol or 1e-8

        self.ndim = ndim
        # low-discrepancy rule (chaospy)
        self.rule = 'L' if ld_rule is None else ld_rule
        self.elitist = elitist

        # set strategy parameter for selection
        def_pop = 4 + int(3*np.log(ndim))
        self.lam = popsize or def_pop
        
        mu = int(def_pop/2.) if scaled else mu
        # set number of parents/points/solutions for recombination
        self.mu = int(mu or (self.lam / 2))

        self.maxfev = maxfev or 100*self.lam + 150*(ndim+3)**2*self.lam**.5

        if active is None:
            active = False if scaled else True

        if active:
            self.weights, self.mueff = self.recombination_weights()
        else:
            # set non-negative recombination weights & normalize them
            weights = np.zeros(self.lam)
            weights[:self.mu] = np.log(self.mu + 0.5) - np.log(np.arange(self.mu) + 1)
            self.weights = weights/np.sum(weights[:self.mu])
            # variance-effectiveness of sum w_i x_i
            self.mueff = np.sum(
                self.weights[:self.mu])**2 / np.sum(self.weights[:self.mu]**2)

        # set strategy parameter adaptation:
        # set time constant for cumulation for COV
        def_cc = (4 + self.mueff/ndim) / (ndim+4 + 2 * self.mueff/ndim)
        self.cc = def_cc if cc is None else cc

        # define time constant for cumulation of sigma control
        def_cs = (self.mueff + 2) / (ndim + self.mueff + 5)
        self.cs = def_cs if cs is None else cs

        # define learning rate of rank-one update
        def_c1 = 2 / ((ndim + 1.3)**2 + self.mueff)
        self.c1 = def_c1 if c1 is None else c1

        # define learning rate of rank-mu update
        def_cmu = np.minimum(
            1 - self.c1, 2 * (self.mueff - 2 + 1/self.mueff) / ((ndim + 2)**2 + self.mueff))
        self.cmu = def_cmu if cmu is None else cmu

        # define damping for sigma (usually close to 1)
        self.damps = self.mueff/self.mu + 0.3 + self.cs

        if active:
            self.finalize_weights()

        prt_str0 = '(%d' % (self.mu) + ',%d' % (self.lam) + ')-' + 'CMA-ES'
        prt_str0 += ' (mu_w=%2.1f,w_1=%d%%)' % (self.mueff,
                                                int(100 * self.weights[0]))
        prt_str0 += ' in %d dimensions (seed=%s)' % (ndim,
                                                     np.random.get_state()[1][0])

        prt_str1 = '[cc=%1.2f' % self.cc
        prt_str1 += '(%1.2f)' % def_cc if def_cc != self.cc else ''
        prt_str1 += ', cs=%1.2f' % self.cs
        prt_str1 += '(%1.2f)' % def_cs if def_cs != self.cs else ''
        prt_str1 += ', c1=%1.2f' % self.c1
        prt_str1 += '(%1.2f)' % def_c1 if def_c1 != self.c1 else ''
        prt_str1 += ', cmu=%1.2f' % self.cmu
        prt_str1 += '(%1.2f)]' % def_cmu if def_cmu != self.cmu else ']'
        print('[cma-es:]'.ljust(15, ' ') + prt_str0)
        print(''.ljust(15, ' ') + prt_str1)

        if scaled and active:
            print(''.ljust(15, ' ') + 'warning: scaled CMA works better without active adaptation.')

    def recombination_weights(self):

        weights = np.log(self.lam + 1) - np.log(np.arange(self.lam)+1) - np.log(2)

        mu = np.sum(weights > 0)
        weights /= np.sum(weights[:mu])

        # define variance-effectiveness
        mueff = 1 / np.sum(weights[:mu]**2)
        sum_neg = np.sum(weights[mu:])

        if sum_neg != 0:
            weights[mu:] /= -sum_neg

        return weights, mueff

    def finalize_weights(self):

        if self.weights[-1] < 0:
            if self.cmu > 0:

                value = np.abs(1 + self.c1 / self.cmu)

                if self.weights[-1] < 0:
                    factor = np.abs(value / np.sum(self.weights[self.mu:]))
                    self.weights[self.mu:] *= factor

                value = np.abs((1 - self.c1 - self.cmu) / self.cmu / self.ndim)

                # if nothing to limit
                if np.sum(self.weights[self.mu:]) < -value:  
                    factor = np.abs(value / np.sum(self.weights[self.mu:]))
                    if factor < 1:
                        self.weights[self.mu:] *= factor

            sum_neg = np.sum(self.weights[self.mu:])
            mueffminus = sum_neg**2 / \
                np.sum(self.weights[self.mu:]**2) if sum_neg else 0
            value = np.abs(1 + 2 * mueffminus / (self.mueff + 2))

            # if nothing to limit
            if sum_neg < -value:  
                factor = np.abs(value / sum_neg)
                if factor < 1:
                    self.weights[self.mu:] *= factor

        return


class CMAES(object):

    def __init__(self, xstart, sigma, popsize, biject, **args):

        # number of dimensions
        N = len(xstart)
        self.params = CMAESParameters(N, popsize, **args)

        # define bijection function
        self.tfunc = (lambda x: np.log(1/x - 1)) if biject else (lambda x: x)

        # use low-discrepancy series
        try:
            if not self.params.rule:
                raise ModuleNotFoundError()
            import chaospy
            self.randn = lambda size: chaospy.MvNormal(np.zeros(size[1]), np.eye(size[1])).sample(size=size[0], rule=self.params.rule).T
        except ModuleNotFoundError:
            self.randn = lambda size: np.random.normal(0, 1, size=size)

        # initialize dynamic state variables
        self.sigma = self.tfunc(.5-sigma) if biject else sigma
        self.xmean = self.tfunc(xstart)

        # initialize evolution path for COV
        self.pc = np.zeros(N)
        # initialize evolution path for sigma
        self.ps = np.zeros(N)
        self.C = np.eye(N)
        self.counteval = 0
        self.best = BestSolution()
        self.biject = biject

    def run(self):
        """update the evolution paths and the distribution parameters m,
        sigma, and C within CMA-ES.
        """

        par = self.params

        self.eigenvalues, self.eigenbasis = np.linalg.eigh(self.C)
        self.condition_number = np.linalg.cond(self.C)
        self.invsqrt = np.linalg.inv(np.linalg.cholesky(self.C))

        z = self.sigma * self.eigenvalues**0.5 * self.randn(size=(par.lam, par.ndim))
        y = self.eigenbasis @ z.T
        xs = self.xmean + y.T

        res = self.pool.imap(self.objective_fct, xs)
        fs = map2arr(res)

        # interrupt if things don't go nicely
        if np.isinf(fs).all():
            raise ValueError('Sample returns infs only.')

        self.counteval += len(fs)
        N = len(self.xmean)
        xold = self.xmean.copy()

        if par.elitist and hasattr(self, 'fs'):
            fsold = self.fs.copy()
            xsold = self.xs.copy()
            fs = np.hstack((fs,fsold))
            xs = np.vstack((xs,xsold))

        # sort by fitness
        xs = xs[fs.argsort()][:par.lam]
        self.fs = np.sort(fs)[:par.lam]
        self.xs = xs

        self.best.update(xs[0], self.fs[0], self.counteval)

        # compute new weighted mean value via recombination
        self.xmean = xs[0:par.mu].T @ par.weights[:par.mu]

        # update evolution paths via cumulation: 
        y = self.xmean - xold
        z = self.invsqrt @ y
        csn = (par.cs * (2 - par.cs) * par.mueff)**0.5 / self.sigma
        # update evolution path
        self.ps = (1 - par.cs) * self.ps + csn * z

        ccn = (par.cc * (2 - par.cc) * par.mueff)**0.5 / self.sigma
        # turn off rank-one accumulation when sigma increases quickly
        hsig = par.cs and (np.sum(self.ps**2) / N
                           # account for initial value of ps
                           / (1-(1-par.cs)**(2*self.counteval/par.lam))
                           < 2 + 4/(N+1))

        self.pc = (1 - par.cc) * self.pc + ccn * hsig * y

        # covariance matrix adaption:
        # minor adjustment for the variance loss from hsig
        c1a = par.c1 * (1 - (1-hsig**2) * par.cc * (2-par.cc))
        self.C *= 1 - c1a - par.cmu * np.sum(par.weights)
        # rank-one update
        self.C += par.c1 * np.outer(self.pc, self.pc)

        # rank-mu update
        for k, wk in enumerate(par.weights):
            # guaranty positive definiteness
            if wk < 0:
                mahalano = np.sum((self.invsqrt @ (xs[k] - xold))**2)
                wk *= N * self.sigma**2 / mahalano
            self.C += wk * par.cmu / self.sigma**2 * \
                np.outer(xs[k] - xold, xs[k] - xold)

        # adapt step-size
        cn, sum_square_ps = par.cs / par.damps, np.sum(self.ps**2)
        self.sigma *= np.exp(np.minimum(1, cn * (sum_square_ps / N - 1) / 2))

    def stop(self):
        """Check termination criteria and return string.
        """

        if not self.counteval or np.isinf(self.fs).any():
            return False

        if len(self.fs) > 1 and self.fs[-1] - self.fs[0] < self.params.fatol:
            return 'fatol of %1.0e.' % self.params.fatol
        if len(self.fs) > 1 and self.fs[-1]/self.fs[0] - 1 < self.params.frtol:
            return 'frtol of %1.0e.' % self.params.frtol
        if self.sigma * np.max(self.eigenvalues)**0.5 < self.params.xtol:
            return 'xtol of %1.0e.' % self.params.xtol
        if self.counteval > self.params.maxfev:
            return 'maxfev of %1.0e.' % self.params.maxfev

        return False

    @property
    def result(self):

        return (self.bfunc(self.best.x),
                self.best.f,
                self.best.evals,
                self.counteval,
                int(self.counteval / self.params.lam),
                self.bfunc(self.xmean),
                self.sigma * np.diag(self.C)**0.5)

    def disp(self, verb_modulo=1):
        """print well-formated iteration info 
        """

        if verb_modulo is None:
            verb_modulo = 20
        if not verb_modulo:
            return
        iteration = self.counteval / self.params.lam

        do_print = False
        do_print |= iteration <= 3
        do_print |= iteration % verb_modulo < 1

        if iteration == 1 or iteration % (10 * verb_modulo) < 1:
            print('evals:  sigma   ax-ratio   std (min/max)     f-value     t(mm:ss)')

        if do_print:
            frac_inf = np.sum(np.isinf(self.fs))
            diag_sqrt = np.diagonal(self.C)**0.5
            info_str = str(self.counteval).rjust(5) + ': ' + ' %2.1e  %6.1e %7.0e %7.0e  %8.8e ' % (self.sigma,
                                                                                                    np.linalg.cond(self.C)**0.5, self.sigma*np.min(diag_sqrt), self.sigma*np.max(diag_sqrt), self.fs[0])
            info_str += ' (%02d:%02d)' % divmod(time.time() - self.stime, 60)
            info_str += ' -> %02d%% inf' % (frac_inf /
                                            self.params.lam*100) if frac_inf else ''

            print(info_str)


class BestSolution(object):
    """Container to keep track of the best solution seen.
    """

    def __init__(self, x=None, f=None, evals=None):
        """Take `x`, `f`, and `evals` to initialize the best solution.
        """
        self.x, self.f, self.evals = x, f, evals

    def update(self, x, f, evals=None):
        """Update the best solution if ``f < self.f``.
        """
        if self.f is None or f < self.f:
            self.x = x
            self.f = f
            self.evals = evals
        return self

    @property
    def all(self):
        """``(x, f, evals)`` of the best seen solution.
        """
        return self.x, self.f, self.evals
