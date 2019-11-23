#!/bin/python
# -*- coding: utf-8 -*-

import tqdm
from sys import stdout
import numpy as np
from grgrlib import map2arr


def cmaes(objective_fct, xstart, sigma, popsize=None, mueff=None, pool=None, maxfev=None, verb_disp=100, verb_save=1000, **args):

    es = CMAES(xstart, sigma, mueff=mueff, maxfev=maxfev, popsize=int(popsize), **args)

    es.objective_fct = objective_fct
    es.pool = pool

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

    def __init__(self, N, popsize, mueff=None, maxsigma=None, fatol=None, frtol=None, xtol=None):

        self.ndim = N
        self.maxsigma = maxsigma
        self.chiN = (1 - 1. / (4 * N) + 1. / (21 * N**2))
        self.fatol = fatol or 1e-8
        self.frtol = frtol or 1e-8
        self.xtol = xtol or 1e-8

        self.default_popsize = int(4 + 3*np.log(N))
        self.lam = popsize or self.default_popsize

        # number of parents/points/solutions for recombination
        self.mu = int(self.lam / 2)

        weights = np.zeros(self.lam)
        weights[:self.mu] = np.log(self.lam / 2 + 0.2) - np.log(np.arange(self.mu) + 1)

        self.weights = weights / np.sum(weights)
        # variance-effectiveness of sum w_i x_i
        self.mueff = mueff or np.sum(self.weights)**2 / np.sum(self.weights**2)

        print('[cma-es:]'.ljust(15, ' ') + 
              '(%d' % (self.mu) + ',%d' % (self.lam) + ')-' + 'CMA-ES' + 
              ' (mu_w=%2.1f,w_1=%d%%)' % (self.mueff, int(100 * self.weights[0])) + 
              ' in %d dimensions (seed=%s)' % (N, np.random.get_state()[1][0]))

        # Strategy parameter setting: Adaptation
        # time constant for cumulation for C
        self.cc = (4 + self.mueff/N) / (N + 4 + 2*self.mueff/N)
        # time constant for cumulation for sigma control
        self.cs = (self.mueff + 2) / (N + self.mueff + 5)
        # learning rate for rank-one update of C
        self.c1 = 2 / ((N + 1.3)**2 + self.mueff)
        # and for rank-mu update
        self.cmu = np.minimum(1 - self.c1, 2 * (self.mueff - 2 + 1/self.mueff) /((N + 2)**2 + self.mueff))  
        # damping for sigma, usually close to 1
        self.damps = 2 * self.mueff/self.lam + 0.3 + self.cs  


class CMAES(object):  # could also inherit from object
    def __init__(self, xstart, sigma, popsize=None, mueff=None, maxfev=None, show_pbar=True, **args):
        """Instantiate `CMAES` object instance using `xstart` and `sigma`.

        Parameters
        ----------
            `xstart`: `list`
                of numbers (like ``[3, 2, 1.2]``), initial
                solution vector
            `sigma`: `float`
                initial step-size (standard deviation in each coordinate)
            `popsize`: `int` or `str`
                population size, number of candidate samples per iteration
            `maxfev`: `int` or `str`
                maximal number of function evaluations, a string is
                evaluated with ``N`` as search space dimension

        Details: this method initializes the dynamic state variables and
        creates a `CMAESParameters` instance for static parameters.
        """
        # process some input parameters and set static parameters
        # number of objective variables/problem dimension
        N = len(xstart)  
        self.params = CMAESParameters(N, popsize, mueff, **args)
        popsize = self.params.default_popsize
        self.maxfev = maxfev or 100*popsize + 150*(N + 3)**2*popsize**0.5
        self.show_pbar = show_pbar

        # initializing dynamic state variables
        # initial point, distribution mean, a copy
        self.xmean = np.array(xstart)
        self.sigma = sigma
        # evolution path for C
        self.pc = np.zeros(N)  
        # and for sigma
        self.ps = np.zeros(N)  
        self.C = np.eye(N)
        # countiter should be equal to counteval / lam
        self.counteval = 0  
        self.best = BestSolution()

    def run(self):
        """update the evolution paths and the distribution parameters m,
        sigma, and C within CMA-ES.

        Parameters
        ----------
            `arx`: `list` of "row vectors"
                a list of candidate solution vectors, presumably from
                calling `ask`. ``arx[k][i]`` is the i-th element of
                solution vector k.
            `fitvals`: `list`
                the corresponding objective function values, to be
                minimised
        """

        self.eigenvalues, self.eigenbasis = np.linalg.eigh(self.C)  # O(N**3)

        wrap = tqdm.tqdm if self.show_pbar else lambda x, **args: x

        def func(s):

            np.random.seed(s)

            f = np.inf
            niter = 0

            while np.isinf(f):
                z = self.sigma * self.eigenvalues**0.5 * np.random.normal(size=self.params.ndim)
                y = self.eigenbasis @ z
                x = self.xmean + y
                f = self.objective_fct(x)
                niter += 1

            return f, x, niter

        seeds = np.random.randint(2**32-2, size=self.params.lam)
        res = self.pool.imap(func, seeds)
        res = wrap(res, total=self.params.lam, unit='sample(s)', dynamic_ncols=True)

        fvals, xvals, niters = map2arr(res)
        self.show_pbar = np.mean(niters) > 3

        # bookkeeping and convenience short cuts
        # evaluations used within tell
        self.counteval += len(fvals)  
        par = self.params
        # not a copy, xmean is assigned anew later
        xold = self.xmean  

        # sort by fitness
        xvals = np.array(xvals)[np.argsort(fvals)]
        # used for termination and display only
        self.fvals = np.sort(fvals)  
        self.best.update(xvals[0], self.fvals[0], self.counteval)

        # recombination, compute new weighted mean value
        self.xmean = xvals.T @ par.weights

        invsqrt = np.linalg.inv(np.linalg.cholesky(self.C))

        # Cumulation: update evolution paths
        y = self.xmean - xold
        # z = C**(-1/2) * (xnew - xold)
        z = invsqrt @ y  
        csn = (par.cs * (2 - par.cs) * par.mueff)**0.5 / self.sigma
        self.ps = (1 - par.cs) * self.ps + csn * z
        ccn = (par.cc * (2 - par.cc) * par.mueff)**0.5 / self.sigma
        # turn off rank-one accumulation when sigma increases quickly
        hsig = (np.sum(self.ps**2) / self.params.ndim  # ||ps||^2 / N is 1 in expectation
                # account for initial value of ps
                / (1-(1-par.cs)**(2*self.counteval/par.lam))
                # should be smaller than 2 + ...
                < 2 + 4./(self.params.ndim+1))  
        self.pc = (1 - par.cc)*self.pc + ccn*hsig*y

        # Adapt covariance matrix C
        # minor adjustment for the variance loss from hsig
        c1a = par.c1*(1 - (1-hsig**2)*par.cc*(2-par.cc))
        # C *= 1 - c1 - cmu * sum(w)
        self.C *= 1 - c1a - par.cmu
        self.C += par.c1*np.outer(self.pc, self.pc)
        self.C += par.cmu/self.sigma**2 * (xvals-xold).T @ np.diag(par.weights) @ (xvals-xold) 

        # Adapt step-size sigma
        cn, sum_square_ps = par.cs / par.damps, np.sum(self.ps**2)
        self.sigma *= np.exp(np.minimum(1, cn * (sum_square_ps / self.params.ndim - 1) / 2))

        if self.params.maxsigma is not None:
            self.sigma = np.minimum(self.sigma, self.params.maxsigma)

    def stop(self):
        """return satisfied termination conditions in a dictionary,

        generally speaking like ``{'termination_reason':value, ...}``,
        for example ``{'tolfun':1e-12}``, or the empty `dict` ``{}``.
        """

        if not self.counteval:
            return False

        if len(self.fvals) > 1 and self.fvals[-1] - self.fvals[0] < self.params.fatol:
            return 'fatol of %1.0e.' %self.params.fatol
        if len(self.fvals) > 1 and self.fvals[-1]/self.fvals[0] - 1 < self.params.frtol:
            return 'frtol of %1.0e.' %self.params.frtol
        if self.sigma * np.max(self.eigenvalues)**0.5 < self.params.xtol:
            return 'xtol of %1.0e.' %self.params.xtol
        if self.counteval > self.maxfev:
            return 'maxfev of %1.0e.' %self.maxfev

        return False

    @property
    def result(self):
        """the `tuple` ``(xbest, f(xbest), evaluations_xbest, evaluations,
        iterations, xmean, stds)``
        """

        return (self.best.x,
                self.best.f,
                self.best.evals,
                self.counteval,
                int(self.counteval / self.params.lam),
                self.xmean,
                self.sigma * np.diagonal(self.C)**.5)

    def disp(self, verb_modulo=1):
        """`print` some iteration info to `stdout`
        """

        if verb_modulo is None:
            verb_modulo = 20
        if not verb_modulo:
            return
        iteration = self.counteval / self.params.lam

        if iteration == 1 or iteration % (10 * verb_modulo) < 1:
            print('evals: ax-ratio   std (min/max)     f-value')

        if iteration <= 2 or iteration % verb_modulo < 1:
            print(str(self.counteval).rjust(5) + ': ' +
                  ' %6.1e %8.1e %8.1e  %8.8e ' % (np.linalg.cond(self.C)**0.5,
                                                 self.sigma *
                                                 np.min(np.diagonal(
                                                     self.C))**0.5,
                                                 self.sigma *
                                                 np.max(np.diagonal(
                                                     self.C))**0.5,
                                                 self.fvals[0]))
                                                 # self.fvals[0]) + "\r", end='')
            stdout.flush()


class BestSolution(object):
    """container to keep track of the best solution seen"""

    def __init__(self, x=None, f=None, evals=None):
        """take `x`, `f`, and `evals` to initialize the best solution
        """
        self.x, self.f, self.evals = x, f, evals

    def update(self, x, f, evals=None):
        """update the best solution if ``f < self.f``
        """
        if self.f is None or f < self.f:
            self.x = x
            self.f = f
            self.evals = evals
        return self

    @property
    def all(self):
        """``(x, f, evals)`` of the best seen solution"""
        return self.x, self.f, self.evals
