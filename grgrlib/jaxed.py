#!/bin/python
# -*- coding: utf-8 -*-

import jax
import time
import scipy.sparse as ssp


def newton_jax(func, init, jac=None, maxit=30, tol=1e-8, sparse=False, verbose=False):
    """Newton method for root finding using automatic differenciation with jax. The argument `func` must be jittable with jax.

    ...

    Parameters
    ----------
    func : callable 
        Function f for which f(x)=0 should be found. Must be jittable with jax
    init : array
        Initial values of x
    jac : callable, optional
        Funciton that returns the jacobian. If not provided, jax.jacfwd is used
    maxit : int, optional
        Maximum number of iterations
    tol : float, optional
        Random seed. Defaults to 0
    sparse : bool, optional
        Whether to use a sparse solver. If `true`, and jac is supplied, this should return a sparse matrix
    verbose : bool, optional
        Whether to display messages

    Returns
    -------
    res: dict 
        A dictionary of results similar to the output from scipy.optimize.root
    """

    st = time.time()

    if jac is None:
        if sparse:
            jac = lambda x: ssp.csr_array(jax.jacfwd(func)(x))
        else:
            jac = jax.jacfwd(func)

    if sparse:
        solver = ssp.linalg.spsolve
    else:
        solver = jax.scipy.linalg.solve

    res = {}
    cnt = 0
    xi = jax.numpy.array(init)

    while True:
        cnt += 1
        xold = xi.copy()
        xi -= solver(jac(xi), func(xi))
        eps = jax.numpy.abs(xi - xold).max()

        if verbose:
            ltime = time.time() - st
            print(f'    Iteration {cnt:3d} | max error {eps:.2e} | lapsed %ss' %str(ltime)[:6])

        if cnt == maxit:
            res['success'] = False
            res['message'] = f"Maximum number of {maxit} iterations reached."
            break

        if eps < 1e-8:
            res['success'] = True
            res['message'] = "The solution converged."
            break

        if jax.numpy.isnan(eps):
            raise Exception('Newton method returned `NaN`')

    res['x'], res['fun'], res['niter'] = xi, func(xi), cnt

    return res
