#!/bin/python
# -*- coding: utf-8 -*-

import jax
import jax.numpy as np
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
            def jac(x): return ssp.csr_array(jax.jacfwd(func)(x))
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
            print(
                f'    Iteration {cnt:3d} | max error {eps:.2e} | lapsed %ss' % str(ltime)[:6])

        if cnt == maxit:
            res['success'] = False
            res['message'] = f"Maximum number of {maxit} iterations reached."
            break

        if eps < tol:
            res['success'] = True
            res['message'] = "The solution converged."
            break

        if jax.numpy.isnan(eps):
            raise Exception('Newton method returned `NaN` in iter %s' % cnt)

    res['x'], res['fun'], res['niter'] = xi, func(xi), cnt

    return res


def newton_jax_jittable(func, init, jac=None, maxit=30, tol=1e-8):
    """Newton method for root finding using automatic differenciation with jax BUT running in pure jitted jax. The argument `func` must be jittable with jax. Remember to check the error flags!

    Note that when compiling this function without context, it is necessary to have the function as a static argument. This would imply that AD does not work on functions including a jitted version of this function, which renders jax rather useless. The major advantage of having this function is to include the jittable version (not the jitted one) directly into to-be-jitted code _together with the function for which the root is needed_.

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

    Returns
    -------
    res: (xopt, fopt, niter, success)
    """

    if jac is None:
        jac = jax.jacfwd(func)

    xi = np.array(init)

    def cond_func(tain):
        xi, xold, cnt = tain
        eps = np.abs(xi - xold).max()

        cond = cnt < maxit
        cond &= eps > tol
        cond &= ~jax.numpy.isnan(eps)

        return cond

    def body_func(tain):
        (xi, _, cnt) = tain
        cnt += 1
        xold = xi
        xi -= .1*jax.scipy.linalg.solve(jac(xi), func(xi))
        return (xi, xold, cnt)

    tain = jax.lax.while_loop(cond_func, body_func, (xi, xi + 1, 0))
    eps = np.abs(tain[0] - tain[1]).max()

    return tain[0], func(tain[0]), tain[2], eps < tol


newton_jax_jit = jax.jit(newton_jax_jittable, static_argnums=(0, 2, 3, 4))
