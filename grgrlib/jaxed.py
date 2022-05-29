#!/bin/python
# -*- coding: utf-8 -*-

import jax
import time
import functools
import jax.numpy as jnp
import scipy.sparse as ssp
from numpy import ndarray, isnan
from .plots import spy
from jax.experimental.host_callback import id_print as jax_print
# from jaxlib.xla_extension import DeviceArray


def value_and_jac_inner(f, x, sparse):
    """Return value and Jacobian of x.
    """

    pushfwd = functools.partial(jax.jvp, f, (x,))
    basis = jnp.eye(x.size, dtype=x.dtype)
    y, jac = jax.vmap(pushfwd, out_axes=(None, 1))((basis,))

    if sparse:
        return y, ssp.csr_array(jac)

    return y, jac


def value_and_jac(f, sparse=False):
    """Return function that returns value and Jacobian of x.
    """
    return lambda x: value_and_jac_inner(f, x, sparse)


def newton_jax(func, init, jac=None, maxit=30, tol=1e-8, sparse=False, solver=None, func_returns_jac=False, inspect_jac=False, verbose=False, verbose_jac=False):
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
        Whether to calculate a sparse jacobian. If `true`, and jac is supplied, this should return a sparse matrix
    solver : callable, optional
        Provide a custom solver
    func_returns_jac : bool, optional
        Set to `True` if the function also returns the jacobian.
    inspect_jac : bool, optional
        If `True`, use grgrlib.plots.spy to visualize the jacobian
    verbose : bool, optional
        Whether to display messages
    verbose_jac : bool, optional
        Whether to supply additional information on the determinant of the jacobian (computationally more costly).

    Returns
    -------
    res: dict
        A dictionary of results similar to the output from scipy.optimize.root
    """

    st = time.time()
    verbose_jac |= inspect_jac
    verbose |= verbose_jac

    if jac is None and not func_returns_jac:
        if sparse:
            def jac(x): return ssp.csr_array(jax.jacfwd(func)(x))
        else:
            jac = jax.jacfwd(func)

    if solver is None:
        if sparse:
            solver = ssp.linalg.spsolve
        else:
            solver = jax.scipy.linalg.solve

    res = {}
    cnt = 0
    xi = jnp.array(init)

    while True:

        cnt += 1
        xold = xi.copy()

        if func_returns_jac:
            fval, jacval = func(xi)
        else:
            fval, jacval = func(xi), jac(xi)

        jac_is_nan = isnan(jacval.data) if isinstance(
            jacval, ssp._arrays.csr_array) else jnp.isnan(jacval)
        if jac_is_nan.any():
            res['success'] = False
            res['message'] = "The Jacobian contains `NaN`s."
            break

        eps_fval = jnp.abs(fval).max()
        if eps_fval < tol:
            res['success'] = True
            res['message'] = "The solution converged."
            break

        xi -= solver(jacval, fval)
        eps = jnp.abs(xi - xold).max()

        if verbose:
            ltime = time.time() - st
            info_str = f'    Iteration {cnt:3d} | max error {eps:.2e} | lapsed {ltime:3.4f}'
            if verbose_jac:
                jacval = jacval.toarray() if sparse else jacval
                info_str += f' | det {jnp.linalg.det(jacval):1.5g} | rank {jnp.linalg.matrix_rank(jacval)}/{jacval.shape[0]}'
                if inspect_jac:
                    spy(jacval)

            print(info_str)

        if cnt == maxit:
            res['success'] = False
            res['message'] = f"Maximum number of {maxit} iterations reached."
            break

        if eps < tol:
            res['success'] = True
            res['message'] = "The solution converged."
            break

        if jnp.isnan(eps):
            res['success'] = False
            res['message'] = f"Function returns 'NaN's"
            break

    jacval = jacval.toarray() if isinstance(
        jacval, ssp._arrays.csr_array) else jacval

    res['x'], res['niter'] = xi, cnt
    res['fun'] = func(xi)[0] if func_returns_jac else func(xi)
    res['jac'] = jacval
    res['det'] = jnp.linalg.det(jacval)

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

    xi = jnp.array(init)

    def cond_func(tain):
        xi, xold, cnt = tain
        eps = jnp.abs(xi - xold).max()

        cond = cnt < maxit
        cond &= eps > tol
        cond &= ~jnp.isnan(eps)

        return cond

    def body_func(tain):
        (xi, _, cnt) = tain
        cnt += 1
        xold = xi
        xi -= jax.scipy.linalg.solve(jac(xi), func(xi))
        return (xi, xold, cnt)

    tain = jax.lax.while_loop(cond_func, body_func, (xi, xi + 1, 0))
    eps = jnp.abs(tain[0] - tain[1]).max()

    return tain[0], func(tain[0]), tain[2], eps < tol


newton_jax_jit = jax.jit(newton_jax_jittable, static_argnums=(0, 2, 3, 4))
