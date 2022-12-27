#!/bin/python
# -*- coding: utf-8 -*-

import jax
import time
import functools
import jax.numpy as jnp
import scipy.sparse as ssp
from numpy import ndarray, isnan
from numpy.linalg import det
from .plots import spy
from jax.experimental.host_callback import id_print as jax_print
from jax._src.api import (_check_callable, _check_input_dtype_jacfwd, _check_input_dtype_jacrev, _check_output_dtype_jacfwd, _check_output_dtype_jacrev, _ensure_index,
                          _jvp, _vjp, _std_basis, _jacfwd_unravel, _jacrev_unravel, lu, argnums_partial, tree_map, tree_structure, tree_transpose, partial, Callable, Sequence, Union, vmap)


def jvp_vmap(fun: Callable, argnums: Union[int, Sequence[int]] = 0):
    """Vectorized (forward-mode) jacobian-vector product of ``fun``. This is by large adopted from the implementation of jacfwd in jax._src.api.

    Args:
      fun: Function whose value and Jacobian is to be computed.
      argnums: Optional, integer or sequence of integers. Specifies which
        positional argument(s) to differentiate with respect to (default ``0``).

    Returns:
      A function with the same arguments as ``fun``, that evaluates the value and Jacobian of
      ``fun`` using forward-mode automatic differentiation.
    """
    _check_callable(fun)
    argnums = _ensure_index(argnums)

    def jvpfun(args, tangents, **kwargs):

        f = lu.wrap_init(fun, kwargs)
        f_partial, dyn_args = argnums_partial(
            f, argnums, args, require_static_args_hashable=False)
        pushfwd = partial(_jvp, f_partial, dyn_args)
        y, jac = vmap(pushfwd, out_axes=(None, -1), in_axes=-1)(tangents)

        return y, jac

    return jvpfun


def vjp_vmap(fun: Callable, argnums: Union[int, Sequence[int]] = 0):
    """Vectorized (reverse-mode) vector-jacobian product of ``fun``. This is by large adopted from the implementation of jacrev in jax._src.api.

    Args:
      fun: Function whose value and Jacobian are to be computed.
      argnums: Optional, integer or sequence of integers. Specifies which
        positional argument(s) to differentiate with respect to (default ``0``).

    Returns:
      A function with the same arguments as ``fun``, that evaluates the value and Jacobian of
      ``fun`` using reverse-mode automatic differentiation.
    """
    _check_callable(fun)

    def vjpfun(args, tangents, **kwargs):
        f = lu.wrap_init(fun, kwargs)
        f_partial, dyn_args = argnums_partial(f, argnums, args,
                                              require_static_args_hashable=False)
        y, pullback = _vjp(f_partial, *dyn_args)
        jac = vmap(pullback)(tangents)
        return y, jac

    return vjpfun


def jacfwd_and_val(fun: Callable, argnums: Union[int, Sequence[int]] = 0,
                   has_aux: bool = False, holomorphic: bool = False) -> Callable:
    """Value and Jacobian of ``fun`` evaluated column-by-column using forward-mode AD. Apart from returning the function value, this is one-to-one adopted from
  jax._src.api.

    Args:
      fun: Function whose value and Jacobian is to be computed.
      argnums: Optional, integer or sequence of integers. Specifies which
        positional argument(s) to differentiate with respect to (default ``0``).
      has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the
        first element is considered the output of the mathematical function to be
        differentiated and the second element is auxiliary data. Default False.
      holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
        holomorphic. Default False.

    Returns:
      A function with the same arguments as ``fun``, that evaluates the value and Jacobian of
      ``fun`` using forward-mode automatic differentiation. If ``has_aux`` is True
      then a tuple of (value, jacobian, auxiliary_data) is returned.
    """
    _check_callable(fun)
    argnums = _ensure_index(argnums)

    def jacfun(*args, **kwargs):
        f = lu.wrap_init(fun, kwargs)
        f_partial, dyn_args = argnums_partial(f, argnums, args,
                                              require_static_args_hashable=False)
        tree_map(partial(_check_input_dtype_jacfwd, holomorphic), dyn_args)
        if not has_aux:
            pushfwd = partial(_jvp, f_partial, dyn_args)
            y, jac = vmap(pushfwd, out_axes=(None, -1))(_std_basis(dyn_args))
        else:
            pushfwd = partial(_jvp, f_partial, dyn_args, has_aux=True)
            y, jac, aux = vmap(pushfwd, out_axes=(
                None, -1, None))(_std_basis(dyn_args))
        tree_map(partial(_check_output_dtype_jacfwd, holomorphic), y)
        example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
        jac_tree = tree_map(partial(_jacfwd_unravel, example_args), y, jac)
        if not has_aux:
            return y, jac_tree
        else:
            return y, jac_tree, aux

    return jacfun


def jacrev_and_val(fun: Callable, argnums: Union[int, Sequence[int]] = 0,
                   has_aux: bool = False, holomorphic: bool = False, allow_int: bool = False) -> Callable:
    """Value and Jacobian of ``fun`` evaluated row-by-row using reverse-mode AD.

    Args:
      fun: Function whose value and Jacobian are to be computed.
      argnums: Optional, integer or sequence of integers. Specifies which
        positional argument(s) to differentiate with respect to (default ``0``).
      has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the
        first element is considered the output of the mathematical function to be
        differentiated and the second element is auxiliary data. Default False.
      holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
        holomorphic. Default False.
      allow_int: Optional, bool. Whether to allow differentiating with
        respect to integer valued inputs. The gradient of an integer input will
        have a trivial vector-space dtype (float0). Default False.

    Returns:
      A function with the same arguments as ``fun``, that evaluates the value and Jacobian of
      ``fun`` using reverse-mode automatic differentiation. If ``has_aux`` is True
      then a pair of (jacobian, auxiliary_data) is returned.
    """
    _check_callable(fun)

    def jacfun(*args, **kwargs):
        f = lu.wrap_init(fun, kwargs)
        f_partial, dyn_args = argnums_partial(f, argnums, args,
                                              require_static_args_hashable=False)
        tree_map(partial(_check_input_dtype_jacrev,
                 holomorphic, allow_int), dyn_args)
        if not has_aux:
            y, pullback = _vjp(f_partial, *dyn_args)
        else:
            y, pullback, aux = _vjp(f_partial, *dyn_args, has_aux=True)
        tree_map(partial(_check_output_dtype_jacrev, holomorphic), y)
        jac = vmap(pullback)(_std_basis(y))
        jac = jac[0] if isinstance(argnums, int) else jac
        example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
        jac_tree = tree_map(partial(_jacrev_unravel, y), example_args, jac)
        jac_tree = tree_transpose(tree_structure(
            example_args), tree_structure(y), jac_tree)
        if not has_aux:
            return y, jac_tree
        else:
            return y, jac_tree, aux

    return jacfun


def newton_jax(func, init, jac=None, maxit=30, tol=1e-8, rtol=None, sparse=False, solver=None, func_returns_jac=False, inspect_jac=False, verbose=False, verbose_jac=False):
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
    rtol = rtol or tol

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

        xold = xi.copy()
        jacold = jacval.copy() if cnt else None
        cnt += 1

        if func_returns_jac:
            fval, jacval = func(xi)
            if sparse and not isinstance(jacval, ssp._arrays.csr_array):
                jacval = ssp.csr_array(jacval)
        else:
            fval, jacval = func(xi), jac(xi)

        jac_is_nan = isnan(jacval.data) if isinstance(
            jacval, ssp._arrays.csr_array) else jnp.isnan(jacval)
        if jac_is_nan.any():
            res['success'] = False
            res['message'] = "The Jacobian contains `NaN`s."
            jacval = jacold if jacold is not None else jacval
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
                jacdet = jnp.linalg.det(jacval) if (
                    jacval.shape[0] == jacval.shape[1]) else 0
                info_str += f' | det {jacdet:1.5g} | rank {jnp.linalg.matrix_rank(jacval)}/{jacval.shape[0]}'
                if inspect_jac:
                    spy(jacval)

            print(info_str)

        if cnt == maxit:
            res['success'] = False
            res['message'] = f"Maximum number of {maxit} iterations reached."
            break

        if eps < rtol:
            res['success'] = True
            res['message'] = "The solution converged."
            break

        if jnp.isnan(eps):
            res['success'] = False
            res['message'] = f"Function returns 'NaN's"
            break

    jacval = jacval.toarray() if isinstance(
        jacval, (ssp._arrays.csr_array, ssp._arrays.lil_array)) else jacval

    res['x'], res['niter'] = xi, cnt
    res['fun'], res['jac'] = fval, jacval

    if verbose_jac:
        # only calculate determinant if requested
        res['det'] = det(jacval) if (jacval.shape[0] == jacval.shape[1]) else 0
    else:
        res['det'] = None

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

    return tain[0], func(tain[0]), tain[2] < maxit, eps < tol


amax = jax.jit(lambda x: jnp.abs(x).max())
newton_jax_jit = jax.jit(newton_jax_jittable, static_argnums=(0, 2, 3, 4))
