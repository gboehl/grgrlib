#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from numba import njit

from interpolation.splines import UCGrid, nodes, eval_linear
from interpolation.splines import extrap_options as xto


def simulate_noeps(t_func, T, transition_phase, initial_state):

    x = initial_state

    for t in range(transition_phase):
        x = t_func(x)

    res = np.empty((T,)+x.shape)
    for t in range(T):
        x = t_func(x)
        res[t] = x

    return res


def simulate(t_func, T=None, transition_phase=0, initial_state=None, eps=None, numba_jit=True, show_warnings=True):
    """Generic simulation command

    Hopefully one day merged with pydsge.stuff.simulate
    """

    if T is None:
        if eps is not None:
            T = eps.shape[0]
        else:
            UnboundLocalError("Either `T` or `eps` must be given.")

    if initial_state is None:
        if ndim is not None:
            initial_state = np.zeros(ndim)
        else:
            UnboundLocalError(
                "Either `initial_state` or `ndim` must be given.")

    if numba_jit:
        res = simulate_noeps_jit(t_func, T, transition_phase, initial_state)
    else:
        res = simulate_noeps(t_func, T, transition_phase, initial_state)

    return res


def pfi_t_func(pfunc, grid, numba_jit=True):
    """Wrapper to return a jitted transition function based on the policy function and the grid
    """

    def pfi_t_func_wrap(state):

        newstate = eval_linear(grid, pfunc, state, xto.LINEAR)
        # newstate = eval_linear(grid, pfunc, state)

        return newstate

    if numba_jit:
        return njit(pfi_t_func_wrap, nogil=True)
    else:
        return pfi_t_func_wrap


def pfi_determinisic(func, xfromv, pars, args, grid_shape, grid, gp, eps_max):

    ndim = len(grid)

    eps = 1e9

    values = func(pars, gp, 0., args=args)[0]

    while eps > eps_max:

        values_old = values.copy()
        svalues = values.reshape(grid_shape)
        xe = xfromv(eval_linear(grid, svalues, values, xto.LINEAR))
        # xe = xfromv(eval_linear(grid, svalues, values))
        values = func(pars, gp, xe, args=args)[0]
        eps = np.linalg.norm(values - values_old)

    return values.reshape(grid_shape)


def pfi(grid, model=None, func=None, pars=None, xfromv=None, system_type=None, eps_max=1e-8, numba_jit=True):
    """Somewhat generic policy function iteration

    For now only deterministic solutions are supported. This assumes a form of

        v_t = f(E_t x_{t+1}, v_{t-1})

    where f(.) is `func` and x_t = h(v_t) (where h(.) is `xfromv` can be directly retrieved from v_t. The returning function `p_func` is then the array repesentation of the solution v_t = g(v_{t-1}).

    In the future this should also allow for 

        y_t = f(E_t x_{t+1}, w_t, v_{t-1})

    with implied existing functions x_t = h_1(y_t), v_t = h_2(y_t) and w_t = h_3(y_t).

    Parameters
    ----------
    func : dynamic system as described above. Takes the argument `pars', `state`,  and `expect` (must be jitted)
    xfrom : see above (must be jitted)
    pars: list of parameters to func
    grid: for now only UCGrid from interpolate.py are supported and tested
    system_type: str (eiter 'deterministic' or 'stochastic'
    eps_max: float of maximum error tolerance

    Returns
    -------
    numpy array 
    """

    if system_type is None:
        system_type = 'deterministic'

    if numba_jit:
        pfi_determinisic_func = pfi_determinisic_jit
    else:
        pfi_determinisic_func = pfi_determinisic

    flag = 0

    if model is not None:
        func = model.func
        xfromv = model.xfromv
        pars = np.ascontiguousarray(model.pars)
        args = np.ascontiguousarray(model.args)
    else:
        if func is None or pars is None or xfromv is None:
            SyntaxError(
                "If no model object is given, 'func', 'pars' and 'xfromv' must be provided.")
        pars = np.ascontiguousarray(pars)
        args = np.ascontiguousarray(args)

    gp = nodes(grid)
    grid_shape = tuple(g_spec[2] for g_spec in grid)
    grid_shape = grid_shape + (len(grid),)

    if system_type == 'deterministic':

        p_func = pfi_determinisic_func(
            func, xfromv, pars, args, grid_shape, grid, gp, eps_max)
        if np.isnan(p_func).any():
            flag = 1

    else:
        NotImplementedError('Only deterministic systems supported by now...')

    if flag:
        print('Error in pfi. Error no.:', flag)

    return p_func


simulate_noeps_jit = njit(simulate_noeps, nogil=True, fastmath=True)
pfi_determinisic_jit = njit(pfi_determinisic, nogil=True, fastmath=True)
