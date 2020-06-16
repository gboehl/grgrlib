#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from numba import njit
from .njitted import numba_rand_norm
from .core import model

bh_par_names = ['discount_factor', 'intensity_of_choice',
                'bias', 'degree_trend_extrapolation', 'costs']
bh_pars = np.array([1/.99, 1., 1., 0., 0.])
bh_arg_names = ['rational', 'noise']
bh_args = np.array([0, 0])


@njit(nogil=True, cache=True)
def bh_func(pars, state, expect, args):

    rational, noise = args
    dis, dlt, bet, gam, cos = pars

    xe = expect

    xm1 = np.ascontiguousarray(state[:, 0])
    xm2 = np.ascontiguousarray(state[:, 1])

    if state.shape[1] < 3:
        xm3 = np.zeros_like(xm1)
    else:
        xm3 = np.ascontiguousarray(state[:, 2])

    x_shp = np.shape(xm1)

    if rational:
        prof0 = (xm1 - dis*xm2)**2 - cos
    else:
        prof0 = -(xm1 - dis*xm2)*xm2 - cos

    prof1 = (xm1 - dis*xm2) * (gam*xm3 + bet - dis*xm2)

    if bet == 0:
        prof2 = np.zeros_like(prof1)
    else:
        prof2 = (xm1 - dis*xm2) * (gam*xm3 - bet - dis*xm2)

    # exprof0 = np.exp(dlt*prof0)
    # exprof1 = np.exp(dlt*prof1)
    # exprof2 = np.exp(dlt*prof2)
    # psum = exprof0 + exprof1 + exprof2

    # frac0 = exprof0/psum
    # frac1 = exprof1/psum
    # frac2 = exprof2/psum

    frac0 = 1/(1 + np.exp(dlt*(prof1-prof0)) + bool(bet) * np.exp(dlt*(prof2-prof0)))
    frac1 = 1/(1 + np.exp(dlt*(prof0-prof1)) + bool(bet) * np.exp(dlt*(prof2-prof1)))
    frac2 = bool(bet) / (1 + np.exp(dlt*(prof0-prof2)) + np.exp(dlt*(prof1-prof2)))

    x = (frac0*xe + (frac1+frac2)*gam*xm1 + (frac1-frac2)*bet)/dis

    if noise:
        x = x + numba_rand_norm(scale=1e-16, size=np.shape(x))

    if state.shape[1] < 3:
        res = np.concatenate((
            x.reshape(x_shp+(1,)),
            xm1.reshape(x_shp+(1,))),
            axis=1)
    else:
        res = np.concatenate((
            x.reshape(x_shp+(1,)),
            xm1.reshape(x_shp+(1,)),
            xm2.reshape(x_shp+(1,))),
            axis=1)

    return res, frac0, frac1, frac2


@njit(nogil=True, cache=True)
def bh_xfromv(v):
    return v[:, 0]


bh1998 = model(bh_func, bh_par_names, bh_pars,
               bh_arg_names, bh_args, bh_xfromv)
