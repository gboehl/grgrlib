#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from numba import njit, prange
from grgrlib.numerics import chebychev

"""EXPERIMENTAL!!!"""


@njit(cache=True)
def grgrdist_matrix(cdf, order):

    # use to construct OLS matrices
    X = np.empty((cdf.shape[1], order[0]))
    Y = np.empty((cdf.shape[1], order[1]+order[2]))

    for k in prange(0, order[0]):

        if k % 2:
            X[:, k] = chebychev((k-1)/2, 2*cdf[0]-1)*np.log(cdf[0]/(1-cdf[0]))
        else:
            X[:, k] = chebychev(k/2, 2*cdf[0]-1)

    for k in prange(0, order[1]):

        if k % 2:
            Y[:, k] = chebychev((k-1)/2, 2*cdf[1]-1)*np.log(cdf[1]/(1-cdf[1]))
        else:
            Y[:, k] = chebychev(k/2, 2*cdf[1]-1)

    for k in prange(0, order[2]):

        if k % 2:
            Y[:, order[1]+k] = chebychev((k+1)/2, 2*cdf[0]-1)
        else:
            Y[:, order[1]+k] = chebychev(k/2, 2 *
                                         cdf[0]-1)*np.log(cdf[0]/(1-cdf[0]))

    return X, Y
