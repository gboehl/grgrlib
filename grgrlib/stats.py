#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from numba import njit

_cond = 1e-9


@njit(cache=True)
def psd_func(M):
    """Compute the symmetric eigendecomposition

    Note that eigh takes care of array conversion, chkfinite, and assertion that the matrix is square.
    """

    s, u = np.linalg.eigh(M)
    eps = _cond * np.max(np.abs(s))
    d = s[s > eps]

    s_pinv = [0 if abs(x) <= eps else 1/x for x in s]
    s_pinv_np = np.array(s_pinv)
    U = np.multiply(u, np.sqrt(s_pinv_np))

    return U, np.sum(np.log(d)), len(d)


@njit(cache=True)
def logpdf(x, mean, cov):
    """log-PDF of multivariate normal
    """

    _LOG_2PI = np.log(2 * np.pi)

    dim = mean.shape[0]
    prec_U, log_det_cov, rank = psd_func(cov)
    dev = x - mean
    maha = np.sum(np.square(np.dot(dev, prec_U)), axis=-1)
    out = -0.5 * (rank * _LOG_2PI + log_det_cov + maha)

    return out


def percentile(x, q=.01):
    """Find share owned by q richest individuals"""

    xsort = np.sort(x)
    n = len(x)
    return np.sum(xsort[-int(np.ceil(n*q)):])/np.sum(x)


def mode(x):
    """Find mode of (unimodal) univariate distribution"""

    p, lb, ub = fast_kde(x)
    xs = np.linspace(lb, ub, p.shape[0])
    return xs[p.argmax()]


def gini(x):
    """Calculate the Gini coefficient of a numpy array

    Stolen from https://github.com/oliviaguest/gini
    """

    # All values are treated equally, arrays must be 1d:
    x = x.flatten()
    if np.amin(x) < 0:
        # Values cannot be negative:
        x -= np.amin(x)
    # Values cannot be 0:
    x += 1e-10
    # Values must be sorted:
    x = np.sort(x)
    # Index per array element:
    index = np.arange(1, x.shape[0]+1)
    # Number of array elements:
    n = x.shape[0]

    # Gini coefficient:
    return ((np.sum((2 * index - n - 1) * x)) / (n * np.sum(x)))
