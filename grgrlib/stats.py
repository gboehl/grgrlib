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


def fast_kde(x, bw=4.5):
    """
    A fft-based Gaussian kernel density estimate (KDE)
    The code was adapted from https://github.com/mfouesneau/faststats

    Parameters
    ----------
    x : Numpy array or list
    bw : float
        Bandwidth scaling factor for the KDE. Should be larger than 0. The higher this number the
        smoother the KDE will be. Defaults to 4.5 which is essentially the same as the Scott's rule
        of thumb (the default rule used by SciPy).

    Returns
    -------
    density: A gridded 1D KDE of the input points (x)
    xmin: minimum value of x
    xmax: maximum value of x
    """

    from scipy.signal import gaussian, convolve
    from scipy.stats import entropy

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    nx = 200

    xmin, xmax = np.min(x), np.max(x)

    dx = (xmax - xmin) / (nx - 1)
    std_x = entropy((x - xmin) / dx) * bw
    if ~np.isfinite(std_x):
        std_x = 0.
    grid, _ = np.histogram(x, bins=nx)

    scotts_factor = n ** (-0.2)
    kern_nx = int(scotts_factor * 2 * np.pi * std_x)
    kernel = gaussian(kern_nx, scotts_factor * std_x)

    npad = min(nx, 2 * kern_nx)
    grid = np.concatenate([grid[npad: 0: -1], grid, grid[nx: nx - npad: -1]])
    density = convolve(grid, kernel, mode='same')[npad: npad + nx]

    norm_factor = n * dx * (2 * np.pi * std_x ** 2 * scotts_factor ** 2) ** 0.5

    density = density / norm_factor

    return density, xmin, xmax
