#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from numba import njit

_cond    = 1e-9

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
