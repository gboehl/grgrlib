#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from numba import njit


@njit(nogil=True, cache=True)
def subt(A, B):
    res = A
    for i in range(len(A)):
        for j in range(len(A)):
            res[i][j] = A[i][j] - B[i][j]
    return res


@njit(nogil=True, cache=True)
def cholesky(A):
    """Performs a Cholesky decomposition of on symmetric, pos-def A. Returns lower-triangular L (full sized, zeroed above diag)
    """
    n = A.shape[0]
    L = np.zeros_like(A)

    # Perform the Cholesky decomposition
    for row in range(n):
        for col in range(row+1):
            tmp_sum = np.dot(L[row, :col], L[col, :col])
            if (row == col):  # Diagonal elements
                L[row, col] = np.sqrt(max(A[row, row] - tmp_sum, 0))
            elif np.abs(L[col, col]) < 1e-5:
                L[row, col] = 0
            else:
                L[row, col] = (1.0 / L[col, col]) * (A[row, col] - tmp_sum)
    return L


@njit(nogil=True, cache=True)
def numba_rand_norm(loc=0, scale=1, size=1):
    """A numba interface to create the user experience of np.random.normal with the size argument.
    """

    out = np.empty(size)

    for i in np.ndindex(size):
        out[i] = np.random.normal(loc, scale)

    return out
