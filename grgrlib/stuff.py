#!/bin/python2
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as nl
import scipy.linalg as sl
import warnings
from numba import njit
import time


def eig(M):
    return np.sort(np.abs(nl.eig(M)[0]))[::-1]


def invertible_subm(A):
    """
    For an (m times n) matrix A with n > m this function finds the m columns that are necessary to construct a nonsingular submatrix of A.
    """

    q, r, p = sl.qr(A, mode='economic', pivoting=True)

    res = np.zeros(A.shape[1], dtype=bool)
    res[p[:A.shape[0]]] = True

    return res


def nul(n):
    return np.zeros((n, n))


def iuc(x, y):
    out = np.empty_like(x, dtype=bool)
    nonzero = (y != 0)
    # handles (x, y) = (0, 0) too
    out[~nonzero] = False
    # rounding is necessary to avoid false round-offs
    out[nonzero] = (abs(x[nonzero]/y[nonzero]).round(3) < 1.0)
    return out


def re_bc(N, d_endo):

    n = N.shape[0]

    MM, PP, alp, bet, Q, Z = sl.ordqz(N, np.eye(n), sort=iuc)

    if not fast0(Q @ MM @ Z.T - N, 2):
        raise ValueError('Numerical errors in QZ')

    Z21 = Z.T[-d_endo:, :d_endo]
    Z22 = Z.T[-d_endo:, d_endo:]

    return -nl.inv(Z21) @ Z22


def fast0(A, mode=-1):

    if mode == -1:
        return np.isclose(A, 0)
    elif mode == 0:
        return np.isclose(A, 0).all(axis=0)
    elif mode == 1:
        return np.isclose(A, 0).all(axis=1)
    else:
        return np.allclose(A, 0)


def nearestPSD(A):

    B = (A + A.T)/2
    H = sl.polar(B)[1]

    return (B + H)/2


def quarterlyzator(ts):
    """Takes a series of years where quarters are expressed as decimal numbers and returns strings of the form "'YYQQ"
    """
    qts = []
    for date in ts:

        rest = date - int(date)
        if rest == .25:
            qstr = 'Q2'
        elif rest == .5:
            qstr = 'Q3'
        elif rest == .75:
            qstr = 'Q4'
        else:
            qstr = 'Q1'
        qts.append("'"+str(int(date))[-2:]+qstr)
    return qts


def map2list(iterator, return_np_array=True):
    """Function to cast result from `map` to a tuple of stacked results

    By default, this returns numpy arrays. Automatically checks if the map object is a tuple, and if not, just one object is returned (instead of a tuple). Be warned, this does not work if the result of interest of the mapped function is a single tuple.

    Parameters
    ----------
    iterator : iter
        the iterator returning from `map`

    Returns
    -------
    numpy array (optional: list)
    """

    res = ()
    mode = 0

    for obj in iterator:

        if not mode:
            if isinstance(obj, tuple):
                for entry in obj:
                    res = res + ([entry],)
                mode = 1
            else:
                res = [obj]
                mode = 2

        else:
            if mode == 1:
                for no, entry in enumerate(obj):
                    res[no].append(entry)
            else:
                res.append(obj)

    if return_np_array:
        if mode == 1:
            res = tuple(np.array(tupo) for tupo in res)
        else:
            res = np.array(res)

    return res


@njit(cache=True)
def subt(A, B):
    res = A
    for i in range(len(A)):
        for j in range(len(A)):
            res[i][j] = A[i][j] - B[i][j]
    return res


@njit(cache=True)
def cholesky(A):
    """
       Performs a Cholesky decomposition of on symmetric, pos-def A.
       Returns lower-triangular L (full sized, zeroed above diag)
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


@njit(cache=True)
def numba_rand_norm(loc=0, scale=1, size=1):
    """A numba interface to create the user experience of np.random.normal with the size argument.
    """
    
    out = np.empty(size)

    for i in np.ndindex(size):
        out[i] = np.random.normal(loc, scale)

    return out
