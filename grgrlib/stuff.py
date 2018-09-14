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

    q, r, p     = sl.qr(A, mode='economic', pivoting=True)

    res                     = np.zeros(A.shape[1], dtype=bool)
    res[p[:A.shape[0]]]     = True

    return res


@njit(cache=True)
def subt(A, B):
	res 	= A
	for i in range(len(A)):
		for j in range(len(A)):
			res[i][j] = A[i][j] - B[i][j]
	return res


def nul(n):
    return np.zeros((n,n))


def iuc(x, y):
    out = np.empty_like(x, dtype=bool)
    nonzero = (y != 0)
    # handles (x, y) = (0, 0) too
    out[~nonzero] = False
    ## rounding is necessary to avoid false round-offs
    out[nonzero] = (abs(x[nonzero]/y[nonzero]).round(3) < 1.0)
    return out

def re_bc(N, d_endo):

    n   = N.shape[0]

    MM, PP, alp, bet, Q, Z    = sl.ordqz(N,np.eye(n),sort=iuc)

    if not fast0(Q @ MM @ Z.T - N, 2):
        # warnings.warn('Numerical errors in QZ')
        raise ValueError('Numerical errors in QZ')

    Z21     = Z.T[-d_endo:,:d_endo]
    Z22     = Z.T[-d_endo:,d_endo:]

    return -nl.inv(Z21) @ Z22


def fast0(A, mode=None):

    if mode == None:
        return np.isclose(A, 0)
    elif mode == 0:
        return np.isclose(A, 0).all(axis=0)
    elif mode == 1:
        return np.isclose(A, 0).all(axis=1)
    else:
        return np.allclose(A, 0)


def nearestPSD(A):

    B   = (A + A.T)/2

    H   = sl.polar(B)[1]

    return (B + H)/2

