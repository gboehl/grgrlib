# -*- coding: utf-8 -*-

import numpy as np
from numba import njit, vectorize
from math import erfc, pi


SQRT2 = np.sqrt(2.0)
SQRT2pi = np.sqrt(2.0 * pi)


@njit(nogil=True, cache=True)
def subt(A, B):
    res = A
    for i in range(len(A)):
        for j in range(len(A)):
            res[i][j] = A[i][j] - B[i][j]
    return res


@njit(nogil=True, cache=True)
def cholesky(A):
    """Performs a Cholesky decomposition of on symmetric, pos-def A. Returns lower-triangular L (full sized, zeroed above diag)"""
    n = A.shape[0]
    L = np.zeros_like(A)

    # Perform the Cholesky decomposition
    for row in range(n):
        for col in range(row + 1):
            tmp_sum = np.dot(L[row, :col], L[col, :col])
            if row == col:  # Diagonal elements
                L[row, col] = np.sqrt(max(A[row, row] - tmp_sum, 0))
            elif np.abs(L[col, col]) < 1e-5:
                L[row, col] = 0
            else:
                L[row, col] = (1.0 / L[col, col]) * (A[row, col] - tmp_sum)
    return L


@njit(nogil=True, cache=True)
def numba_rand_norm(loc=0, scale=1, size=1):
    """A numba interface to create the user experience of np.random.normal with the size argument."""

    out = np.empty(size)

    for i in np.ndindex(size):
        out[i] = np.random.normal(loc, scale)

    return out


@vectorize(nopython=True, cache=True)
def normal_cdf(x, mu, sig):
    return erfc((mu - x) / sig / SQRT2) / 2.0


@njit(nogil=True, cache=True)
def normal_pdf(x, mu, sig):
    return 1 / (sig * SQRT2pi) * np.exp(-0.5 * ((x - mu) / sig) ** 2)


@njit(nogil=True, cache=True)
def histogram_weights(a, bins, weights):

    sorting_index = np.argsort(a)
    sa = a[sorting_index]
    sw = weights[sorting_index]
    cw = np.hstack((np.zeros(1), sw.cumsum()))

    bin_index = np.hstack(
        (
            np.searchsorted(sa, bins[:-1], "left"),
            np.searchsorted(sa, bins[-1:], "right"),
        )
    )

    cum_n = cw[bin_index]

    n = np.diff(cum_n)

    return n, bins


@njit(cache=True, nogil=True)
def shredder_basic(M, tol, verbose):

    m, n = M.shape
    Q = np.eye(m)

    j = 0

    for i in range(min(m, n)):

        while j < n:

            a = np.ascontiguousarray(Q.T[i:]) @ np.ascontiguousarray(M[:, j])
            do = False
            for ia in a:
                if abs(ia) > tol:
                    do = True
                    break

            if do:
                # apply Householder transformation
                v, tau = householder_reflection(a)

                H = np.identity(m - i)
                H -= tau * np.outer(v, v)

                Q[:, i:] = np.ascontiguousarray(Q[:, i:]) @ H

                if verbose:
                    print("...shredding row", j, "and col", i)
                break

            else:
                j += 1

    return Q


@njit(cache=True, nogil=True)
def shredder_pivoting(M, tol, verbose):
    """The DS-decomposition with pivoting"""

    m, n = M.shape
    M = M.copy().astype(np.float64)
    Q = np.eye(m)
    P = np.arange(n)

    j = 0

    for i in range(min(m, n)):

        # pivoting loop
        k = i
        while True:
            if k == n:
                break
            if np.any(np.abs(M[i:, k]) > tol):
                if k != i:
                    # move columns
                    M[:, np.array([i, k])] = M[:, np.array([k, i])]
                    P[np.array([i, k])] = P[np.array([k, i])]
                break
            k += 1

        while j < n:
            if np.any(np.abs(M[i:, j]) > tol):

                # apply Householder transformation
                a = M[i:, j]
                v, tau = householder_reflection(a)

                H = np.identity(m - i)
                H -= tau * np.outer(v, v)
                M[i:] = H @ M[i:]
                Q[:, i:] = np.ascontiguousarray(Q[:, i:]) @ H

                if verbose:
                    print("...shredding row", j, "and col", i)
                break

            else:
                j += 1

    return Q, M, P


@njit(cache=True, nogil=True)
def shredder_non_pivoting(M, tol, verbose):
    """The DS-decomposition with anti-pivoting"""

    m, n = M.shape
    M = M.copy().astype(np.float64)
    Q = np.eye(m)
    P = np.arange(n)

    j = 0

    for i in range(min(m, n)):

        # anti-pivoting
        # for each i move those columns forward that do not need transformation

        for k in range(j, n):
            if np.all(np.abs(M[i:, k]) < tol):
                # move columns
                if k != j:
                    M[:, np.array([j, k])] = M[:, np.array([k, j])]
                    P[np.array([j, k])] = P[np.array([k, j])]

                if verbose:
                    print("...switching rows", j, "and", k)
                j += 1

        while j < n:
            if np.any(np.abs(M[i:, j]) > tol):

                # apply Householder transformation
                a = M[i:, j]
                v, tau = householder_reflection(a)

                H = np.identity(m - i)
                H -= tau * np.outer(v, v)
                M[i:] = H @ M[i:]
                Q[:, i:] = np.ascontiguousarray(Q[:, i:]) @ H

                if verbose:
                    print("...shredding row", j, "and col", i)
                break

            else:
                j += 1

    return Q, M, P


def shredder(M, pivoting=None, tol=1e-11, verbose=False):
    """The QS decomposition from "Efficient Solution of Models with Occasionally Binding Constraints" (Gregor Boehl)

    The QS decomposition uses Householder reflections to bring a system in the row-echolon form. This is a dispatcher for the sub-functions.
    """

    if pivoting is None:
        q = shredder_basic(M, tol, verbose)
        return q, q.T @ M
    elif pivoting:
        return shredder_pivoting(M, tol, verbose)
    elif not pivoting:
        return shredder_non_pivoting(M, tol, verbose)
    else:
        raise NotImplementedError


@njit(cache=True, nogil=True, fastmath=True)
def householder_reflection(x):
    alpha = x[0]
    s = np.linalg.norm(x[1:]) ** 2
    v = x.copy()

    if s == 0:
        tau = 0.0
    else:
        t = np.sqrt(alpha ** 2 + s)
        v[0] = alpha - t if alpha <= 0 else -s / (alpha + t)

        tau = 2 * v[0] ** 2 / (s + v[0] ** 2)
        v /= v[0]

    return v, tau


@njit(cache=True, nogil=True, fastmath=True)
def householder_reflection_right(x):
    alpha = x[-1]
    s = np.linalg.norm(x[:-1]) ** 2
    v = x.copy()

    if s == 0:
        tau = 0.0
    else:
        t = np.sqrt(alpha ** 2 + s)
        v[-1] = alpha - t if alpha <= 0 else -s / (alpha + t)

        tau = 2 * v[-1] ** 2 / (s + v[-1] ** 2)
        v /= v[-1]

    return v, tau


_cond = 1e-9


@njit(cache=True)
def psd_func(M):
    """Compute the symmetric eigendecomposition

    Note that eigh takes care of array conversion, chkfinite, and assertion that the matrix is square.
    """

    s, u = np.linalg.eigh(M)
    eps = _cond * np.max(np.abs(s))
    d = s[s > eps]

    s_pinv = [0 if abs(x) <= eps else 1 / x for x in s]
    s_pinv_np = np.array(s_pinv)
    U = np.multiply(u, np.sqrt(s_pinv_np))

    return U, np.sum(np.log(d)), len(d)


@njit(cache=True)
def mvn_logpdf(x, mean, cov):
    """log-PDF of multivariate normal"""

    _LOG_2PI = np.log(2 * np.pi)

    dim = mean.shape[0]
    prec_U, log_det_cov, rank = psd_func(cov)
    dev = x - mean
    maha = np.sum(np.square(np.dot(dev, prec_U)), axis=-1)
    out = -0.5 * (rank * _LOG_2PI + log_det_cov + maha)

    return out
