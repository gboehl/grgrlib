# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as plt


def H(arr):
    """conjugate transpose"""
    return arr.T.conj()


def eig(M):
    return np.sort(np.abs(nl.eig(M)[0]))[::-1]


def truncate_rank(s, threshold, avoid_pathological):
    "Find r such that s[:r] contains the threshold proportion of s."
    assert isinstance(threshold, float)
    if threshold == 1.0:
        r = len(s)
    elif threshold < 1.0:
        r = np.sum(np.cumsum(s) / np.sum(s) < threshold)
        r += 1  # Hence the strict inequality above
        if avoid_pathological:
            # If not avoid_pathological, then the last 4 diag. entries of
            # reconst( *tsvd(eye(400),0.99) )
            # will be zero. This is probably not intended.
            r += np.sum(np.isclose(s[r - 1], s[r:]))
    else:
        raise ValueError
    return r


def is_int(a):
    return np.issubdtype(type(a), np.integer)


def tsvd(A, threshold=0.99999, avoid_pathological=True):
    """Truncated svd.

    Also automates 'full_matrices' flag.

    - threshold:

      - if float, < 1.0 then "rank" = lowest number
        such that the "energy" retained >= threshold
      - if int,  >= 1   then "rank" = threshold

    - avoid_pathological: avoid truncating (e.g.) the identity matrix.
      NB: only applies for float threshold.
    """

    M, N = A.shape
    full_matrices = False

    if is_int(threshold):
        # Assume specific number is requested
        r = threshold
        assert 1 <= r <= max(M, N)
        if r > min(M, N):
            full_matrices = True
            r = min(M, N)

    U, s, VT = sl.svd(A, full_matrices)

    if isinstance(threshold, float):
        # Assume proportion is requested
        r = truncate_rank(s, threshold, avoid_pathological)

    # Truncate
    U = U[:, :r]
    VT = VT[:r]
    s = s[:r]
    return U, s, VT


def tinv(A, *kargs, **kwargs):
    """
    Inverse based on truncated svd.
    Also see sl.pinv2().
    """
    U, s, VT = tsvd(A, *kargs, **kwargs)
    return (VT.T * s ** (-1.0)) @ U.T


def invertible_subm(A):
    """
    For an (m times n) matrix A with n > m this function finds the m columns that are necessary to construct a nonsingular submatrix of A.
    """

    q, r, p = sl.qr(A, mode="economic", pivoting=True)

    res = np.zeros(A.shape[1], dtype=bool)
    res[p[: A.shape[0]]] = True

    return res


def givens_rotation(a, b):
    """Compute matrix entries for Givens rotation."""
    r = np.hypot(a, b)
    c = a / r
    s = -b / r

    return (c, s)


def qr_gr(A):
    """Perform QR decomposition of matrix A using Givens rotation."""
    num_rows, num_cols = np.shape(A)

    # Initialize orthogonal matrix Q and upper triangular matrix R.
    Q = np.identity(num_rows)
    R = np.copy(A)

    # Iterate over lower triangular matrix.
    (rows, cols) = np.tril_indices(num_rows, -1, num_cols)
    for (row, col) in zip(rows, cols):

        # Compute Givens rotation matrix and
        # zero-out lower triangular matrix entries.
        if R[row, col] != 0:
            (c, s) = givens_rotation(R[col, col], R[row, col])

            G = np.identity(num_rows)
            G[[col, row], [col, row]] = c
            G[row, col] = s
            G[col, row] = -s

            R = np.dot(G, R)
            Q = np.dot(Q, G.T)

    return (Q, R)


def qr_hh(A):
    """Perform QR decomposition of matrix A using Householder reflections."""

    M = A.copy()
    m, n = M.shape
    Q = np.eye(m)

    for i in range(min(m, n)):

        a = M[i:, i]

        v, tau = householder_reflection(a)

        H = np.identity(m - i)
        H -= tau * np.outer(v, v)

        M[i:] = H @ M[i:]
        Q[:, i:] = Q[:, i:] @ H

    return Q, M


def rq_hh(A):
    """Perform RQ decomposition of matrix A using Householder reflections."""

    M = A.copy()
    m, n = M.shape
    Q = np.eye(n)

    for i in range(min(m, n)):

        a = M[-1 - i, : n - i]

        v, tau = householder_reflection_right(a)

        H = np.identity(n - i)
        H -= tau * np.outer(v, v)

        M[:, : n - i] = M[:, : n - i] @ H
        Q[: n - i, :] = H @ Q[: n - i, :]

    return M, Q


def ql(M):
    """Perform QL decomposition of matrix A using scipy"""
    r, q = sl.rq(M.T)
    return q.T, r.T


def nearest_psd(A, eps=0):

    B = (A + A.T) / 2
    H = sl.polar(B)[1]

    return (B + H) / 2 + eps


def nul(n):
    return np.zeros((n, n))


def iuc(x, y):
    """
    Checks if pair of generalized EVs x,y is inside the unit circle. Here for legacy reasons
    """

    out = np.empty_like(x, dtype=bool)
    nonzero = y != 0

    # handles (x, y) = (0, 0) too
    out[~nonzero] = False
    out[nonzero] = abs(x[nonzero] / y[nonzero]) < 1.0

    return out


def ouc(x, y):
    """
    Check if pair of generalized EVs x,y is outside the unit circle. Here for legacy reasons
    """

    # stolen from scipy and inverted
    out = np.empty_like(x, dtype=bool)
    nonzero = y != 0
    # handles (x, y) = (0, 0) too
    out[~nonzero] = True
    out[nonzero] = abs(x[nonzero] / y[nonzero]) > 1.0

    return out


def fast0(A, mode=-1, tol=1e-08):

    con = abs(A) < tol
    if mode == -1:
        return con
    elif mode == 0:
        return con.all(axis=0)
    elif mode == 1:
        return con.all(axis=1)
    else:
        return con.all()


def shuffle(a, axis=-1):
    """Shuffle along single axis"""

    shape = a.shape
    res = a.reshape(-1, a.shape[axis])
    np.random.shuffle(res)

    return res.reshape(shape)


def sabs(x, eps=1e-10):
    """absolute value but smooth around 0"""
    return np.sqrt(x ** 2 + eps)
