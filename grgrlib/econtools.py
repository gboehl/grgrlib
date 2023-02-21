# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as nl
import scipy.linalg as sl
import time
from .linalg import ouc, fast0


def klein(A, B=None, nstates=None, verbose=False, force=False):
    """
    Klein's method
    """

    st = time.time()
    if B is None:
        B = np.eye(A.shape[0])

    SS, TT, alp, bet, Q, Z = sl.ordqz(A, B, sort="ouc")

    if np.any(np.isclose(alp, bet)):
        mess = " Warning: unit root detected!"
    else:
        mess = ""

    # check for precision
    if not fast0(Q @ SS @ Z.T - A, 2):
        raise ValueError("Numerical errors in QZ")

    if verbose > 1:
        out = np.empty_like(alp)
        nonzero = bet != 0
        out[~nonzero] = np.inf * np.abs(alp[~nonzero])
        out[nonzero] = alp[nonzero] / bet[nonzero]

        print(
            "[RE solver:]".ljust(15, " ") +
            " Generalized EVs:\n", np.sort(np.abs(out))
        )

    # check for Blanchard-Kahn
    out = ouc(alp, bet)

    if not nstates:
        nstates = sum(out)
    else:
        if not nstates == sum(out):
            mess = (
                "B-K condition not satisfied: %s states but %s Evs inside the unit circle."
                % (nstates, sum(out))
                + mess
            )

            if not force:
                raise ValueError(mess)
            elif verbose:
                print(mess)

    S11 = SS[:nstates, :nstates]
    T11 = TT[:nstates, :nstates]

    Z11 = Z[:nstates, :nstates]
    Z21 = Z[nstates:, :nstates]

    # changed from sl to nl because of stability:
    omg = Z21 @ nl.inv(Z11)
    lam = Z11 @ nl.inv(S11) @ T11 @ nl.inv(Z11)

    if verbose:
        print(
            "[RE solver:]".ljust(15, " ")
            + " Done in %s. Determinant of `Z11` is %1.2e. There are %s EVs o.u.c. (of %s)."
            % (np.round((time.time() - st), 5), nl.det(Z11), sum(out), len(out))
            + mess
        )

    return omg, lam


def lti(AA, BB, CC, dimp, dimq, tol=1e-6, check=False, verbose=False):
    """standard linear time iteration"""

    if check:
        pass

    g = np.eye(dimq + dimp)

    norm = tol + 1

    icnt = 0
    while norm > tol:
        gn = g
        g = -nl.solve(BB + AA @ g, CC)
        norm = np.max(np.abs(gn - g))
        icnt += 1

    if verbose:
        print(icnt)

    omg = g[dimq:, :dimq]
    lam = g[:dimq, :dimq]

    return omg, lam


def speed_kills(A, B, dimp, selector=None, tol=1e-6, check=False, max_iter=1000, verbose=False):
    """Improved linear time iteration"""

    dimq = A.shape[0] - dimp
    q, A = nl.qr(A)
    B = q.T @ B

    B11i = nl.inv(B[dimq:, dimq:])

    A[dimq:] = B11i @ A[dimq:]
    B[dimq:] = B11i @ B[dimq:]

    A[:dimq] -= B[:dimq, dimq:] @ A[dimq:]
    B[:dimq, :dimq] -= B[:dimq, dimq:] @ B[dimq:, :dimq]

    B[:dimq, dimq:] = 0
    B[dimq:, dimq:] = np.eye(dimp)

    A1 = A[:dimq, :dimq]
    A3 = A[dimq:, dimq:]
    A2 = A[:dimq, dimq:]
    B1 = B[:dimq, :dimq]
    B2 = B[dimq:, :dimq]

    g = -B2

    norm = tol + 1
    icnt = 0

    icnt = 0
    while norm > tol:
        gn = g
        g = A3 @ g @ nl.solve(A1 + A2 @ g, B1) - B2
        if selector is not None:
            norm = np.max(np.abs(gn - g)[selector])
        else:
            norm = np.max(np.abs(gn - g))
        icnt += 1

        if icnt == max_iter:
            raise Exception("(speed_kills:) iteration did not converge")

    if verbose:
        print(icnt)

    return g, -nl.inv(A[:dimq, :dimq] + A2 @ g) @ B1
