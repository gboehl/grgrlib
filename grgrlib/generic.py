#!/bin/python
# -*- coding: utf-8 -*-

import time
import os
import numpy as np
import numpy.linalg as nl
import scipy.linalg as sl
import scipy.stats as ss
from .linalg import ouc


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


def map2arr(iterator, return_np_array=True, check_nones=True):
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

        if check_nones and obj is None:
            continue

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


def napper(cond, interval=0.1):

    import time

    start_time = time.time()

    while not cond():

        elt = round(time.time() - start_time, 3)
        print("Zzzz... " + str(elt) + "s", end="\r", flush=True)
        time.sleep(interval)

    print("Zzzz... " + str(elt) + "s.")


def timeprint(s, round_to=5, full=False):

    if s < 60:
        if full:
            return str(np.round(s, round_to)) + " seconds"
        return str(np.round(s, round_to)) + "s"

    m, s = divmod(s, 60)

    if m < 60:
        if full:
            return "%s minutes, %s seconds" % (int(m), int(s))
        return "%sm%ss" % (int(m), int(s))

    h, m = divmod(m, 60)

    if full:
        return "%s hours, %s minutes, %s seconds" % (int(h), int(m), int(s))
    return "%sh%sm%ss" % (int(h), int(m), int(s))


def shuffle(a, axis=-1):
    """Shuffle along single axis"""

    shape = a.shape
    res = a.reshape(-1, a.shape[axis])
    np.random.shuffle(res)

    return res.reshape(shape)


def print_dict(d):

    for k in d.keys():
        print(str(k) + ":", d[k])

    return 0


def sabs(x, eps=1e-10):
    """absolute value but smooth around 0"""
    return np.sqrt(x ** 2 + eps)


def parse_yaml(mfile):
    """parse from yaml file"""
    import yaml

    f = open(mfile)
    mtxt = f.read()
    f.close()

    # get dict
    return yaml.safe_load(mtxt)


def load_as_module(path):

    import importlib.machinery
    import importlib.util

    modname = os.path.splitext(os.path.basename(path))[0]
    loader = importlib.machinery.SourceFileLoader(modname, path)
    spec = importlib.util.spec_from_loader(modname, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)

    return module


# aliases
map2list = map2arr
indof = np.searchsorted
