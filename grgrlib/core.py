#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as nl
import scipy.linalg as sl
import scipy.stats as ss
import time

aca = np.ascontiguousarray


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
        r = np.sum(np.cumsum(s)/np.sum(s) < threshold)
        r += 1  # Hence the strict inequality above
        if avoid_pathological:
            # If not avoid_pathological, then the last 4 diag. entries of
            # reconst( *tsvd(eye(400),0.99) )
            # will be zero. This is probably not intended.
            r += np.sum(np.isclose(s[r-1], s[r:]))
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
    return (VT.T * s**(-1.0)) @ U.T


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
    """
    Checks if pair of generalized EVs x,y is inside the unit circle. Here for legacy reasons
    """

    out = np.empty_like(x, dtype=bool)
    nonzero = (y != 0)

    # handles (x, y) = (0, 0) too
    out[~nonzero] = False
    out[nonzero] = (abs(x[nonzero]/y[nonzero]) < 1.0)

    return out


def ouc(x, y):
    """
    Check if pair of generalized EVs x,y is outside the unit circle. Here for legacy reasons
    """

    # stolen from scipy and inverted
    out = np.empty_like(x, dtype=bool)
    nonzero = (y != 0)
    # handles (x, y) = (0, 0) too
    out[~nonzero] = True
    out[nonzero] = abs(x[nonzero]/y[nonzero]) > 1.0

    return out

def klein(A, B=None, nstates=None, verbose=False, force=False):
    """
    Klein's method
    """

    st = time.time()
    if B is None:
        B = np.eye(A.shape[0])

    SS, TT, alp, bet, Q, Z = sl.ordqz(A, B, sort='ouc')


    if np.any(np.isclose(alp, bet)):
        mess0 = 'Warning: unit root detected! '
    else:
        mess0 = ''

    # check for precision
    if not fast0(Q @ SS @ Z.T - A, 2):
        raise ValueError('Numerical errors in QZ')

    if verbose > 1:
        print('[RE solver:]'.ljust(15, ' ') + ' Generalized EVs: ', np.sort(np.abs(alp/bet)))

    # check for Blanchard-Kahn
    out = ouc(alp, bet)

    if not nstates:
        nstates = sum(out)
    else:
        if not nstates == sum(out):
            mess1 = 'B-K condition not satisfied: %s states but %s Evs inside the unit circle. ' % (nstates, sum(out))
        else:
            mess1 = ''

        if mess1 and not force:
            raise ValueError(mess1+mess0)
        elif mess1 and verbose:
            print(mess1+mess0)

    S11 = SS[:nstates, :nstates]
    T11 = TT[:nstates, :nstates]

    Z11 = Z[:nstates, :nstates]
    Z21 = Z[nstates:, :nstates]

    omg = Z21 @ sl.inv(Z11)
    lam = Z11 @ sl.inv(S11) @ T11 @ sl.inv(Z11)

    if verbose:
        print('[RE solver:]'.ljust(
            15, ' ')+' Done in %s. Determinant of `Z11` is %1.2e. There are %s EVs o.u.c. ' % (np.round((time.time() - st), 5), nl.det(Z11), sum(out)) + mess0)

    return omg, lam


def re_bk(A, B=None, d_endo=None, verbose=False, force=False):
    """
    Klein's method
    """
    # TODO: rename this
    print('[RE solver:]'.ljust(15, ' ') + ' `re_bk` is depreciated. Use `klein` instead.')

    if B is None:
        B = np.eye(A.shape[0])

    MM, PP, alp, bet, Q, Z = sl.ordqz(A, B, sort='iuc')

    if not fast0(Q @ MM @ Z.T - A, 2):
        raise ValueError('Numerical errors in QZ')

    if verbose > 1:
        print('[RE solver:]'.ljust(15, ' ') +
              ' Pairs of `alp` and `bet`:\n', np.vstack((alp, bet)).T)

    out = ouc(alp, bet)

    if not d_endo:
        d_endo = sum(out)
    else:
        if sum(out) > d_endo:
            mess = 'B-K condition not satisfied: %s EVs outside the unit circle for %s forward looking variables.' % (
                sum(out), d_endo)
        elif sum(out) < d_endo:
            mess = 'B-K condition not satisfied: %s EVs outside the unit circle for %s forward looking variables.' % (
                sum(out), d_endo)
        else:
            mess = ''

        if mess and not force:
            raise ValueError(mess)
        elif mess and verbose:
            print(mess)

    Z21 = Z.T[-d_endo:, :d_endo]
    Z22 = Z.T[-d_endo:, d_endo:]

    if verbose:
        print('[RE solver:]'.ljust(
            15, ' ')+' Determinant of `Z21` is %1.2e. There are %s EVs o.u.c.' % (nl.det(Z21), sum(out)))

    return -nl.inv(Z21) @ Z22


def lti(AA, BB, CC, dimp, dimq, tol=1e-6, check=False, verbose=False):
    """standard linear time iteration
    """

    if check:
        pass

    g = np.eye(dimq+dimp)

    norm = tol + 1

    icnt = 0
    while norm > tol:
        gn = g
        g = -nl.solve(BB + AA @ g, CC)
        norm = np.max(np.abs(gn-g))
        icnt += 1

    if verbose:
        print(icnt)

    omg = g[dimq:,:dimq]
    lam = g[:dimq,:dimq]

    return omg, lam


def speed_kills(A, B, dimp, dimq, selector=None, tol=1e-6, check=False, verbose=False):
    """Improved linear time iteration
    """

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
            norm = np.max(np.abs(gn-g)[selector])
        else:
            norm = np.max(np.abs(gn-g))
        icnt += 1

    if verbose:
        print(icnt)

    if icnt == max_iter:
        raise Exception("iteration did not converge")

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
        print("Zzzz... "+str(elt)+"s", end='\r', flush=True)
        time.sleep(interval)

    print("Zzzz... "+str(elt)+"s.")


def find_ss(ss_func, par, init_par, init_guess=None, ndim=None, max_iter=500, tol=None, method=None, debug=False):
    """Finds steady states for parameters give a set of parameters where the steady state is known. This is useful if you don't have a nice initial guess, but know some working parameters.
    ...

    Parameters
    ----------
    ss_func : callable
        A vector function to find a root of.
    par : list or ndarray
        Paramters for which you want to solve for the steady state
    init_par : list or ndarray
        Parameters for which you know that the steady state can be found given the initial guess `init_guess`
    init_guess : list or ndarray (optional)
        Initial guess which leads to the solution of the root problem of `ss_func` with `init_par`. Defaults to a vector of ones.
    ndim : dimensionality of problem (optional, only if `init_guess` is not given)
    max_iter : int
    debug : bool

    Returns
    -------
    list
        The root / steady state

    Raises
    -------
    ValueError
        If the given problem cannot be solved for the initial parameters and guess
    """
    import scipy.optimize as so

    # convert to np.arrays to allow for math
    par = np.array(par)
    cur_par = np.array(init_par)
    last_par = cur_par

    if init_guess is None:
        # very stupid first guess
        sval = np.ones(ndim)
    else:
        sval = init_guess

    cnt = 0

    if method is None:
        method = 'hybr'

    if debug:
        res = so.root(lambda x: ss_func(x, list(cur_par)),
                      sval, tol=tol, method=method)
        return res

    while last_par is not par:

        try:
            res = so.root(lambda x: ss_func(x, list(cur_par)),
                          sval, tol=tol, method=method)
            suc = res['success']
        except:
            # if this is not even evaluable set success to False manually
            suc = False

        if not suc:

            if cnt == 0:
                raise ValueError(
                    "Can not find steady state of initial parameters.")
            # if unsuccessful, chose parameters closer to last working parameters
            cur_par = .5*last_par + .5*cur_par

        else:
            # if successful, update last working parameter and try final paramter
            last_par = cur_par
            cur_par = par
            sval = res['x']

        cnt += 1
        if cnt >= max_iter:
            print("Steady state could not be found after %s iterations. Message from last attempt: %s" % (
                max_iter, res['message']))
            break

    return res


def timeprint(s, round_to=5, full=False):

    if s < 60:
        if full:
            return str(np.round(s, round_to)) + ' seconds'
        return str(np.round(s, round_to)) + 's'

    m, s = divmod(s, 60)

    if m < 60:
        if full:
            return '%s minutes, %s seconds' % (int(m), int(s))
        return '%sm%ss' % (int(m), int(s))

    h, m = divmod(m, 60)

    if full:
        return '%s hours, %s minutes, %s seconds' % (int(h), int(m), int(s))
    return '%sh%sm%ss' % (int(h), int(m), int(s))


def shuffle(a, axis=-1):
    """Shuffle along single axis
    """

    shape = a.shape
    res = a.reshape(-1, a.shape[axis])
    np.random.shuffle(res)

    return res.reshape(shape)


def serializer(func):
    """Dirty hack that transforms the non-serializable function to a serializable one (when using dill)
    ...
    Don't try that at home!
    """

    fname = func.__name__
    exec('dark_%s = func' % fname, locals(), globals())

    def vodoo(*args, **kwargs):
        return eval('dark_%s(*args, **kwargs)' % fname)

    return vodoo


def sabs(x, eps=1e-10):
    """absolute value but smooth around 0
    """
    return np.sqrt(x**2 + eps)


# aliases
map2list = map2arr
indof = np.searchsorted
