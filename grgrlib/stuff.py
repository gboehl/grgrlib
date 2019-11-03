#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as nl
import scipy.linalg as sl
import warnings
import time


class GPP:
    """Generic PYGMO problem. Assumes maximization.
    """

    name = 'GPP'

    def __init__(self, func, bounds):

        self.func = func
        self.bounds = bounds

    def fitness(self, x):
        return [-self.func(x)]

    def get_bounds(self):
        return self.bounds


def eig(M):
    return np.sort(np.abs(nl.eig(M)[0]))[::-1]


def truncate_rank(s,threshold,avoid_pathological):
  "Find r such that s[:r] contains the threshold proportion of s."
  assert isinstance(threshold,float)
  if threshold == 1.0:
    r = len(s)
  elif threshold < 1.0:
    r = np.sum(np.cumsum(s)/np.sum(s) < threshold)
    r += 1 # Hence the strict inequality above
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
  M,N = A.shape
  full_matrices = False

  if is_int(threshold):
    # Assume specific number is requested
    r = threshold
    assert 1 <= r <= max(M,N)
    if r > min(M,N):
      full_matrices = True
      r = min(M,N)

  U,s,VT = sl.svd(A, full_matrices)

  if isinstance(threshold,float):
    # Assume proportion is requested
    r = truncate_rank(s,threshold,avoid_pathological)

  # Truncate
  U  = U [:,:r]
  VT = VT[  :r]
  s  = s [  :r]
  return U,s,VT


def tinv(A,*kargs,**kwargs):
  """
  Inverse based on truncated svd.
  Also see sl.pinv2().
  """
  U,s,VT = tsvd(A,*kargs,**kwargs)
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


def map2arr(iterator, return_np_array=True):
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


class model(object):

    def __init__(self, func, par_names, par_values, arg_names, arg_values, xfromv=None):

        self.func = func
        self.par_names = par_names
        self.pars = par_values.copy()
        self.init_pars = par_values.copy()
        self.arg_names = arg_names
        self.args = arg_values.copy()
        self.init_args = arg_values.copy()

        if xfromv is None:

            @njit
            def xfromv(v):
                return v

        self.xfromv = xfromv

    def __repr__(self):
        return "A generic representation of a model"

    def reset(self):
        self.pars = self.init_pars.copy()
        self.args = self.init_args.copy()

    def get_args(self):

        arg_dict = dict(zip(self.arg_names, self.args))

        return arg_dict

    def set_args(self, **args):
        for a in zip(args, args.values()):
            self.args[self.arg_names.index(a[0])] = a[1]


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


def mvn_sobol(mean=None, cov=None, size=1):
    """Multivariate normal with sobol.

    Creates a sample of size `size` using the SOBOL quasi-random number sequences.
    """

    import sobol_seq
    from scipy.stats import norm
    import sobol_new

    err_msg = 'Argument cov must be a dxd ndarray, with d>1, defining a symmetric positive matrix.'

    if mean is None:
        if cov is None:
            raise ValueError(err_msg)
        dim = cov.shape[0]
        mean = np.zeros(dim)
    elif isinstance(mean, float):
        dim = cov.shape[0]
        mean = np.ones(dim)*mean
    elif cov is None:
        dim = mean.shape[0]
        cov = np.eye(dim)
    else:
        dim = mean.shape[0]
        assert cov.shape == mean.shape*2, err_msg

    try:
        L = np.linalg.cholesky(cov)
    except:
        raise ValueError(err_msg)

    shape = np.multiply(*size) if isinstance(size, tuple) else size
    size = size if isinstance(size, tuple) else (size,)

    sobols = sobol_new.sobol_points(shape+1,dim)[1:]
    z = norm.ppf(sobols)
    res = mean + np.dot(z, L.T)

    return res.reshape(size + (dim,))


def shuffle(a, axis=-1):
    """Shuffle along single axis
    """

    shape = a.shape
    res = a.reshape(-1,a.shape[axis])
    np.random.shuffle(res)

    return res.reshape(shape)


def blow_matrix(X, cov):
    """Analitical stretch to transform X -> Y with Y = X + Z and Z ~ N(0,cov).
    """

    mean = np.mean(X, axis=0)

    X = mean + (X-mean) @ sl.sqrtm(c2 @ nl.inv(np.cov(X.T)) + np.eye(X.shape[1]))

    return X


def mode(x):
    """Find mode of (unimodal) univariate distribution"""

    xs = np.sort(x)
    kde = ss.gaussian_kde(xs)
    p = kde.evaluate(xs)
    return xs[p.argmax()]


# aliases 
map2list = map2arr
indof = np.searchsorted
