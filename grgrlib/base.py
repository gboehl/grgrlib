#!/bin/python2
# -*- coding: utf-8 -*-

directory = '/home/gboehl/repos/'
import os, sys, importlib
for i in os.listdir(directory):
    sys.path.append(directory+i)
sys.path.append('/home/gboehl/rsh/bs18/code/')

import numpy as np
import numpy.linalg as nl
import scipy.linalg as sl
import warnings
from grgrlib.pyzlb import *
import pydsge
import matplotlib.pyplot as plt
from numba import njit
import time


def eig(M):
    return np.sort(np.abs(nl.eig(M)[0]))[::-1]

def sorter(x, y):
    out     = np.empty_like(x, dtype=bool)
    out[:]  = False
    zero_y      = np.isclose(y,0)
    zero_x      = np.isclose(x,0)
    out[zero_y] = True
    out[zero_x] = True
    return out

def invertible_subm(A):
    """
    For a m times n matrix A with n > m this function finds the m columns that are necessary to construct a nonsingular submatrix of A.
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

def get_sys(self, par, care_for = None, info = False):

    st  = time.time()

    if not self.const_var:
        warnings.warn('Code is only meant to work with OBCs')

    vv_v    = np.array(self.variables)
    vv_x    = np.array(self.variables)

    dim_v   = len(vv_v)

    ## obtain matrices from pydsge
    ## this can be further accelerated by getting them directly from the equations in pydsge
    AA  = self.AA(par)              # forward
    BB  = self.BB(par)              # contemp
    CC  = self.CC(par)              # backward
    b   = self.bb(par).flatten()    # constraint

    ## define transition shocks -> state
    D   = self.PSI(par)
    H   = - D.copy()
    # H   = self.PSI(par)
    hit     = ~fast0(D, 1)

    ## mask those vars that are either forward looking or part of the constraint
    in_x       = ~fast0(AA, 0) | ~fast0(b[:dim_v])

    ## reduce x vector
    vv_x2   = vv_x[in_x]
    A1      = AA[:,in_x]
    b1      = np.hstack((b[:dim_v][in_x], b[dim_v:]))

    dim_x   = len(vv_x2)

    ## define actual matrices
    M       = np.block([[np.zeros(A1.shape), CC], 
                        [np.eye(dim_x), np.zeros((dim_x,dim_v))]])

    P       = np.block([[A1, -BB],
                        [np.zeros((dim_x,dim_x)), np.eye(dim_v)[in_x]]])

    H1      = np.block([[H],
                        [np.zeros((dim_x,H.shape[1]))]])

    c_arg       = list(vv_x2).index(self.const_var)

    ## c contains information on how the constraint var affects the system
    c_M     = M[:,c_arg]
    c_P     = P[:,c_arg]

    ## get rid of constrained var
    b2      = np.delete(b1, c_arg)
    M1      = np.delete(M, c_arg, 1)
    P1      = np.delete(P, c_arg, 1)
    vv_x3   = np.delete(vv_x2, c_arg)

    ## decompose P in singular & nonsingular rows
    U, s, V     = nl.svd(P1)
    s0  = fast0(s)

    P2  = np.diag(s) @ V
    M2  = U.T @ M1
    H2  = U.T @ H1

    c1  = U.T @ c_M

    if not fast0(c1[s0], 2) or not fast0(U.T[s0] @ c_P, 2):
        warnings.warn('\nNot implemented: the system depends directly or indirectly on whether the constraint holds in the future or not.\n')
        
    ## actual desingularization by iterating equations in M forward
    P2[s0]  = M2[s0]

    try:
        x_bar       = par[[p.name for p in self.parameters].index('x_bar')]
    except ValueError:
        warnings.warn("\nx_bar (maximum value of the constraint) not specified. Assuming x_bar = -1 for now.\n")
        x_bar       = -1

    ## create the stuff that the algorithm needs
    N       = nl.inv(P2) @ M2 
    A       = nl.inv(P2) @ (M2 + np.outer(c1,b2))
    H3      = nl.inv(P2) @ H2

    if sum(eig(A).round(3) >= 1) - len(vv_x3):
        # warnings.warn('BC *not* satisfied.')
        raise ValueError('BC *not* satisfied.')

    dim_x       = len(vv_x3)
    OME         = re_bc(A, dim_x)
    J 			= np.hstack((np.eye(dim_x), -OME))
    cx 		    = nl.inv(P2) @ c1*x_bar

    ## check condition:
    n1  = N[:dim_x,:dim_x]
    n3  = N[dim_x:,:dim_x]
    cc1  = cx[:dim_x]
    cc2  = cx[dim_x:]
    bb1  = b2[:dim_x]

    if info == 1:
        print('Creation of system matrices finished in %ss. Condition value is %s.' 
              % (np.round(time.time() - st,3), (bb1 @ nl.inv(n1 - OME @ n3) @ (cc1 - OME @ cc2)).round(4)))

    ## reduce size of matrices if possible
    if care_for is None or care_for is 'obs':
        care_for    = [ o.name for o in self['observables'] ] 
    if care_for == 'all':
        care_for    = [ o.name for o in self.variables ] 

    var_str     = [ v.name for v in vv_v ]
    out_msk     = fast0(N, 0) & fast0(A, 0) & fast0(b2) & fast0(cx)
    out_msk[-len(vv_v):]    = out_msk[-len(vv_v):] & np.array([v not in care_for for v in var_str])

    ## add everything to the DSGE object
    self.vv     = vv_v[~out_msk[-len(vv_v):]]
    self.obs_arg        = [ list(self.vv).index(ob) for ob in self['observables'] ]
    self.observables    = self['observables']
    self.par    = par
    self.SIG    = (BB.T @ D)[~out_msk[-len(vv_v):]]
    self.sys 	= N[~out_msk][:,~out_msk], A[~out_msk][:,~out_msk], J[:,~out_msk], H3[~out_msk], cx[~out_msk], b2[~out_msk], x_bar

    """
    ## add everything to the DSGE object
    self.vv     = vv_v
    self.obs_arg        = [ list(vv_v).index(ob) for ob in self['observables'] ]
    self.observables    = self['observables']
    self.par    = par
    self.SIG    = BB.T @ D
    self.sys 	= N, A, J, H3, cx, b2, x_bar
    # """


def irfs(self, shocklist, wannasee = None, plot = True):

    ## returns time series of impule responses 
    ## shocklist: takes list of tuples of (shock, size, timing) 
    ## wannasee: list of strings of the variables to be plotted and stored

    labels      = [v.name.replace('_','') for v in self.vv]
    if wannasee is not None:
        args_see    = [labels.index(v) for v in wannasee]
    else:
        args_see    = self.obs_arg.copy()

    st_vec          = np.zeros(len(self.vv))

    Y   = []
    K   = []
    L   = []
    superflag   = False

    for t in range(30):

        shk_vec     = np.zeros(len(self.shocks))
        for vec in shocklist: 
            if vec[2] == t:

                shock       = vec[0]
                shocksize   = vec[1]

                shock_arg           = [v.name for v in self.shocks].index(shock)
                shk_vec[shock_arg]  = shocksize

                shk_process     = (self.SIG @ shk_vec).nonzero()

                args_see += shk_process

        st_vec, (l,k), flag     = boehlgorithm(self, st_vec, shk_vec)

        if flag: 
            superflag   = True

        Y.append(st_vec)
        K.append(k)
        L.append(l)

    Y   = np.array(Y)
    K   = np.array(K)
    L   = np.array(L)

    care_for    = np.unique(args_see)

    X   = Y[:,care_for]

    if superflag:
        warnings.warn('Numerical errors in boehlgorithm, did not converge')

    return X, self.vv[care_for], (Y, K, L)


def t_func(self, state, noise = None, return_k = False):

    if noise is not None:
        newstate, (l,k), flag   = boehlgorithm(self, state, noise)
    else:
        zro     = np.zeros(self.SIG.shape[1])
        newstate, (l,k), flag   = boehlgorithm(self, state, zro)

    if return_k: 	return newstate, (l,k), flag
    else: 			return newstate


def create_filter(self, alpha = .2, scale_obs = .2):

    from filterpy.kalman import UnscentedKalmanFilter as UKF
    # from filterpy.kalman import ReducedScaledSigmaPoints
    # from filterpy.kalman import MerweScaledSigmaPoints
    from filterpy.kalman import SigmaPoints_ftl

    dim_v       = len(self.vv)
    beta_ukf 	= 2.
    kappa_ukf 	= 3 - dim_v

    if not hasattr(self, 'Z'):
        warnings.warn('No time series of observables provided')
    else:
        sig_obs 	= np.std(self.Z, 0)*scale_obs

    exo_args    = ~fast0(self.SIG,1)

    ## ReducedScaledSigmaPoints are an attemp to reduce the number of necessary sigma points. 
    ## As of yet not functional
    # spoints     = ReducedScaledSigmaPoints(alpha, beta_ukf, kappa_ukf, exo_args)
    spoints     = SigmaPoints_ftl(dim_v,alpha, beta_ukf, kappa_ukf)
    ukf 		= UKF(dim_x=dim_v, dim_z=self.ny, hx=self.obs_arg, fx=self.t_func, points=spoints)
    ukf.x 		= np.zeros(dim_v)
    ukf.R 		= np.diag(sig_obs)**2

    CO          = self.SIG @ self.QQ(self.par)
    ukf.Q 		= CO @ CO.T

    self.ukf    = ukf


def run_filter(self, use_rts=False, info=False):

    if info == 1:
        st  = time.time()

    exo_args    = ~fast0(self.SIG,1)

    X1, cov, Y, ll     = self.ukf.batch_filter(self.Z)

    ## the actual filter seems to work better than the smoother. The implemented version (succesfully) 
    ## uses the pseudoinverse to deal with the fact that the co-variance matrix is singular
    if use_rts:
        X1, _, _            = self.ukf.rts_smoother(X1, cov)

    self.filtered_Z     = X1[:,self.obs_arg]
    self.filtered_X     = X1
    self.filtered_V     = X1[:,exo_args]
    self.ll             = ll
    self.residuals      = Y[:,exo_args]

    if info == 1:
        print('Filtering done in '+str(np.round(time.time()-st,3))+'seconds.')


def pplot(X, labels, yscale=None, title='', style='-', savepath=None, Y=None):
    plt_no      = X.shape[1] // 4 + bool(X.shape[1]%4)
    if yscale is None:
        yscale  = np.arange(X.shape[0])
    for i in range(plt_no):
        ax  = plt.subplots(2,2)[1].flatten()
        for j in range(4):
            if 4*i+j >= X.shape[1]:
                ax[j].set_visible(False)
            else:
                if X.shape[1] > 4*i+j:
                    ax[j].plot(yscale, X[:,4*i+j], style, lw=2)
                if Y is not None:
                    if Y.shape[1] > 4*i+j:
                        ax[j].plot(yscale, Y[:,4*i+j], style, lw=2)
                ax[j].tick_params(axis='both', which='both', top=False, right=False, labelsize=12)
                ax[j].spines['top'].set_visible(False)
                ax[j].spines['right'].set_visible(False)
                ax[j].set_xlabel(labels[4*i+j], fontsize=14)
        if title:
            plt.suptitle('%s %s' %(title,i+1), fontsize=16)
        plt.tight_layout()
        if savepath is not None:
            plt.savefig(savepath+title+str(i+1)+'.pdf')
        plt.show()

from pympler import muppy
from pympler import tracker

def bayesian_estimation(self, alpha = 0.2, scale_obs = 0.2, draws = 500, tune = 500, no_cores = None, use_find_MAP = True, info = False):

    import pymc3 as pm
    import theano.tensor as tt
    from theano.compile.ops import as_op
    import multiprocessing

    if no_cores is None:
        # no_cores    = multiprocessing.cpu_count() - 1
        no_cores    = 4

    ## dry run before the fun beginns
    self.create_filter(scale_obs = scale_obs)
    self.ukf.R[-1,-1]  /= 100
    self.run_filter()
    print("Model operational. Ready for estimation.")

    ## from here different from filtering
    par_active  = np.array(self.par).copy()

    p_names     = [ p.name for p in self.parameters ]
    priors      = self['__data__']['estimation']['prior']
    prior_arg   = [ p_names.index(pp) for pp in priors.keys() ]

    init_par    = dict(zip(np.array(p_names)[prior_arg], np.array(self.par)[prior_arg]))

    tlist   = []
    for i in range(len(priors)):
        tlist.append(tt.dscalar)

    tr = tracker.SummaryTracker()

    @as_op(itypes=tlist, otypes=[tt.dvector])
    def get_ll(*parameters):
        st  = time.time()

        tr.print_diff()

        try: 
            par_active[prior_arg]  = parameters
            par_active_lst  = list(par_active)

            self.get_sys(par_active_lst)
            self.preprocess(info=info)

            self.create_filter(scale_obs = scale_obs)
            self.ukf.R[-1,-1]  /= 100
            self.run_filter(info=info)

            if info == 2:
                print('Sample took '+str(np.round(time.time() - st))+'s.')
            return self.ll.reshape(1)

        except:

            if info == 2:
                print('Sample took '+str(np.round(time.time() - st))+'s. (failure)')
            return np.array(-sys.maxsize - 1, dtype=float).reshape(1)


    with pm.Model() as model:
        
        be_pars_lst     = []
        for pp in priors:
            dist    = priors[str(pp)]
            pmean = dist[1]
            pstdd = dist[2]

            if str(dist[0]) == 'uniform':
                be_pars_lst.append( pm.Uniform(str(pp), lower=dist[1], upper=dist[2]) )
            elif str(dist[0]) == 'inv_gamma':
                alp     = pmean**2/pstdd**2 + 2
                bet     = pmean*(alp - 1)
                be_pars_lst.append( pm.InverseGamma(str(pp), alpha=alp, beta=bet) )
            elif str(dist[0]) == 'normal':
                be_pars_lst.append( pm.Normal(str(pp), mu=pmean, sd=pstdd) )
            elif str(dist[0]) == 'gamma':
                be_pars_lst.append( pm.Gamma(str(pp), mu=pmean, sd=pstdd) )
            elif str(dist[0]) == 'beta':
                be_pars_lst.append( pm.Beta(str(pp), mu=pmean, sd=pstdd) )
            else:
                print('Distribution not implemented')
            print('Adding parameter %s as %s to the prior distributions.' %(pp, dist[0]))

        be_pars = tuple(be_pars_lst)

        pm.Potential('logllh', get_ll(*be_pars))
        
        if use_find_MAP:
            self.MAP = pm.find_MAP(start=init_par)
            # self.MAP = pm.find_MAP(start=init_par, method='Nelder-Mead')
        else:
            self.MAP = init_par
        step = pm.Metropolis()
        self.trace = pm.sample(draws=draws, tune=tune, step=step, start=self.MAP, cores=no_cores, random_seed=list(np.arange(no_cores)))

    return be_pars

pydsge.DSGE.DSGE.get_sys            = get_sys
pydsge.DSGE.DSGE.t_func             = t_func
pydsge.DSGE.DSGE.irfs               = irfs
pydsge.DSGE.DSGE.create_filter      = create_filter
pydsge.DSGE.DSGE.run_filter         = run_filter
pydsge.DSGE.DSGE.bayesian_estimation    = bayesian_estimation
