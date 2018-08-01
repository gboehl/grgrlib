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
from pyzlb import *
import dsge
import matplotlib.pyplot as plt
from numba import njit

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


"""     ## old version using eigenvector-eigenvalue decomposition
def re_bc(N, d_endo):

    eigvals, eigvecs 	= nl.eig(N)

    idx             	= np.abs(eigvals).argsort()[::-1]

    eigvecs      		= nl.inv(eigvecs[:,idx])     

    res     = nl.inv(eigvecs[:d_endo,:d_endo]) @ eigvecs[:d_endo,d_endo:]

    if not fast0(res.imag,2):
        warnings.warn('Non-neglible imaginary parts in OMEGA')

    return -res.real
    """


def re_bc(N, d_endo):

    n   = N.shape[0]

    MM, PP, alp, bet, Q, Z    = sl.ordqz(N,np.eye(n),sort='iuc')

    if not fast0(Q @ MM @ Z.T - N, 2):
        warnings.warn('Numerical errors in QZ')

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

def get_sys(self, par):

    if not self.const_var:
        warnings.warn('Code is only meant to work with OBCs')

    vv_v    = np.array(self.variables)
    vv_x    = np.array(self.variables)

    dim_v   = len(vv_v)

    A   = self.AA(par) # forward
    B   = self.BB(par) # contemp
    C   = self.CC(par) # backward

    D   = self.PSI(par)
    D2  = B.T @ D

    b           = self.bb(par).flatten()

    in_x       = ~fast0(A, 0) | ~fast0(b[:dim_v])

    ## suit to x/y system
    vv_x2   = vv_x[in_x]
    A1      = A[:,in_x]
    b1      = np.hstack((b[:dim_v][in_x], b[dim_v:]))

    dim_x   = len(vv_x2)

    M       = np.block([[np.zeros(A1.shape), C], 
                        [np.eye(dim_x), np.zeros((dim_x,dim_v))]])

    P       = np.block([[A1, -B],
                        [np.zeros((dim_x,dim_x)), np.eye(dim_v)[in_x]]])

    c_arg       = list(vv_x2).index(self.const_var)

    c_M     = M[:,c_arg]
    c_P     = P[:,c_arg]

    b2      = np.delete(b1, c_arg)
    M1      = np.delete(M, c_arg, 1)
    P1      = np.delete(P, c_arg, 1)

    vv_x3   = np.delete(vv_x2, c_arg)

    U, s, V     = nl.svd(P1)

    s0  = fast0(s)

    P2  = np.diag(s) @ V
    M2  = U.T @ M1

    c1  = U.T @ c_M

    if not fast0(c1[s0], 2) or not fast0(U.T[s0] @ c_P, 2):
        ## write propper warnings
        warnings.warn('\nNot implemented: the system depends directly or indirectly on whether the constraint holds in the future or not.\n')
        
    ## actual desingularization
    P2[s0]  = M2[s0]

    ## create all the crazy stuff I need
    try:
        x_bar       = par[[p.name for p in self.parameters].index('x_bar')]
    except ValueError:
        warnings.warn("x_bar (maximum value of the constraint) not specified. Assuming x_bar = -1 for now.")
        x_bar       = -1

    N       = nl.inv(P2) @ M2 
    A       = nl.inv(P2) @ (M2 + np.outer(c1,b2))

    if sum(eig(A).round(3) >= 1) - len(vv_x3):
        warnings.warn('BC *not* satisfied.')

    dim_x       = len(vv_x3)
    OME         = re_bc(A, dim_x)
    J 			= np.hstack((np.eye(dim_x), -OME))
    cx 		    = nl.inv(P2) @ c1*x_bar

    self.vv     = vv_x3, vv_v

    self.par    = par

    self.sys 	= N, J, A, cx, dim_x, dim_v + dim_x, b2, x_bar, D2


def irfs(mod, shock, shocksize=1, wannasee = ['y', 'Pi', 'r']):

    shk_vec             = np.zeros(len(mod.shocks))
    shock_arg           = [v.name for v in mod.shocks].index(shock)
    shk_vec[shock_arg]  = shocksize

    st_vec          = mod.sys[-1] @ shk_vec
    shk_process     = np.where(~fast0(st_vec))[0]

    labels      = [v.name.replace('_','') for v in mod.vv[1]]
    args_see    = [labels.index(v) for v in wannasee]

    care_for    = np.unique(np.hstack((args_see,shk_process)))

    X   = []
    Y   = []
    for t in range(30):
        st_vec, _   = boehlgorithm(mod, st_vec)
        X.append(st_vec[care_for])
        Y.append(st_vec)
    X   = np.array(X)
    Y   = np.array(Y)

    fig, ax     = plt.subplots()
    for i, l in enumerate(care_for):
        plt.plot(X[:,i], lw=3, label=labels[l])
    ax.tick_params(axis='both', which='both', top=False, right=False, labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.locator_params(nbins=8,axis='x')
    ax.legend(frameon=0)
    plt.tight_layout()
    plt.show()

    return Y

@njit(cache=True)
def geom_series(M, n):
    res  = np.zeros(M.shape)
    for i in range(n):
        gs_add(res,nl.matrix_power(M,i))
    return res


@njit(cache=True)
def gs_add(A, B):
	for i in range(len(A)):
		for j in range(len(A)):
			A[i][j] += B[i][j]


dsge.DSGE.DSGE.get_sys   = get_sys
