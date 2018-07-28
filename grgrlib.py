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


def desingularize(M, P, vv, b=None, c=None):
    """
    Given a linear dynamic system of the form

          | x_t     |      | E_t x_{t+1} |
        M |         |  = P |             |
          | y_{t-1} |      |     y_t     |
          
    this algorythm reduces the vector of endogenous variables such that P is nonsingular.
    'vv' is a tuple containing the names of the variables in x_t (as dsge.symbols.Variables) and likewise the names of the variables in y_t.
    """
    ## all this is might simpler be done by QZ
    ## but not sure about how to obtain b & c with QZ

    vv_x, vv_v  = vv
    dim_x       = len(vv_x)

    ## find singular rows of old matrices
    U, s, V     = nl.svd(P)

    s0  = fast0(s)
    if not sum(s0):
        print('P is probably already nonsingular')
        return M, P, vv, np.empty(0)

    P1  = np.diag(s) @ V
    M1  = U.T @ M

    c   = U.T @ c

    ## deconstructing old matrices
    m23     = M1[s0][:,dim_x:]
    m21_22  = M1[s0][:,:dim_x]
    m13     = M1[~s0][:,dim_x:]
    m11_12  = M1[~s0][:,:dim_x]

    p3      = P1[~s0][:,dim_x:]
    p1_2    = P1[~s0][:,:dim_x]

    ## find variables in z_t
    sub_ind     = invertible_subm(m21_22)

    ## collecting building blocks
    m11         = m11_12[:,~sub_ind]
    m12         = m11_12[:,sub_ind]
    m21         = m21_22[:,~sub_ind]
    m22         = m21_22[:,sub_ind]
    p1          = p1_2[:,~sub_ind]
    p2          = p1_2[:,sub_ind]

    if type(c) == np.ndarray:
        b1          = b[:dim_x][~sub_ind]
        b2          = b[:dim_x][sub_ind]
        b3          = b[dim_x:]

    ## just a tiny little bit faster
    m22_inv     = nl.inv(m22)

    ## constructing building blocks
    mh1     = m11 - m12 @ m22_inv @ m21
    mh2     = m13 - m12 @ m22_inv @ m23
    ph1     = p1 - p2 @ m22_inv @ m21
    ph2     = p3 - p2 @ m22_inv @ m23

    ## constructing final matrices
    M2  = np.block([[mh1, mh2]])
    P2  = np.block([[ph1, ph2]])

    bh1     = b1 - b2 @ m22_inv @ m21
    bh2     = b3 - b2 @ m22_inv @ m23

    c1  = c[~s0]
    c2  = m22_inv @ c[s0]
    
    if not fast0(c2,2):
        warnings.warn('z_t also depends on the constrained variable')
        if not fast0(p2 @ c2,2):
            warnings.warn('System further depends on expectations of constrained value trough z_t')

    bb2 = np.hstack((bh1, bh2))

    ## collecting information about the new state system
    vv2         = vv_x[~sub_ind], vv_x[sub_ind], vv_v
    
    return M2, P2, vv2, bb2, c1


def get_sys(mod, par):

    const_var   = mod.const_var

    if not const_var:
        warnings.warn('Code is only meant to work with OBCs')

    vv_v        = np.array(mod.variables)
    vv_x        = np.array(mod.variables)
    vv          = vv_x, vv_v

    dim         = len(vv_x)

    A   = mod.AA(par) # forward
    B   = mod.BB(par) # contemp
    C   = mod.CC(par) # backward
    D   = mod.PSI(par)
    D2  = B.T @ D

    const_arg   = list(vv_x).index(const_var)
    b           = mod.bb(par).flatten()
    if not fast0(A[:,const_arg],2):
        warnings.warn('\n   Not implemented: system depends on expectations of constrained variable\n')

    M       = np.block([[B,C], [np.eye(dim), np.zeros((dim,dim))]])
    P       = np.block([[A,np.zeros(C.shape)],[np.zeros((dim,dim)), np.eye(dim)]])

    c   = M[:,const_arg]
    M   = np.delete(M, const_arg, 1)
    P   = np.delete(P, const_arg, 1)
    b2      = -np.delete(b, const_arg)/b[const_arg]
    vv_x    = np.delete(vv_x, const_arg)

    M3, P3, vv2, b3, c2  = desingularize(M, P, (vv_x, vv_v), b2, c)

    if np.abs(nl.det(P3)) < 1e-3:

        M3, P3, vv3, b3, c2     = desingularize(M3, P3, (vv2[0],vv2[2]), b3, c2)

        vv_z    = np.hstack((vv2[1], vv3[1]))
        vv3     = vv3[0], vv_z, vv3[2]
    else:
        vv3     = vv2

    dim_x   = len(vv3[0])
    dim_y   = len(vv3[2]) + dim_x

    A   = nl.inv(P3) @ (M3 + np.outer(c2,b3))
    N   = nl.inv(P3) @ M3

    if not len(vv3[0]) == sum(eig(A) >= 1):
        warnings.warn('\n   BC *not* satisfied!\n')

    try:
        x_bar       = par[[p.name for p in mod.parameters].index('x_bar')]
    except ValueError:
        warnings.warn("x_bar (maximum value of the constraint) not specified. Assuming x_bar = -1 for now.")
        x_bar       = -1
        
    OME         = re_bc(A, dim_x)
    J 			= np.hstack((np.eye(dim_x), -OME))
    cx 		    = nl.inv(P3) @ c2*x_bar
    IN 			= nl.inv(np.identity(dim_y) - N)

    mod.vv      = vv3
    mod.sys 	= N, J, A, IN, cx, dim_x, dim_y, b3, x_bar, D2


def irfs(mod, shock, shocksize=1, wannasee = ['y', 'Pi', 'r']):

    shk_vec             = np.zeros(len(mod.shocks))
    shock_arg           = [v.name for v in mod.shocks].index(shock)
    shk_vec[shock_arg]  = shocksize

    st_vec          = mod.sys[-1] @ shk_vec
    shk_process     = np.where(~fast0(st_vec))[0]

    labels      = [v.name.replace('_','') for v in mod.vv[2]]
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

