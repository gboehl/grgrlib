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


def desingularize(M, P, D, vv, b=None, c=None, return_sub_ind=False):
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

    ## get_z is probably completely unnecessary since z_t is included in y_t

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
    D1  = U.T @ D

    if type(c) == np.ndarray:
        c   = U.T @ c

    ## deconstructing old matrices
    m23     = M1[s0][:,dim_x:]
    m21_22  = M1[s0][:,:dim_x]
    m13     = M1[~s0][:,dim_x:]
    m11_12  = M1[~s0][:,:dim_x]

    p3      = P1[~s0][:,dim_x:]
    p1_2    = P1[~s0][:,:dim_x]

    d1      = D[~s0]
    d2      = D[s0]

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

    D2      = d1 - m12 @ m22_inv @ d2

    if type(c) == np.ndarray:
        if not fast0(b2 @ m22_inv @ d2, 2):
            warnings.warn('Potential approximation error trough ommited shocks in the constraint')

        bh1     = b1 - b2 @ m22_inv @ m21
        bh2     = b3 - b2 @ m22_inv @ m23

        c1  = c[~s0]
        c2  = m22_inv @ c[s0]
        
        if not fast0(c2,2):
            warnings.warn('z_t also depends on the constrained variable')
            if not fast0(p2 @ c2,2):
                warnings.warn('System futher depends on expectations of constrained value trough z_t')

    ## constructing final matrices
    M2  = np.block([[mh1, mh2]])
    P2  = np.block([[ph1, ph2]])

    if type(c) == np.ndarray:
        bb2 = np.hstack((bh1, bh2))

    ## collecting information about the new state system
    vv2         = vv_x[~sub_ind], vv_x[sub_ind], vv_v
    get_z       = -np.hstack([m22_inv @ m21, m22_inv @ m23])
    
    if type(c) == np.ndarray:
        if return_sub_ind:
            return M2, P2, D2, vv2, get_z, bb2, c1, sub_ind
        else:
            return M2, P2, D2, vv2, get_z, bb2, c1
    else:
        if return_sub_ind:
            return M2, P2, D2, vv2, get_z, sub_ind
        else:
            return M2, P2, D2, vv2, get_z


def get_sys(mod, par):
    ## get_z is probably completely unnecessary since z_t is included in y_t

    const_var   = mod.const_var

    vv_v        = np.array(mod.variables)
    vv_x        = np.array(mod.variables)
    vv          = vv_x, vv_v

    dim         = len(vv_x)

    A   = mod.AA(par) # forward
    B   = mod.BB(par) # contemp
    C   = mod.CC(par) # backward
    D   = mod.PSI(par)

    if const_var:
        const_arg   = list(vv_x).index(const_var)
        b           = mod.bb(par).flatten()
        if not fast0(A[:,const_arg],2):
            warnings.warn('\n   Not implemented: system depends on expectations of constrained variable\n')

    M       = np.block([[B,C], [np.eye(dim), np.zeros((dim,dim))]])
    P       = np.block([[A,np.zeros(C.shape)],[np.zeros((dim,dim)), np.eye(dim)]])
    D       = np.block([[D], [np.zeros((dim,D.shape[1]))]])

    if const_var:
        c   = M[:,const_arg]
        M   = np.delete(M, const_arg, 1)
        P   = np.delete(P, const_arg, 1)
        b2      = -np.delete(b, const_arg)/b[const_arg]
        vv_x    = np.delete(vv_x, const_arg)

        M3, P3, D2, vv2, get_z, b3, c2  = desingularize(M, P, D, (vv_x, vv_v), b2, c)
    else:
        M3, P3, D2, vv2, get_z          = desingularize(M, P, D, (vv_x, vv_v))

    if np.abs(nl.det(P3)) < 1e-3:

        M3, P3, D2, vv3, get_z2, b3, c2, sub_ind    = desingularize(M3, P3, D2, (vv2[0],vv2[2]), b3, c2, return_sub_ind=True)

        u   = np.hstack((get_z[:,:len(vv2[0])][:,~sub_ind], get_z[:,len(vv2[0]):])) + get_z[:,:len(vv2[0])][:,sub_ind] @ get_z2
        w   = get_z2
        get_z3  = np.vstack((u,w))

        vv_z    = np.hstack((vv2[1], vv3[1]))
        vv3     = vv3[0], vv_z, vv3[2]
    else:
        vv3     = vv2
        get_z3  = get_z

    MM  = M3
    PP  = P3
    DD  = D2

    if const_var:
        bb  = b3
        cc  = c2
        N       = nl.inv(PP) @ (MM + np.outer(cc, bb))
    else:
        N       = nl.inv(PP) @ MM

    if not len(vv3[0]) == sum(eig(N) >= 1):
        warnings.warn('\n   BC *not* satisfied!\n')

    mod.vv          = vv3
    if const_var:
        return MM, PP, DD, bb, cc, get_z3
    else:
        return MM, PP, DD, get_z3

@njit(cache=True)
def subt(A, B):
	res 	= A
	for i in range(len(A)):
		for j in range(len(A)):
			res[i][j] = A[i][j] - B[i][j]
	return res

@njit(cache=True)
def LL_jit(mod,l, k, s, v):
    N, J, A, JIN, Mcx, dim_x, nr_dims, IN 	= mod
    ## as in paper
    k0 		= max(s-l, 0)
    l0 		= min(l, s)
    matrices 		= nl.matrix_power(N,k0) @ nl.matrix_power(A,l0)
    N_k 		    = nl.matrix_power(N.copy(),k0)
    subt_part       = subt(np.identity(nr_dims), N_k)
    term			= IN @ subt_part @ Mcx
    return matrices @ np.hstack((SS_jit(mod[:7], l, k, v), v)) + term

@njit(cache=True)
def LL_jit(mod,l, k, s, v):
    N, J, A, JIN, Mcx, dim_x, nr_dims, IN 	= mod
    ## as in paper
    if k == 0:
        l = s
    k0 		= max(s-l, 0)
    l0 		= min(l, s)
    matrices 		= nl.matrix_power(N,k0) @ nl.matrix_power(A,l0)
    N_k 		    = nl.matrix_power(N.copy(),k0)
    subt_part       = subt(np.identity(nr_dims), N_k)
    term			= IN @ subt_part @ Mcx
    return matrices @ np.hstack((SS_jit(mod[:7], l, k, v), v)) + term

@njit(cache=True)
def boehlgorithm_jit(vals, v, k_max = 20):

    N, J, A, JIN, Mcx, dim_x, nr_dims, IN, P, b, x_bar, c = vals

    l, k 		= 0, 0
    l1, k1 		= 1, 1

    while (l, k) != (l1, k1):
        l1, k1 		= l, k
        if l: l 		-= 1
        while np.dot(b,LL_jit(vals[:8],l, k, l, v)) - x_bar > 0:
            if l > k_max:
                l = 0
                break
            l 	+= 1
        if (l) == (l1):
            if k: k 		-= 1
            while np.dot(b,LL_jit(vals[:8], l, k, l+k, v)) - x_bar < 0: 
                k +=1
                if k > k_max:
                    # warnings.warn('k_max reached, exiting')
                    print('k_max reached, exiting')
                    break

    v_new 	= LL_jit(vals[:8], l, k, 1, v)[dim_x:]
    return v_new, (l, k)
