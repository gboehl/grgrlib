#!/bin/python2
# -*- coding: utf-8 -*-

directory = '/home/gboehl/repos/'
import os, sys, importlib, time
for i in os.listdir(directory):
    sys.path.append(directory+i)

import numpy as np
import numpy.linalg as nl
import scipy.linalg as sl
import warnings
import pydsge
from grgrlib.base import *
from numba import njit

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

@njit(cache=True)
def preprocess_jit(vals, ll_max, kk_max):

    N, A, J, H, cx, b, x_bar  = vals

    dim_x, dim_y    = J.shape
    dim_v           = dim_y - dim_x
    dim_s           = H.shape[1]

    ss_max 	= ll_max + kk_max
    LL_mat 	= np.empty((ll_max,ss_max, dim_y, dim_y))
    SS_mat 	= np.empty((ll_max,kk_max, dim_x, dim_v))
    LL_term = np.empty((ll_max,ss_max, dim_y))
    SS_term = np.empty((ll_max,kk_max, dim_x))
    LL_tran = np.empty((ll_max,ss_max, dim_y, dim_s))
    SS_tran = np.empty((ll_max,kk_max, dim_x, dim_s))

    for ll in range(ll_max):
        for kk in range(kk_max):
            SS_mat[ll,kk], SS_term[ll,kk], SS_tran[ll,kk] 	= create_SS(vals[:5],ll,kk)
        for ss in range(ss_max):
            LL_mat[ll,ss], LL_term[ll,ss], LL_tran[ll,ss] 	= create_LL(vals[:5],ll,0,ss)
            ## here is minimal potiental for speed up:
            # if ss >= ll-1: LL_mat[ll,ss], LL_term[ll,ss] 	= create_LL(vals[:6],ll,0,ss)

    return SS_mat, SS_term, SS_tran, LL_mat, LL_term, LL_tran


def preprocess(self, ll_max = 5, kk_max = 20, info = False):
    st  = time.time()
    self.precalc_mat    = preprocess_jit(self.sys, ll_max, kk_max)
    if info == 1: 
        print('Preproceccing finished within %s s.' % np.round((time.time() - st), 3))


@njit(cache=True)
def create_SS(vals, l, k):

    N, A, J, H, cx  = vals
    dim_x, dim_y    = J.shape

    term 		= J @ geom_series(N, k) @ cx
    if l:
        N_k 		= nl.matrix_power(N,k)
        A_k         = nl.matrix_power(A,l-1)
        tran        = J @ N_k @ A_k
        JN			= J @ N_k @ A_k @ A
    elif k:
        N_k 		= nl.matrix_power(N,k-1)
        tran        = J @ N_k
        JN			= J @ N_k @ N
    else:
        tran    = JN    = J
    core        = -nl.inv(JN[:,:dim_x]) 

    return core @ JN[:,dim_x:], core @ term, core @ tran @ H


@njit(cache=True)
def create_LL(vals, l, k, s):

    N, A, J, H, cx  = vals
    dim_x, dim_y    = J.shape

    k0 		= max(s-l, 0)
    l0 		= min(l, s)
    if l0:
        N_k 		    = nl.matrix_power(N,k0)
        A_k             = nl.matrix_power(A,l0-1)
        matrices 		= N_k @ A_k @ A
        tran 		    = N_k @ A_k
    elif k0:
        N_k 		    = nl.matrix_power(N,k0-1)
        matrices 		= N_k @ N
        tran 		    = N_k 
    else:
        matrices    = tran  = np.eye(dim_y)
    term			= geom_series(N, k0) @ cx

    return matrices, term, tran @ H


@njit(cache=True)
def LL_pp(l, k, s, v, eps, SS_mat, SS_term, SS_tran, LL_mat, LL_term, LL_tran):

    # SS_mat, SS_term, SS_tran, LL_mat, LL_term, LL_tran  = precalc_mat

    dim_x   = SS_mat.shape[2]

    SS 	= SS_mat[l,k] @ v + SS_term[l,k] + SS_tran[l,k] @ eps

    matrices 	= LL_mat[l,s]
    term 		= LL_term[l,s]

    return matrices[:,:dim_x] @ SS + matrices[:,dim_x:] @ v + term + LL_tran[l,s] @ eps


@njit(cache=True)
def SS_jit(vals, l, k, v, eps):

    N, A, J, H, cx  = vals
    dim_x, dim_y    = J.shape

    term 		= J @ geom_series(N, k) @ cx
    if l:
        N_k 		= nl.matrix_power(N,k)
        A_k         = nl.matrix_power(A,l-1)
        tran        = J @ N_k @ A_k
        JN			= J @ N_k @ A_k @ A
    elif k:
        N_k 		= nl.matrix_power(N,k-1)
        tran        = J @ N_k
        JN			= J @ N_k @ N
    else:
        tran    = JN    = J
    core        = -nl.inv(JN[:,:dim_x]) 
    return core @ JN[:,dim_x:] @ v + core @ term + core @ tran @ H @ eps

@njit(cache=True)
def LL_jit(l, k, s, v, eps, vals):

    N, A, J, H, cx  = vals
    dim_x, dim_y    = J.shape

    k0 		= max(s-l, 0)
    l0 		= min(l, s)
    if l0:
        N_k 		    = nl.matrix_power(N,k0)
        A_k             = nl.matrix_power(A,l0-1)
        matrices 		= N_k @ A_k @ A
        tran 		    = N_k @ A_k
    elif k0:
        N_k 		    = nl.matrix_power(N,k0-1)
        matrices 		= N_k @ N
        tran 		    = N_k 
    else:
        matrices    = tran  = np.eye(dim_y)
    term			= geom_series(N, k0) @ cx

    return matrices[:,:dim_x] @ SS_jit(vals, l, k, v, eps) + matrices[:,dim_x:] @ v + term + tran @ H @ eps


@njit(cache=True)
def boehlgorithm_pp(N, A, J, H, cx, b, x_bar , v, eps, SS_mat, SS_term, SS_tran, LL_mat, LL_term, LL_tran, max_cnt):

    dim_x, dim_y    = J.shape

    l, k 		= 0, 0
    l1, k1 		= 1, 1

    l_max   = SS_mat.shape[0] - 1
    k_max   = SS_mat.shape[1] - 1

    cnt     = 0
    flag    = False
    while (l, k) != (l1, k1):
        if cnt  > max_cnt:
            flag    = True
            break
        l1, k1 		= l, k
        if l: l 		-= 1
        while b @ LL_pp(l, k, l, v, eps, SS_mat, SS_term, SS_tran, LL_mat, LL_term, LL_tran) - x_bar > 0:
            if l >= l_max:
                l = 0
                break
            l 	+= 1
        if (l) == (l1):
            if k: k 		-= 1
            while b @ LL_pp(l, k, l+k, v, eps, SS_mat, SS_term, SS_tran, LL_mat, LL_term, LL_tran) - x_bar < 0: 
                k +=1
                if k >= k_max:
                    # print('k_max reached, exiting')
                    break
        cnt += 1

    if not k: l = 1
    v_new 	= LL_pp(l, k, 1, v, eps, SS_mat, SS_term, SS_tran, LL_mat, LL_term, LL_tran)[dim_x:]

    return v_new, (l, k), flag


@njit(cache=True)
def boehlgorithm_jit(vals, v, eps, max_cnt, k_max = 20, l_max = 20):

    N, A, J, H, cx, b, x_bar    = vals
    dim_x, dim_y    = J.shape
    
    l, k 		= 0, 0
    l1, k1 		= 1, 1

    cnt     = 0
    flag    = False
    while (l, k) != (l1, k1):
        cnt += 1
        if cnt  > max_cnt:
            flag    = True
            break
        l1, k1 		= l, k
        if l: l -= 1
        while b @ LL_jit(l, k, l, v, eps, vals[:5]) - x_bar > 0:
            if l > l_max:
                l       = 0
                break
            l 	+= 1
        if (l) == (l1):
            if k: k 		-= 1
            while b @ LL_jit(l, k, l+k, v, eps, vals[:5]) - x_bar < 0: 
                k +=1
                if k > k_max:
                    # print('k_max reached, exiting')
                    break
        cnt += 1

    if not k: l = 1
    v_new 	= LL_jit(l, k, 1, v, eps, vals[:5])[dim_x:]

    return v_new, (l, k), flag


def boehlgorithm(model_obj, v, eps, max_cnt = 5e1):

    if hasattr(model_obj, 'precalc_mat'):

        ## numba does not like tuples of numpy arrays
        SS_mat, SS_term, SS_tran, LL_mat, LL_term, LL_tran  = model_obj.precalc_mat
        N, A, J, H, cx, b, x_bar                            = model_obj.sys

        return boehlgorithm_pp(N, A, J, H, cx, b, x_bar, v, eps, SS_mat, SS_term, SS_tran, LL_mat, LL_term, LL_tran, max_cnt)
    else:
        return boehlgorithm_jit(model_obj.sys, v, eps, max_cnt)

pydsge.DSGE.DSGE.preprocess   = preprocess
