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
import matplotlib.pyplot as plt
# from itertools import *

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

def get_sys(mod, par):

    A   = mod.AA(par)
    B   = mod.BB(par)
    C   = mod.CC(par)

    nums, II    = mod.info_vec['numbers'], mod.info_vec['II'] 
    M   = np.block([[-B,C],[II, np.zeros((nums[0],nums[0]))]])
    P   = np.block([[A,np.zeros(C.shape)],[np.zeros(C.T.shape), np.eye(nums[0])]])

    MM, PP, alp, bet, Q, Z  = sl.ordqz(M,P, sort=sorter)

    if not np.allclose(Q@MM@Z.T, M):
        print('problem with QZ')

    ind     = ~sorter(alp, bet)

    M1  = MM[ind][:,ind]
    P1  = PP[ind][:,ind]

    if not nl.det(P1) or not nl.det(M1):
        print('Singularities! P,M:', nl.det(P1), nl.det(M1))

    # active_ind      = ~np.isclose(Z.T[ind],0).all(axis=0)

    # if not np.allclose( PP[~ind]@Z.T[:,~active_ind], 0 ):
    if not np.allclose( PP[~ind][:,~ind], 0 ):
        print('Apparent dependency on redundant future vals!')

    N   = nl.inv(M1) @ P1

    # return N
    return N, MM, PP, Q, Z, ind

def re_bc(N, d_endo):

    eigvals, eigvecs 	= nl.eig(nl.inv(N))

    idx             	= np.abs(eigvals).argsort()[::-1]

    eigvecs      		= nl.inv(eigvecs[:,idx])     

    res     = nl.inv(eigvecs[:d_endo,:d_endo]) @ eigvecs[:d_endo,d_endo:]

    return -res.real

def fast0(A):

    return np.isclose(A, 0)

