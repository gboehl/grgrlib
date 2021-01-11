#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np


def uncompress(x, ce_dict):

    nm, nk, ny, _ = ce_dict['npars'].astype(int)

    zeta = np.zeros(nm*nk*ny)
    zeta[ce_dict['c_inds'].astype(int)-1] = x

    zeta = zeta.reshape(nm, nk, ny, order='f')
    for i in range(ny):
        zeta[:, :, i] = ce_dict['idc1'] @ zeta[:, :, i] @ ce_dict['dc2']

    for i in range(nm):
        zeta[i, :, :] = zeta[i, :, :] @ ce_dict['dc3']

    return zeta.flatten('f')


def consumption_equivalent(x, ce_dict):

    util_par = ce_dict['npars'][-1]

    theta = uncompress(x, ce_dict)

    VT = (ce_dict['VSS'] + theta)**(1 - util_par)

    return (VT / ce_dict['VSS'])**(1/(1 - util_par)) - 1
