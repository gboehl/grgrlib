#!/bin/python
# -*- coding: utf-8 -*-

import os
import numpy as np

def mat2dict(path, shock_states):

    # load stuff
    AA = np.loadtxt(os.path.join(path, 'AA.txt'), delimiter=',')
    BB = np.loadtxt(os.path.join(path, 'BB.txt'), delimiter=',')
    CC = np.loadtxt(os.path.join(path, 'CC.txt'), delimiter=',')
    vv = np.loadtxt(os.path.join(path, 'list_of_vars.txt'), delimiter=',', dtype=str)

    vv = np.array([v[2:-1] for v in vv])

    # lets stick with UTF-8
    if 'π' in vv:
        vv[list(vv).index('π')] = 'Pi'
    if 'πw' in vv:
        vv[list(vv).index('πw')] = 'Piw'

    # DD is the mapping from shocks to states
    shocks = ['e_' + s for s in shock_states]

    DD = np.zeros((len(vv), len(shocks)))
    for i, v in enumerate(shock_states):
        DD[:, shock_states.index(v)] = CC[:, list(vv).index(v)]

    rdict = {}
    rdict['AA'] = AA
    rdict['BB'] = BB
    rdict['CC'] = CC
    rdict['DD'] = DD
    rdict['vars'] = vv
    rdict['shock_states'] = shock_states
    rdict['shocks'] = shocks
    
    return rdict
