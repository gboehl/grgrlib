#!/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from numba import njit


def chebychev(order, use_numba=True):
    """Returns a function of x that is the Chebychev polynomial of given order. The function is build dynamically to avoid recursions at runtime.
    """

    if not order: 
        return np.ones_like

    def chebystr(k):
        if k == 0:
            return '1'
        elif k == 1:
            return 'x'
        else:
            return '2*x*(%s) - (%s)' %(chebystr(k-1),chebystr(k-2))

    fstr = 'lambda x: %s' %chebystr(order)
    func = eval(fstr)

    if use_numba:
        return njit(func)
    else:
        return func
