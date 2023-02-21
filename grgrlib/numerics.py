# -*- coding: utf-8 -*-

from .__pycache__.chebychev import chebychev
import os

CHEBY_RANGE = 12


def chebystr(k):
    if k == 0:
        return 'np.ones_like(x)'
    elif k == 1:
        return 'x'
    else:
        return '2*x*(%s) - (%s)' % (chebystr(k-1), chebystr(k-2))


rows = "\n ".join([f"if order == {i}:\n  return %s" %
                  chebystr(i) for i in range(CHEBY_RANGE)])
chebyfactory = """
from numba import njit\n
import numpy as np\n
@njit(cache=True)\n
def chebychev(order, x):\n %s\n else:\n  print('order not implemented')
""" % rows

pth = os.path.dirname(__file__)
cache_path = os.path.join(pth, '__pycache__', 'chebychev.py')

try:
    cache_file = open(cache_path, "r")
    content = cache_file.read()
    cache_file.close()
    assert content == chebyfactory
except:
    cache_file = open(cache_path, "w")
    cache_file.write(chebyfactory)
    cache_file.close()
