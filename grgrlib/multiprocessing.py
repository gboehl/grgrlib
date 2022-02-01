#!/bin/python
# -*- coding: utf-8 -*-
import dill
from sys import platform


def serializer_unix(func):
    """Dirty hack that transforms the non-serializable function to a serializable one (when using dill)
    ...
    Don't try that at home!
    """

    fname = func.__name__
    exec("dark_%s = func" % fname, locals(), globals())

    def vodoo(*args, **kwargs):
        return eval("dark_%s(*args, **kwargs)" % fname)

    return vodoo


def serializer(*functions):
    """Serialize functions to use multiprocessing"""

    if platform == "darwin" or platform == "linux":
        rtn = []
        for func in functions:
            rtn.append(serializer_unix(func))
    else:
        fstr = dill.dumps(functions, recurse=True)
        rtn = dill.loads(fstr)
    if len(functions) == 1:
        return rtn[0]
    else:
        return rtn


class JoblibPoolDummy(object):
    """joblib Parallel workers pool pretending to behave like a multiprocessing pool"""

    def __init__(self, func=None, njobs=None, **kwargs):

        from joblib import Parallel, delayed
        import multiprocessing

        if njobs is None:
            self.njobs = multiprocessing.cpu_count()
        else:
            self.njobs = njobs

        self.func = func
        self.parallel = Parallel(n_jobs=self.njobs, **kwargs)
        self.parallel.__enter__()

    def map(self, func, iterable):
        return self.parallel(map(delayed(func), iterable))

    def vectorized_func(self, iterable):
        return self.parallel(delayed(self.func)(val) for val in list(iterable))

    def close(self):
        self.parallel.__exit__(None, None, None)
