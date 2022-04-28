#!/bin/python
# -*- coding: utf-8 -*-

import time
import os
import sys
import numpy as np
import numpy.linalg as nl
import scipy.linalg as sl
import scipy.stats as ss
from .linalg import ouc


def map2arr(iterator, return_np_array=True, check_nones=True):
    """Function to cast result from `map` to a tuple of stacked results

    By default, this returns numpy arrays. Automatically checks if the map object is a tuple, and if not, just one object is returned (instead of a tuple). Be warned, this does not work if the result of interest of the mapped function is a single tuple.

    Parameters
    ----------
    iterator : iter
        the iterator returning from `map`

    Returns
    -------
    numpy array (optional: list)
    """

    res = ()
    mode = 0

    for obj in iterator:

        if check_nones and obj is None:
            continue

        if not mode:
            if isinstance(obj, tuple):
                for entry in obj:
                    res = res + ([entry],)
                mode = 1
            else:
                res = [obj]
                mode = 2

        else:
            if mode == 1:
                for no, entry in enumerate(obj):
                    res[no].append(entry)
            else:
                res.append(obj)

    if return_np_array:
        if mode == 1:
            res = tuple(np.array(tupo) for tupo in res)
        else:
            res = np.array(res)

    return res


def napper(cond, interval=0.1):

    import time

    start_time = time.time()

    while not cond():

        elt = round(time.time() - start_time, 3)
        print("Zzzz... " + str(elt) + "s", end="\r", flush=True)
        time.sleep(interval)

    print("Zzzz... " + str(elt) + "s.")


def timeprint(s, round_to=5, full=False):
    """Print time. Needs urgent overhault.
    """

    if s < 60:
        if full:
            return str(np.round(s, round_to)) + " seconds"
        return str(np.round(s, round_to)) + "s"

    m, s = divmod(s, 60)

    if m < 60:
        if full:
            return "%s minutes, %s seconds" % (int(m), int(s))
        return "%sm%ss" % (int(m), int(s))

    h, m = divmod(m, 60)

    if full:
        return "%s hours, %s minutes, %s seconds" % (int(h), int(m), int(s))
    return "%sh%sm%ss" % (int(h), int(m), int(s))


def print_dict(d):

    for k in d.keys():
        print(str(k) + ":", d[k])

    return 0


def parse_yaml(mfile):
    """parse from yaml file"""
    import yaml

    f = open(mfile)
    mtxt = f.read()
    f.close()

    # get dict
    return yaml.safe_load(mtxt)


def load_as_module(path, add_to_path=True):

    if add_to_path:
        directory = os.path.dirname(path)
        sys.path.append(directory)

    # necessary for first option:
    # import importlib.machinery
    # import importlib.util

    # modname = os.path.splitext(os.path.basename(path))[0]
    # loader = importlib.machinery.SourceFileLoader(modname, path)
    # spec = importlib.util.spec_from_loader(modname, loader)
    # module = importlib.util.module_from_spec(spec)
    # loader.exec_module(module)

    # slighly different:
    import importlib.util as iu

    modname = os.path.splitext(os.path.basename(path))[0]
    spec = iu.spec_from_file_location(modname, path)
    module = iu.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# aliases
map2list = map2arr
indof = np.searchsorted
