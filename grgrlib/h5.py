# -*- coding: utf-8 -*-

import os
import h5py
import numpy as np


def init(path, dataset_name, data, compression='gzip'):
    """Initialize a dataset. Assumes `data` to be a numpy array.
    """

    f = h5py.File(path, 'a')
    f.create_dataset(dataset_name, data=np.array(data)[
                     np.newaxis], compression=compression, chunks=True, maxshape=(None, *np.shape(data)))
    f.close()

    return


def append(path_or_h5py, dataset_names, data):
    """Append to a dataset. Batch append takes care that the append to each dataset has the same index!
    """

    if isinstance(path_or_h5py, str):
        f = h5py.File(path_or_h5py, 'a')
    else:
        f = path_or_h5py

    if isinstance(dataset_names, str):
        dataset_names = dataset_names,
        data = data,

    # batch append assumes that all datasets have the same lenght!
    # if this is irrelevant, just call append independently for each dataset
    oldlenght = np.shape(f[dataset_names[0]])[0]
    for i, name in enumerate(dataset_names):
        f[name].resize(oldlenght + 1, axis=0)
        f[name][oldlenght] = data[i]

    f.close()

    return


def write_at(path_or_h5py, dataset_names, index, data):
    """Append to a dataset. Batch append takes care that the append to each dataset has the same index!
    """

    if isinstance(path_or_h5py, str):
        f = h5py.File(path_or_h5py, 'a')
    else:
        f = path_or_h5py

    if isinstance(dataset_names, str):
        dataset_names = dataset_names,
        data = data,

    for i, name in enumerate(dataset_names):
        f[name][index] = data[i]

    f.close()

    return


def read(path, dataset_name):
    """Get content from specific dataset.
    """

    f = h5py.File(path, 'r')
    res = f[dataset_name][:]
    f.close()

    return res


def rm(path):
    """Remove backend.
    """

    os.remove(path)

    return
