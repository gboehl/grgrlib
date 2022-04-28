#!/bin/python
# -*- coding: utf-8 -*-

import h5py
import numpy as np


def init(path, dataset_name, data, compression='gzip'):
    """Initialize a dataset. Assumes `data` to be a numpy array.
    """

    f = h5py.File(path, 'w')
    f.create_dataset(dataset_name, data=data[np.newaxis],
                     compression=compression, chunks=True, maxshape=(None, *data.shape))
    f.close()

    return


def append(path, dataset_name, data):
    """Append to a dataset. Only appends one new observation at a time!
    """

    f = h5py.File(path, 'a')
    oldlenght = f[dataset_name].shape[0]
    f[dataset_name].resize(oldlenght + 1, axis=0)
    f[dataset_name][oldlenght:] = data
    f.close()

    return


def read(path, dataset_name):
    """Get content from specific dataset.
    """

    f = h5py.File(path, 'r')
    res = f[dataset_name][:]
    f.close()

    return res
