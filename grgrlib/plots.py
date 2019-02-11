#!/bin/python2
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from .stuff import fast0


def pplot(X, yscale=None, labels=None, title='', style='-', legend=None, ax=None, figsize=None, sigma=0.05, alpha=0.3):

    if not isinstance(X, tuple):
        # make it a tuple
        X = X,

    if yscale is None:
        yscale = np.arange(X[0].shape[-2])
    elif isinstance(yscale, tuple):
        yscale = np.arange(yscale[0], yscale[0] +
                           X[0].shape[-2]*yscale[1], yscale[1])

    if labels is None:
        labels  = np.arange(X[0].shape[1]) + 1
    else:
        labels  = np.array(labels)

    # yet we can not be sure about the number of dimensions
    selector = np.zeros(X[0].shape[-1], dtype=bool)

    X_list = []
    for x in X:
        # X.shape[0] is the number of time series
        # X.shape[1] is the len of the x axis (e.g. time)
        # X.shape[2] is the no of different objects (e.g. states)
        if x.ndim == 2:
            # be sure that X has 3 dimensions
            x = x.reshape(1, *x.shape)

        line = None
        interval = None

        if x.shape[0] == 1:
            line = x[0]
        if x.shape[0] == 2:
            interval = x
        if x.shape[0] == 3:
            line = x[1]
            interval = x[[0, 2]]
        if x.shape[0] > 3:
            interval = np.percentile(
                x, [sigma*100/2, (1 - sigma/2)*100], axis=0)
            line = np.median(x, axis=0)

        # check if there are states that are always zero
        if line is not None:
            selector += ~fast0(line, 0)
        if interval is not None:
            selector += ~fast0(interval[0], 0)
            selector += ~fast0(interval[1], 0)

        X_list.append((line, interval))

    no_states = sum(selector)

    # first create axes as an iterateble if it does not exist
    if ax is None:
        ax = []
        figs = []
        rest = no_states % 4
        plt_no = no_states // 4 + bool(rest)
        # assume we want to have two rows per plot
        no_rows = 2
        for i in range(plt_no):

            if 4*(i+1) - no_states > 1:
                no_rows -= 1

            if figsize is None:
                figsize_loc = (8, no_rows*3)

            fig, ax_of4 = plt.subplots(no_rows, 2, figsize=figsize_loc)
            ax_flat = ax_of4.flatten()

            # assume we also want two cols per plot
            for j in range(no_rows*2):

                if 4*i+j >= no_states:
                    ax_flat[j].set_visible(False)
                else:
                    ax.append(ax_flat[j])

            if title:
                if plt_no > 1:
                    plt.suptitle('%s %s' % (title, i+1), fontsize=16)
                else:
                    plt.suptitle('%s' % (title), fontsize=16)
            figs.append(fig)
    else:
        [ axis.set_prop_cycle(None) for axis in ax ]
        figs = None

    for obj_no, obj in enumerate(X_list):

        if legend is not None:
            legend_tag = legend[obj_no]
        else:
            legend_tag = None

        line, interval = obj
        # ax is a list of all the subplots
        for i in range(no_states):

            if line is not None:
                ax[i].plot(yscale, line[:, selector][:, i],
                           style, lw=2, label=legend_tag)
            if interval is not None:
                ax[i].fill_between(
                    yscale, *interval[:, :, selector][:, :, i], lw=0, alpha=alpha, label=legend_tag)

            ax[i].tick_params(axis='both', which='both',
                              top=False, right=False, labelsize=12)

            ax[i].set_xlabel(labels[selector][i], fontsize=14)

    if figs is not None:
        [fig.tight_layout() for fig in figs]

    return figs, ax
