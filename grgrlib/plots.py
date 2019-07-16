#!/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm, SymLogNorm
from .stuff import fast0


def grplot(X, yscale=None, labels=None, title='', style=None, legend=None, bulk_plot=False, ax=None, figsize=None, nlocbins=None, sigma=0.05, alpha=0.3):

    if not isinstance(X, tuple):
        # make it a tuple
        X = X,

    if yscale is None:
        if X[0].ndim > 1:
            yscale = np.arange(X[0].shape[-2])
        else:
            yscale = np.arange(len(X[0]))
    elif isinstance(yscale, tuple):
        yscale = np.arange(yscale[0], yscale[0] +
                           X[0].shape[-2]*yscale[1], yscale[1])

    if labels is None:
        if X[0].shape[-1] > 1 and X[0].ndim > 1:
            labels = np.arange(X[0].shape[-1]) + 1
        else:
            labels = np.array([None])
    else:
        labels = np.array(labels)

    if style is None:
        style = '-'

    if not isinstance(style, tuple):
        style = np.repeat(style, len(X))

    if nlocbins is None:
        nlocbins = 'auto'

    # yet we can not be sure about the number of dimensions
    if X[0].ndim == 1:
        selector = False
    else:
        selector = np.zeros(X[0].shape[-1], dtype=bool)

    X_list = []
    for x in X:
        # X.shape[0] is the number of time series
        # X.shape[1] is the len of the x axis (e.g. time)
        # X.shape[2] is the no of different objects (e.g. states)
        if x.ndim == 1:
            # be sure that X has 3 dimensions
            x = x.reshape(1, len(x), 1)
        if x.ndim == 2:
            x = x.reshape(1, *x.shape)

        line = None
        interval = None
        bulk = None

        if x.shape[0] == 1:
            line = x[0]
        if x.shape[0] == 2:
            interval = x
        if x.shape[0] == 3:
            line = x[1]
            interval = x[[0, 2]]
        if x.shape[0] > 3:
            if not bulk_plot:
                interval = np.percentile(
                    x, [sigma*100/2, (1 - sigma/2)*100], axis=0)
                line = np.median(x, axis=0)
            else:
                bulk = x

        # check if there are states that are always zero
        if line is not None:
            selector += ~fast0(line, 0)
        if interval is not None:
            selector += ~fast0(interval[0], 0)
            selector += ~fast0(interval[1], 0)
        if bulk is not None:
            selector[:] = 1

        X_list.append((line, interval, bulk))

    no_states = sum(selector)

    # first create axes as an iterateble if it does not exist
    if ax is None:
        ax = []
        figs = []
        rest = no_states % 4
        plt_no = no_states // 4 + bool(rest)

        # assume we want to have two rows and cols per plot
        no_rows = 2
        no_cols = 2
        for i in range(plt_no):

            no_rows -= 4*(i+1) - no_states > 1
            no_cols -= 4*(i+1) - no_states > 2

            if figsize is None:
                figsize_loc = (no_cols*4, no_rows*3)
            else:
                figsize_loc = figsize

            fig, ax_of4 = plt.subplots(no_rows, no_cols, figsize=figsize_loc)
            ax_flat = np.array(ax_of4).flatten()

            # assume we also want two cols per plot
            for j in range(no_rows*no_cols):

                if 4*i+j >= no_states:
                    ax_flat[j].set_visible(False)
                else:
                    ax.append(ax_flat[j])

            if title:
                if plt_no > 1:
                    plt.suptitle('%s %s' % (title, i+1))
                else:
                    plt.suptitle('%s' % (title))
            figs.append(fig)
    else:
        [axis.set_prop_cycle(None) for axis in ax]
        figs = None

    locator = MaxNLocator(nbins=nlocbins, steps=[1, 2, 4, 8, 10])

    for obj_no, obj in enumerate(X_list):

        if legend is not None:
            legend_tag = legend[obj_no]
        else:
            legend_tag = None

        line, interval, bulk = obj
        # ax is a list of all the subplots
        for i in range(no_states):

            if line is not None:
                ax[i].plot(yscale, line[:, selector][:, i],
                           style[obj_no], lw=2, label=legend_tag)
            if interval is not None:
                ax[i].fill_between(
                    yscale, *interval[:, :, selector][:, :, i], lw=0, alpha=alpha, label=legend_tag if line is None else None)
            elif bulk is not None:
                if len(X_list) > 1:
                    color = 'C'+str(obj_no)
                else:
                    color = 'maroon'
                ax[i].plot(yscale, bulk[..., i].swapaxes(
                    0, 1), c=color, alpha=.04)
            ax[i].tick_params(axis='both', which='both',
                              top=False, right=False)
            ax[i].set_xlabel(labels[selector][i])
            ax[i].xaxis.set_major_locator(locator)

    if figs is not None:
        [fig.tight_layout() for fig in figs]

    return figs, ax


def bifplot(y, X=None, plot_dots=None, ax=None, color='k', ylabel=None, xlabel=None):
    """A bifurcation diagram

    (add further documentation)
    """

    if X is None:
        X = y
        y = np.arange(y.shape[0])

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    if plot_dots is None:
        if X.shape[0] > 50:
            plot_dots = False
        else:
            plot_dots = True

    if not plot_dots:
        ax.plot(y, X, '.', color=color, markersize=0.01)
    else:
        ax.plot(y, X, 'o', color=color)

    ax.set_xlim(np.min(y), np.max(y))
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    if fig is not None:
        fig.tight_layout()

    return fig, ax


def grheat(X, gridbounds, xlabel=None, ylabel=None, zlabel=None):
    """Simple interface to a heatmap (uses matplotlib's `imshow`).

    Parameters
    ----------
    X : numpy.array
        a matrix-like object 
    gridbounds : float or tuple
        the bounds of the grid. If a float, -/+ this value is taken as the bounds
    xlabel : str (optional)
    ylabel : str (optional)
    zlabel : str (optional)
    """

    fig, ax = plt.subplots()

    if isinstance(gridbounds, tuple):
        if isinstance(gridbounds[0], tuple):
            extent = [*gridbounds[0], *gridbounds[1], ]
        else:
            extent = [-gridbounds[0], gridbounds[0], -
                      gridbounds[1], gridbounds[1], ]
    else:
        extent = [-gridbounds, gridbounds, -gridbounds, gridbounds, ]

    plt.imshow(X, cmap="hot", extent=extent, vmin=np.nanmin(
        X), vmax=np.nanmax(X), norm=SymLogNorm(1, linscale=1))

    clb = plt.colorbar()

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(False)
    clb.set_label(zlabel)

    plt.tight_layout()


pplot = grplot
