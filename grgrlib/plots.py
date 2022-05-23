#!/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm, SymLogNorm


def grplot(
    X,
    xscale=None,
    labels=None,
    title="",
    styles=None,
    colors=None,
    legend=None,
    bulk_plot=False,
    ax=None,
    fig=None,
    figsize=None,
    nlocbins=None,
    sigma=0.05,
    alpha=None,
    tol=1e-8,
    stat=np.nanmedian,
    **plotargs
):

    if not isinstance(X, tuple):
        # make it a tuple
        X = (X,)

    # use first object in X to get some infos
    X0 = np.array(X[0])

    if xscale is None:
        if isinstance(X[0], pd.DataFrame):
            xscale = X[0].index
        elif X0.ndim > 1:
            xscale = np.arange(X[0].shape[-2])
        else:
            xscale = np.arange(len(X0))
    elif isinstance(xscale, tuple):
        xscale = np.arange(xscale[0], xscale[0] +
                           X0.shape[-2] * xscale[1], xscale[1])

    if labels is None:
        if isinstance(X[0], pd.DataFrame):
            labels = np.array(X[0].keys())
        elif X0.shape[-1] > 1 and X0.ndim > 1:
            labels = np.arange(X0.shape[-1]) + 1
        else:
            labels = np.array([None])
    else:
        labels = np.ascontiguousarray(labels)

    if styles is None:
        styles = "-"

    if not isinstance(styles, tuple):
        styles = np.repeat(styles, len(X))

    if nlocbins is None:
        nlocbins = "auto"

    # yet we can not be sure about the number of dimensions
    if X0.ndim == 1:
        selector = False
    else:
        selector = np.zeros(np.shape(X[0])[-1], dtype=bool)

    X_list = []
    for x_raw in X:

        x = np.array(x_raw)

        # X.shape = (no of time series, len of x axis (e.g. time), no of different objects (e.g. states))
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
                interval = np.nanpercentile(
                    x, [sigma * 100 / 2, (1 - sigma / 2) * 100], axis=0
                )
                line = stat(x, axis=0)
            else:
                bulk = x

        # check if there are states that are always zero
        if line is not None:
            selector += np.nanstd(line, 0) > tol
        if interval is not None:
            selector += np.nanstd(interval[0], 0) > tol
            selector += np.nanstd(interval[1], 0) > tol
        if bulk is not None:
            selector[:] = 1

        X_list.append((line, interval, bulk))

    colors = colors or [None] * len(X_list)
    if isinstance(colors, str):
        colors = (colors,)
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

            no_rows -= 4 * (i + 1) - no_states > 1
            no_cols -= 4 * (i + 1) - no_states > 2

            if figsize is None:
                figsize_loc = (no_cols * 4, no_rows * 3)
            else:
                figsize_loc = figsize

            fig, ax_of4 = plt.subplots(no_rows, no_cols, figsize=figsize_loc)
            ax_flat = np.array(ax_of4).flatten()

            # assume we also want two cols per plot
            for j in range(no_rows * no_cols):

                if 4 * i + j >= no_states:
                    ax_flat[j].set_visible(False)
                ax.append(ax_flat[j])

            if title:
                if plt_no > 1:
                    plt.suptitle("%s %s" % (title, i + 1))
                else:
                    plt.suptitle("%s" % (title))
            figs.append(fig)
    else:
        try:
            len(ax)
        except TypeError:
            ax = (ax,)
        [axis.set_prop_cycle(None) for axis in ax]
        figs = fig or None

    if not isinstance(xscale, pd.DatetimeIndex):
        locator = MaxNLocator(nbins=nlocbins, steps=[1, 2, 4, 8, 10])

    handles = []
    for obj_no, obj in enumerate(X_list):

        if legend is not None:
            legend_tag = np.ascontiguousarray(legend)[obj_no]
        else:
            legend_tag = None

        subhandles = []
        line, interval, bulk = obj
        # ax is a list of all the subplots
        for i in range(no_states):

            if line is not None:
                lalpha = alpha if (
                    interval is None and len(X_list) == 1) else 1
                lline = ax[i].plot(
                    xscale,
                    line[:, selector][:, i],
                    styles[obj_no],
                    color=colors[obj_no],
                    label=legend_tag,
                    alpha=lalpha,
                    **plotargs
                )
                subhandles.append(lline)

            if interval is not None:

                label = legend_tag if line is None else None
                color = lline[-1].get_color() if line is not None else colors[obj_no]

                if color:
                    ax[i].fill_between(
                        xscale,
                        *interval[:, :, selector][:, :, i],
                        lw=0,
                        color=color,
                        alpha=alpha or 0.3,
                        label=label,
                        **plotargs
                    )
                else:
                    ax[i].fill_between(
                        xscale,
                        *interval[:, :, selector][:, :, i],
                        lw=0,
                        alpha=alpha or 0.3,
                        label=label,
                        **plotargs
                    )

            elif bulk is not None:
                color = colors[obj_no] or "maroon"

                ax[i].plot(
                    xscale, bulk[..., i].swapaxes(0, 1), c=color, alpha=alpha or 0.05
                )
            ax[i].tick_params(axis="both", which="both",
                              top=False, right=False)
            if not isinstance(xscale, pd.DatetimeIndex):
                ax[i].xaxis.set_major_locator(locator)

        handles.append(subhandles)

    if figs is not None:
        [fig.autofmt_xdate() for fig in figs]

    for i in range(no_states):
        ax[i].set_title(labels[selector][i])

    # the notebook `inline` backend does not like `tight_layout`. But better don't use it...
    # shell = get_ipython().__class__.__name__
    # if not shell == 'ZMQInteractiveShell' and figs is not None:
    # [fig.tight_layout() for fig in figs]

    if figs is not None:
        [fig.tight_layout() for fig in figs]

    return figs, ax, handles


def bifplot(
    y,
    X=None,
    plot_dots=None,
    ax=None,
    markersize=0.02,
    color="k",
    ylabel=None,
    xlabel=None,
):
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
        ax.plot(y, X, ".", color=color, markersize=markersize)
    else:
        ax.plot(y, X, "o", color=color)

    ax.set_xlim(np.min(y), np.max(y))
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    if fig is not None:
        fig.tight_layout()

    return fig, ax


def figurator(nrows=2, ncols=2, nfigs=1, tight_layout=True, format_date=True, **args):
    """Create list of figures and axes with (potentially) more than one graph

    Parameters
    ----------
    nrows : int, optional
        Number of rows per figure, defaults to 2
    ncols : int, optional
        Number of cols per figure, defaults to 2
    nfigs : int, optional
        Number of figures, defaults to 1
    args : keyword arguments, optional
        keyword arguments that will be forwarded to `matplotlib.pyplot.subplots`

    Returns
    -------
    fig, ax : list, list
        A tuple of two lists: the first list are all figure handlers, the second is a list of all the axis
    """

    fax = [plt.subplots(nrows, ncols, **args) for _ in range(nfigs)]
    axs = np.array([f[1] for f in fax]).flatten()
    figs = [f[0] for f in fax]

    if format_date:
        [fig.autofmt_xdate() for fig in figs]

    if tight_layout:
        [fig.tight_layout() for fig in figs]

    return figs, axs


def axformater(ax, mode="rotate"):
    """Rotate ax as in `autofmt_xdate`"""

    if mode == "rotate":
        return plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    elif mode == "off":
        return ax.set_axis_off()
    else:
        raise NotImplementedError("No such modus: %s" % mode)


def save_png2pdf(fig, path, **args):
    """Save as a .png and use unix `convert` to convert to PDF."""

    if not path:
        print("[save_png2pdf:]".ljust(15, " ") +
              " No path provided, I'll pass...")
        return

    import os

    fig.savefig(path + ".png", **args)
    os.system("convert %s.png %s.pdf" % (path, path))

    return


def spy(M, ax=None, cmap="inferno"):
    """Visualize a matrix nicely"""
    M = np.array(M)
    s0, s1 = M.shape
    fig_exists = False

    if ax is None:
        fig_exists = True
        frc = max(min(s0 / s1, 2), 0.5)
        fig, ax = plt.subplots(1, 1, figsize=(5 + 2 / frc, frc * 5 + 2))

    ax.imshow(np.log10(1e-15 + np.abs(M)), cmap=cmap)

    if fig_exists:
        fig.tight_layout()
        return fig, ax

    return ax


def wunstify(figs, axs):

    for ax in axs:
        ax.axis("off")

    for fig in figs:
        fig.tight_layout()

    return


def grheat(
    X,
    bounds,
    xlabel=None,
    ylabel=None,
    zlabel=None,
    ax=None,
    draw_colorbar=None,
    cmap=None,
):
    """Simple interface to a heatmap (uses matplotlib's `imshow`).

    Parameters
    ----------
    X : numpy.array
        a matrix-like object
    bounds : float or tuple
        the bounds of the grid. If a float, -/+ this value is taken as the bounds
    xlabel : str (optional)
    ylabel : str (optional)
    zlabel : str (optional)
    """

    draw_colorbar = True if draw_colorbar is None else draw_colorbar
    cmap = "hot" if cmap is None else cmap

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    if isinstance(bounds, tuple):
        if isinstance(bounds[0], tuple):
            extent = [
                *bounds[0],
                *bounds[1],
            ]
        else:
            extent = [
                bounds[0],
                bounds[1],
                bounds[0],
                bounds[1],
            ]
    elif isinstance(bounds, (float, int)):
        extent = [
            -bounds,
            bounds,
            -bounds,
            bounds,
        ]
    else:
        extent = bounds

    img = ax.imshow(X, cmap=cmap, extent=extent,
                    vmin=np.nanmin(X), vmax=np.nanmax(X))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)

    if draw_colorbar:
        clb = plt.colorbar(img, ax=ax)
        clb.set_label(zlabel)

    if fig is not None:
        fig.tight_layout()

    return fig, ax, img


def grhist2d(x, y=None, bins=10, ax=None, alpha=None):

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

    if y is None:
        x, y = x

    hist, xedges, yedges = np.histogram2d(x, y, bins=bins)

    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    dx = dy = 0.5 * np.ones_like(zpos)
    dz = hist.ravel()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', alpha=alpha)

    return ax, (xedges, yedges)


def grbar3d(x, bounds=None, xedges=None, yedges=None, width=1, depth=1, ax=None, figsize=None, **kwargs):

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d')

    xedges = np.linspace(*bounds[0], x.shape[0]) if xedges is None else xedges
    yedges = np.linspace(*bounds[1], x.shape[1]) if yedges is None else yedges

    # xpos, ypos = np.meshgrid(xedges, yedges)
    xpos, ypos = np.meshgrid(xedges, yedges, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(x.ravel())

    dx = dy = 0.5 * np.ones_like(zpos)
    dz = x.ravel()

    ax.bar3d(xpos, ypos, zpos, width, depth, dz, shade=True, **kwargs)

    return ax, (xedges, yedges)


pplot = grplot
