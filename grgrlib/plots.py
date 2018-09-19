#!/bin/python2
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

def pplot(X, yscale = None, labels = None, title = '', style = '-', Y = None, ax = None): 

    plt_no      = X.shape[1] // 4 + bool(X.shape[1]%4)

    if yscale is None:
        yscale  = np.arange(X.shape[0])

    if labels is None:
        labels  = np.arange(X.shape[1]) + 1

    if ax is None:
        axs      = []
        figs     = []
        for i in range(plt_no):

            fig, axis     = plt.subplots(2,2)
            axi     = axis.flatten()

            for j in range(4):

                if 4*i+j >= X.shape[1]:
                    axi[j].set_visible(False)

                else:
                    if X.shape[1] > 4*i+j:
                        axi[j].plot(yscale, X[:,4*i+j], style, lw=2)

                    if Y is not None:
                        if Y.shape[1] > 4*i+j:
                            axi[j].plot(yscale, Y[:,4*i+j], style, lw=2)

                    axi[j].tick_params(axis='both', which='both', top=False, right=False, labelsize=12)
                    axi[j].spines['top'].set_visible(False)
                    axi[j].spines['right'].set_visible(False)
                    axi[j].set_xlabel(labels[4*i+j], fontsize=14)

            if title:
                plt.suptitle('%s %s' %(title,i+1), fontsize=16)

            plt.tight_layout()
            axs.append(axi)
            figs.append(fig)

        return figs, axs
    else:
        for i, axi in enumerate(ax):
            axi.plot(yscale, X[:,i], style, lw=2)

            if Y is not None:
                axi.plot(yscale, Y[:,i], style, lw=2)

            axi.tick_params(axis='both', which='both', top=False, right=False, labelsize=12)
            axi.spines['top'].set_visible(False)
            axi.spines['right'].set_visible(False)
            axi.set_xlabel(labels[i], fontsize=14)
        return ax


