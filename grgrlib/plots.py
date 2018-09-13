#!/bin/python2
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

def pplot(X, labels=None, yscale=None, title='', style='-', savepath=None, Y=None):

    plt_no      = X.shape[1] // 4 + bool(X.shape[1]%4)

    if yscale is None:
        yscale  = np.arange(X.shape[0])

    if labels is None:
        labels  = np.arange(X.shape[1]) + 1

    axs     = []
    for i in range(plt_no):

        ax  = plt.subplots(2,2)[1].flatten()

        for j in range(4):

            if 4*i+j >= X.shape[1]:
                ax[j].set_visible(False)

            else:
                if X.shape[1] > 4*i+j:
                    ax[j].plot(yscale, X[:,4*i+j], style, lw=2)

                if Y is not None:
                    if Y.shape[1] > 4*i+j:
                        ax[j].plot(yscale, Y[:,4*i+j], style, lw=2)

                ax[j].tick_params(axis='both', which='both', top=False, right=False, labelsize=12)
                ax[j].spines['top'].set_visible(False)
                ax[j].spines['right'].set_visible(False)
                ax[j].set_xlabel(labels[4*i+j], fontsize=14)

        if title:
            plt.suptitle('%s %s' %(title,i+1), fontsize=16)

        plt.tight_layout()

        if savepath is not None:
            plt.savefig(savepath+title+str(i+1)+'.pdf')

        axs.append(ax)

    return axs

