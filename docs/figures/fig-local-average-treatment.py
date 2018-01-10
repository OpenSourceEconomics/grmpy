# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 21:25:29 2018

@author: master
"""

"""This module contains the code for a local average treatment graph.

"""
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm

from grmpy.simulate.simulate_auxiliary import simulate_covariates
from grmpy.simulate.simulate_auxiliary import construct_covariance_matrix
from grmpy.simulate.simulate_auxiliary import mte_information
from grmpy.read.read import read

from bld.project_paths import project_paths_join as ppj

filename=ppj("IN_FIGURES", "tutorial.grmpy.ini")

GRID = [i / 100 for i in range(1, 100, 1)]
init_dict = read(filename)



def plot_local_average_treatment(mte):
    ax = plt.figure().add_subplot(111)

    # Plot the mte
    ax.plot(GRID, mte)

    # Plot vertical lines and dots
    for xtick in [0.2, 0.3, 0.4, 0.6, 0.7, 0.8]:
        index = GRID.index(xtick)
        height = mte[index]
        ax.plot((xtick, xtick), (0, height), color='grey', alpha=0.7)
        endpoints = [(xtick, xtick), (0, height)]
        if xtick == 0.3:
            ax.scatter(*endpoints, color='black')
            ax.annotate(
                '$LATE(p_2, p_1)$', xy=(xtick - 0.005, height + 0.1),
                xytext=(xtick - 0.1, height + 0.5),
                arrowprops=dict(facecolor='black'))
            ax.annotate(
                '$u_S(p_2, p_1)$', xy=(xtick, 0),
                xytext=(xtick - 0.05, -0.15), bbox=dict(facecolor='white'))
        elif xtick == 0.7:
            ax.scatter(*endpoints, color='black')
            ax.annotate(
                '$LATE(p_4, p_3)$', xy=(xtick - 0.005, height + 0.1),
                xytext=(xtick - 0.1, height + 0.5),
                arrowprops=dict(facecolor='black'))
            ax.annotate(
                '$u_S(p_4, p_3)$', xy=(xtick, 0),
                xytext=(xtick - 0.05, -0.15), bbox=dict(facecolor='white'))

    # Set squared braces
    for xtick in [0.195, 0.594]:
        index = GRID.index(round(xtick, 2))
        height = mte[index]
        ax.text(x=xtick, y=-0.08, s='[', fontsize=30)
        ax.text(x=xtick, y=height - 0.15, s='[', fontsize=30)
    for xtick in [0.39, 0.79]:
        index = GRID.index(round(xtick, 2))
        height = mte[index]
        ax.text(x=xtick, y=-0.08, s=']', fontsize=30)
        ax.text(x=xtick, y=height - 0.15, s=']', fontsize=30)

    ax.set_xlabel('$u_S$')
    ax.set_ylabel(r'$B^{MTE}$')

    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xticklabels([0, '$p_1$', '$p_2$', '$p_3$', '$p_4$', 1])
    ax.tick_params(axis='x', which='major', pad=15)
    ax.set_yticks([])
    ax.set_ylim([1, 4.5])

    plt.tight_layout()
    plt.savefig(ppj("OUT_FIGURES", 'fig-local-average-treatment.png'))


if __name__ == '__main__':
    
    coeffs_untreated = init_dict['UNTREATED']['all']
    coeffs_treated = init_dict['TREATED']['all']
    cov = construct_covariance_matrix(init_dict)
    x = np.loadtxt(ppj("OUT_DATA", "X.csv"), delimiter=",")
    x = x.reshape(165, 2)
    

    mte = mte_information(coeffs_treated, coeffs_untreated, cov, GRID, x) 
    
    plot_local_average_treatment(mte)