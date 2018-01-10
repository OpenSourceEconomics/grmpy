# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 12:54:45 2018

@author: master
"""
import numpy as np
import matplotlib.pyplot as plt

from grmpy.simulate.simulate_auxiliary import simulate_covariates
from grmpy.simulate.simulate_auxiliary import construct_covariance_matrix
from grmpy.simulate.simulate_auxiliary import mte_information
from grmpy.read.read import read

from bld.project_paths import project_paths_join as ppj

filename=ppj("IN_FIGURES", "tutorial.grmpy.ini")

GRID = np.linspace(0.01, 0.99, num=99, endpoint=True)
init_dict = read(filename)

def save_data(sample):
    sample.tofile(ppj("OUT_DATA", "X.csv"), sep=",")

def plot_marginal_treatment_effect(pres, abs_):
    ax = plt.figure().add_subplot(111)
    
    ax.set_ylabel(r"$B^{MTE}$")
    ax.set_xlabel("$u_S$")
    ax.plot(GRID, pres, label='Presence')
    ax.plot(GRID, abs_, label='Absence', linestyle = '--')
    
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(ppj("OUT_FIGURES", 'fig-eh-marginal-effect.png'))
    
if __name__ == '__main__':
    coeffs_untreated = init_dict['UNTREATED']['all']
    coeffs_treated = init_dict['TREATED']['all']
    cov = construct_covariance_matrix(init_dict)
    x = simulate_covariates(init_dict, 'TREATED')
    save_data(x)
    
    MTE_pres = mte_information(coeffs_treated, coeffs_untreated, cov, GRID, x)
    
    para_diff = coeffs_treated - coeffs_untreated
    
    MTE_abs = []
    for i in GRID:
        if cov[2, 2] == 0.00:
            MTE_abs += ['---']
        else:
            MTE_abs += [
                np.mean(np.dot(para_diff, x.T))]
            
    plot_marginal_treatment_effect(MTE_pres, MTE_abs)