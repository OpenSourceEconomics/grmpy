"""This script replicates the estimation results from Cainero 2011 via the grmpy estimation method.
Additionally it returns a figure of the Marginal treatment effect based on the estimation results.
"""
import matplotlib.pyplot as plt
from os.path import join
from shutil import move
import pandas as pd
import numpy as np
import json
import os

from grmpy.estimate.estimate_auxiliary import calculate_mte
from grmpy.estimate.estimate import estimate

def plot_est_mte(rslt, data_frame):
    """This function calculates the marginal treatment effect for different quartiles of the
    unobservable V. ased on the calculation results."""

    # Define the Quantiles and read in the original results
    quantiles = [0.0001] + np.arange(0.01, 1., 0.01).tolist() + [0.9999]
    mte_original = json.load(open('mte_original.json', 'r'))

    #Calculate the MTE
    mte = calculate_mte(rslt, data_frame, quantiles)
    mte = [i/4 for i in mte]

    # Plot both curves
    ax = plt.figure().add_subplot(111)

    ax.set_ylabel(r"$B^{MTE}$")
    ax.set_xlabel("$u_S$")
    ax.plot(quantiles, mte, label='grmpy MTE')
    ax.plot(quantiles, mte_original, label='original MTE')
    ax.set_ylim([-0.4, 0.5])

    plt.legend()

    plt.tight_layout()
    plt.savefig('fig-marginal-benefit-parametric-replication.png')

if __name__ == '__main__':

    directory = os.path.dirname(os.path.realpath(__file__))
    target = os.path.split(os.path.split(os.path.split(directory)[0])[0])[0] + '/docs/figures'
    filename = 'fig-marginal-benefit-parametric-replication.png'

    # Estimate the coefficients
    rslt = estimate('replication.grmpy.ini')

    # Calculate and plot the marginal treatment effect
    data = pd.read_pickle('aer-replication-mock.pkl')
    plot_est_mte(rslt, data)

    # Move the plot to the documentation directory
    move(join(directory, filename), join(target, filename))
