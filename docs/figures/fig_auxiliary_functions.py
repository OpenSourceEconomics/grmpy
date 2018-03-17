"""This module contains supporting functions for the lecture."""
# TODO: For some reason this leads to a breakdown of the Python kernel
#import warnings
#warnings.simplefilter(action='ignore', category=[FutureWarning, UserWarning])

import linecache
import shlex

import matplotlib.pyplot as plt
import seaborn.apionly as sns
from scipy import stats
import numpy as np

from grmpy.test.random_init import print_dict
from grmpy.read.read import read


def get_effect_grmpy():
    """This function simply reads the average treatment effect from the output file of the
    package."""
    linecache.clearcache()
    line = linecache.getline('mc_data.grmpy.info', 25)
    stat = float(shlex.split(line)[1])
    return stat


def get_marginal_effect_grmpy(fname):
    """This function simply ready the marginal effect of treatment."""
    linecache.clearcache()
    stats = []
    for num in range(40, 60):
        line = linecache.getline(fname, num)
        stats += [float(shlex.split(line)[1])]
    return stats


def get_model_dict(fname):
    """This function returns the model definition."""
    model_dict = read(fname)
    return model_dict


def print_model_dict(model_dict, fname='mc_init'):
    """This function prints a model specification."""
    print_dict(model_dict, fname)


def plot_estimates(true, estimates):
    """This function plots the estimates from a pure randomization approach."""
    ax = plt.figure().add_subplot(111)

    grid = np.linspace(0.00, 0.99, len(estimates))
    true = np.tile(true, len(estimates))

    ax.set_xlim(0, 1)
    ax.set_ylim(0.4, 0.6)
    ax.set_ylabel(r"Effect")
    ax.set_xlabel(r"$\rho_{U_1, V}$")
    ax.plot(grid, estimates, label="Estimate")

    ax.plot(grid, true, label="True")

    ax.yaxis.get_major_ticks()[0].set_visible(False)

    plt.legend()

    plt.show()


def plot_distribution_of_benefits(df):
    """This function plots the distribution of benefits for a simulated dataset."""
    kernel = stats.gaussian_kde(df['Y1'] - df['Y0'])
    grid = np.linspace(0.25, 0.75, df.shape[0])
    density = kernel.evaluate(grid)

    ax = plt.figure().add_subplot(111)

    ax.set_ylim(density.min(), density.max() * 1.1)
    ax.set_xlim(grid.min(), grid.max())

    # Remove x and y ticks
    ax.set_xticks([0])
    ax.set_yticks([])

    ax.set_ylabel('Density')
    ax.set_xlabel('$Y_1 - Y_0$')

    ax.plot(grid, density, linestyle='-')

    plt.show()


def plot_effects(effects):
    """This function plots the effects of treatment."""
    effects = np.array(effects)

    ax = plt.figure().add_subplot(111)

    grid = np.linspace(0.00, 0.99, len(effects[:, 0]))

    ax.set_xlim(0, 1)
    ax.set_ylim(0.4, 0.6)
    ax.set_ylabel(r"Effect")
    ax.set_xlabel(r"$\rho_{U_1, V}$")

    ax.plot(grid, effects[:, 0], label=r"$ATE$")
    ax.plot(grid, effects[:, 1], label=r"$TT$")

    ax.yaxis.get_major_ticks()[0].set_visible(False)

    plt.legend()

    plt.show()


def plot_joint_distribution_unobservables(df):
    """This function plots the joint distribution of the relevant unobservables."""
    sns.jointplot(df['V'], df['U1'], stat_func=None).set_axis_labels('$V$', '$U_1$')


def plot_joint_distribution_potential(df):
    """This function plots the joint distribution of potential outcomes."""
    sns.jointplot(df['Y1'], df['Y0'], stat_func=None).set_axis_labels('$Y_1$', r'$Y_0$')


def plot_joint_distribution_benefits_surplus(model_dict, df):
    """This function plots the joint distribution of benefits and surplus."""
    coeffs_cost = model_dict['COST']['all']

    B = df['Y1'] - df['Y0']
    Z = df[['Z_0', 'Z_1']]

    C = np.dot(coeffs_cost, Z.T) + df['UC']
    S = B - C
    sns.jointplot(S, B, stat_func=None).set_axis_labels('$S$', r'$B$')


def plot_marginal_effect(parameter):
    """This function plots the marginal effect of treatment."""
    ax = plt.figure().add_subplot(111)

    ax.set_xlim(0, 1)
    ax.set_ylabel(r"$B^{MTE}$")
    ax.set_xlabel("$u_S$")

    if parameter.count(parameter[0]) == len(parameter):
        label = 'Absence'
    else:
        label = 'Presence'

    grid = np.linspace(0.01, 1, num=20, endpoint=True)
    ax.plot(grid, parameter, label=label)

    plt.legend()

    plt.show()