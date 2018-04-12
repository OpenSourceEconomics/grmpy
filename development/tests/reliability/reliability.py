"""This module contains a monte carlo example that illustrates the advantages of the grmpy
estimation strategy. Additionally the module creates four different figures for the reliability
section of the documentation.
"""
from grmpy.test.random_init import print_dict
from grmpy.estimate.estimate import estimate
from grmpy.simulate.simulate import simulate
from grmpy.test.auxiliary import cleanup
from grmpy.read.read import read

from statsmodels.sandbox.regression.gmm import IV2SLS
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import linecache
import shlex

def update_correlation_structure(model_dict, rho):
    """This function takes a valid model specification and updates the correlation structure
    among the unobservables."""

    # We first extract the baseline information from the model dictionary.
    sd_v = model_dict['DIST']['all'][-1]
    sd_u = model_dict['DIST']['all'][0]

    # Now we construct the implied covariance, which is relevant for the initialization file.
    cov = rho * sd_v * sd_u
    model_dict['DIST']['all'][2] = cov

    # We print out the specification to an initialization file with the name mc_init.grmpy.ini.
    print_dict(model_dict)

def get_effect_grmpy(dict_):
    """This function simply reads the average treatment effect from the output file of the
    package."""
    name = dict_['SIMULATION']['source']
    linecache.clearcache()
    line = linecache.getline('{}.grmpy.info'.format(name), 25)
    print(line)
    stat = float(shlex.split(line)[1])
    return stat


def monte_carlo(file, grid_points):
    """This function estimates the ATE for a sample with different correlation structures between U1
     and V. Four different strategies for , OLS, 2SLS, LATE and perfect randomization are implemented.
     """
    # Define a dictionary with a key for each estimation strategy
    effects = {}
    for key_ in ['random', 'grmpy', '2sls', 'ols']:
        effects[key_] = []

    # Loop over different correlations between V and U_1
    for rho in np.linspace(0.00, 0.99, grid_points):

        # Readjust the initialization file values to add correlation
        model_spec = read(file)
        update_correlation_structure(model_spec, rho)

        # Simulate a Data set and specify exogeneous and endogeneous variables
        df_mc = simulate(file)

        endog, exog, instr = df_mc['Y'], df_mc[['X_0', 'D']], df_mc[['X_0', 'X_1']]
        d_treated = df_mc['D'] == 1

        # Effect randomization
        stat = np.mean(endog.loc[d_treated]) - np.mean(endog.loc[~d_treated])
        effects['random'] += [stat]

        # Estimate  via grmpy
        rslt = estimate('test.grmpy.ini')
        stat = rslt['TREATED']['all'][0] - rslt['UNTREATED']['all'][0]
        effects['grmpy'] += [stat]

        # Estimate via 2SLS
        stat = IV2SLS(endog, exog, instr).fit().params[1]
        effects['2sls'] += [stat]

        # Estimate via OLS
        stat = sm.OLS(endog, exog).fit().params[1]
        effects['ols'] += [stat]

    return effects

def create_plots(effects, true):
    """The function creates the """

    # Determine the title for each strategy plot
    for strategy in ['random', 'grmpy', '2sls', 'ols']:
        if strategy == 'random':
            title = 'Perfect randomization'
        elif strategy == 'grmpy':
            title= 'Local Average Treatment Effect'
        elif strategy == '2sls':
            title = 'Instrumental Variable'
        elif strategy == 'ols':
            title = 'Ordinary Least Squares'

    # Create a figure for each estimation strategy
        ax = plt.figure().add_subplot(111)

        grid = np.linspace(0.00, 0.99, len(effects[strategy]))
        true_ = np.tile(true, len(effects[strategy]))

        ax.set_xlim(0, 1)
        ax.set_ylim(0.4, 0.6)
        ax.set_ylabel(r"Effect")
        ax.set_xlabel(r"$\rho_{U_1, V}$")
        ax.plot(grid, effects[strategy], label="Estimate")

        ax.plot(grid, true_, label="True")

        ax.yaxis.get_major_ticks()[0].set_visible(False)
        plt.title(title)
        plt.legend()
        file_name = 'fig_{}_average_effect_estimation.png'.format(strategy)
        plt.savefig(file_name)


x = monte_carlo('test.grmpy.ini', 10)
create_plots(x, 0.5)


