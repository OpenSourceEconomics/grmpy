"""The module provides an estimation process given the simulated data set and the initialization
file."""
import os

from scipy.optimize import minimize
from scipy.stats import norm
import pandas as pd
import numpy as np

from grmpy.estimate.estimate_auxiliary import adjust_output_maxiter_zero
from grmpy.estimate.estimate_auxiliary import distribute_parameters
from grmpy.estimate.estimate_auxiliary import _prepare_arguments
from grmpy.estimate.estimate_auxiliary import calculate_criteria
from grmpy.estimate.estimate_auxiliary import optimizer_options
from grmpy.estimate.estimate_auxiliary import adjust_output
from grmpy.estimate.estimate_auxiliary import start_values
from grmpy.estimate.estimate_auxiliary import bfgs_dict
from grmpy.read.read import read


def log_likelihood_old(data_frame, init_dict, rslt, dict_=None):
    """The function provides the loglikelihood function for the minimization process."""
    beta1, beta0, gamma, sd1, sd0, sdv, rho1v, rho0v, choice = \
        _prepare_arguments(init_dict, rslt)
    likl = np.tile(np.nan, data_frame.shape[0])

    for observation in range(data_frame.shape[0]):
        target = data_frame.loc[observation]
        X = target.filter(regex=r'^X\_')
        Z = target.filter(regex=r'^Z\_')
        g = pd.concat((X, Z))
        choice_ = np.dot(choice, g)
        if target['D'] == 1.00:
            beta, gamma, rho, sd, sdv = beta1, gamma, rho1v, sd1, sdv
        else:
            beta, gamma, rho, sd, sdv = beta0, gamma, rho0v, sd0, sdv
        part1 = (target['Y'] - np.dot(beta, X.T)) / sd
        part2 = (choice_ - rho * sdv * part1) / (np.sqrt((1 - rho ** 2) * sdv ** 2))

        dist_1, dist_2 = norm.pdf(part1), norm.cdf(part2)

        if target['D'] == 1.00:
            contrib = (1.0 / sd) * dist_1 * dist_2
        else:
            contrib = (1.0 / sd) * dist_1 * (1.0 - dist_2)
        likl[observation] = contrib
    likl = - np.mean(np.log(np.clip(likl, 1e-20, np.inf)))
    if dict_ is None:
        pass
    else:
        dict_['crit'][str(len(dict_['crit']))] = likl

    return likl


def minimizing_interface_old(start_values, data_frame, init_dict, dict_):
    """The function provides the minimization interface for the estimation process."""
    # Collect arguments
    rslt = distribute_parameters(init_dict, start_values, dict_)

    # Calculate liklihood for pre specified arguments
    likl = log_likelihood_old(data_frame, init_dict, rslt, dict_)

    return likl


def estimate_old(init_file):
    """The function estimates the coefficients of the simulated data set."""
    # Import init file as dictionary
    assert os.path.isfile(init_file)
    dict_ = read(init_file)

    data_file = dict_['SIMULATION']['source'] + '.grmpy.txt'
    assert os.path.isfile(data_file)

    # Set Start option
    option = dict_['ESTIMATION']['start']

    # Read data frame
    data = pd.read_table(data_file, delim_whitespace=True, header=0)

    # define starting values
    x0 = start_values(dict_, data, option)
    opts, method = optimizer_options(dict_)
    dict_['AUX']['criteria'] = calculate_criteria(dict_, data, x0)
    if opts['maxiter'] == 0:
        rslt = adjust_output_maxiter_zero(dict_, x0)
    else:
        rslt_dict = bfgs_dict()
        opt_rslt = minimize(
            minimizing_interface_old, x0, args=(data, dict_, rslt_dict), method=method,
            options=opts)
        rslt = adjust_output(opt_rslt, dict_, opt_rslt['x'], rslt_dict)

    # Finishing
    return rslt

def calculate_criteria_old(init_dict, data_frame, start_values):
    """The function calculates the criteria function value."""
    rslt = distribute_parameters(init_dict, start_values)
    criteria = log_likelihood_old(data_frame, init_dict, rslt)
    return criteria
