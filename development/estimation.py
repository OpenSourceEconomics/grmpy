"""The module provides an estimation process given the simulated data set and the initialization
file."""
import os

from scipy.optimize import minimize
import pandas as pd

from development.estimation_auxiliary import minimizing_interface
from development.estimation_auxiliary import distribute_parameters
from development.estimation_auxiliary import start_values
from grmpy.read.read import read


def estimate(init_file, option):
    """The function estimates the coefficients of the simulated data set."""
    # Import init file as dictionary
    assert os.path.isfile(init_file)
    dict_ = read(init_file)

    data_file = dict_['SIMULATION']['source'] + '.grmpy.txt'
    assert os.path.isfile(data_file)

    # Read data frame
    data = pd.read_table(data_file, delim_whitespace=True, header=0)

    # define starting values
    x0 = start_values(dict_, data, option)
    opts = {'maxiter': 10}

    opt_rslt = minimize(minimizing_interface, x0, args=(data, dict_), method='BFGS', options=opts)
    x_rslt, fun = opt_rslt['x'], opt_rslt['fun']
    success = opt_rslt['success']

    rslt = distribute_parameters(x_rslt, dict_)

    rslt['fval'], rslt['success'] = fun, success

    # Finishing
    return rslt
