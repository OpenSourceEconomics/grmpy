"""The module provides an estimation process given the simulated data set and the initialization
file."""
import os

from scipy.optimize import minimize
import pandas as pd

from grmpy.estimate.estimate_auxiliary import distribute_parameters
from grmpy.estimate.estimate_auxiliary import minimizing_interface
from grmpy.estimate.estimate_auxiliary import calculate_criteria
from grmpy.estimate.estimate_auxiliary import write_descriptives
from grmpy.estimate.estimate_auxiliary import optimizer_options
from grmpy.estimate.estimate_auxiliary import print_logfile
from grmpy.estimate.estimate_auxiliary import start_values
from grmpy.read.read import read


def estimate(init_file, option, optimizer):
    """The function estimates the coefficients of the simulated data set."""
    # Import init file as dictionary
    assert os.path.isfile(init_file)
    dict_ = read(init_file)

    data_file = dict_['ESTIMATION']['file']
    assert os.path.isfile(data_file)

    # Read data frame
    data = pd.read_table(data_file, delim_whitespace=True, header=0)

    # define starting values
    x0 = start_values(dict_, data, option)
    opts, method = optimizer_options(dict_, optimizer)
    dict_['AUX']['criteria'] = calculate_criteria(x0, dict_, data)
    if opts['maxiter'] == 0:
        rslt = distribute_parameters(x0, dict_)
        fun, success, status = calculate_criteria(x0, dict_, data), False, 2
        message, nfev = '---', 0
    else:
        opt_rslt = minimize(
            minimizing_interface, x0, args=(data, dict_), method=method, options=opts)
        x_rslt, fun, success = opt_rslt['x'], opt_rslt['fun'], opt_rslt['success']
        status, nfev, message = opt_rslt['status'], opt_rslt['nfev'], opt_rslt['message']
        rslt = distribute_parameters(x_rslt, dict_)
    rslt['fval'], rslt['success'], rslt['status'] = fun, success, status
    rslt['message'], rslt['nfev'], rslt['crit'] = message, nfev, fun

    # Print Output files
    print_logfile(rslt, dict_)
    write_descriptives(data, rslt, dict_)

    return rslt
