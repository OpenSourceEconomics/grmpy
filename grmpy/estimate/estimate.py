"""The module provides an estimation process given the simulated data set and the initialization
file."""
import os

from scipy.optimize import minimize
import pandas as pd

from grmpy.estimate.estimate_auxiliary import distribute_parameters
from grmpy.estimate.estimate_auxiliary import minimizing_interface
from grmpy.estimate.estimate_auxiliary import calculate_criteria
from grmpy.estimate.estimate_auxiliary import print_logfile
from grmpy.estimate.estimate_auxiliary import start_values
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
    opts = {'maxiter': dict_['ESTIMATION']['maxfun']}
    method = dict_['ESTIMATION']['optimizer'].split('-')[1]
    calculate_criteria(x0,dict_, data)

    opt_rslt = minimize(minimizing_interface, x0, args=(data, dict_), method=method, options=opts)
    x_rslt, fun = opt_rslt['x'], opt_rslt['fun']
    success, message = opt_rslt['success'], opt_rslt['message']
    status, nfev = opt_rslt['status'], opt_rslt['nfev']
    crit_opt = opt_rslt['fun']
    rslt = distribute_parameters(x_rslt, dict_)

    rslt['fval'], rslt['success'], rslt['status'] = fun, success, status
    rslt['message'], rslt['nfev'], rslt['crit'] = message, nfev, crit_opt
    print_logfile(rslt, dict_)


    # Finishing
    return rslt