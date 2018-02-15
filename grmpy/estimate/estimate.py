"""The module provides an estimation process given the simulated data set and the initialization
file."""
import os

from scipy.optimize import minimize
import pandas as pd
import numpy as np

from grmpy.estimate.estimate_auxiliary import adjust_output_maxiter_zero
from grmpy.estimate.estimate_auxiliary import minimizing_interface
from grmpy.estimate.estimate_auxiliary import calculate_criteria
from grmpy.estimate.estimate_auxiliary import optimizer_options
from grmpy.estimate.estimate_auxiliary import write_comparison
from grmpy.estimate.estimate_auxiliary import adjust_output
from grmpy.estimate.estimate_auxiliary import print_logfile
from grmpy.estimate.estimate_auxiliary import start_values
from grmpy.estimate.estimate_auxiliary import bfgs_dict
from grmpy.check.check import check_initialization_dict
from grmpy.check.check import check_init_file
from grmpy.check.check import UserError
from grmpy.read.read import read


def estimate(init_file):
    """The function estimates the coefficients of the simulated data set."""
    # Import init file as dictionary
    if not os.path.isfile(init_file):
        msg = '{}: There is no such file or directory.'.format(init_file)
        raise UserError(msg)

    dict_ = read(init_file)
    np.random.seed(dict_['SIMULATION']['seed'])
    # Check if the initialization file specifications are appropriate for the estimation process

    # We perform some basic consistency checks regarding the user's request.
    check_initialization_dict(dict_)
    check_init_file(dict_)

    data_file = dict_['ESTIMATION']['file']
    if not os.path.isfile(data_file):
        msg = 'The data file specified in your initialization file doesn`t exist.'
        raise UserError(msg)

    # Start value option
    option = dict_['ESTIMATION']['start']

    # Read data frame
    data = pd.read_table(data_file, delim_whitespace=True, header=0)
    # define starting values
    x0 = start_values(dict_, data, option)
    opts, method = optimizer_options(dict_)
    dict_['AUX']['criteria'] = calculate_criteria(dict_, data, x0)
    if opts['maxiter'] == 0:
        rslt = adjust_output_maxiter_zero(dict_, x0)
        print(rslt['TREATED'])
    else:
        rslt_dict = bfgs_dict()
        opt_rslt = minimize(
            minimizing_interface, x0, args=(dict_, data, rslt_dict), method=method, options=opts)

        rslt = adjust_output(opt_rslt, dict_, opt_rslt['x'], rslt_dict)
    # Print Output files
    print_logfile(dict_, rslt)
    write_comparison(dict_, data, rslt)

    return rslt
