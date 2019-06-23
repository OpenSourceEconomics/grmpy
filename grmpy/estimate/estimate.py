"""The module provides an estimation process given the simulated data set and the
initialization file.
"""
from scipy.optimize import minimize
import numpy as np

from grmpy.estimate.estimate_auxiliary import backward_transformation
from grmpy.estimate.estimate_auxiliary import minimizing_interface
from grmpy.estimate.estimate_auxiliary import calculate_criteria
from grmpy.estimate.estimate_auxiliary import optimizer_options
from grmpy.check.check import check_presence_estimation_dataset
from grmpy.estimate.estimate_output import write_comparison
from grmpy.estimate.estimate_auxiliary import adjust_output
from grmpy.estimate.estimate_auxiliary import start_values
from grmpy.estimate.estimate_auxiliary import process_data
from grmpy.estimate.estimate_output import print_logfile
from grmpy.estimate.estimate_auxiliary import bfgs_dict
from grmpy.check.check import check_initialization_dict
from grmpy.check.check import check_presence_init
from grmpy.check.check import check_init_file
from grmpy.check.auxiliary import read_data
from grmpy.read.read import read


def fit(init_file):
    """The function estimates the coefficients of the simulated data set."""
    check_presence_init(init_file)

    dict_ = read(init_file)
    np.random.seed(dict_["SIMULATION"]["seed"])

    # We perform some basic consistency checks regarding the user's request.
    check_presence_estimation_dataset(dict_)
    check_initialization_dict(dict_)
    check_init_file(dict_)

    # Distribute initialization information.
    data = read_data(dict_["ESTIMATION"]["file"])
    num_treated = dict_["AUX"]["num_covars_treated"]
    num_untreated = num_treated + dict_["AUX"]["num_covars_untreated"]

    _, X1, X0, Z1, Z0, Y1, Y0 = process_data(data, dict_)

    if dict_["ESTIMATION"]["maxiter"] == 0:
        option = "init"
    else:
        option = dict_["ESTIMATION"]["start"]

    # Read data frame

    # define starting values
    x0 = start_values(dict_, data, option)
    opts, method = optimizer_options(dict_)
    dict_["AUX"]["criteria"] = calculate_criteria(dict_, X1, X0, Z1, Z0, Y1, Y0, x0)
    dict_["AUX"]["starting_values"] = backward_transformation(x0)
    rslt_dict = bfgs_dict()
    if opts["maxiter"] == 0:
        rslt = adjust_output(None, dict_, x0, X1, X0, Z1, Z0, Y1, Y0, rslt_dict)
    else:
        opt_rslt = minimize(
            minimizing_interface,
            x0,
            args=(dict_, X1, X0, Z1, Z0, Y1, Y0, num_treated, num_untreated, rslt_dict),
            method=method,
            options=opts,
        )
        rslt = adjust_output(
            opt_rslt, dict_, opt_rslt["x"], X1, X0, Z1, Z0, Y1, Y0, rslt_dict
        )
    # Print Output files
    print_logfile(dict_, rslt)

    if "comparison" in dict_["ESTIMATION"].keys():
        if dict_["ESTIMATION"]["comparison"] == 0:
            pass
        else:
            write_comparison(data, rslt)
    else:
        write_comparison(data, rslt)

    return rslt
