"""This module provides several auxiliary functions for the check module."""

import numpy as np
import pandas as pd

from grmpy.simulate.simulate_auxiliary import construct_covariance_matrix


def is_pos_def(dict_):
    """The function tests if the specified covariance matrix is positive semi
    definite.
    """
    return np.all(np.linalg.eigvals(construct_covariance_matrix(dict_)) >= 0)


def read_data(data_file):
    """This function uses different data import methods which depend on the format that
    is specified in the initialization file."""

    if data_file[-4:] == ".pkl":
        data = pd.read_pickle(data_file)
    elif data_file[-4:] == ".txt":
        data = pd.read_csv(data_file, delim_whitespace=True, header=0)
    elif data_file[-4:] == ".dta":
        data = pd.read_stata(data_file)
        data = data.drop(["index"], axis=1)
    return data


def check_special_conf(dict_):
    """This function ensures that an init file uses appropriate specifications for
    binary and categorical variables.
    """
    msg = (
        "The specified probability that a binary variable is equal to 1 has to be "
        "sufficiently lower than one."
    )

    for variable in dict_["VARTYPES"]:
        if isinstance(dict_["VARTYPES"][variable], list):
            if dict_["VARTYPES"][variable][1] >= 0.9:
                return True, msg

    return False, " "
