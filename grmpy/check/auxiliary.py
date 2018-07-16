"""This module provides several auxiliary functions for the check module."""
import pandas as pd
import numpy as np

from grmpy.simulate.simulate_auxiliary import construct_covariance_matrix


def is_pos_def(dict_):
    """The function tests if the specified covariance matrix is positive semi definite."""
    return np.all(np.linalg.eigvals(construct_covariance_matrix(dict_)) >= 0)


def read_data(data_file):
    """This function uses different data import methods which depend on the format that is specified
    in the initialization file."""

    if data_file[-4:] == '.pkl':
        data = pd.read_pickle(data_file)
    elif data_file[-4:] == '.txt':
        data = pd.read_table(data_file, delim_whitespace=True, header=0)
    elif data_file[-4:] == '.dta':
        data = pd.read_stata(data_file)
        data = data.drop(['index'], axis=1)
    return data


def check_special_conf(dict_):
    """This function ensures that an init file uses appropriate specifications for binary and
    categorical variables.
    """
    for key_ in ['TREATED', 'UNTREATED', 'CHOICE']:
        for x in dict_[key_]['types']:
            invalid = False
            msg = ' '
            if isinstance(x, list):
                # Check for binary variables
                str_ = 'The specified probability that a {} variable is equal to {} has to be ' \
                       'sufficiently lower than one.'
                if x[0] == 'binary':
                    if x[1] >= 0.9:
                        msg = str_.format(x[0], 'one')
                        invalid = True
                # Check for categorical variables
                elif x[0] == 'categorical':
                    if any(i >= 0.9 for i in x[2]):
                        msg = str_.format(x[0], 'a specific category')
                        invalid = True
                    elif not np.isclose(sum(x[2]), 1., 0.01):
                        msg = 'The specified probability for all possible categories of a ' \
                              'categorical variable have to sum up to 1.'
                        invalid = True

                return invalid, msg

    return invalid, ' '
