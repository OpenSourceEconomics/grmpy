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

    return data
