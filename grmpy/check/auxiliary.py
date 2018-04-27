"""This module provides several auxiliary functions for the check module."""
import numpy as np

from grmpy.simulate.simulate_auxiliary import construct_covariance_matrix


def is_pos_def(dict_):
    """The function tests if the specified covariance matrix is positive semi definite."""
    return np.all(np.linalg.eigvals(construct_covariance_matrix(dict_)) >= 0)
