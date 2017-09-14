"""The module provides the simulation process."""
import os

import numpy as np

from grmpy.simulate.simulate_auxiliary import construct_covariance_matrix
from grmpy.simulate.simulate_auxiliary import simulate_unobservables
from grmpy.simulate.simulate_auxiliary import simulate_covariates
from grmpy.simulate.simulate_auxiliary import simulate_outcomes
from grmpy.simulate.simulate_auxiliary import write_output
from grmpy.simulate.simulate_auxiliary import print_info
from grmpy.read.read import read


def simulate(init_file):
    """This function simulates a user-specified version of the generalized Roy model."""
    init_dict = read(init_file)

    # Distribute information
    num_agents = init_dict['SIMULATION']['agents']
    source = init_dict['SIMULATION']['source']
    seed = init_dict['SIMULATION']['seed']
    np.random.seed(seed)

    # Construct covariance matrix directly from the initialization file.
    cov = construct_covariance_matrix(init_dict)

    Y1_coeffs = init_dict['TREATED']['all']
    Y0_coeffs = init_dict['UNTREATED']['all']
    C_coeffs = init_dict['COST']['all']
    coeffs = [Y0_coeffs, Y1_coeffs, C_coeffs]
    Dist_coeffs = init_dict['DIST']['all']

    # Simulate observables
    X = simulate_covariates(init_dict, 'TREATED', num_agents)
    Z = simulate_covariates(init_dict, 'COST', num_agents)

    # Simulate unobservables
    U, V = simulate_unobservables(cov, num_agents)

    # Simulate endogeneous variables
    Y, D, Y_1, Y_0 = simulate_outcomes([X, Z], U, coeffs)

    # Write output file
    df = write_output([Y, D, Y_1, Y_0], [X, Z], [U, V], source)

    print_info(df, [Y0_coeffs, Y1_coeffs, C_coeffs, Dist_coeffs], source)

    return df