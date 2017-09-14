"""The module provides the simulation process."""
import os

import numpy as np

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

    # Simulate observables of the model
    X = simulate_covariates(init_dict, 'TREATED', num_agents)
    Z = simulate_covariates(init_dict, 'COST', num_agents)

    # Simulate unobservables of the model
    U, V = simulate_unobservables(init_dict)

    # Simulate endogeneous variables of the model
    Y, D, Y_1, Y_0 = simulate_outcomes(init_dict, X, Z, U)

    # Write output file
    df = write_output([Y, D, Y_1, Y_0], [X, Z], [U, V], source)

    print_info(init_dict, df)

    return df