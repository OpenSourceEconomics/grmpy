import numpy as np

from grmpy.simulation.simulation_auxiliary import simulate_unobservables
from grmpy.simulation.simulation_auxiliary import simulate_outcomes
from grmpy.simulation.simulation_auxiliary import write_output
from grmpy.simulation.simulation_auxiliary import print_info
from grmpy.read.read import read


def simulate(init_file):
    """ This function simulated a user-specified version of the generalized Roy model
    """

    # Transform init file to dictionary

    init_dict = read(init_file)
    # Distribute information
    is_deterministic = init_dict['DETERMINISTIC']
    num_agents = init_dict['SIMULATION']['agents']
    source = init_dict['SIMULATION']['source']

    Y1_coeffs = init_dict['TREATED']['all']
    Y0_coeffs = init_dict['UNTREATED']['all']
    C_coeffs = init_dict['COST']['all']
    coeffs = [Y0_coeffs, Y1_coeffs, C_coeffs]

    U0_sd, U1_sd, V_sd = init_dict['DIST']['all'][:3]
    vars_ = [U0_sd ** 2, U1_sd ** 2, V_sd ** 2]
    U01, U0_V, U1_V = init_dict['DIST']['all'][3:]
    covar_ = [U01**2, U0_V**2, U1_V**2]
    Dist_coeffs = init_dict['DIST']['all']

    num_covars_out = Y1_coeffs.shape[0]
    num_covars_cost = C_coeffs.shape[0]

    # Simulate observables
    means = np.tile(.0, num_covars_out)
    covs = np.identity(num_covars_out)
    X = np.random.multivariate_normal(means, covs, num_agents)

    means = np.tile(.0, num_covars_cost)
    covs = np.identity(num_covars_cost)
    Z = np.random.multivariate_normal(means, covs, num_agents)
    Z[:, 0], X[:, 0] = 1.0, 1.0

    # Simulate unobservables
    # Read information about the distribution and the specific means from the init dic

    U, V = simulate_unobservables(covar_, vars_, num_agents)

    # Simulate endogeneous variables

    Y, D, Y_1, Y_0 = simulate_outcomes([X, Z], U, coeffs)

    # Write output file
    df = write_output([Y, D, Y_1, Y_0], [X, Z], [U, V], source, is_deterministic)

    print_info(df, [Y0_coeffs, Y1_coeffs, C_coeffs, Dist_coeffs], source)

    return df, Y, Y_1, Y_0, D, X, Z, U, V
