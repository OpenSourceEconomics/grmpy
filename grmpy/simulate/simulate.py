"""The module provides the simulation process."""
import numpy as np

from grmpy.simulate.simulate_auxiliary import simulate_unobservables
from grmpy.simulate.simulate_auxiliary import simulate_covariates
from grmpy.estimate.estimate_auxiliary import calculate_criteria
from grmpy.simulate.simulate_auxiliary import simulate_outcomes
from grmpy.simulate.simulate_auxiliary import write_output
from grmpy.estimate.estimate_auxiliary import start_values
from grmpy.simulate.simulate_auxiliary import print_info
from grmpy.check.check import check_initialization_dict
from grmpy.read.read import read


def simulate(init_file):
    """This function simulates a user-specified version of the generalized Roy model."""
    init_dict = read(init_file)

    # We perform some basic consistency checks regarding the user's request.
    check_initialization_dict(init_dict)

    # Distribute information
    seed = init_dict['SIMULATION']['seed']

    # Set random seed to ensure recomputabiltiy
    np.random.seed(seed)

    # Simulate unobservables of the model
    U = simulate_unobservables(init_dict)

    # Simulate observables of the model
    X = simulate_covariates(init_dict)

    # Simulate endogeneous variables of the model
    df = simulate_outcomes(init_dict, X, U)

    # Write output file
    df = write_output(init_dict, df)

    # Calculate Criteria function value
    if not init_dict['DETERMINISTIC']:
        x0 = start_values(init_dict, df, 'init')
        init_dict['AUX']['criteria_value'] = calculate_criteria(init_dict, df, x0)

    # Print Log file
    print_info(init_dict, df)

    return df
