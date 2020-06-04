"""The module provides the simulation process."""
import numpy as np

from grmpy.simulate.simulate_auxiliary import simulate_unobservables
from grmpy.simulate.simulate_auxiliary import simulate_covariates
from grmpy.estimate.estimate_par import calculate_criteria

from grmpy.simulate.simulate_auxiliary import simulate_outcomes
from grmpy.simulate.simulate_auxiliary import write_output
from grmpy.simulate.simulate_auxiliary import print_info

from grmpy.estimate.estimate_par import start_values
from grmpy.estimate.estimate_par import process_data
from grmpy.check.check import check_sim_init_dict
from grmpy.read.read import read_simulation


def simulate(init_file):
    """This function simulates a user-specified version of the generalized Roy model."""
    init_dict = read_simulation(init_file)

    # We perform some basic consistency checks regarding the user's request.
    check_sim_init_dict(init_dict)

    # Distribute information
    seed = init_dict["SIMULATION"]["seed"]

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
    if not init_dict["DETERMINISTIC"]:
        x0 = start_values(init_dict, df, "init")
        _, X1, X0, Z1, Z0, Y1, Y0 = process_data(df, init_dict)
        init_dict["AUX"]["criteria_value"] = calculate_criteria(
            init_dict, X1, X0, Z1, Z0, Y1, Y0, x0
        )

    # Print Log file
    print_info(init_dict, df)

    return df
