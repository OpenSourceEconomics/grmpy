"""The module provides the simulation process."""
import numpy as np

from grmpy.check.check import check_sim_init_dict
from grmpy.estimate.estimate_par import calculate_criteria, process_data, start_values
from grmpy.read.read import read_simulation
from grmpy.simulate.simulate_auxiliary import (
    print_info,
    simulate_covariates,
    simulate_outcomes,
    simulate_unobservables,
    write_output,
)


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
        D, X1, X0, Z1, Z0, Y1, Y0 = process_data(df, init_dict)
        x0 = start_values(init_dict, D, X1, X0, Z1, Z0, Y1, Y0, "init")
        init_dict["AUX"]["criteria_value"] = calculate_criteria(
            x0, X1, X0, Z1, Z0, Y1, Y0
        )

    # Print Log file
    print_info(init_dict, df)

    return df
