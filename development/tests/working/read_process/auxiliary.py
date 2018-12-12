"""The module provides the simulation process."""
import numpy as np

from development.tests.working.read_process.init_file_import_process import read_new

from grmpy.simulate.simulate_auxiliary import simulate_unobservables
from grmpy.simulate.simulate_auxiliary import simulate_covariates
from grmpy.simulate.simulate_auxiliary import simulate_outcomes
from grmpy.simulate.simulate_auxiliary import write_output
from grmpy.simulate.simulate_auxiliary import print_info
from grmpy.check.check import check_initialization_dict


def simulate_new(init_file):
    """This function simulates a user-specified version of the generalized Roy model."""
    init_dict = read_new(init_file)

    # We perform some basic consistency checks regarding the user's request.
    check_initialization_dict(init_dict)

    # Distribute information
    seed = init_dict['SIMULATION']['seed']

    # Set random seed to ensure recomputabiltiy
    np.random.seed(seed)

    # Simulate unobservables of the model
    U, V = simulate_unobservables(init_dict)

    # Simulate observables of the model
    X = simulate_covariates(init_dict)

    # Simulate endogeneous variables of the model
    Y, D, Y_1, Y_0 = simulate_outcomes(init_dict, X, U, V)

    # Write output file
    df = write_output(init_dict, Y, D, X, Y_1, Y_0, U, V)

    # Calculate Criteria function value
#
    # Print Log file
    print_info(init_dict, df)

    return df


def attr_dict_to_init_dict(attr, old=False):
    """This function converts a already imported attr dict into an initalization dict."""
    init = {}
    for key in ['TREATED', 'UNTREATED', 'CHOICE']:
        init[key] = {'params': list(attr[key]['all']),
                     'order': [attr['varnames'][j - 1] for j in attr[key]['order']]}
    init['DIST'] = {'params': list(attr['DIST']['all'])}
    for key in ['ESTIMATION', 'SCIPY-BFGS', 'SCIPY-POWELL', 'SIMULATION']:
        init[key] = attr[key]
    init['VARTYPES'] = {}
    for name in attr['varnames']:
        index = attr['varnames'].index(name)

        init['VARTYPES'][name] = attr['AUX']['types'][index]

    return init


def dict_transformation(dict_):
    varnames = []
    vartypes = {}
    for section in ['TREATED', 'UNTREATED', 'CHOICE']:
        for variable in dict_[section]['order']:
            if dict_[section]['order'] not in varnames:
                varnames += [variable]
                vartypes[variable] = dict_[section]['types'][
                    dict_[section]['order'].index(variable)]
    dict_['varnames'] = varnames

    dict_['VARTYPES'] = vartypes
    return dict_
