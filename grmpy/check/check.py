"""This module provides some capabilities to check the integrity of the package."""
import numpy as np
import os

from grmpy.check.custom_exceptions import UserError
from grmpy.check.auxiliary import is_pos_def


def check_presence_init(fname):
    """This function checks whether the model initialization file does in fact exist."""
    if not os.path.isfile(fname):
        msg = '{}: There is no such file or directory.'.format(fname)
        raise UserError(msg)


def check_presence_estimation_dataset(init_dict):
    """This function checks whether the estimation dataset does exist."""
    data_file = init_dict['ESTIMATION']['file']
    if not os.path.isfile(data_file):
        msg = 'The data file specified in your initialization file doesn`t exist.'
        raise UserError(msg)

def check_initialization_dict(dict_):
    """This function performs some basic checks regarding the integrity of the user's request.
    There should be no uncontrolled terminations of the package once these checks are passed.
    """
    # Distribute details
    num_agents_sim = dict_['SIMULATION']['agents']

    # This are just two example for a whole host of tests.
    if num_agents_sim <= 0:
        msg = 'The number of simulated individuals needs to be larger than zero.'
        raise UserError(msg)

    if dict_['DETERMINISTIC'] is False:
        if not is_pos_def(dict_):
            msg = 'The specified covariance matrix has to be positive semidefinite.'
            raise UserError(msg)
    for key_ in ['TREATED', 'UNTREATED', 'CHOICE']:
        if len(dict_[key_]['order']) > len(set(dict_[key_]['order'])):
            msg = 'There is a problem in the {} section of the initialization file. \n         ' \
                  'Probably you specified two coefficients for one covariate in the same section.'\
                .format(key_)
            raise UserError(msg)
        for x in dict_[key_]['types']:
            if isinstance(x, list):
                # Check for binary variables
                if x[0] == 'binary':
                    if x[1] >= 0.9:
                        msg = 'The specified probability that a binary variable is equal to one \
                                   has to be sufficiently lower than one.'
                        raise UserError(msg)
                # Check for categorical variables
                elif x[0] == 'categorical':
                    if 1 in x[1]:
                        index = x[1].index(1)
                        if x[2][index] >= 0.9:
                            msg = 'The specified probability that a categorical variable is equal\
                                        to one has to be sufficiently lower than one.'
                            raise UserError(msg)
                    if np.testing.assert_approx_equal(sum(x[2]),1.):
                        msg = 'The specified probability for all possible categories of a ' \
                              'categorical variable have to sum up to 1.'
                        raise UserError(msg)

    if dict_['ESTIMATION']['file'][-4:] not in ['.pkl', '.txt', 'dta']:
        msg = 'The {} format specified in the Estimation section of the initialization file' \
              ' is currently not supported by grmpy. \n' \
              '         Please use either .txt, .pkl or .dta files.'\
            .format(dict_['ESTIMATION']['file'][-4:])
        raise UserError(msg)


def check_init_file(dict_):
    """This function checks if the specified initialization file meets the requirements for the
    estimation process.
    """
    if all(dist_elements == 0 for dist_elements in dict_['DIST']['all']):
        msg = 'The distributional characteristics have to be undeterministic.'
        raise UserError(msg)
    elif dict_['DIST']['all'][5] == 0:
        msg = 'The standard deviation of the collected unobservables have to be larger than zero.'
        raise UserError(msg)
    for key_ in ['TREATED', 'UNTREATED', 'CHOICE']:
        if len(set(dict_[key_]['order'])) != len(dict_[key_]['order']):
            msg = 'There are two start coefficients {} Section'.format(key_)
            raise UserError(msg)






