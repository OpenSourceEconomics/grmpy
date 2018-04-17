"""This module provides some capabilities to check the integrity of the package."""
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
        # TODO: Please review the whole code in light of the second answer here
        # https://stackoverflow.com/questions/18922407/boolean-and-type-checking-in-python-vs-numpy
        # Don't compare boolean values to True or False using ==.
        if not is_pos_def(dict_):
            msg = 'The specified covariance matrix has to be positive semidefinite.'
            raise UserError(msg)
    for key_ in ['TREATED', 'UNTREATED', 'COST']:
        if len(dict_[key_]['order']) > len(set(dict_[key_]['order'])):
            msg = 'There is a problem in the {} section of the initialization file. \n         ' \
                  'Probably you specified two coefficients for one covariate in the same section.'\
                .format(key_)
            raise UserError(msg)
        for x in dict_[key_]['types']:
            if isinstance(x, list):
                if x[1] >= 0.9:
                    msg = 'The specified probability that a binary variable is equal to one has to be \\\
                           suffiently lower than one.'
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






