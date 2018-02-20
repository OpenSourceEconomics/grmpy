"""This module provides some capabilities to check the integrity of the package."""
from grmpy.check.custom_exceptions import UserError
from grmpy.check.auxiliary import is_pos_def


def check_initialization_dict(dict_):
    """This function performs some basic checks regarding the integrity of the user's request.
    There should be no uncontrolled terminations of the package once these checks are passed.
    """
    # Distribute details
    num_coeffs_untreated = len(dict_['UNTREATED']['all'])
    num_coeffs_treated = len(dict_['TREATED']['all'])
    num_agents_sim = dict_['SIMULATION']['agents']

    # This are just two example for a whole host of tests.
    if num_agents_sim <= 0:
        msg = 'The number of simulated individuals needs to be larger than zero.'
        raise UserError(msg)

    if num_coeffs_treated != num_coeffs_untreated:
        msg = 'The number of covariates determining potential outcomes needs to be identical.'
        raise UserError(msg)
    if dict_['DETERMINISTIC'] is False:
        if is_pos_def(dict_) is False:
            msg = 'The specified covariance matrix has to be positive semidefinite.'
            raise UserError(msg)
    for key_ in ['TREATED', 'UNTREATED', 'COST']:
        if len(dict_[key_]['order']) > len(set(dict_[key_]['order'])):
            msg = 'There is a problem in the {} section of the initialization file. \n         ' \
                  'Probably you specified two coefficients for one covariate in the same section.'\
                .format(key_)
            raise UserError(msg)



def check_init_file(dict_):
    """This fuction checks if the specified initialization file meets the requirements for the
    estimation process.
    """
    if all(dist_elements == 0 for dist_elements in dict_['DIST']['all']):
        msg = 'The distributional characteristics have to be undeterministic.'
        raise UserError(msg)
    elif dict_['DIST']['all'][5] == 0:
        msg = 'The standard deviation of the collected unobservables have to be larger than zero.'
        raise UserError(msg)






