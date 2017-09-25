"""This module provides some capabilities to check the integrity of the package."""

from grmpy.check.custom_exceptions import UserError


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
        raise UserError('The number of simulated individuals needs to be larger than zero.')

    if num_coeffs_treated != num_coeffs_untreated:
        msg = 'The number of covariates determining potential outcomes needs to be identical.'
        raise UserError(msg)
