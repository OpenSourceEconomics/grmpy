"""The module contains the main function of the init file import process."""
import yaml

from grmpy.read.read_auxiliary import create_attr_dict_sim
from grmpy.read.read_auxiliary import create_attr_dict_est
from grmpy.check.check import check_presence_init


def read(file, semipar=False, include_constant=False):
    """This function processes the initialization file
    for the estimation process.
     """
    # Check if there is a init file with the specified filename
    check_presence_init(file)

    # Load the initialization file
    with open(file) as y:
        init_dict = yaml.load(y)

    # Process the initialization file
    attr_dict = create_attr_dict_est(init_dict, semipar, include_constant)

    return attr_dict


def read_simulation(file):
    """Process the initialization file for
    simulation purposes
    """
    # Check if there is a init file with the specified filename
    check_presence_init(file)

    # Load the initialization file
    with open(file) as y:
        init_dict = yaml.load(y)

    # Process the initialization file
    attr_dict = create_attr_dict_sim(init_dict)

    return attr_dict
