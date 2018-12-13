"""The module contains the main function of the init file import process."""
import yaml

from grmpy.read.read_auxiliary import init_dict_to_attr_dict
from grmpy.check.check import check_presence_init


def read(file):
    """This function processes the initialization file so that it can be used for simulation as well
     as estimation purposes.
     """
    # Check if there is a init file with the specified filename
    check_presence_init(file)

    # Load the initialization file
    with open(file) as y:
        init_dict = yaml.load(y)

    # Process the initialization file
    attr_dict = init_dict_to_attr_dict(init_dict)


    return attr_dict
