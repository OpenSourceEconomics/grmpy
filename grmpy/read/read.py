"""The module contains the main function of the init file import process."""
import numpy as np
import yaml

from grmpy.check.check import check_presence_init
from grmpy.read.read_auxiliary import create_attr_dict_est, create_attr_dict_sim


def read(file, semipar=False, include_constant=False):
    """This function processes the initialization file
    for the estimation process.
    """
    # Check if there is an init file with the specified filename
    check_presence_init(file)

    # Load the initialization file
    with open(file) as y:
        init_dict = yaml.load(y, Loader=yaml.FullLoader)

    # If missing, add generic covariance matrix of the unobservables
    if semipar is False:
        try:
            init_dict["DIST"]
        except KeyError:
            init_dict["DIST"] = {"params": np.array([0.1, 0.0, 0.0, 0.1, 0.0, 1.0])}
        else:
            pass
    else:
        pass

    # Process the initialization file
    attr_dict = create_attr_dict_est(init_dict, semipar, include_constant)

    return attr_dict


def read_simulation(file):
    """Process initialization file for the simulation."""
    # Check if there is an init file with the specified filename
    check_presence_init(file)

    # Load the initialization file
    with open(file) as y:
        init_dict = yaml.load(y, Loader=yaml.FullLoader)

    # Process the initialization file
    attr_dict = create_attr_dict_sim(init_dict)

    return attr_dict


def check_append_constant(init_file, dict_, data, semipar=False):
    """Check if constant already provided by user.
    If not, add auto-generated constant.
    Pass in case there is a constant in first position of the data frame,
    but with a name other than 'const'.
    """
    if (
        "const" not in data
        and np.array_equal(np.asarray(data.iloc[:, 0]), np.ones(len(data))) is False
    ):
        dict_ = read(init_file, semipar, include_constant=True)
        data.insert(0, "const", 1.0)

    else:
        pass

    return dict_, data
