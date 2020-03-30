"""The module provides an estimation process given the simulated data set and the
initialization file.
"""
from grmpy.check.check import check_presence_estimation_dataset
from grmpy.check.check import check_presence_init
from grmpy.check.check import check_est_init_dict
from grmpy.check.check import check_par_init_file
from grmpy.check.auxiliary import read_data

from grmpy.read.read import read, check_append_constant

from grmpy.estimate.estimate_semipar import semipar_fit
from grmpy.estimate.estimate_par import par_fit


def fit(init_file, semipar=False):
    """This function estimates the MTE based on a parametric normal model
    or, alternatively, via the semiparametric method of
    local instrumental variables (LIV).
    """

    # Load the estimation file
    check_presence_init(init_file)
    dict_ = read(init_file, semipar)

    # Perform some consistency checks given the user's request
    check_presence_estimation_dataset(dict_)
    check_est_init_dict(dict_)

    # Semiparametric LIV Model
    if semipar is True:
        # Distribute initialization information.
        data = read_data(dict_["ESTIMATION"]["file"])
        dict_, data = check_append_constant(init_file, dict_, data, semipar=True)

        rslt = semipar_fit(dict_, data)

    # Parametric Normal Model
    else:
        # Perform some extra checks
        check_par_init_file(dict_)

        # Distribute initialization information.
        data = read_data(dict_["ESTIMATION"]["file"])
        dict_, data = check_append_constant(init_file, dict_, data, semipar=False)

        rslt = par_fit(dict_, data)

    return rslt
