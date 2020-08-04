"""The module provides an estimation process given the simulated data set and the
initialization file.
"""
from grmpy.check.auxiliary import read_data
from grmpy.check.check import (
    check_est_init_dict,
    check_par_init_file,
    check_presence_estimation_dataset,
)
from grmpy.estimate.estimate_par import par_fit
from grmpy.estimate.estimate_semipar import semipar_fit
from grmpy.read.read import check_append_constant, read


def fit(init_file, semipar=False):
    """This function estimates the MTE based on a parametric normal model
    or, alternatively, via the semiparametric method of
    local instrumental variables (LIV).

    Parameters
    ----------
    init_file: yaml
        Initialization file containing parameters for the estimation
        process.

    Returns
    ------
    rslt: dict
        Result dictionary containing
        - quantiles
        - mte
        - mte_x
        - mte_u
        - mte_min
        - mte_max
        - X
        - b1
        - b0
    """

    # Load the estimation file
    dict_ = read(init_file, semipar)

    # Perform some consistency checks given the user's request
    check_presence_estimation_dataset(dict_)
    check_est_init_dict(dict_)

    # Semiparametric LIV Model
    if semipar:
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
