"""The module provides an estimation process given the simulated data set and the
initialization file.
"""
import numpy as np

from grmpy.check.auxiliary import read_data
from grmpy.check.check import (
    check_basic_init_basic,
    check_par_init_dict,
    check_presence_estimation_dataset,
    check_presence_init,
)
from grmpy.estimate.estimate_par import par_fit
from grmpy.estimate.estimate_semipar import semipar_fit
from grmpy.read.read import read


def fit(init_file, semipar=False):
    """This function estimates the MTE based on a parametric normal model or,
    alternatively, via the semiparametric method of local instrumental variables (LIV)"""

    # Load the estimation file
    check_presence_init(init_file)
    dict_ = read(init_file, semipar)

    # Perform some consistency checks given the user's request
    check_presence_estimation_dataset(dict_)

    # Semiparametric LIV Model
    if semipar is True:
        check_basic_init_basic(dict_)

        # Distribute initialization information.
        data = read_data(dict_["ESTIMATION"]["file"])

        # Check if constant already provided by user, but with name
        # other than 'const'. If so, drop auto-generated constant.
        if np.array_equal(np.asarray(data.iloc[:, 0]), np.ones(len(data))) is False:
            dict_ = read(init_file, semipar, include_constant=True)

        rslt = semipar_fit(dict_)

    # Parametric Normal Model
    else:
        check_par_init_dict(dict_)

        # Distribute initialization information.
        data = read_data(dict_["ESTIMATION"]["file"])

        # Check if constant already provided by user, but with name
        # other than 'const'. If so, drop auto-generated constant.
        if np.array_equal(np.asarray(data.iloc[:, 0]), np.ones(len(data))) is False:
            dict_ = read(init_file, semipar=False, include_constant=True)

        rslt = par_fit(dict_)

    return rslt
