"""The module provides an estimation process given the simulated data set and the
initialization file.
"""
from grmpy.check.check import check_presence_estimation_dataset
from grmpy.check.check import check_par_init_dict
from grmpy.check.check import check_presence_init
from grmpy.check.check import check_basic_init_basic
from grmpy.check.check import check_par_init_file
from grmpy.read.read import read

from grmpy.estimate.estimate_semipar import semipar_fit
from grmpy.estimate.estimate_par import par_fit


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
        rslt = semipar_fit(dict_)

    # Parametric Normal Model
    else:
        check_par_init_dict(dict_)
        check_par_init_file(dict_)
        rslt = par_fit(dict_)

    return rslt
