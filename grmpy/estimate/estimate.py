"""The module provides an estimation process given the simulated data set and the
initialization file.
"""
import numpy as np

from grmpy.check.check import check_presence_estimation_dataset
from grmpy.check.check import check_initialization_dict
from grmpy.check.check import check_presence_init
from grmpy.check.check import check_par
from grmpy.read.read import read

from grmpy.estimate.estimate_semipar import semipar_fit
from grmpy.estimate.estimate_par import par_fit


def fit(init_file, semipar=False):
    """This function estimates the MTE based on a parametric normal model or,
    alternatively, via the semiparametric method of local instrumental variables (LIV)"""

    # Load the estimation file
    check_presence_init(init_file)
    dict_ = read(init_file)

    # Perform some consistency checks given the user's request
    check_presence_estimation_dataset(dict_)
    check_initialization_dict(dict_)

    # Semiparametric LIV Model
    if semipar is True:
        quantiles, mte_u, X, b1_b0 = semipar_fit(dict_)  # change to dict_

        # Construct the MTE
        # Calculate the MTE component that depends on X
        mte_x = np.dot(X, b1_b0)

        # Put the MTE together
        mte = mte_x.mean(axis=0) + mte_u

        # Account for variation in X
        mte_min = np.min(mte_x) + mte_u
        mte_max = np.max(mte_x) + mte_u

        rslt = {
            "quantiles": quantiles,
            "mte": mte,
            "mte_x": mte_x,
            "mte_u": mte_u,
            "mte_min": mte_min,
            "mte_max": mte_max,
            "X": X,
            "b1-b0": b1_b0,
        }

    # Parametric Normal Model
    else:
        check_par(dict_)
        rslt = par_fit(dict_)

    return rslt
