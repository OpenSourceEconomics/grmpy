"""
This module contains a simple test for the equality of the results of
R's locpoly function and grmpy's locpoly function. Therefore,
the mock data set from Carneiro et al (2011) is used.
"""
import pandas as pd
import numpy as np

from numpy.testing import assert_equal
from grmpy.estimate.estimate import fit

import sys

sys.path.append("..")


def test_semipar_replication():
    """
    This function asserts equality between the test mte_u and the replicated
    mte_u
    """
    test_rslt = fit("promotion/grmpy_tutorial_notebook/files/tutorial_semipar.yml", semipar=True)

    expected_mte_u = pd.read_pickle("resources/replication-results-mte_u.pkl")

    assert_equal(np.round(test_rslt['mte_u'], 6), np.round(expected_mte_u, 6))


if __name__ == "__main__":
    test_semipar_replication()
    print("Everything passed")
