"""This file contains tests for the semipar estimation routine."""

import numpy as np
import pandas as pd

from grmpy.estimate.estimate import fit
from grmpy.grmpy_config import TEST_RESOURCES_DIR
from grmpy.read.read import read
from grmpy.test.random_init import print_dict


def test1():
    """
    This module contains a simple test for the equality of the results of
    R's locpoly function and grmpy's locpoly function. Therefore,
    the mock data set from Carneiro et al (2011) is used.
    """
    init_dict = read(TEST_RESOURCES_DIR + "/replication_semipar.yml")
    init_dict["ESTIMATION"]["file"] = TEST_RESOURCES_DIR + "/aer-replication-mock.pkl"
    print_dict(init_dict, TEST_RESOURCES_DIR + "/replication_semipar")
    test_rslt = fit(TEST_RESOURCES_DIR + "/replication_semipar.grmpy.yml", semipar=True)

    expected_mte_u = pd.read_pickle(
        TEST_RESOURCES_DIR + "/replication-results-mte_u.pkl"
    )

    np.testing.assert_array_almost_equal(test_rslt["mte_u"], expected_mte_u, 6)
