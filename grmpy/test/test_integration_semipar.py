"""This file contains integration tests for the semiparametric estimation routine."""
import numpy as np
import pandas as pd
import pickle

from grmpy.estimate.estimate import fit
from grmpy.grmpy_config import TEST_RESOURCES_DIR
from grmpy.read.read import read
from grmpy.simulate.simulate import simulate
from grmpy.test.random_init import print_dict


def test_replication_carneiro():
    """
    This function checks the equality of the results of
    R's locpoly function and grmpy's locpoly function. The mock data set
    from Carneiro et al (2011) is used and both the mte_u and the final
    mte are compared.
    """
    init_dict = read(TEST_RESOURCES_DIR + "/replication_semipar.yml")
    init_dict["ESTIMATION"]["file"] = TEST_RESOURCES_DIR + "/aer-replication-mock.pkl"
    print_dict(init_dict, TEST_RESOURCES_DIR + "/replication_semipar")
    test_rslt = fit(TEST_RESOURCES_DIR + "/replication_semipar.grmpy.yml", semipar=True)

    expected_mte_u = pd.read_pickle(
        TEST_RESOURCES_DIR + "/replication-results-mte_u.pkl"
    )
    expected_mte = pd.read_pickle(TEST_RESOURCES_DIR + "/replication-results-mte.pkl")

    np.testing.assert_array_almost_equal(test_rslt["mte_u"], expected_mte_u, 6)
    np.testing.assert_array_almost_equal(test_rslt["mte"], expected_mte, 6)


def test_rslt_dictionary():
    """
    This test checks if the elements of the estimation dictionary are equal
    to their expected values when the initialization file of the
    semipar tutorial is used.
    """
    fname = TEST_RESOURCES_DIR + "/tutorial-semipar.grmpy.yml"
    simulate(fname)

    rslt = fit(fname, semipar=True)
    expected_rslt = pickle.load(
        open(TEST_RESOURCES_DIR + "/tutorial-semipar-results.pkl", "rb")
    )

    np.testing.assert_equal(rslt["quantiles"], expected_rslt["quantiles"])
    np.testing.assert_almost_equal(rslt["mte"], expected_rslt["mte"], 7)
    np.testing.assert_almost_equal(rslt["mte_u"], expected_rslt["mte_u"], 7)
    np.testing.assert_almost_equal(rslt["mte_min"], expected_rslt["mte_min"], 5)
    np.testing.assert_almost_equal(rslt["mte_max"], expected_rslt["mte_max"], 5)
    np.testing.assert_almost_equal(rslt["b0"], expected_rslt["b0"], 7)
    np.testing.assert_almost_equal(rslt["b1"], expected_rslt["b1"], 7)
