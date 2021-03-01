"""This file contains unit tests for the semiparametric estimation routine."""
import numpy as np
import pandas as pd
import pytest
import random

from grmpy.estimate.estimate_semipar import (
    _construct_Xp,
    _define_common_support,
    _generate_residuals,
    double_residual_reg,
    estimate_treatment_propensity,
    process_primary_inputs,
    process_secondary_inputs,
    trim_support,
)
from grmpy.grmpy_config import TEST_RESOURCES_DIR
from grmpy.read.read import check_append_constant, read
from grmpy.simulate.simulate import simulate


@pytest.fixture
def simulate_test_data():
    """
    Simulate test dict_ and data.
    """
    fname = TEST_RESOURCES_DIR + "/tutorial.grmpy.yml"
    data = simulate(fname)
    dict_ = read(fname)
    dict_, data = check_append_constant(
        TEST_RESOURCES_DIR + "/tutorial.grmpy.yml", dict_, data, semipar=True
    )

    return dict_, data


@pytest.fixture
def loess_example_data():
    """
    Generate data used in the loess test functions.
    """
    exog = np.array(
        [
            0.5578196,
            2.0217271,
            2.5773252,
            3.4140288,
            4.3014084,
            4.7448394,
            5.1073781,
            6.5411662,
            6.7216176,
            7.2600583,
            8.1335874,
            9.1224379,
            11.9296663,
            12.3797674,
            13.2728619,
            14.2767453,
            15.3731026,
            15.6476637,
            18.5605355,
            18.5866354,
            18.7572812,
        ]
    )

    endog = np.array(
        [
            18.63654,
            103.49646,
            150.35391,
            190.51031,
            208.70115,
            213.71135,
            228.49353,
            233.55387,
            234.55054,
            223.89225,
            227.68339,
            223.91982,
            168.01999,
            164.95750,
            152.61107,
            160.78742,
            168.55567,
            152.42658,
            221.70702,
            222.69040,
            243.18828,
        ]
    )

    return exog, endog


def expected_data_no_trim(dict_, data):
    """
    Generate test data for the trim test cases.
    """
    data = data.sort_values(by="prop_score", ascending=True)
    X_expected = data[dict_["TREATED"]["order"]]
    Y_expected = data[[dict_["ESTIMATION"]["dependent"]]]
    prop_score_expected = data["prop_score"]

    return X_expected, Y_expected, prop_score_expected


def test_inputs():
    """
    Check whether default parameters are provided if user does not
    specify the input parameters herself.
    """
    init_dict = read(TEST_RESOURCES_DIR + "/replication_semipar.yml")
    init_dict.pop("ESTIMATION")

    nbins, logit, bandwidth, gridsize, start_grid, end_grid = process_primary_inputs(
        init_dict
    )
    trim, rbandwidth, reestimate_p, show_output = process_secondary_inputs(init_dict)

    default_input = (trim, rbandwidth, reestimate_p)
    user_input = (nbins, logit, bandwidth, gridsize, start_grid, end_grid)

    default_input_expected = (True, 0.05, False)
    user_input_expected = (25, True, 0.32, 500, 0.005, 0.995)

    np.testing.assert_equal(default_input, default_input_expected)
    np.testing.assert_equal(user_input, user_input_expected)


def test_propensity_score(simulate_test_data):
    """
    Check whether propensity score has the same number of observation as
    the input data frame (for both the logit and probit model).
    """
    dict_, data = simulate_test_data

    ps_logit = estimate_treatment_propensity(dict_, data, logit=True)
    ps_probit = estimate_treatment_propensity(dict_, data, logit=False)

    np.testing.assert_equal(len(ps_logit), data.shape[0])
    np.testing.assert_equal(len(ps_logit), len(ps_probit))


def test_trim(simulate_test_data):
    """
    Test whether original data is returned if *trim* is set to False
    but *reestimate_p* to True.
    """
    dict_, data = simulate_test_data

    data = estimate_treatment_propensity(dict_, data, logit=True)
    X_expected, Y_expected, prop_score_expected = expected_data_no_trim(dict_, data)

    logit, trim, reestimate_p = False, False, True
    X, Y, prop_score = trim_support(dict_, data, logit, 25, trim, reestimate_p)

    pytest.X_testing = X
    pytest.Y_testing = Y
    pytest.prop_score_testing = prop_score

    np.testing.assert_array_equal(X, X_expected)
    np.testing.assert_array_equal(Y, Y_expected)
    np.testing.assert_array_equal(prop_score, prop_score_expected)


def test_trim2(simulate_test_data):
    """
    Test whether trim function returns original data when common support
    is set to the entire unit interval.
    """
    dict_, data = simulate_test_data

    data = estimate_treatment_propensity(dict_, data, logit=True)

    logit, trim, reestimate_p = True, True, False
    prop_score = data["prop_score"]
    common_support = [0, 1]

    # Trim the data. Recommended.
    if trim is True:
        # data, prop_score = trim_data(prop_score, common_support, data)
        data_trim = data[
            (data.prop_score >= common_support[0])
            & (data.prop_score <= common_support[1])
        ]
        prop_score_trim = prop_score[
            (prop_score >= common_support[0]) & (prop_score <= common_support[1])
        ]

        # Optional. Not recommended
        # Re-estimate baseline propensity score on the trimmed sample
        if reestimate_p is True:
            # Re-estimate the parameters of the decision equation based
            # on the new trimmed data set
            data_trim = estimate_treatment_propensity(dict_, data_trim, logit)

        else:
            pass
    else:
        data_trim = data
        prop_score_trim = prop_score

    data_trim = data_trim.sort_values(by="prop_score", ascending=True)
    X_trim = data_trim[dict_["TREATED"]["order"]]
    Y_trim = data_trim[[dict_["ESTIMATION"]["dependent"]]]
    prop_score_trim = np.sort(prop_score_trim)

    X_expected, Y_expected, prop_score_expected = expected_data_no_trim(dict_, data)

    np.testing.assert_array_equal(X_trim, X_expected)
    np.testing.assert_array_equal(Y_trim, Y_expected)
    np.testing.assert_array_equal(prop_score_trim, prop_score_expected)


def test_common_support():
    """
    Test whether common support is indeed zero if treatment propensity
    is 0.5 for everyone.
    """
    fname = TEST_RESOURCES_DIR + "/tutorial.grmpy.yml"
    data = simulate(fname)
    dict_ = read(fname)

    prop_score = pd.Series(np.ones(len(data))) * 0.5
    data.loc[:, "prop_score"] = prop_score

    estimated_support = _define_common_support(dict_, data)
    expected_support = [0.5, 0.5]

    np.testing.assert_equal(estimated_support, expected_support)


def test_loess_residuals(loess_example_data):
    """
    Test whether the correct loess residuals are generated when endogeneous
    variable is a column vector.
    """
    random.seed(123)

    exog, endog = loess_example_data

    res = _generate_residuals(exog, endog, 0.5)

    res_expected = np.array(
        [
            -16.35514377,
            -1.60290726,
            19.43891125,
            23.2356233,
            11.68133431,
            4.7969777,
            13.99253927,
            7.97312287,
            8.61448786,
            -3.08724342,
            1.40557558,
            4.90562643,
            -10.19734474,
            -9.42800137,
            -14.0138517,
            -2.91374618,
            -10.91497562,
            -30.97944944,
            0.19833045,
            0.8296881,
            18.89072403,
        ]
    )

    np.testing.assert_almost_equal(res, res_expected, 7)


def test_loess_residuals2(loess_example_data):
    """
    Test whether the correct loess residuals are generated when endogeneous
    variable is a multi-column matrix.
    """
    random.seed(123)

    exog, endog = loess_example_data
    y2 = pd.DataFrame(np.array([endog, endog * 2]).T)

    res2 = _generate_residuals(exog, y2, 0.5)

    res_expected2 = np.array(
        [
            [-16.35514377, -32.71028754],
            [-1.60290726, -3.20581451],
            [19.43891125, 38.87782249],
            [23.2356233, 46.47124659],
            [11.68133431, 23.36266863],
            [4.7969777, 9.5939554],
            [13.99253927, 27.98507854],
            [7.97312287, 15.94624574],
            [8.61448786, 17.22897572],
            [-3.08724342, -6.17448684],
            [1.40557558, 2.81115117],
            [4.90562643, 9.81125286],
            [-10.19734474, -20.39468947],
            [-9.42800137, -18.85600274],
            [-14.0138517, -28.0277034],
            [-2.91374618, -5.82749235],
            [-10.91497562, -21.82995124],
            [-30.97944944, -61.95889889],
            [0.19833045, 0.3966609],
            [0.8296881, 1.6593762],
            [18.89072403, 37.78144806],
        ]
    )

    np.testing.assert_almost_equal(res2, res_expected2, 7)


def test_double_residual_reg(loess_example_data):
    """
    Test whether the double residual regression returns the
    correct beta coefficients for a multi-column X matrix.
    """
    random.seed(123)

    exog, endog = loess_example_data
    X_data = pd.DataFrame({"X0": endog, "X1": endog * 2})
    Y_data = pd.DataFrame({"Y": exog})

    prop_score = pd.Series(np.linspace(0.2, 0.8, 21))

    b0, b1_b0 = double_residual_reg(X_data, Y_data, prop_score, 0.2)

    b0_expected = np.array([-0.00529223, -0.01058445])
    b1_b0_expected = np.array([0.01267241, 0.02534481])

    np.testing.assert_almost_equal(b0, b0_expected, 8)
    np.testing.assert_almost_equal(b1_b0, b1_b0_expected, 8)


def test_constructXp(loess_example_data):
    """
    Check whether generated variable Xp = X * P(Z) is a pandas.DataFrame
    and has the same shape as the imput X matrix.
    """
    exog, endog = loess_example_data
    X_data = pd.DataFrame({"X0": endog, "X1": endog * 2})

    prop_score = pd.Series(np.linspace(0.2, 0.8, 21))

    Xp = _construct_Xp(X_data, prop_score)

    assert isinstance(Xp, pd.DataFrame) is True
    np.testing.assert_equal(X_data.shape, Xp.shape)
