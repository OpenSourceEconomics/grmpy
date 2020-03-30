"""The module includes an integration and a regression test for the simulation and the
estiomation process.
 """
import json

import numpy as np
import pytest

from grmpy.check.auxiliary import check_special_conf
from grmpy.check.check import (
    check_sim_distribution,
    check_sim_init_dict,
    check_start_values,
)
from grmpy.check.custom_exceptions import UserError
from grmpy.estimate.estimate import fit
from grmpy.estimate.estimate_par import calculate_criteria, process_data, start_values
from grmpy.estimate.estimate_output import simulate_estimation
from grmpy.grmpy_config import TEST_RESOURCES_DIR
from grmpy.read.read import read
from grmpy.simulate.simulate import simulate
from grmpy.test.auxiliary import cleanup, dict_transformation, read_desc
from grmpy.test.random_init import generate_random_dict, print_dict


def test1():
    """The test runs a loop to check the consistency of the random init file generating
    process and the following simulation.
    """
    for _ in range(10):
        dict_ = generate_random_dict()
        print_dict(dict_)
        simulate("test.grmpy.yml")


def test2():
    """This test runs a random selection of five regression tests from the our old
    regression test battery.
    """
    fname = TEST_RESOURCES_DIR + "/old_regression_vault.grmpy.json"
    tests = json.load(open(fname))
    random_choice = np.random.choice(range(len(tests)), 5)
    tests = [tests[i] for i in random_choice]

    for test in tests:
        stat, dict_, criteria = test
        print_dict(dict_transformation(dict_))
        df = simulate("test.grmpy.yml")
        init_dict = read("test.grmpy.yml")
        start = start_values(init_dict, df, "init")
        _, X1, X0, Z1, Z0, Y1, Y0 = process_data(df, init_dict)

        criteria_ = calculate_criteria(init_dict, X1, X0, Z1, Z0, Y1, Y0, start)
        np.testing.assert_almost_equal(np.sum(df.sum()), stat)
        np.testing.assert_array_almost_equal(criteria, criteria_)


def test3():
    """The test checks if the criteria function value of the simulated and the
    'estimated' sample is equal if both samples include an identical number of
    individuals.
    """
    for _ in range(5):
        constr = dict()
        constr["DETERMINISTIC"], constr["AGENTS"], constr["START"] = False, 1000, "init"
        constr["OPTIMIZER"], constr["SAME_SIZE"] = "SCIPY-BFGS", True
        generate_random_dict(constr)
        df1 = simulate("test.grmpy.yml")
        rslt = fit("test.grmpy.yml")
        init_dict = read("test.grmpy.yml")
        _, df2 = simulate_estimation(rslt)
        start = start_values(init_dict, df1, "init")

        criteria = []
        for data in [df1, df2]:
            _, X1, X0, Z1, Z0, Y1, Y0 = process_data(data, init_dict)
            criteria += [calculate_criteria(init_dict, X1, X0, Z1, Z0, Y1, Y0, start)]
        np.testing.assert_allclose(criteria[1], criteria[0], rtol=0.1)


def test4():
    """The test checks if the estimation process works if the Powell algorithm is
    specified as the optimizer option.
    """
    for _ in range(5):
        constr = dict()
        constr["DETERMINISTIC"], constr["AGENTS"], constr["start"] = (
            False,
            10000,
            "init",
        )
        constr["optimizer"] = "SCIPY-Powell"
        generate_random_dict(constr)

        simulate("test.grmpy.yml")
        fit("test.grmpy.yml")


def test5():
    """The test checks if the estimation process works properly when maxiter is set to
    zero.
    """
    for _ in range(5):
        constr = dict()
        constr["DETERMINISTIC"], constr["MAXITER"] = False, 0
        generate_random_dict(constr)
        simulate("test.grmpy.yml")
        fit("test.grmpy.yml")


def test6():
    """Additionally to test5 this test checks if the comparison file provides the
    expected output when maxiter is set to zero and the estimation process uses the
    initialization file values as start values.
    """
    for _ in range(5):
        constr = dict()
        constr["DETERMINISTIC"], constr["MAXITER"], constr["AGENTS"] = False, 0, 15000
        constr["START"], constr["SAME_SIZE"] = "init", True
        dict_ = generate_random_dict(constr)
        dict_["DIST"]["params"][1], dict_["DIST"]["params"][5] = 0.0, 1.0
        print_dict(dict_)
        simulate("test.grmpy.yml")
        fit("test.grmpy.yml")
        dict_ = read_desc("comparison.grmpy.info")
        for section in ["ALL", "TREATED", "UNTREATED"]:
            np.testing.assert_equal(len(set(dict_[section]["Number"])), 1)
            np.testing.assert_almost_equal(
                dict_[section]["Observed Sample"],
                dict_[section]["Simulated Sample (finish)"],
                0.001,
            )
            np.testing.assert_array_almost_equal(
                dict_[section]["Simulated Sample (finish)"],
                dict_[section]["Simulated Sample (start)"],
                0.001,
            )


def test7():
    """This test ensures that the estimation process returns an UserError if one tries
    to execute an estimation process with initialization file values as start values for
    an deterministic setting.
    """
    fname_falsespec1 = TEST_RESOURCES_DIR + "/test_falsespec1.grmpy.yml"
    fname_falsespec2 = TEST_RESOURCES_DIR + "/test_falsespec2.grmpy.yml"
    fname_noparams = TEST_RESOURCES_DIR + "/test_noparams.grmpy.yml"
    fname_binary = TEST_RESOURCES_DIR + "/test_binary.grmpy.yml"
    fname_vzero = TEST_RESOURCES_DIR + "/test_vzero.grmpy.yml"
    fname_possd = TEST_RESOURCES_DIR + "/test_npsd.grmpy.yml"
    fname_zero = TEST_RESOURCES_DIR + "/test_zero.grmpy.yml"

    for _ in range(5):
        constr = dict()
        constr["AGENTS"], constr["DETERMINISTIC"] = 1000, True
        generate_random_dict(constr)
        dict_ = read("test.grmpy.yml")
        pytest.raises(UserError, check_sim_distribution, dict_)
        pytest.raises(UserError, fit, "test.grmpy.yml")

        generate_random_dict(constr)
        dict_ = read("test.grmpy.yml")
        if len(dict_["CHOICE"]["order"]) == 1:
            dict_["CHOICE"]["params"] = list(dict_["CHOICE"]["params"])
            dict_["CHOICE"]["params"] += [1.000]
            dict_["CHOICE"]["order"] += [2]

        dict_["CHOICE"]["order"][1] = "X1"
        print_dict(dict_)
        pytest.raises(UserError, check_sim_init_dict, dict_)
        pytest.raises(UserError, simulate, "test.grmpy.yml")
        pytest.raises(UserError, fit, "test.grmpy.yml")

        constr["AGENTS"] = 0
        generate_random_dict(constr)
        dict_ = read("test.grmpy.yml")
        pytest.raises(UserError, check_sim_init_dict, dict_)
        pytest.raises(UserError, simulate, "test.grmpy.yml")

        length = np.random.randint(2, 100)
        array = np.random.rand(length, 1)
        subsitute = np.random.randint(0, len(array) - 1)
        array[subsitute] = np.inf
        pytest.raises(UserError, check_start_values, array)

    dict_ = read(fname_possd)
    pytest.raises(UserError, check_sim_init_dict, dict_)
    pytest.raises(UserError, simulate, fname_possd)

    dict_ = read(fname_zero)
    pytest.raises(UserError, check_sim_distribution, dict_)
    pytest.raises(UserError, fit, fname_zero)

    dict_ = read(fname_vzero)
    pytest.raises(UserError, check_sim_distribution, dict_)
    pytest.raises(UserError, fit, fname_vzero)

    dict_ = read(fname_noparams)
    pytest.raises(UserError, check_sim_distribution, dict_)
    pytest.raises(UserError, fit, fname_noparams)

    dict_ = read(fname_falsespec1)
    pytest.raises(UserError, check_sim_init_dict, dict_)
    pytest.raises(UserError, fit, fname_noparams)

    dict_ = read(fname_falsespec2)
    pytest.raises(UserError, check_sim_init_dict, dict_)
    pytest.raises(UserError, fit, fname_noparams)

    dict_ = read(fname_binary)
    status, _ = check_special_conf(dict_)
    np.testing.assert_equal(status, True)
    pytest.raises(UserError, check_sim_init_dict, dict_)
    pytest.raises(UserError, fit, fname_noparams)


def test8():
    """The test checks if an UserError occurs if wrong inputs are specified for a
    different functions/methods.
    """
    constr = dict()
    constr["DETERMINISTIC"], constr["AGENTS"] = False, 1000
    generate_random_dict(constr)
    df = simulate("test.grmpy.yml")
    dict_ = read("test.grmpy.yml")
    a = list()
    dict_["ESTIMATION"]["file"] = "data.grmpy.yml"
    print_dict(dict_, "false_data")
    pytest.raises(UserError, fit, "tast.grmpy.yml")
    pytest.raises(UserError, fit, "false_data.grmpy.yml")
    pytest.raises(UserError, simulate, "tast.grmpy.yml")
    pytest.raises(UserError, read, "tast.grmpy.yml")
    pytest.raises(UserError, start_values, a, df, "init")
    pytest.raises(UserError, generate_random_dict, a)


def test9():
    """This test ensures that the random initialization file generating process, the
    read in process and the simulation process works if the constraints function allows
    for different number of covariates for each treatment state and the occurence of
    cost-benefit shifters."""
    for _ in range(5):
        constr = dict()
        constr["DETERMINISTIC"], constr["AGENT"], constr["STATE_DIFF"] = (
            False,
            1000,
            True,
        )
        constr["OVERLAP"] = True
        generate_random_dict(constr)
        read("test.grmpy.yml")
        simulate("test.grmpy.yml")
        fit("test.grmpy.yml")

    cleanup()
