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
from grmpy.grmpy_config import TEST_RESOURCES_DIR
from grmpy.read.read import read
from grmpy.simulate.simulate import simulate
from grmpy.test.auxiliary import cleanup, dict_transformation
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
        D, X1, X0, Z1, Z0, Y1, Y0 = process_data(df, init_dict)
        start = start_values(init_dict, D, X1, X0, Z1, Z0, Y1, Y0, "init")

        criteria_ = calculate_criteria(start, X1, X0, Z1, Z0, Y1, Y0)
        np.testing.assert_almost_equal(np.sum(df.sum()), stat)
        np.testing.assert_array_almost_equal(criteria, criteria_)


def test3():
    """The test checks if the estimation process works if the Powell algorithm is
    specified as the optimizer option.
    """
    constr = {
        "DETERMINISTIC": False,
        "AGENTS": 10000,
        "START": "init",
        "OPTIMIZER": "POWELL",
    }

    for _ in range(5):
        generate_random_dict(constr)
        simulate("test.grmpy.yml")
        fit("test.grmpy.yml")


def test4():
    """The test checks if the estimation process works properly when maxiter is set to
    zero.
    """
    constr = {"DETERMINISTIC": False, "MAXITER": 0}
    for _ in range(5):
        generate_random_dict(constr)
        simulate("test.grmpy.yml")
        fit("test.grmpy.yml")


def test5():
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


def test6():
    """The test checks if an UserError occurs if wrong inputs are specified for a
    different functions/methods.
    """
    constr = {"DETERMINISTIC": False, "AGENTS": 1000}
    generate_random_dict(constr)
    simulate("test.grmpy.yml")
    dict_ = read("test.grmpy.yml")
    a = []
    dict_["ESTIMATION"]["file"] = "data.grmpy.yml"
    print_dict(dict_, "false_data")
    pytest.raises(UserError, fit, "tast.grmpy.yml")
    pytest.raises(UserError, fit, "false_data.grmpy.yml")
    pytest.raises(UserError, simulate, "tast.grmpy.yml")
    pytest.raises(UserError, read, "tast.grmpy.yml")
    pytest.raises(UserError, generate_random_dict, a)


def test7():
    """This test ensures that the random initialization file generating process, the
    read in process and the simulation process works if the constraints function allows
    for different number of covariates for each treatment state and the occurence of
    cost-benefit shifters.
    """

    constr = {
        "DETERMINISTIC": False,
        "AGENTS": 1000,
        "STATE_DIFF": True,
        "OVERLAP": True,
    }

    for _ in range(5):
        generate_random_dict(constr)
        read("test.grmpy.yml")
        simulate("test.grmpy.yml")
        fit("test.grmpy.yml")

    cleanup()
