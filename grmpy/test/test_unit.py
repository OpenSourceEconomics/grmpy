"""The module provides unit tests for different aspects of the simulation process."""
import numpy as np
import pandas as pd
import pytest

from grmpy.check.auxiliary import read_data
from grmpy.estimate.estimate import fit
from grmpy.estimate.estimate_par import (
    adjust_output,
    backward_transformation,
    calculate_criteria,
    calculate_p_values,
    calculate_se,
    check_rslt_parameters,
    process_data,
    process_output,
    start_value_adjustment,
    start_values,
)
from grmpy.grmpy_config import TEST_RESOURCES_DIR
from grmpy.read.read import read
from grmpy.simulate.simulate import simulate
from grmpy.simulate.simulate_auxiliary import (
    construct_covariance_matrix,
    mte_information,
)
from grmpy.test.auxiliary import cleanup
from grmpy.test.random_init import generate_random_dict, print_dict


def test1():
    """The first test tests whether the relationships in the simulated datasets are
    appropriate in a deterministic and an un-deterministic setting.
    """
    constr = dict()
    for case in ["deterministic", "undeterministic"]:
        if case == "deterministic":
            constr["DETERMINISTIC"] = True
        else:
            constr["DETERMINISTIC"] = True
        for _ in range(10):
            generate_random_dict(constr)
            df = simulate("test.grmpy.yml")
            dict_ = read("test.grmpy.yml")
            x_treated = df[dict_["TREATED"]["order"]]
            y_treated = (
                pd.DataFrame.sum(dict_["TREATED"]["params"] * x_treated, axis=1) + df.U1
            )
            x_untreated = df[dict_["UNTREATED"]["order"]]
            y_untreated = (
                pd.DataFrame.sum(dict_["UNTREATED"]["params"] * x_untreated, axis=1)
                + df.U0
            )

            np.testing.assert_array_almost_equal(df.Y1, y_treated, decimal=5)
            np.testing.assert_array_almost_equal(df.Y0, y_untreated, decimal=5)
            np.testing.assert_array_equal(df.Y[df.D == 1], df.Y1[df.D == 1])
            np.testing.assert_array_equal(df.Y[df.D == 0], df.Y0[df.D == 0])


def test2():
    """The second test  checks whether the relationships hold if the coefficients are zero in
    different setups.
    """
    for _ in range(10):
        for case in ["ALL", "TREATED", "UNTREATED", "CHOICE", "TREATED & UNTREATED"]:
            constr = dict()
            constr["DETERMINISTIC"] = False
            dict_ = generate_random_dict(constr)

            if case == "ALL":
                for section in ["TREATED", "UNTREATED", "CHOICE"]:
                    dict_[section]["params"] = np.array(
                        [0.0] * len(dict_[section]["params"])
                    )
            elif case == "TREATED & UNTREATED":
                for section in ["TREATED", "UNTREATED"]:
                    dict_[section]["params"] = np.array(
                        [0.0] * len(dict_[section]["params"])
                    )
            else:
                dict_[case]["params"] = np.array([0.0] * len(dict_[case]["params"]))

            print_dict(dict_)

            dict_ = read("test.grmpy.yml")
            df = simulate("test.grmpy.yml")
            x_treated = df[dict_["TREATED"]["order"]]
            x_untreated = df[dict_["UNTREATED"]["order"]]

            if case == "ALL":
                np.testing.assert_array_equal(df.Y1, df.U1)
                np.testing.assert_array_equal(df.Y0, df.U0)
            elif case == "TREATED & UNTREATED":
                np.testing.assert_array_equal(df.Y1, df.U1)
                np.testing.assert_array_equal(df.Y0, df.U0)
                np.testing.assert_array_equal(df.Y[df.D == 1], df.U1[df.D == 1])
                np.testing.assert_array_equal(df.Y[df.D == 0], df.U0[df.D == 0])
            elif case == "TREATED":
                y_untreated = (
                    pd.DataFrame.sum(dict_["UNTREATED"]["params"] * x_untreated, axis=1)
                    + df.U0
                )
                np.testing.assert_array_almost_equal(df.Y0, y_untreated, decimal=5)
                np.testing.assert_array_equal(df.Y1, df.U1)

            elif case == "UNTREATED":
                y_treated = (
                    pd.DataFrame.sum(dict_["TREATED"]["params"] * x_treated, axis=1)
                    + df.U1
                )
                np.testing.assert_array_almost_equal(df.Y1, y_treated, decimal=5)
                np.testing.assert_array_equal(df.Y0, df.U0)
            else:
                y_treated = (
                    pd.DataFrame.sum(dict_["TREATED"]["params"] * x_treated, axis=1)
                    + df.U1
                )
                y_untreated = (
                    pd.DataFrame.sum(dict_["UNTREATED"]["params"] * x_untreated, axis=1)
                    + df.U0
                )
                np.testing.assert_array_almost_equal(df.Y1, y_treated, decimal=5)
                np.testing.assert_array_almost_equal(df.Y0, y_untreated, decimal=5)

            np.testing.assert_array_equal(df.Y[df.D == 1], df.Y1[df.D == 1])
            np.testing.assert_array_equal(df.Y[df.D == 0], df.Y0[df.D == 0])


def test3():
    """The fourth test checks whether the simulation process works if there are only
    treated or untreated Agents by setting the number of agents to one. Additionally the
    test checks if the start values for the estimation process are set to the init-
    ialization file values due to perfect separation.
    """
    constr = dict()
    constr["AGENTS"], constr["DETERMINISTIC"] = 1, False
    for _ in range(10):
        generate_random_dict(constr)
        dict_ = read("test.grmpy.yml")
        df = simulate("test.grmpy.yml")
        start = start_values(dict_, df, "auto")
        np.testing.assert_equal(dict_["AUX"]["init_values"][:(-6)], start[:(-4)])


def test4():
    """The fifth test tests the random init file generating process and the import
    process. It generates an random init file, imports it again and compares the entries
    in  both dictionaries.
    """
    for _ in range(10):
        gen_dict = generate_random_dict()
        init_file_name = gen_dict["SIMULATION"]["source"]
        print_dict(gen_dict, init_file_name)
        imp_dict = read(init_file_name + ".grmpy.yml")
        dicts = [gen_dict, imp_dict]
        for section in ["TREATED", "UNTREATED", "CHOICE", "DIST"]:
            np.testing.assert_array_almost_equal(
                gen_dict[section]["params"], imp_dict[section]["params"], decimal=4
            )
            if section in ["TREATED", "UNTREATED", "CHOICE"]:
                for dict_ in dicts:
                    if not dict_[section]["order"] == dict_[section]["order"]:
                        raise AssertionError()
                    if len(dict_[section]["order"]) != len(
                        set(dict_[section]["order"])
                    ):
                        raise AssertionError()
                    if dict_[section]["order"][0] != "X1":
                        raise AssertionError()

        for variable in gen_dict["VARTYPES"].keys():
            if variable not in imp_dict["VARTYPES"].keys():
                raise AssertionError()

            if gen_dict["VARTYPES"][variable] != imp_dict["VARTYPES"][variable]:
                raise AssertionError

        if gen_dict["VARTYPES"]["X1"] != "nonbinary":
            raise AssertionError

        for subkey in ["source", "agents", "seed"]:
            if not gen_dict["SIMULATION"][subkey] == imp_dict["SIMULATION"][subkey]:
                raise AssertionError()

        for subkey in [
            "agents",
            "file",
            "optimizer",
            "start",
            "maxiter",
            "dependent",
            "indicator",
            "comparison",
            "output_file",
        ]:
            if not gen_dict["ESTIMATION"][subkey] == imp_dict["ESTIMATION"][subkey]:
                raise AssertionError()


def test5():
    """The tests checks if the simulation process works even if the covariance between
    U1 and V and U0 and V is equal. Further the test ensures that the mte_information
    function returns the same value for each quantile.
    """
    for _ in range(10):
        generate_random_dict()
        init_dict = read("test.grmpy.yml")

        # We impose that the covariance between the random components of the potential
        # outcomes and the random component determining choice is identical.
        init_dict["DIST"]["params"][2] = init_dict["DIST"]["params"][4]

        # Distribute information
        coeffs_untreated = init_dict["UNTREATED"]["params"]
        coeffs_treated = init_dict["TREATED"]["params"]

        # Construct auxiliary information
        cov = construct_covariance_matrix(init_dict)

        df = simulate("test.grmpy.yml")

        x = df[
            list(set(init_dict["TREATED"]["order"] + init_dict["UNTREATED"]["order"]))
        ]

        q = [0.01] + list(np.arange(0.05, 1, 0.05)) + [0.99]
        mte = mte_information(coeffs_treated, coeffs_untreated, cov, q, x, init_dict)

        # We simply test that there is a single unique value for the marginal treatment
        #  effect.
        np.testing.assert_equal(len(set(mte)), 1)


def test6():
    """The test ensures that the cholesky decomposition and re-composition works
    appropriately. For this purpose the test creates a positive smi definite matrix from
    a Wishart distribution, decomposes this matrix with, reconstruct it and compares the
    matrix with the one that was specified as the input for the decomposition process.
    """
    for _ in range(1000):

        cov = np.random.uniform(0, 1, 2)
        var = np.random.uniform(1, 2, 3)
        aux = [var[0], var[1], cov[0], var[2], cov[1], 1.0]
        dict_ = {"DIST": {"params": aux}}
        before = [var[0], cov[0] / var[0], var[2], cov[1] / var[2]]
        x0 = start_value_adjustment([], dict_, "init")
        after = backward_transformation(x0)
        np.testing.assert_array_almost_equal(before, after, decimal=6)


def test7():
    """We want to able to smoothly switch between generating and printing random
    initialization files.
    """
    for _ in range(10):
        generate_random_dict()
        dict_1 = read("test.grmpy.yml")
        print_dict(dict_1)
        dict_2 = read("test.grmpy.yml")
        np.testing.assert_equal(dict_1, dict_2)


def test8():
    """This test ensures that the random process handles the constraints dict
    appropriately if there the input dictionary is not complete.
    """
    for _ in range(10):
        constr = dict()
        constr["MAXITER"] = np.random.randint(0, 1000)
        constr["START"] = np.random.choice(["start", "init"])
        constr["AGENTS"] = np.random.randint(1, 1000)
        dict_ = generate_random_dict(constr)
        np.testing.assert_equal(constr["AGENTS"], dict_["SIMULATION"]["agents"])
        np.testing.assert_equal(constr["START"], dict_["ESTIMATION"]["start"])
        np.testing.assert_equal(constr["MAXITER"], dict_["ESTIMATION"]["maxiter"])


def test9():
    """This test checks if the start_values function returns the init file values if the
    start option is set to init.
    """
    for _ in range(10):
        constr = dict()
        constr["DETERMINISTIC"] = False
        generate_random_dict(constr)
        dict_ = read("test.grmpy.yml")
        true = []
        for key_ in ["TREATED", "UNTREATED", "CHOICE"]:
            true += list(dict_[key_]["params"])
        df = simulate("test.grmpy.yml")
        x0 = start_values(dict_, df, "init")[:-4]

        np.testing.assert_array_equal(true, x0)


def test10():
    """This test checks if the refactor auxiliary function returns an unchanged init
    file if the maximum number of iterations is set to zero.
    """

    for _ in range(10):
        constr = dict()
        constr["DETERMINISTIC"], constr["AGENTS"] = False, 1000
        constr["MAXITER"], constr["START"] = 0, "init"
        generate_random_dict(constr)
        init_dict = read("test.grmpy.yml")
        df = simulate("test.grmpy.yml")
        start = start_values(init_dict, df, "init")
        start = backward_transformation(start)

        rslt = fit("test.grmpy.yml")

        np.testing.assert_equal(start, rslt["AUX"]["x_internal"])


def test11():
    """This test ensures that the tutorial configuration works as intended."""
    fname = TEST_RESOURCES_DIR + "/tutorial.grmpy.yml"
    simulate(fname)
    fit(fname)


def test12():
    """This test checks if our data import process is able to handle .txt, .dta and .pkl
     files.
     """

    pkl = TEST_RESOURCES_DIR + "/data.grmpy.pkl"
    dta = TEST_RESOURCES_DIR + "/data.grmpy.dta"
    txt = TEST_RESOURCES_DIR + "/data.grmpy.txt"

    real_sum = -3211.20122
    real_column_values = [
        "Y",
        "D",
        "X1",
        "X2",
        "X3",
        "X5",
        "X4",
        "Y1",
        "Y0",
        "U1",
        "U0",
        "V",
    ]

    for data in [pkl, dta, txt]:
        df = read_data(data)
        sum_ = np.sum(df.sum())
        columns = list(df)
        np.testing.assert_array_almost_equal(sum_, real_sum, decimal=5)
        np.testing.assert_equal(columns, real_column_values)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test13():
    """This test checks if functions that affect the estimation output adjustment work as
    intended.
    """
    for _ in range(5):
        generate_random_dict({"DETERMINISTIC": False})
        df = simulate("test.grmpy.yml")
        init_dict = read("test.grmpy.yml")
        start = start_values(init_dict, dict, "init")
        _, X1, X0, Z1, Z0, Y1, Y0 = process_data(df, init_dict)
        init_dict["AUX"]["criteria"] = calculate_criteria(
            init_dict, X1, X0, Z1, Z0, Y1, Y0, start
        )
        init_dict["AUX"]["starting_values"] = backward_transformation(start)

        aux_dict1 = {"crit": {"1": 10}}

        x0, se = [np.nan] * len(start), [np.nan] * len(start)
        index = np.random.randint(0, len(x0) - 1)
        x0[index], se[index] = np.nan, np.nan

        p_values, t_values = calculate_p_values(se, x0, df.shape[0])
        np.testing.assert_array_equal(
            [p_values[index], t_values[index]], [np.nan, np.nan]
        )

        x_processed, crit_processed, _ = process_output(
            init_dict, aux_dict1, x0, "notfinite"
        )

        np.testing.assert_equal(
            [x_processed, crit_processed],
            [init_dict["AUX"]["starting_values"], init_dict["AUX"]["criteria"]],
        )

        check1, flag1 = check_rslt_parameters(
            init_dict, X1, X0, Z1, Z0, Y1, Y0, aux_dict1, start
        )
        check2, flag2 = check_rslt_parameters(
            init_dict, X1, X0, Z1, Z0, Y1, Y0, aux_dict1, x0
        )

        np.testing.assert_equal([check1, flag1], [False, None])
        np.testing.assert_equal([check2, flag2], [True, "notfinite"])

        opt_rslt = {
            "fun": 1.0,
            "success": 1,
            "status": 1,
            "message": "msg",
            "nfev": 10000,
        }
        rslt = adjust_output(
            opt_rslt, init_dict, start, X1, X0, Z1, Z0, Y1, Y0, dict_=aux_dict1
        )
        np.testing.assert_equal(rslt["crit"], opt_rslt["fun"])
        np.testing.assert_equal(rslt["warning"][0], "---")

        x_linalign = [0.0000000000000001] * len(x0)
        num_treated = init_dict["AUX"]["num_covars_treated"]
        num_untreated = num_treated + init_dict["AUX"]["num_covars_untreated"]
        se, hess_inv, conf_interval, p_values, t_values, _ = calculate_se(
            x_linalign, init_dict, X1, X0, Z1, Z0, Y1, Y0, num_treated, num_untreated
        )
        np.testing.assert_equal(se, [np.nan] * len(x0))
        np.testing.assert_equal(hess_inv, np.full((len(x0), len(x0)), np.nan))
        np.testing.assert_equal(conf_interval, [[np.nan, np.nan]] * len(x0))
        np.testing.assert_equal(t_values, [np.nan] * len(x0))
        np.testing.assert_equal(p_values, [np.nan] * len(x0))

    cleanup()
