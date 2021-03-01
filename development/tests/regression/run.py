#!/usr/bin/env python
"""The test provides the basic capabilities to work with the regression tests of the
package.
"""
import os

import argparse
import json
import numpy as np

import grmpy
from grmpy.estimate.estimate_par import calculate_criteria, process_data, start_values
from grmpy.read.read import read
from grmpy.simulate.simulate import simulate
from grmpy.test.auxiliary import cleanup, dict_transformation
from grmpy.test.random_init import generate_random_dict, print_dict


def process_arguments(parser):
    """This function parses the input arguments."""
    args = parser.parse_args()
    # Distribute input arguments
    request = args.request
    if "num_tests" in args:
        num_tests = int(args.num_tests)
    else:
        num_tests = None

    # Test validity of input arguments
    if request not in ["check", "create"]:
        raise AssertionError()
    if num_tests not in [i for i in np.arange(1001)]:
        raise AssertionError(9)
    return request, num_tests


def create_vault(num_tests=100, seed=123):
    """This function creates a new regression vault."""
    np.random.seed(seed)

    tests = []
    for _ in range(num_tests):
        dict_ = generate_random_dict()
        init_dict = read("test.grmpy.yml")
        df = simulate("test.grmpy.yml")
        _, X1, X0, Z1, Z0, Y1, Y0 = process_data(df, init_dict)
        x0 = start_values(init_dict, df, "init")
        criteria = calculate_criteria(init_dict, X1, X0, Z1, Z0, Y1, Y0, x0)
        stat = np.sum(df.sum())
        tests += [(stat, dict_, criteria)]
        cleanup()

    json.dump(tests, open("regression_vault.grmpy.json", "w"))


def check_vault(num_tests=100):
    """This function checks the complete regression vault that is distributed as part of
    the package.
    """
    fname = (
        os.path.dirname(grmpy.__file__)
        + "/test/resources/old_regression_vault.grmpy.json"
    )
    tests = json.load(open(fname))

    if num_tests > len(tests):
        print(
            "The specified number of evaluations is larger than the number"
            " of entries in the regression_test vault.\n"
            "Therefore the test runs the complete test battery."
        )
    else:
        tests = [tests[i] for i in np.random.choice(len(tests), num_tests)]

    for test in tests:
        stat, dict_, criteria = test
        print_dict(dict_transformation(dict_))
        init_dict = read("test.grmpy.yml")
        df = simulate("test.grmpy.yml")
        _, X1, X0, Z1, Z0, Y1, Y0 = process_data(df, init_dict)
        x0 = start_values(init_dict, df, "init")
        criteria_ = calculate_criteria(init_dict, X1, X0, Z1, Z0, Y1, Y0, x0)
        np.testing.assert_almost_equal(criteria_, criteria)
        np.testing.assert_almost_equal(np.sum(df.sum()), stat)
        cleanup("regression")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Work with regression tests for package."
    )

    parser.add_argument(
        "--request",
        action="store",
        dest="request",
        required=True,
        choices=["check", "create"],
        help="request",
    )

    parser.add_argument(
        "--num_tests",
        action="store",
        dest="num_tests",
        required=False,
        choices=[str(i) for i in np.arange(1001)],
        help="num_tests",
    )

    request, num_tests = process_arguments(parser)

    if request == "check":
        if num_tests is None:
            check_vault()
        else:
            check_vault(num_tests)
    elif request == "create":
        if num_tests is None:
            create_vault()
        else:
            create_vault(num_tests)
