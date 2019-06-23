"""The test provides a regression test battery. In the first step it loops over 100
different seeds, generates a random init file for each seed, simulates the resulting
dataframe, sums it up and saves the sum and the generated dictionary in
/grmpy/test/resources/ as a json file. In the second step it reads the json file, and
loops over all entries. For each element it prints an init file, simulates the dataframe,
sums it up again and compares the sum from the first step with the one from the second
step.
 """

import json
import os

import numpy as np

from grmpy.estimate.estimate_auxiliary import (
    calculate_criteria,
    process_data,
    start_values,
)
from grmpy.read.read import read
from grmpy.simulate.simulate import simulate
from grmpy.test.auxiliary import cleanup
from grmpy.test.random_init import generate_random_dict, print_dict

NUM_TESTS = 100

np.random.seed(1234235)
seeds = np.random.randint(0, 1000, size=NUM_TESTS)
directory = os.path.dirname(__file__)
file_dir = os.path.join(directory, "old_regression_vault.grmpy.json")

if True:
    tests = []
    for seed in seeds:
        np.random.seed(seed)
        constr = dict()
        constr["DETERMINISTIC"], constr["CATEGORICAL"] = False, False
        dict_ = generate_random_dict(constr)
        df = simulate("test.grmpy.yml")
        stat = np.sum(df.sum())
        init_dict = read("test.grmpy.yml")
        start = start_values(init_dict, df, "init")
        _, X1, X0, Z1, Z0, Y1, Y0 = process_data(df, init_dict)
        criteria = calculate_criteria(init_dict, X1, X0, Z1, Z0, Y1, Y0, start)
        tests += [(stat, dict_, criteria)]
    json.dump(tests, open(file_dir, "w"))

if True:
    tests = json.load(open(file_dir, "r"))

    for test in tests:
        stat, dict_, criteria = test
        print_dict(dict_)
        init_dict = read("test.grmpy.yml")
        df = simulate("test.grmpy.yml")
        start = start_values(init_dict, df, "init")
        criteria_ = calculate_criteria(init_dict, df, start)
        np.testing.assert_array_almost_equal(criteria, criteria_)
        np.testing.assert_almost_equal(np.sum(df.sum()), stat)

cleanup("regression")
