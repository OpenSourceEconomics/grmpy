"""The test provides a regression test battery. In the first step it loops over 100 different seeds,
generates a random init file for each seed, simulates the resulting dataframe, sums it up and saves
the sum and the generated dictionary in /grmpy/test/resources/ as a json file. In the second step it
reads the json file, and loops over all entries. For each element it prints an init file, simulates
the dataframe, sums it up again and compares the sum from the first step with the one from the
second step.
 """

import json
import os

import numpy as np

from grmpy.estimate.estimate_auxiliary import calculate_criteria
from grmpy.estimate.estimate_auxiliary import start_values
from grmpy.test.random_init import generate_random_dict
from grmpy.test.random_init import print_dict
from grmpy.simulate.simulate import simulate
from grmpy.test.auxiliary import cleanup
from grmpy.read.read import read
NUM_TESTS = 100

np.random.seed(1234235)
seeds = np.random.randint(0, 1000, size=NUM_TESTS)
dir = os.path.dirname(__file__)
file_dir = os.path.join(dir, 'regression_vault.grmpy.json')

if True:
    tests = []
    for seed in seeds:
        np.random.seed(seed)
        dict_ = generate_random_dict()
        df = simulate('test.grmpy.ini')
        if 0.00 in dict_['DIST']['coeff']:
            stat = np.sum(df.sum())
            tests += [(stat, dict_)]
        else:
            stat = np.sum(df.sum())
            init_dict = read('test.grmpy.ini')
            start = start_values(init_dict, df, 'true_values')
            criteria = calculate_criteria(start, init_dict, df)
            tests += [(stat, dict_, criteria)]
    json.dump(tests, open(file_dir, 'w'))

if True:
    tests = json.load(open(file_dir, 'r'))

    for test in tests:
        if len(test) == 2:
            stat, dict_ = test
        else:
            stat, dict_, criteria = test
        print_dict(dict_)
        init_dict = read('test.grmpy.ini')
        df = simulate('test.grmpy.ini')
        np.testing.assert_almost_equal(np.sum(df.sum()), stat)
        if len(test) == 3:
            start = start_values(init_dict, df, 'true_values')
            criteria_ = calculate_criteria(start, init_dict, df)
            np.testing.assert_array_almost_equal(criteria, criteria_)

cleanup('regression')