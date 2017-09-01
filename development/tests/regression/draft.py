"""The module serves as a first draft for a regression tests battery."""
import random
import json
import os

import numpy as np

from grmpy.test.random_init import generate_random_dict
from grmpy.test.random_init import constraints
from grmpy.test.random_init import print_dict
from grmpy.simulate.simulate import simulate
from grmpy.read.read import read

NUM_TESTS = 100

np.random.seed(123)
# TODO: I do not want us to introduce randomness in the package through the random package as
# well. Please look for a numpy replacement.
random.seed(123)
seeds = np.random.randint(0, 1000, size=NUM_TESTS)

if not os.path.exists("grmpy/test/regression_test/"):
    os.makedirs("grmpy/test/regression_test")

if True:
    tests = []
    for seed in seeds:

        np.random.seed(seed)
        dict_ = generate_random_dict()
        df = simulate('test.grmpy.ini')
        stat = np.sum(df.sum())

        tests += [(stat, dict_)]

    json.dump(tests, open('grmpy/test/regression_test/regression_vault.grmpy.json', 'w'))

if True:
    tests = json.load(open('grmpy/test/regression_test/regression_vault.grmpy.json', 'r'))

    for test in tests:
        stat, dict_ = test
        df = simulate('test.grmpy.ini')

        np.testing.assert_almost_equal(np.sum(df.sum()), stat)





