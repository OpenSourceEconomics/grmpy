"""The test provides a regression test battery. In the first step it loops over 100 different seeds,
generates a random init file for each seed, simulates the resulting dataframe, sums it up and saves
the sum and the generated dictionary in /grmpy/test/resources/ as a json file. In the second step it
reads the json file, and loops over all entries. For each element it prints an init file, simulates
the dataframe, sums it up again and compares the sum from the first step with the one from the
second step."""

import glob
import json
import os

import numpy as np

import grmpy
from grmpy.test.random_init import generate_random_dict
from grmpy.test.random_init import print_dict
from grmpy.simulate.simulate import simulate

PACKAGE_DIR = os.path.dirname(grmpy.__file__)

NUM_TESTS = 100

np.random.seed(1234235)
seeds = np.random.randint(0, 1000, size=NUM_TESTS)
dir = os.path.dirname(__file__)
file_dir = os.path.join(dir, 'regression_vault.grmpy.json')

if False:
    tests = []
    for seed in seeds:
        np.random.seed(seed)
        dict_ = generate_random_dict()
        df = simulate('test.grmpy.ini')
        stat = np.sum(df.sum())
        tests += [(stat, dict_)]

    json.dump(tests, open(file_dir, 'w'))

if True:

    fname = PACKAGE_DIR + '/test/resources/regression_vault.grmpy.json'
    tests = json.load(open(fname, 'r'))

    for test in tests:
        stat, dict_ = test
        print_dict(dict_)
        df = simulate('test.grmpy.ini')
        np.testing.assert_almost_equal(np.sum(df.sum()), stat)
#
# for f in glob.glob("*.grmpy.*"):
#     if f.startswith('regression'):
#         pass
#     else:
#         os.remove(f)
