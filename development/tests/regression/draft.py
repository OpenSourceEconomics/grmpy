"""The module serves as a first draft for a regression tests battery."""
import random
import json

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

if True:
    tests = []
    for seed in seeds:

        np.random.seed(seed)
        dict_ = generate_random_dict()
        print_dict(dict_)

        df = simulate('test.grmpy.ini')
        stat = np.sum(df.sum())

        # TODO: Can we simply move from array's to list in the random generation procedure?
        for key in dict_.keys():
            if 'coeff' in dict_[key].keys():
                dict_[key]['coeff'] = dict_[key]['coeff'].tolist()

        tests += [(stat, dict_)]

    json.dump(tests, open('regression_vault.grmpy.json', 'w'))

if True:
    tests = json.load(open('regression_vault.grmpy.json', 'r'))

    for test in tests:
        stat, dict_ = test

        print_dict(dict_)
        df = simulate('test.grmpy.ini')

        np.testing.assert_almost_equal(np.sum(df.sum()), stat)




    print(tests)

