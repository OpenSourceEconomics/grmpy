"""The module includes an integration and a regression test for the simulation process."""
import glob
import os
import json

import numpy as np

from grmpy.test.random_init import generate_random_dict
from grmpy.test.random_init import print_dict
from grmpy.simulate.simulate import simulate


class TestClass:
    def test1(self):
        """The test runs a loop to check the consistency of the random init file generating process
        and the following simulation.
        """
        for _ in range(10):

            dict_ = generate_random_dict()
            print_dict(dict_)
            simulate('test.grmpy.ini')

            for f in glob.glob("*.grmpy.*"):
                os.remove(f)

    def test2(self):
        """The test provides a regression test battery. In the first step it loops over 100
        different seeds, generates a random init file for each seed, simulates the resulting data-
        frame, sums it up and saves the sum and the generated dictionary in /grmpy/test/regress-
        ion_test/ dir as a json file. In the second step it reads the json file, prints an init
        file, simulates the dataframe, sums it up again and compares the sum from the first step
        with the one from the second step.
         """
        NUM_TESTS = 100

        np.random.seed(1234235)
        seeds = np.random.randint(0, 1000, size=NUM_TESTS)
        testdir = os.path.dirname('grmpy/test/regression_test/')
        if not os.path.exists(testdir):
            os.makedirs(testdir)

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
                print_dict(dict_)
                df = simulate('test.grmpy.ini')
                np.testing.assert_almost_equal(np.sum(df.sum()), stat)

        for f in glob.glob("*.grmpy.*"):
            os.remove(f)
