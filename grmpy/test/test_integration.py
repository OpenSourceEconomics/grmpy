"""The module includes an integration and a regression test for the simulation process."""
import glob
import json
import os


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
        """The test takes a subsample of 5 random entries from the regression battery test list
        (resources/regression_vault.grmpy.json), simulates the specific output again, sums the
        resulting data frame up and checks if the sum is equal to the regarding entry in the test
        list eement.
        """
        tests = json.load(open('grmpy/test/resources/regression_vault.grmpy.json', 'r'))

        subsample_indices = np.random.choice(len(tests), 5)
        subsample = []
        for i in subsample_indices:
            subsample += [tests[i]]

        for test in subsample:
            stat, dict_ = test
            print_dict(dict_)
            df = simulate('test.grmpy.ini')
            np.testing.assert_almost_equal(np.sum(df.sum()), stat)

        for f in glob.glob("*.grmpy.*"):
            os.remove(f)







