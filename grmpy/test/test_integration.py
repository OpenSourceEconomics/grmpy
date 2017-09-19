"""The module includes an integration and a regression test for the simulation process."""
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


class TestClass:
    def test1(self):
        """The test runs a loop to check the consistency of the random init file generating process
        and the following simulation.
        """
        for _ in range(10):
            dict_ = generate_random_dict()
            print_dict(dict_)
            simulate('test.grmpy.ini')

    def test2(self):
        """The test takes a subsample of 5 random entries from the regression battery test list
        (resources/regression_vault.grmpy.json), simulates the specific output again, sums the
        resulting data frame up and checks if the sum is equal to the regarding entry in the test
        list eement.
        """
        if os.path.isfile(os.getcwd() + '/test/resources/regression_vault.grmpy.json'):
            tests = json.load(
                open('{}'.format(os.getcwd()) + '/test/resources/regression_vault.grmpy.json', 'r'))
        else:
            tests = json.load(open('grmpy/test/resources/regression_vault.grmpy.json', 'r'))

        subsample_indices = np.random.choice(len(tests), 5)
        subsample = []
        for i in subsample_indices:
            subsample += [tests[i]]

        for test in subsample:
            if len(test) == 2:
                stat, dict_ = test
            else:
                stat, dict_, criteria = test
            print_dict(dict_)
            df = simulate('test.grmpy.ini')
            np.testing.assert_almost_equal(np.sum(df.sum()), stat)
            if len(test) == 3:
                init_dict = read('test.grmpy.ini')
                start = start_values(init_dict, df, 'true_values')
                criteria_ = calculate_criteria(start, init_dict, df)
                np.testing.assert_array_almost_equal(criteria, criteria_)
        cleanup()
