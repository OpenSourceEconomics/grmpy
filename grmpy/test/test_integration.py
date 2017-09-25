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
        """This test runs a random selection of five regression tests from the package's
        regression test vault.
        """
        fname = os.path.dirname(grmpy.__file__) + '/test/resources/regression_vault.grmpy.json'
        tests = json.load(open(fname))

        for i in np.random.choice(range(len(tests)), size=5):
            stat, dict_, criteria = tests[i]
            print_dict(dict_)
            df = simulate('test.grmpy.ini')
            init_dict = read('test.grmpy.ini')
            start = start_values(init_dict, df, 'true_values')
            criteria_ = calculate_criteria(start, init_dict, df)
            np.testing.assert_array_almost_equal(criteria, criteria_)
            np.testing.assert_almost_equal(np.sum(df.sum()), stat)
            cleanup()
