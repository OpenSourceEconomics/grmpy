"""The module includes an integration and a regression test for the simulation process."""
import pytest
import json
import os

import numpy as np
from grmpy.estimate.estimate_auxiliary import calculate_criteria
from grmpy.estimate.estimate_auxiliary import simulate_estimation
from grmpy.estimate.estimate_auxiliary import start_values
from grmpy.test.random_init import generate_random_dict
from grmpy.test.random_init import constraints
from grmpy.test.random_init import print_dict
from grmpy.estimate.estimate import estimate
from grmpy.simulate.simulate import simulate
from grmpy.test.auxiliary import read_desc
from grmpy.test.auxiliary import cleanup
from grmpy.read.read import read
import grmpy


@pytest.mark.usefixtures('fresh_directory', 'set_seed')
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
            start = start_values(init_dict, df, 'init')
            criteria_ = calculate_criteria(init_dict, df, start)
            np.testing.assert_array_almost_equal(criteria, criteria_)
            np.testing.assert_almost_equal(np.sum(df.sum()), stat)

    def test3(self):
        """The test checks if the criteria function value of the simulated and the 'estimated'
        sample is equal if both samples include an identical number of individuals.
        """
        for _ in range(5):
            constr = constraints(probability=0.0, agents=10000, start='init',
                                 optimizer='SCIPY-BFGS')
            dict_ = generate_random_dict(constr)
            print_dict(dict_)

            df1 = simulate('test.grmpy.ini')
            rslt = estimate('test.grmpy.ini')
            init_dict = read('test.grmpy.ini')
            df2 = simulate_estimation(init_dict, rslt, df1)
            start = start_values(init_dict, df1, 'init')

            criteria = []
            for data in [df1, df2]:
                criteria += [calculate_criteria(init_dict, data, start)]
            np.testing.assert_allclose(criteria[1], criteria[0], rtol=0.1)

    def test4(self):
        """The test checks if the estimation process works if the Powell algorithm is specified as
        the optimizer option.
        """
        for _ in range(5):
            constr = constraints(probability=0.0, agents=10000, start='init',
                                 optimizer='SCIPY-POWELL')
            generate_random_dict(constr)

            simulate('test.grmpy.ini')
            estimate('test.grmpy.ini')

    def test5(self):
        """The test checks if the estimation process works properly when maxiter is set to
        zero.
        """
        for _ in range(10):
            constr = constraints(probability=0.0, maxiter=0)
            generate_random_dict(constr)
            simulate('test.grmpy.ini')
            estimate('test.grmpy.ini')

    def test6(self):
        """Additionally to test5 this test checks if the descriptives file provides the expected
        output when maxiter is set to zero and the estimation process uses the initialization file
        values as start values.
        """
        for _ in range(5):
            constr = constraints(probability=0.0, maxiter=0, agents=1000, start='init')
            generate_random_dict(constr)
            simulate('test.grmpy.ini')
            estimate('test.grmpy.ini')
            dict_ = read_desc('descriptives.grmpy.txt')
            for key_ in ['All', 'Treated', 'Untreated']:
                np.testing.assert_equal(len(set(dict_[key_]['Number'])), 1)
                np.testing.assert_array_equal(dict_[key_]['Observed Sample'],
                                              dict_[key_]['Simulated Sample (finish)'])
                np.testing.assert_array_equal(dict_[key_]['Simulated Sample (finish)'],
                                              dict_[key_]['Simulated Sample (start)'])
        cleanup()
