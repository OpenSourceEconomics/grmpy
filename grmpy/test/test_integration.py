"""The module includes an integration and a regression test for the simulation process."""
import json
import os

import numpy as np

from grmpy.test.random_init import generate_random_dict
from grmpy.test.random_init import print_dict
from grmpy.simulate.simulate import simulate
from grmpy.test.auxiliary import cleanup


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
        import grmpy
        fname = os.path.dirname(grmpy.__file__) + '/test/resources/regression_vault.grmpy.json'
        tests = json.load(open(fname))

        for i in np.random.choice(range(len(tests)), size=5):
            stat, dict_ = tests[i]
            print_dict(dict_)
            df = simulate('test.grmpy.ini')
            np.testing.assert_almost_equal(np.sum(df.sum()), stat)
            cleanup()
