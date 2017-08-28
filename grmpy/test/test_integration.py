"""The module includes an integeration test for the simulation process."""

import glob
import os

from grmpy.test.random_init import generate_random_dict
from grmpy.test.random_init import print_dict
from grmpy.simulate.simulate import simulate


class TestClass:
    def test1(self):
        """The test runs a loop to check the consistency of the random init file generating process and the following
        simulation.
        """
        for _ in range(10):

            dict_ = generate_random_dict()
            print_dict(dict_)
            simulate('test.grmpy.ini')

            for f in glob.glob("*.grmpy.*"):
                os.remove(f)
