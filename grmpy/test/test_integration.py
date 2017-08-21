import glob
import os

from grmpy.test.random_init import generate_random_dict
from grmpy.test.random_init import print_dict
from grmpy.simulation.simulation import simulate


class TestClass:
    def test1(self):
        for i in range(1):

            dict_ = generate_random_dict()
            print_dict(dict_)
            simulate('test.grmpy.ini')

            for f in glob.glob("*.grmpy.*"):
                os.remove(f)






