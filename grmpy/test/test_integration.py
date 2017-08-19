import glob
import os

from grmpy.read.read import read
from grmpy.test.random_init import generate_random_dict
from grmpy.test.random_init import print_dict
from grmpy.simulation.simulation import simulate


class TestClass():
    def test1(self):
        for i in range(1):

            dict_ = generate_random_dict()
            print_dict(dict_)
            dict_ = read('test.grmpy.ini')
            simulate(dict_)

            for f in glob.glob("*.grmpy.*"):
                os.remove(f)






