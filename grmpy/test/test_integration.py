import os
import pytest

import sys
import os.path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))



from simulation.simulation import simulate
from random_init import generate_random_dict
from auxiliary.import_process import import_process
from auxiliary.print_init import print_dict


class TestClass():
    def test1(self):
        RUNS = 100
        for i in range(1):

            dict_ = generate_random_dict()
            print_dict(dict_)
            dict_ = import_process('test.grmpy.ini')
            simulate(dict_)

            files = [f for f in os.listdir('.') if os.path.isfile(f)]
            files = [f for f in files if '.grmpy.' in f ]

            for f in files:
                os.remove(f)






