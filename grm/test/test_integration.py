import os
import pytest

import sys
import os.path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))



from simulation.simulation import simulation
from simulation.random_init import generate_random_dict
from simulation.random_init import constraints
from auxiliary.import_process import import_process
from auxiliary.print_init import print_dict


class TestClass():
    def test1(self):
        RUNS = 100
        for i in range(RUNS):

            dict_ = generate_random_dict()
            print_dict(dict_)
            dict_= import_process('test.grmpy.ini')
            simulation(dict_)

            files = [f for f in os.listdir('.') if os.path.isfile(f)]
            files = [f for f in files if '.grmpy.' in f ]

            for f in files:
                os.remove(f)






