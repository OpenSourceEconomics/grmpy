""" This module allows to manually explore the capabilities of the grmpy package.
"""
import sys
from grmpy.simulation.simulation import simulate
from grmpy.read.read import read
from grmpy.test.random_init import generate_random_dict
from grmpy.test.random_init import print_dict

sys.path.insert(0, "../../")

dict_ = generate_random_dict()
print_dict(dict_)
dict_ = read('test.grmpy.ini')
print(dict_)
simulate(dict_)
