""" This module allows to manually explore the capabilities of the grmpy package.
"""
import sys

sys.path.insert(0, "../../")

from grmpy.test.random_init import generate_random_dict
from grmpy.simulation.simulation import simulate
from grmpy.test.random_init import print_dict
from grmpy.test.random_init import constraints
from grmpy.read.read import read

constr= constraints(agents=1)
print(constr)


dict_ = generate_random_dict(constr)
print_dict(dict_)
simulate('test.grmpy.ini')
