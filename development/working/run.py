""" This module allows to manually explore the capabilities of the grmpy package. """
import sys

sys.path.insert(0, "../../")

from grmpy.test.random_init import generate_random_dict
from grmpy.simulate.simulate import simulate
from grmpy.test.random_init import print_dict

dict_ = generate_random_dict()
print_dict(dict_)
simulate('test.grmpy.ini')
