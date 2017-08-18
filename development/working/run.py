""" This module allows to manually explore the capabilities of the grmpy package.
"""
import sys
import os
sys.path.insert(0, "../../grmpy")

from simulation.simulation import simulate
from simulation.random_init import generate_random_dict
from auxiliary.import_process import import_process
from auxiliary.print_init import print_dict

#dict_ = generate_random_dic()
#print_dict(dict_)
dict_ = import_process('test.grmpy.ini')
simulate(dict_)
