

from random_init import generate_random_dict
from print_init import print_dict
from import_process import import_process
from simulation import simulation

RUNS = 100

for i in range(1):

    dict_ = generate_random_dict()
    print_dict(dict_, 'init.ini')
    dict_ = import_process('init.ini')
    simulation(dict_)


