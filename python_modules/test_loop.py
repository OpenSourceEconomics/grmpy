
from random_init import generate_random_dict
from print_init import print_dict
from import_process import import_process
from simulation import simulation


RUNS = 100

for i in range(RUNS):

    is_deterministic, dict_ = generate_random_dict()
    init_file_name= dict_['SIMULATION']['source']
    print_dict(dict_, init_file_name)
    dict_= import_process(init_file_name +'.grmpy.ini', is_deterministic)
    simulation(dict_)



