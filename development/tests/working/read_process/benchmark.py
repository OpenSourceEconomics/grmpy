import json

import numpy as np

from development.tests.working.init_generator.init_file_generator import print_dict_new
from development.tests.working.read_process.init_file_import_process import read_new
from development.tests.working.read_process.auxiliary import attr_dict_to_init_dict
from development.tests.working.init_generator.init_file_generator import first_try
from development.tests.working.read_process.auxiliary import simulate_new
from grmpy.test.random_init import print_dict
from grmpy.test.auxiliary import cleanup
from grmpy.read.read import read


Test1, Test2, Test3, Test4 = False, False, False, True

if Test1:
    for _ in range(1000):
        dict_start = first_try()
        new_dict = read_new('reliability.grmpy.yml')
        dict_end = attr_dict_to_init_dict(new_dict)

        np.testing.assert_equal(dict_start, dict_end)

if Test2:
    old_init = read('tutorial.grmpy.ini')
    new_dict = read_new('reliability.grmpy.yml')

    for entry in old_init.keys():

        if entry in new_dict.keys():
            pass
        else:
            print('\n The entry {} is not in included in the new init dictionary. \n'.format(entry))
        if isinstance(old_init[entry], dict):
            for subentry in old_init[entry].keys():
                if subentry in new_dict[entry].keys():
                    pass
                else:
                    print('\n The subentry {} is not in included in the {} section of the new init'
                          ' dictionary. \n'.format(subentry, entry))

if Test3:
    tests = json.load(open('regression_vault.grmpy.json', 'r'))
    for test in tests:
        stat, dict_, criteria = test
        print_dict(dict_)
        init_dict = read('test.grmpy.ini')

        init_dict = attr_dict_to_init_dict(init_dict)
        print_dict_new(init_dict)

        df = simulate_new('reliability.grmpy.yml')

        np.testing.assert_almost_equal(np.sum(df.sum()), stat)

if Test4:
    init_dict = read('reliability.grmpy.ini')
    dict_ = attr_dict_to_init_dict(init_dict)
    print(dict_)
    print_dict_new(dict_)

