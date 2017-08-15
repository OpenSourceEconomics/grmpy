

import numpy as np
import pandas as pd
import pytest


from simulation import simulation
from random_init import generate_random_dict
from print_init import print_dict
from import_process import import_process


class TestClass():
    def test1(self):
        '''Testing relations in simulated dataset'''

        is_deterministic, dict_ = generate_random_dict()
        init_file_name = dict_['SIMULATION']['source']
        print_dict(dict_, init_file_name)
        dict_ = import_process(init_file_name + '.grmpy.ini', is_deterministic)
        print(dict_)
        df, Y, Y_1, Y_0, D, X, Z, U, V = simulation(dict_)

        # endogeneous variables

        assert Y_1.all() == (
            np.dot(dict_['TREATED']['all'], X.T) + U[0:, 1]).all()
        assert Y_0.all() == (
            np.dot(dict_['UNTREATED']['all'], X.T) + U[0:, 0]).all()
        assert Y[D == 1].all() == Y_1.all()
        assert Y[D == 0].all() == Y_0.all()
        assert V.all() == (U[0:, 2] - U[0:, 1] + U[0:, 0]).all()

    def test2(self):
        '''Testing whether relationships hold if process is deterministic'''

        is_deterministic, dict_= generate_random_dict(deterministic = 1.0)
        init_file_name = dict_['SIMULATION']['source']
        print_dict(dict_, init_file_name)
        dict_ = import_process(init_file_name + '.grmpy.ini', is_deterministic)
        df, Y, Y_1, Y_0, D, X, Z, U, V = simulation(dict_)

        assert Y_1.all() == U[0:, 1].all()
        assert Y_0.all() == U[0:, 0].all()
        assert Y[D == 1].all() == Y_1.all()
        assert Y[D == 0].all() == Y_0.all()
        assert V.all() == (U[0:, 2] - U[0:, 1] + U[0:, 0]).all()
    def test3(self):
        '''Testing whether relationships hold if coefficients are zero in different setups'''


        # All Coefficients
        is_deterministic, dict_= generate_random_dict(deterministic = .0)
        init_file_name = dict_['SIMULATION']['source']
        print_dict(dict_, init_file_name)
        dict_ = import_process(init_file_name + '.grmpy.ini', is_deterministic)

        for key_ in ['TREATED', 'UNTREATED', 'COST']:
            dict_[key_]['all'] = np.array([0]*len(dict_[key_]['all']))

        df, Y, Y_1, Y_0, D, X, Z, U, V = simulation(dict_)

        assert Y_1.all() == U[0:, 1].all()
        assert Y_0.all() == U[0:, 0].all()
        assert Y[D == 1].all() == Y_1.all()
        assert Y[D == 0].all() == Y_0.all()
        assert V.all() == (U[0:, 2] - U[0:, 1] + U[0:, 0]).all()

        # Treated and Untreated
        is_deterministic, dict_= generate_random_dict(deterministic = .0)
        init_file_name = dict_['SIMULATION']['source']
        print_dict(dict_, init_file_name)
        dict_ = import_process(init_file_name + '.grmpy.ini', is_deterministic)

        for key_ in ['TREATED', 'UNTREATED']:
            dict_[key_]['all'] = np.array([0]*len(dict_[key_]['all']))

        df, Y, Y_1, Y_0, D, X, Z, U, V = simulation(dict_)

        assert Y_1.all() == U[0:, 1].all()
        assert Y_0.all() == U[0:, 0].all()
        assert Y[D == 1].all() == Y_1.all()
        assert Y[D == 0].all() == Y_0.all()
        assert V.all() == (U[0:, 2] - U[0:, 1] + U[0:, 0]).all()


        # Only Treated
        is_deterministic, dict_= generate_random_dict(deterministic = .0)
        init_file_name = dict_['SIMULATION']['source']
        print_dict(dict_, init_file_name)
        dict_ = import_process(init_file_name + '.grmpy.ini', is_deterministic)

        dict_['TREATED']['all'] = np.array([0]*len(dict_['TREATED']['all']))

        df, Y, Y_1, Y_0, D, X, Z, U, V = simulation(dict_)

        assert Y_1.all() == U[0:, 1].all()
        assert Y_0.all() == (
            np.dot(dict_['UNTREATED']['all'], X.T) + U[0:, 0]).all()

        assert Y[D == 1].all() == Y_1.all()
        assert Y[D == 0].all() == Y_0.all()
        assert V.all() == (U[0:, 2] - U[0:, 1] + U[0:, 0]).all()

        # Only Untreated
        is_deterministic, dict_= generate_random_dict(deterministic = .0)
        init_file_name = dict_['SIMULATION']['source']
        print_dict(dict_, init_file_name)
        dict_ = import_process(init_file_name + '.grmpy.ini', is_deterministic)

        dict_['UNTREATED']['all'] = np.array([0]*len(dict_['UNTREATED']['all']))

        df, Y, Y_1, Y_0, D, X, Z, U, V = simulation(dict_)

        assert Y_1.all() == (
            np.dot(dict_['TREATED']['all'], X.T) + U[0:, 1]).all()
        assert Y_0.all() == U[0:, 0].all()
        assert Y[D == 1].all() == Y_1.all()
        assert Y[D == 0].all() == Y_0.all()
        assert V.all() == (U[0:, 2] - U[0:, 1] + U[0:, 0]).all()

        # Only Cost
        is_deterministic, dict_= generate_random_dict(deterministic = .0)
        init_file_name = dict_['SIMULATION']['source']
        print_dict(dict_, init_file_name)
        dict_ = import_process(init_file_name + '.grmpy.ini', is_deterministic)

        dict_['COST']['all'] = np.array([0]*len(dict_['COST']['all']))

        df, Y, Y_1, Y_0, D, X, Z, U, V = simulation(dict_)

        assert Y_1.all() == (
            np.dot(dict_['TREATED']['all'], X.T) + U[0:, 1]).all()
        assert Y_0.all() == (
            np.dot(dict_['UNTREATED']['all'], X.T) + U[0:, 0]).all()
        assert Y[D == 1].all() == Y_1.all()
        assert Y[D == 0].all() == Y_0.all()
        assert V.all() == (U[0:, 2] - U[0:, 1] + U[0:, 0]).all()














