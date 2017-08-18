

import numpy as np
import pandas as pd
import pytest
import os
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
        '''Testing relations in simulated dataset'''
        for i in range(100):
            dict_ = generate_random_dict()
            print_dict(dict_)
            dict_ = import_process('test.grmpy.ini')
            df, Y, Y_1, Y_0, D, X, Z, U, V = simulation(dict_)

            # endogeneous variables

            assert Y_1.all() == (
                np.dot(dict_['TREATED']['all'], X.T) + U[0:, 1]).all()
            assert Y_0.all() == (
                np.dot(dict_['UNTREATED']['all'], X.T) + U[0:, 0]).all()
            assert Y[D == 1].all() == Y_1.all()
            assert Y[D == 0].all() == Y_0.all()
            assert V.all() == (U[0:, 2] - U[0:, 1] + U[0:, 0]).all()

            files = [f for f in os.listdir('.') if os.path.isfile(f)]
            files = [f for f in files if '.grmpy.' in f]

            for f in files:
                os.remove(f)

    def test2(self):
        '''Testing if relationships hold if process is deterministic'''
        constr = constraints(probability=1.)
        for i in range(100):
            dict_ = generate_random_dict(constr)
            print_dict(dict_)
            dict_ = import_process('test.grmpy.ini')
            df, Y, Y_1, Y_0, D, X, Z, U, V = simulation(dict_)

            assert Y_1.all() == U[0:, 1].all()
            assert Y_0.all() == U[0:, 0].all()
            assert Y[D == 1].all() == Y_1.all()
            assert Y[D == 0].all() == Y_0.all()
            assert V.all() == (U[0:, 2] - U[0:, 1] + U[0:, 0]).all()

            files = [f for f in os.listdir('.') if os.path.isfile(f)]
            files = [f for f in files if '.grmpy.' in f]

            for f in files:
                os.remove(f)

    def test3(self):
        '''Testing if relationships hold if coefficients are zero in different setups'''
        for i in range(100):
        # All Coefficients
            dict_ = generate_random_dict()
            print_dict(dict_)
            dict_ = import_process('test.grmpy.ini')

            for key_ in ['TREATED', 'UNTREATED', 'COST']:
                dict_[key_]['all'] = np.array([0] * len(dict_[key_]['all']))

            df, Y, Y_1, Y_0, D, X, Z, U, V = simulation(dict_)

            assert Y_1.all() == U[0:, 1].all()
            assert Y_0.all() == U[0:, 0].all()
            assert Y[D == 1].all() == Y_1.all()
            assert Y[D == 0].all() == Y_0.all()
            assert V.all() == (U[0:, 2] - U[0:, 1] + U[0:, 0]).all()

            # Treated and Untreated
            dict_ = generate_random_dict()
            print_dict(dict_)
            dict_ = import_process('test.grmpy.ini')

            for key_ in ['TREATED', 'UNTREATED']:
                dict_[key_]['all'] = np.array([0] * len(dict_[key_]['all']))

            df, Y, Y_1, Y_0, D, X, Z, U, V = simulation(dict_)

            assert Y_1.all() == U[0:, 1].all()
            assert Y_0.all() == U[0:, 0].all()
            assert Y[D == 1].all() == Y_1.all()
            assert Y[D == 0].all() == Y_0.all()
            assert V.all() == (U[0:, 2] - U[0:, 1] + U[0:, 0]).all()

            # Only Treated
            dict_ = generate_random_dict()
            print_dict(dict_)
            dict_ = import_process('test.grmpy.ini')

            dict_['TREATED']['all'] = np.array([0] * len(dict_['TREATED']['all']))

            df, Y, Y_1, Y_0, D, X, Z, U, V = simulation(dict_)

            assert Y_1.all() == U[0:, 1].all()
            assert Y_0.all() == (
                np.dot(dict_['UNTREATED']['all'], X.T) + U[0:, 0]).all()

            assert Y[D == 1].all() == Y_1.all()
            assert Y[D == 0].all() == Y_0.all()
            assert V.all() == (U[0:, 2] - U[0:, 1] + U[0:, 0]).all()

            # Only Untreated
            dict_ = generate_random_dict()
            print_dict(dict_)
            dict_ = import_process('test.grmpy.ini')

            dict_['UNTREATED']['all'] = np.array(
                [0] * len(dict_['UNTREATED']['all']))

            df, Y, Y_1, Y_0, D, X, Z, U, V = simulation(dict_)

            assert Y_1.all() == (
                np.dot(dict_['TREATED']['all'], X.T) + U[0:, 1]).all()
            assert Y_0.all() == U[0:, 0].all()
            assert Y[D == 1].all() == Y_1.all()
            assert Y[D == 0].all() == Y_0.all()
            assert V.all() == (U[0:, 2] - U[0:, 1] + U[0:, 0]).all()

            # Only Cost
            dict_ = generate_random_dict()
            print_dict(dict_)
            dict_ = import_process('test.grmpy.ini')

            dict_['COST']['all'] = np.array([0] * len(dict_['COST']['all']))

            df, Y, Y_1, Y_0, D, X, Z, U, V = simulation(dict_)

            assert Y_1.all() == (
                np.dot(dict_['TREATED']['all'], X.T) + U[0:, 1]).all()
            assert Y_0.all() == (
                np.dot(dict_['UNTREATED']['all'], X.T) + U[0:, 0]).all()
            assert Y[D == 1].all() == Y_1.all()
            assert Y[D == 0].all() == Y_0.all()
            assert V.all() == (U[0:, 2] - U[0:, 1] + U[0:, 0]).all()

            files = [f for f in os.listdir('.') if os.path.isfile(f)]
            files = [f for f in files if '.grmpy.' in f]

            for f in files:
                os.remove(f)

    def test4(self):
        constr = constraints(probability=0.0, Agents=1)
        for i in range(100):
            dict_ = generate_random_dict(constr)
            print_dict(dict_)
            dict_ = import_process('test.grmpy.ini')
            df, Y, Y_1, Y_0, D, X, Z, U, V = simulation(dict_)

            files = [f for f in os.listdir('.') if os.path.isfile(f)]
            files = [f for f in files if '.grmpy.' in f]

            for f in files:
                os.remove(f)

    def test5(self):
        '''Test the generating and  import process'''
        for i in range(100):
            gen_dict = generate_random_dict()
            init_file_name = gen_dict['SIMULATION']['source']
            print_dict(gen_dict, init_file_name)
            imp_dict = import_process(init_file_name + '.grmpy.ini')

            for key_ in ['TREATED', 'UNTREATED', 'COST', 'DIST']:
                assert gen_dict[key_]['coeff'].all() == imp_dict[key_]['all'].all()
            for key_ in ['source', 'agents', 'seed']:
                assert gen_dict['SIMULATION'][key_] == imp_dict['SIMULATION'][key_]

            files = [f for f in os.listdir('.') if os.path.isfile(f)]
            files = [f for f in files if '.grmpy.' in f]

            for f in files:
                os.remove(f)





