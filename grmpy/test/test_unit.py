import numpy as np
import os
import glob


from grmpy.read.read import read
from grmpy.test.random_init import generate_random_dict
from grmpy.simulation.simulation import simulate
from grmpy.test.random_init import print_dict
from grmpy.test.random_init import constraints


class TestClass:
    def test1(self):
        """Testing relations in simulated dataset"""
        constr = constraints(probability=0.0)
        for i in range(10):
            dict_ = generate_random_dict(constr)
            print_dict(dict_)
            df, Y, Y_1, Y_0, D, X, Z, U, V = simulate('test.grmpy.ini')
            dict_ = read('test.grmpy.ini')

            # endogeneous variables

            assert np.array_equal(
                Y_1, (np.dot(dict_['TREATED']['all'], X.T) + U[0:, 1]))
            assert np.array_equal(
                Y_0, (np.dot(dict_['UNTREATED']['all'], X.T) + U[0:, 0]))
            assert np.array_equal(Y[D == 1], Y_1[D == 1])
            assert np.array_equal(Y[D == 0], Y_0[D == 0])
            assert np.array_equal(V, (U[0:, 2] - U[0:, 1] + U[0:, 0]))

            for f in glob.glob("*.grmpy.*"):
                os.remove(f)

    def test2(self):
        """Testing if relationships hold if process is deterministic"""
        constr = constraints(probability=1.)
        for i in range(10):
            dict_ = generate_random_dict(constr)
            print_dict(dict_)
            df, Y, Y_1, Y_0, D, X, Z, U, V = simulate('test.grmpy.ini')
            dict_ = read('test.grmpy.ini')

            assert np.array_equal(Y_1, np.dot(dict_['TREATED']['all'], X.T))
            assert np.array_equal(Y_0, np.dot(dict_['UNTREATED']['all'], X.T))
            assert np.array_equal(Y[D == 1], Y_1[D == 1])
            assert np.array_equal(Y[D == 0], Y_0[D == 0])
            assert np.array_equal(V, (U[0:, 2] - U[0:, 1] + U[0:, 0]))

            for f in glob.glob("*.grmpy.*"):
                os.remove(f)

    def test3(self):
        """Testing if relationships hold if coefficients are zero in different setups"""
        for i in range(10):

            # All Coefficients

            constr = constraints(probability=0.0)
            dict_ = generate_random_dict(constr)
            for key_ in ['TREATED', 'UNTREATED', 'COST']:
                dict_[key_]['coeff'] = np.array(
                    [0.] * len(dict_[key_]['coeff']))
            print_dict(dict_)
            df, Y, Y_1, Y_0, D, X, Z, U, V = simulate('test.grmpy.ini')

            assert np.array_equal(Y_1, U[0:, 1])
            assert np.array_equal(Y_0, U[0:, 0])
            assert np.array_equal(Y[D == 1], Y_1[D == 1])
            assert np.array_equal(Y[D == 0], Y_0[D == 0])
            assert np.array_equal(V, U[0:, 2] - U[0:, 1] + U[0:, 0])

            # Treated and Untreated
            constr = constraints(probability=0.0)
            dict_ = generate_random_dict(constr)
            for key_ in ['TREATED', 'UNTREATED']:
                dict_[key_]['coeff'] = np.array(
                    [0.] * len(dict_[key_]['coeff']))
            print_dict(dict_)

            df, Y, Y_1, Y_0, D, X, Z, U, V = simulate('test.grmpy.ini')

            assert np.array_equal(Y_1, U[0:, 1])
            assert np.array_equal(Y_0, U[0:, 0])
            assert np.array_equal(Y[D == 1], U[0:, 1][D == 1])
            assert np.array_equal(Y[D == 0], U[0:, 0][D == 0])
            assert np.array_equal(Y[D == 0], Y_0[D == 0])
            assert np.array_equal(Y[D == 1], Y_1[D == 1])
            assert np.array_equal(V, U[0:, 2] - U[0:, 1] + U[0:, 0])

            # Only Treated
            constraints(probability=0.0)
            dict_ = generate_random_dict(constr)
            dict_['TREATED']['coeff'] = np.array(
                [0.] * len(dict_['TREATED']['coeff']))
            print_dict(dict_)
            dict_ = read('test.grmpy.ini')

            df, Y, Y_1, Y_0, D, X, Z, U, V = simulate('test.grmpy.ini')

            assert np.array_equal(Y_1, U[0:, 1])
            assert np.array_equal(Y_0,
                                  np.dot(dict_['UNTREATED']['all'], X.T) + U[0:, 0])

            assert np.array_equal(Y[D == 1], Y_1[D == 1])
            assert np.array_equal(Y[D == 0], Y_0[D == 0])
            assert np.array_equal(V, U[0:, 2] - U[0:, 1] + U[0:, 0])

            # Only Untreated
            constr = constraints(probability=0.0)
            dict_ = generate_random_dict(constr)
            dict_['UNTREATED']['coeff'] = np.array(
                [0.] * len(dict_['UNTREATED']['coeff']))
            print_dict(dict_)
            dict_ = read('test.grmpy.ini')

            df, Y, Y_1, Y_0, D, X, Z, U, V = simulate('test.grmpy.ini')

            assert np.array_equal(Y_1, np.dot(
                dict_['TREATED']['all'], X.T) + U[0:, 1])
            assert np.array_equal(Y_0, U[0:, 0])
            assert np.array_equal(Y[D == 1], Y_1[D == 1])
            assert np.array_equal(Y[D == 0], Y_0[D == 0])
            assert np.array_equal(V, U[0:, 2] - U[0:, 1] + U[0:, 0])

            # Only Cost
            constr = constraints(probability=0.0)
            dict_ = generate_random_dict(constr)
            dict_['COST']['coeff'] = np.array(
                [0.] * len(dict_['COST']['coeff']))
            print_dict(dict_)
            dict_ = read('test.grmpy.ini')

            df, Y, Y_1, Y_0, D, X, Z, U, V = simulate('test.grmpy.ini')

            assert np.array_equal(Y_1, np.dot(
                dict_['TREATED']['all'], X.T) + U[0:, 1])
            assert np.array_equal(Y_0, np.dot(
                dict_['UNTREATED']['all'], X.T) + U[0:, 0])
            assert np.array_equal(Y[D == 1], Y_1[D == 1])
            assert np.array_equal(Y[D == 0], Y_0[D == 0])
            assert np.array_equal(V, U[0:, 2] - U[0:, 1] + U[0:, 0])

            for f in glob.glob("*.grmpy.*"):
                os.remove(f)

    def test4(self):
        constr = constraints(probability=0.0, agents=1)
        for i in range(10):
            dict_ = generate_random_dict(constr)
            print_dict(dict_)
            df, Y, Y_1, Y_0, D, X, Z, U, V = simulate('test.grmpy.ini')

            for f in glob.glob("*.grmpy.*"):
                os.remove(f)

    def test5(self):
        """Test the generating and  import process"""
        for i in range(10):
            gen_dict = generate_random_dict()
            init_file_name = gen_dict['SIMULATION']['source']
            print_dict(gen_dict, init_file_name)
            imp_dict = read(init_file_name + '.grmpy.ini')

            for key_ in ['TREATED', 'UNTREATED', 'COST', 'DIST']:
                assert np.array_equal(np.around(gen_dict[key_]['coeff'], 4), imp_dict[key_]['all'])
            for key_ in ['source', 'agents', 'seed']:
                assert gen_dict['SIMULATION'][key_] == imp_dict['SIMULATION'][key_]

            for f in glob.glob("*.grmpy.*"):
                os.remove(f)
