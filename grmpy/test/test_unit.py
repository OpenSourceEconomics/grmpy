"""The module provides unit tests for different aspects of the simulation process."""
import glob
import os

import pandas as pd
import numpy as np

from grmpy.test.random_init import generate_random_dict
from grmpy.simulate.simulate import simulate
from grmpy.test.random_init import constraints
from grmpy.test.random_init import print_dict
from grmpy.test.auxiliary import check_series
from grmpy.read.read import read


class TestClass:
    def test1(self):
        """The first test tests whether the relationships in the simulated datasets are appropriate."""
        constr = constraints(probability=0.0)
        for _ in range(10):
            dict_ = generate_random_dict(constr)
            print_dict(dict_)

            df = simulate('test.grmpy.ini')
            dict_ = read('test.grmpy.ini')

            x_ = [col for col in df if col.startswith('X')]
            y_treated = pd.DataFrame.sum(dict_['TREATED']['all'] * df[x_], axis=1) + df.U1
            y_untreated = pd.DataFrame.sum(dict_['UNTREATED']['all'] * df[x_], axis=1) + df.U0

            assert check_series(df.Y1, y_treated)
            assert check_series(df.Y0, y_untreated)
            assert np.array_equal(df.Y[df.D == 1], df.Y1[df.D == 1])
            assert np.array_equal(df.Y[df.D == 0], df.Y0[df.D == 0])
            assert np.array_equal(df.V, (df.UC - df.U1 + df.U0))

            for f in glob.glob("*.grmpy.*"):
                os.remove(f)

    def test2(self):
        """The second test checks whether the relationships hold if the process is deterministic."""
        constr = constraints(probability=1.)
        for _ in range(10):
            dict_ = generate_random_dict(constr)
            print_dict(dict_)

            df = simulate('test.grmpy.ini')
            dict_ = read('test.grmpy.ini')
            x_ = [col for col in df if col.startswith('X')]
            y_treated = pd.DataFrame.sum(dict_['TREATED']['all'] * df[x_], axis=1)
            y_untreated = pd.DataFrame.sum(dict_['UNTREATED']['all'] * df[x_], axis=1)

            assert check_series(df.Y1, y_treated)
            assert check_series(df.Y0, y_untreated)
            assert np.array_equal(df.Y[df.D == 1], df.Y1[df.D == 1])
            assert np.array_equal(df.Y[df.D == 0], df.Y0[df.D == 0])
            assert np.array_equal(df.V, (df.UC - df.U1 + df.U0))

            for f in glob.glob("*.grmpy.*"):
                os.remove(f)

    def test3(self):
        """The third test  checks whether the relationships hold if the coefficients are zero in different setups"""
        for _ in range(10):

            # All Coefficients
            constr = constraints(probability=0.0)
            dict_ = generate_random_dict(constr)
            for key_ in ['TREATED', 'UNTREATED', 'COST']:
                dict_[key_]['coeff'] = np.array(
                    [0.] * len(dict_[key_]['coeff']))

            print_dict(dict_)
            df = simulate('test.grmpy.ini')

            assert np.array_equal(df.Y1, df.U1)
            assert np.array_equal(df.Y0, df.U0)
            assert np.array_equal(df.Y[df.D == 1], df.Y1[df.D == 1])
            assert np.array_equal(df.Y[df.D == 0], df.Y0[df.D == 0])
            assert np.array_equal(df.V, (df.UC - df.U1 + df.U0))

            # Treated and Untreated
            constr = constraints(probability=0.0)
            dict_ = generate_random_dict(constr)
            for key_ in ['TREATED', 'UNTREATED']:
                dict_[key_]['coeff'] = np.array(
                    [0.] * len(dict_[key_]['coeff']))
            print_dict(dict_)

            df = simulate('test.grmpy.ini')

            assert np.array_equal(df.Y1, df.U1)
            assert np.array_equal(df.Y0, df.U0)
            assert np.array_equal(df.Y[df.D == 1], df.U1[df.D == 1])
            assert np.array_equal(df.Y[df.D == 0], df.U0[df.D == 0])
            assert np.array_equal(df.Y[df.D == 0], df.Y0[df.D == 0])
            assert np.array_equal(df.Y[df.D == 1], df.Y1[df.D == 1])
            assert np.array_equal(df.V, (df.UC - df.U1 + df.U0))

            # Only Treated
            constraints(probability=0.0)
            dict_ = generate_random_dict(constr)
            dict_['TREATED']['coeff'] = np.array(
                [0.] * len(dict_['TREATED']['coeff']))
            print_dict(dict_)
            dict_ = read('test.grmpy.ini')

            df = simulate('test.grmpy.ini')
            x_ = [col for col in df if col.startswith('X')]
            y_untreated = pd.DataFrame.sum(dict_['UNTREATED']['all'] * df[x_], axis=1) + df.U0

            assert np.array_equal(df.Y1, df.U1)
            assert check_series(df.Y0, y_untreated)
            assert np.array_equal(df.Y[df.D == 1], df.Y1[df.D == 1])
            assert np.array_equal(df.Y[df.D == 0], df.Y0[df.D == 0])
            assert np.array_equal(df.V, (df.UC - df.U1 + df.U0))

            # Only Untreated
            constr = constraints(probability=0.0)
            dict_ = generate_random_dict(constr)
            dict_['UNTREATED']['coeff'] = np.array(
                [0.] * len(dict_['UNTREATED']['coeff']))
            print_dict(dict_)
            dict_ = read('test.grmpy.ini')

            df = simulate('test.grmpy.ini')
            x_ = [col for col in df if col.startswith('X')]
            y_treated = pd.DataFrame.sum(dict_['TREATED']['all'] * df[x_], axis=1) + df.U1

            assert check_series(df.Y1, y_treated)
            assert np.array_equal(df.Y0, df.U0)
            assert np.array_equal(df.Y[df.D == 1], df.Y1[df.D == 1])
            assert np.array_equal(df.Y[df.D == 0], df.Y0[df.D == 0])
            assert np.array_equal(df.V, (df.UC - df.U1 + df.U0))

            # Only Cost
            constr = constraints(probability=0.0)
            dict_ = generate_random_dict(constr)
            dict_['COST']['coeff'] = np.array(
                [0.] * len(dict_['COST']['coeff']))
            print_dict(dict_)
            dict_ = read('test.grmpy.ini')

            df = simulate('test.grmpy.ini')
            x_ = [col for col in df if col.startswith('X')]
            y_treated = pd.DataFrame.sum(dict_['TREATED']['all'] * df[x_], axis=1) + df.U1
            y_untreated = pd.DataFrame.sum(dict_['UNTREATED']['all'] * df[x_], axis=1) + df.U0

            assert check_series(df.Y1, y_treated)
            assert check_series(df.Y0, y_untreated)
            assert np.array_equal(df.Y[df.D == 1], df.Y1[df.D == 1])
            assert np.array_equal(df.Y[df.D == 0], df.Y0[df.D == 0])
            assert np.array_equal(df.V, (df.UC - df.U1 + df.U0))

            for f in glob.glob("*.grmpy.*"):
                os.remove(f)

    def test4(self):
        """The fourth test checks whether the simulation process works if there are only treated or
        untreated Agents by setting the number of agents to one
        """
        constr = constraints(probability=0.0, agents=1)
        for _ in range(10):
            dict_ = generate_random_dict(constr)
            print_dict(dict_)

            simulate('test.grmpy.ini')

            for f in glob.glob("*.grmpy.*"):
                os.remove(f)

    def test5(self):
        """The fifth test tests the random init file generating process and the  import process. It generates an random
        init file, imports it again and compares the entries in the both dictionaries.
        """
        for _ in range(10):
            gen_dict = generate_random_dict()
            init_file_name = gen_dict['SIMULATION']['source']
            print_dict(gen_dict, init_file_name)
            imp_dict = read(init_file_name + '.grmpy.ini')

            for key_ in ['TREATED', 'UNTREATED', 'COST', 'DIST']:
                assert np.array_equal(
                    np.around(
                        gen_dict[key_]['coeff'], 4), imp_dict[key_]['all']
                )
            for key_ in ['source', 'agents', 'seed']:
                assert gen_dict['SIMULATION'][key_] == \
                       imp_dict['SIMULATION'][key_]

            for f in glob.glob("*.grmpy.*"):
                os.remove(f)
