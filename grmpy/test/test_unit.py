"""The module provides unit tests for different aspects of the simulation process."""
import os

import pandas as pd
import numpy as np

from grmpy.simulate.simulate_auxiliary import construct_covariance_matrix
from grmpy.estimate.estimate_auxiliary import backward_transformation
from grmpy.estimate.estimate_auxiliary import start_value_adjustment
from grmpy.simulate.simulate_auxiliary import mte_information
from grmpy.estimate.estimate_auxiliary import start_values
from grmpy.test.auxiliary import attr_dict_to_init_dict
from grmpy.test.random_init import generate_random_dict
from grmpy.grmpy_config import TEST_RESOURCES_DIR
from grmpy.test.random_init import print_dict
from grmpy.simulate.simulate import simulate
from grmpy.check.auxiliary import read_data
from grmpy.test.auxiliary import cleanup
from grmpy.estimate.estimate import fit
from grmpy.read.read import read
import grmpy


def test1():
    """The first test tests whether the relationships in the simulated datasets are appropriate
    in a deterministic and an un-deterministic setting.
    """
    constr = dict()
    for case in ['deterministic', 'undeterministic']:
        if case == 'deterministic':
            constr['DETERMINISTIC'] = True
        else:
            constr['DETERMINISTIC'] = True
        for _ in range(10):
            generate_random_dict(constr)
            df = simulate('test.grmpy.yml')
            dict_ = read('test.grmpy.yml')
            x_treated = df[[dict_['varnames'][i - 1] for i in dict_['TREATED']['order']]]
            y_treated = pd.DataFrame.sum(dict_['TREATED']['all'] * x_treated, axis=1) + df.U1
            x_untreated = df[[dict_['varnames'][i - 1] for i in dict_['UNTREATED']['order']]]
            y_untreated = pd.DataFrame.sum(dict_['UNTREATED']['all'] * x_untreated, axis=1) + df.U0

            np.testing.assert_array_almost_equal(df.Y1, y_treated, decimal=5)
            np.testing.assert_array_almost_equal(df.Y0, y_untreated, decimal=5)
            np.testing.assert_array_equal(df.Y[df.D == 1], df.Y1[df.D == 1])
            np.testing.assert_array_equal(df.Y[df.D == 0], df.Y0[df.D == 0])


def test2():
    """The second test  checks whether the relationships hold if the coefficients are zero in
    different setups.
    """
    for _ in range(10):
        for i in ['ALL', 'TREATED', 'UNTREATED', 'CHOICE', 'TREATED & UNTREATED']:
            constr = dict()
            constr['DETERMINISTIC'] = False
            dict_ = generate_random_dict(constr)

            if i == 'ALL':
                for key_ in ['TREATED', 'UNTREATED', 'CHOICE']:
                    dict_[key_]['params'] = np.array([0.] * len(dict_[key_]['params']))
            elif i == 'TREATED & UNTREATED':
                for key_ in ['TREATED', 'UNTREATED']:
                    dict_[key_]['params'] = np.array([0.] * len(dict_[key_]['params']))
            else:
                dict_[i]['params'] = np.array([0.] * len(dict_[i]['params']))

            print_dict(dict_)

            dict_ = read('test.grmpy.yml')
            df = simulate('test.grmpy.yml')
            x_treated = df[[dict_['varnames'][i - 1] for i in dict_['TREATED']['order']]]
            x_untreated = df[[dict_['varnames'][i - 1] for i in dict_['UNTREATED']['order']]]

            if i == 'ALL':
                np.testing.assert_array_equal(df.Y1, df.U1)
                np.testing.assert_array_equal(df.Y0, df.U0)
            elif i == 'TREATED & UNTREATED':
                np.testing.assert_array_equal(df.Y1, df.U1)
                np.testing.assert_array_equal(df.Y0, df.U0)
                np.testing.assert_array_equal(df.Y[df.D == 1], df.U1[df.D == 1])
                np.testing.assert_array_equal(df.Y[df.D == 0], df.U0[df.D == 0])
            elif i == 'TREATED':
                y_untreated = pd.DataFrame.sum(
                    dict_['UNTREATED']['all'] * x_untreated, axis=1) + df.U0
                np.testing.assert_array_almost_equal(df.Y0, y_untreated, decimal=5)
                np.testing.assert_array_equal(df.Y1, df.U1)

            elif i == 'UNTREATED':
                y_treated = pd.DataFrame.sum(dict_['TREATED']['all'] * x_treated, axis=1) + df.U1
                np.testing.assert_array_almost_equal(df.Y1, y_treated, decimal=5)
                np.testing.assert_array_equal(df.Y0, df.U0)
            else:
                y_treated = pd.DataFrame.sum(dict_['TREATED']['all'] * x_treated, axis=1) + df.U1
                y_untreated = pd.DataFrame.sum(
                    dict_['UNTREATED']['all'] * x_untreated, axis=1) + df.U0
                np.testing.assert_array_almost_equal(df.Y1, y_treated, decimal=5)
                np.testing.assert_array_almost_equal(df.Y0, y_untreated, decimal=5)

            np.testing.assert_array_equal(df.Y[df.D == 1], df.Y1[df.D == 1])
            np.testing.assert_array_equal(df.Y[df.D == 0], df.Y0[df.D == 0])


def test3():
    """The fourth test checks whether the simulation process works if there are only treated or un-
    treated Agents by setting the number of agents to one. Additionally the test checks if the start
    values for the estimation process are set to the initialization file values due to perfect
    separation.
    """
    constr = dict()
    constr['AGENTS'], constr['DETERMINISTIC'] = 1, False
    for _ in range(10):
        generate_random_dict(constr)
        dict_ = read('test.grmpy.yml')
        df = simulate('test.grmpy.yml')
        start = start_values(dict_, df, 'auto')
        np.testing.assert_equal(dict_['AUX']['init_values'][:(-6)], start[:(-4)])


def test4():
    """The fifth test tests the random init file generating process and the import process. It
    generates an random init file, imports it again and compares the entries in  both dictionaries.
    """
    for _ in range(10):
        gen_dict = generate_random_dict()
        init_file_name = gen_dict['SIMULATION']['source']
        print_dict(gen_dict, init_file_name)
        imp_dict = read(init_file_name + '.grmpy.yml')
        dicts = [gen_dict, imp_dict]
        for key_ in ['TREATED', 'UNTREATED', 'CHOICE', 'DIST']:
            np.testing.assert_array_almost_equal(gen_dict[key_]['params'], imp_dict[key_]['all'],
                                                 decimal=4)
            if key_ in ['TREATED', 'UNTREATED', 'CHOICE']:
                for dict_ in dicts:
                    if not dict_[key_]['order'] == dict_[key_]['order']:
                        raise AssertionError()
                    if len(dict_[key_]['order']) != len(set(dict_[key_]['order'])):
                        raise AssertionError()
                    if dict_[key_]['order'][0] not in ['X1', 1]:
                        raise AssertionError()

                for variable in gen_dict['VARTYPES'].keys():
                    for section in ['TREATED', 'UNTREATED', 'CHOICE']:
                        if variable in gen_dict[section]['order']:
                            index = gen_dict[section]['order'].index(variable)
                            type = imp_dict[section]['types'][index]
                            if gen_dict['VARTYPES'][variable] != type:
                                raise AssertionError()
                        if imp_dict[section]['types'][0] != 'nonbinary' or gen_dict['VARTYPES'][
                            'X1'] \
                                != 'nonbinary':
                            raise AssertionError()

        for key_ in ['source', 'agents', 'seed']:
            if not gen_dict['SIMULATION'][key_] == imp_dict['SIMULATION'][key_]:
                raise AssertionError()


def test5():
    """The tests checks if the simulation process works even if the covariance between U1 and V
    and U0 and V is equal. Further the test ensures that the mte_information function returns
    the same value for each quantile.
    """
    for _ in range(10):
        generate_random_dict()
        init_dict = read('test.grmpy.yml')

        # We impose that the covariance between the random components of the potential
        # outcomes and the random component determining choice is identical.
        init_dict['DIST']['all'][2] = init_dict['DIST']['all'][4]

        # Distribute information
        coeffs_untreated = init_dict['UNTREATED']['all']
        coeffs_treated = init_dict['TREATED']['all']

        # Construct auxiliary information
        cov = construct_covariance_matrix(init_dict)

        df = simulate('test.grmpy.yml')
        help_ = list(set(init_dict['TREATED']['order'] + init_dict['UNTREATED']['order']))
        x = df[[init_dict['varnames'][i - 1] for i in help_]]

        q = [0.01] + list(np.arange(0.05, 1, 0.05)) + [0.99]
        mte = mte_information(coeffs_treated, coeffs_untreated, cov, q, x, init_dict)

        # We simply test that there is a single unique value for the marginal treatment effect.
        np.testing.assert_equal(len(set(mte)), 1)


def test6():
    """The test ensures that the cholesky decomposition and re-composition works appropriately.
    For this purpose the test creates a positive smi definite matrix fom a Wishart distribution,
    decomposes this matrix with, reconstruct it and compares the matrix with the one that was
    specified as the input for the decomposition process.
    """
    for _ in range(1000):

        cov = np.random.uniform(0, 1, 2)
        var = np.random.uniform(1, 2, 3)
        aux = [var[0], var[1], cov[0], var[2], cov[1], 1.0]
        dict_ = {'DIST': {'all': aux}}
        before = [var[0], cov[0] / var[0], var[2], cov[1] / var[2]]
        x0 = start_value_adjustment([], dict_, 'init')
        after = backward_transformation(x0)
        np.testing.assert_array_almost_equal(before, after, decimal=6)


def test7():
    """We want to able to smoothly switch between generating and printing random initialization
    files.
    """
    for _ in range(10):
        generate_random_dict()
        dict_1 = read('test.grmpy.yml')

        print_dict(attr_dict_to_init_dict(dict_1))
        dict_2 = read('test.grmpy.yml')
        np.testing.assert_equal(dict_1, dict_2)


def test8():
    """This test ensures that the random process handles the constraints dict appropriately if there
    the input dictionary is not complete.
    """
    for _ in range(10):
        constr = dict()
        constr['MAXITER'] = np.random.randint(0, 1000)
        constr['START'] = np.random.choice(['start', 'init'])
        constr['AGENTS'] = np.random.randint(1, 1000)
        dict_ = generate_random_dict(constr)
        np.testing.assert_equal(constr['AGENTS'], dict_['SIMULATION']['agents'])
        np.testing.assert_equal(constr['START'], dict_['ESTIMATION']['start'])
        np.testing.assert_equal(constr['MAXITER'], dict_['ESTIMATION']['maxiter'])


def test9():
    """This test checks if the start_values function returns the init file values if the start
    option is set to init.
    """
    for _ in range(10):
        constr = dict()
        constr['DETERMINISTIC'] = False
        generate_random_dict(constr)
        dict_ = read('test.grmpy.yml')
        true = []
        for key_ in ['TREATED', 'UNTREATED', 'CHOICE']:
            true += list(dict_[key_]['all'])
        df = simulate('test.grmpy.yml')
        x0 = start_values(dict_, df, 'init')[:-4]

        np.testing.assert_array_equal(true, x0)


def test10():
    """This test checks if the refactor auxiliary function returns an unchanged init file if the
    maximum number of iterations is set to zero.
    """

    for _ in range(10):
        constr = dict()
        constr['DETERMINISTIC'], constr['AGENTS'] = False, 1000
        constr['MAXITER'], constr['START'] = 0, 'init'
        generate_random_dict(constr)
        init_dict = read('test.grmpy.yml')
        df = simulate('test.grmpy.yml')
        start = start_values(init_dict, df, 'init')
        start = backward_transformation(start)
        rslt = fit('test.grmpy.yml')

        np.testing.assert_equal(start, rslt['AUX']['x_internal'])


def test11():
    """This test ensures that the tutorial configuration works as intended."""
    fname = TEST_RESOURCES_DIR + '/tutorial.grmpy.yml'
    simulate(fname)
    fit(fname)


def test12():
    """This test checks if our data import process is able to handle .txt, .dta and .pkl files."""

    pkl = TEST_RESOURCES_DIR + '/data.grmpy.pkl'
    dta = TEST_RESOURCES_DIR + '/data.grmpy.dta'
    txt = TEST_RESOURCES_DIR + '/data.grmpy.txt'

    real_sum = -3211.20122
    real_column_values = ['Y', 'D', 'X1', 'X2', 'X3', 'X5', 'X4', 'Y1', 'Y0', 'U1', 'U0', 'V']

    for data in [pkl, dta, txt]:
        df = read_data(data)
        sum = np.sum(df.sum())
        columns = list(df)
        np.testing.assert_array_almost_equal(sum, real_sum, decimal=5)
        np.testing.assert_equal(columns, real_column_values)

    cleanup()
