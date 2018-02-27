"""The module provides unit tests for different aspects of the simulation process."""
import os

from scipy.stats import wishart
import pandas as pd
import numpy as np

from grmpy.estimate.estimate_auxiliary import backward_cholesky_transformation
from grmpy.simulate.simulate_auxiliary import construct_covariance_matrix
from grmpy.estimate.estimate_auxiliary import provide_cholesky_decom
from grmpy.simulate.simulate_auxiliary import mte_information
from grmpy.estimate.estimate_auxiliary import start_values
from grmpy.test.random_init import generate_random_dict
from grmpy.test.auxiliary import adjust_output_cholesky
from grmpy.test.auxiliary import refactor_results
from grmpy.test.random_init import constraints
from grmpy.test.random_init import print_dict
from grmpy.simulate.simulate import simulate
from grmpy.estimate.estimate import estimate
from grmpy.test.auxiliary import cleanup
from grmpy.read.read import read
import grmpy


def test1():
    """The first test tests whether the relationships in the simulated datasets are appropriate
    in a deterministic and an undeterministic setting.
    """
    for case in ['deterministic', 'undeterministic']:
        if case == 'deterministic':
            prob = 1.0
        else:
            prob = 0.0
        constr = constraints(probability=prob)
        for _ in range(10):
            generate_random_dict(constr)
            df = simulate('test.grmpy.ini')
            dict_ = read('test.grmpy.ini')

            x_treated = df[['X_{}'.format(i-1) for i in dict_['TREATED']['order']]]
            y_treated = pd.DataFrame.sum(dict_['TREATED']['all'] * x_treated, axis=1) + df.U1
            x_untreated = df[['X_{}'.format(i-1) for i in dict_['UNTREATED']['order']]]
            y_untreated = pd.DataFrame.sum(dict_['UNTREATED']['all'] * x_untreated, axis=1) + df.U0

            np.testing.assert_array_almost_equal(df.Y1, y_treated, decimal=5)
            np.testing.assert_array_almost_equal(df.Y0, y_untreated, decimal=5)
            np.testing.assert_array_equal(df.Y[df.D == 1], df.Y1[df.D == 1])
            np.testing.assert_array_equal(df.Y[df.D == 0], df.Y0[df.D == 0])
            np.testing.assert_array_almost_equal(df.V, (df.UC - df.U1 + df.U0), decimal=7)


def test2():
    """The third test  checks whether the relationships hold if the coefficients are zero in
    different setups.
    """
    for j in range(10):
        for i in ['ALL', 'TREATED', 'UNTREATED', 'COST', 'TREATED & UNTREATED']:
            constr = constraints(probability=0.0)
            dict_ = generate_random_dict(constr)

            if i == 'ALL':
                for key_ in ['TREATED', 'UNTREATED', 'COST']:
                    dict_[key_]['all'] = np.array([0.] * len(dict_[key_]['all']))
            elif i == 'TREATED & UNTREATED':
                for key_ in ['TREATED', 'UNTREATED']:
                    dict_[key_]['all'] = np.array([0.] * len(dict_[key_]['all']))
            else:
                dict_[i]['all'] = np.array([0.] * len(dict_[i]['all']))

            print_dict(dict_)

            dict_ = read('test.grmpy.ini')
            df = simulate('test.grmpy.ini')
            x_treated = df[['X_{}'.format(i-1) for i in dict_['TREATED']['order']]]
            x_untreated = df[['X_{}'.format(i-1) for i in dict_['UNTREATED']['order']]]

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
            np.testing.assert_array_almost_equal(df.V, (df.UC - df.U1 + df.U0))


def test3():
    """The fourth test checks whether the simulation process works if there are only treated or un-
    treated Agents by setting the number of agents to one. Additionally the test checks if the start
    values for the estimation process are set to the initialization file values due to perfect se-
    paration.
    """
    constr = constraints(probability=0.0, agents=1)
    for _ in range(10):
        generate_random_dict(constr)
        dict_ = read('test.grmpy.ini')
        df = simulate('test.grmpy.ini')
        start = start_values(dict_, df, 'auto')
        np.testing.assert_equal(dict_['AUX']['init_values'][:(-6)], start[:(-6)])


def test4():
    """The fifth test tests the random init file generating process and the import process. It gen-
    erates an random init file, imports it again and compares the entries in  both dictionaries.
    """
    for _ in range(10):
        gen_dict = generate_random_dict()
        init_file_name = gen_dict['SIMULATION']['source']
        print_dict(gen_dict, init_file_name)
        imp_dict = read(init_file_name + '.grmpy.ini')
        dicts = [gen_dict, imp_dict]
        for key_ in ['TREATED', 'UNTREATED', 'COST', 'DIST']:
            np.testing.assert_array_almost_equal(gen_dict[key_]['all'], imp_dict[key_]['all'],
                                                 decimal=4)
            if key_ in ['TREATED', 'UNTREATED', 'COST']:
                for dict_ in dicts:

                    if not dict_[key_]['order'] == dict_[key_]['order']:
                        raise AssertionError()
                    if len(dict_[key_]['order']) != len(set(dict_[key_]['order'])):
                        raise AssertionError()
                    if dict_[key_]['order'][0] != 1:
                        raise AssertionError()

                for i in range(len(gen_dict[key_]['types'])):

                    if isinstance(gen_dict[key_]['types'][i], str):
                        if not gen_dict[key_]['types'][i] == imp_dict[key_]['types'][i]:
                            raise AssertionError()
                    elif isinstance(gen_dict[key_]['types'][i], list):
                        if not gen_dict[key_]['types'][i][0] == imp_dict[key_]['types'][i][0]:
                            raise AssertionError()
                        np.testing.assert_array_almost_equal(
                            gen_dict[key_]['types'][i][1], imp_dict[key_]['types'][i][1], 4)

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
        init_dict = read('test.grmpy.ini')

        # We impose that the covariance between the random components of the potential
        # outcomes and the random component determining choice is identical.
        init_dict['DIST']['all'][2] = init_dict['DIST']['all'][4]

        # Distribute information
        coeffs_untreated = init_dict['UNTREATED']['all']
        coeffs_treated = init_dict['TREATED']['all']

        # Construct auxiliary information
        cov = construct_covariance_matrix(init_dict)

        df = simulate('test.grmpy.ini')
        help_ = list(set(init_dict['TREATED']['order'] + init_dict['UNTREATED']['order']))
        x = df[['X_{}'.format(i - 1) for i in help_]]

        q = [0.01] + list(np.arange(0.05, 1, 0.05)) + [0.99]
        mte = mte_information(coeffs_treated, coeffs_untreated, cov, q, x, init_dict)

        # We simply test that there is a single unique value for the marginal treatment effect.
        np.testing.assert_equal(len(set(mte)), 1)


def test6():
    """The test ensures that the cholesky decomposition and recomposition works appropriately.
    For this purpose the test creates a positive smi definite matrix fom a wishart distribution,
    decomposes this matrix with, reconstruct it and compares the matrix with the one that was
    specified as the input for the decomposition process.
    """
    pseudo_dict = {'DIST': {'all': []}, 'AUX': {'init_values': []}}
    for _ in range(20):
        b = wishart.rvs(df=10, scale=np.identity(3), size=1)
        parameter = b[np.triu_indices(3)]
        for i in [0, 3, 5]:
            parameter[i] **= 0.5
        pseudo_dict['DIST']['all'] = parameter
        pseudo_dict['AUX']['init_values'] = parameter
        cov_1 = construct_covariance_matrix(pseudo_dict)
        x0, _ = provide_cholesky_decom(pseudo_dict, [], 'init')
        output = backward_cholesky_transformation(x0, test=True)
        output = adjust_output_cholesky(output)
        pseudo_dict['DIST']['all'] = output
        cov_2 = construct_covariance_matrix(pseudo_dict)
        np.testing.assert_array_almost_equal(cov_1, cov_2)

def test7():
    """This test ensures that setting different variables in the TREATED and UNTREATED section to
    binary in the initialization file leads to the same type lists for both sections. Further it is
    verified that it is not possible to set an intercept variable to a binary one."""
    fname = os.path.dirname(grmpy.__file__) + '/test/resources/test_binary.grmpy.ini'
    dict_ = read(fname)

    for i in set(dict_['TREATED']['order'] + dict_['UNTREATED']['order']):
        if i in dict_['TREATED']['order'] and i in dict_['UNTREATED']['order']:
            index_treated = dict_['TREATED']['order'].index(i)
            index_untreated = dict_['UNTREATED']['order'].index(i)
            if not dict_['TREATED']['types'][index_treated] == dict_['UNTREATED']['types'][index_untreated]:
                raise AssertionError()
    for key_ in ['TREATED', 'UNTREATED', 'COST']:
        if isinstance(dict_[key_]['types'][0], list):
            raise AssertionError()

def test8():
    """We want to able to smoothly switch between generating and printing random initialization
    files."""
    for _ in range(10):
        generate_random_dict()
        dict_1 = read('test.grmpy.ini')
        print_dict(dict_1)
        dict_2 = read('test.grmpy.ini')
        np.testing.assert_equal(dict_1, dict_2)

def test9():
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
        for key_ in ['SCIPY-BFGS', 'SCIPY-POWELL']:
            np.testing.assert_equal(constr['MAXITER'], dict_[key_]['maxiter'])

def test10():
    """This test checks if the start_values function returns the init file values if the start
    option is set to init.
    """
    for _ in range(10):
        constr = constraints(0.0)
        generate_random_dict(constr)
        dict_ = read('test.grmpy.ini')
        print(dict_)
        true = []
        for key_ in ['TREATED', 'UNTREATED', 'COST']:
            true += list(dict_[key_]['all'])
        df = simulate('test.grmpy.ini')
        x0 = start_values(dict_, df, 'init')[:-6]

        np.testing.assert_array_equal(true, x0)

def test11():
    """This test checks if the refactor auxiliary function returns an unchanged init file if the
    maximum number of iterations is set to zero."""

    for _ in range(10):
        constr = constraints(0.0,1000,maxiter=0, start='init' )
        generate_random_dict(constr)
        simulate('test.grmpy.ini')
        rslt = estimate('test.grmpy.ini')
        refactor_results(rslt, 'test.grmpy.ini', 'estimate')
        dict_1 = read('test.grmpy.ini')
        dict_2 = read('estimate.grmpy.ini')
        np.testing.assert_equal(dict_1, dict_2)

    cleanup()
