"""The module provides unit tests for different aspects of the simulation process."""
import os

from scipy.stats import wishart
import pandas as pd
import numpy as np

from grmpy.estimate.estimate_auxiliary import backward_cholesky_transformation
from grmpy.simulate.simulate_auxiliary import construct_covariance_matrix
from grmpy.estimate.estimate_auxiliary import provide_cholesky_decom
from grmpy.simulate.simulate_auxiliary import mte_information
from grmpy.test.random_init import generate_random_dict
from grmpy.test.auxiliary import adjust_output_cholesky
from grmpy.test.random_init import constraints
from grmpy.test.random_init import print_dict
from grmpy.simulate.simulate import simulate
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

            x = df.filter(regex=r'^X\_', axis=1)
            y_treated = pd.DataFrame.sum(dict_['TREATED']['all'] * x, axis=1) + df.U1
            y_untreated = pd.DataFrame.sum(dict_['UNTREATED']['all'] * x, axis=1) + df.U0

            np.testing.assert_array_almost_equal(df.Y1, y_treated, decimal=5)
            np.testing.assert_array_almost_equal(df.Y0, y_untreated, decimal=5)
            np.testing.assert_array_equal(df.Y[df.D == 1], df.Y1[df.D == 1])
            np.testing.assert_array_equal(df.Y[df.D == 0], df.Y0[df.D == 0])
            np.testing.assert_array_almost_equal(df.V, (df.UC - df.U1 + df.U0), decimal=7)


def test2():
    """The third test  checks whether the relationships hold if the coefficients are zero in
    different setups.
    """
    for _ in range(10):
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
            x = df.filter(regex=r'^X\_', axis=1)

            if i == 'ALL':
                np.testing.assert_array_equal(df.Y1, df.U1)
                np.testing.assert_array_equal(df.Y0, df.U0)
            elif i == 'TREATED & UNTREATED':
                np.testing.assert_array_equal(df.Y1, df.U1)
                np.testing.assert_array_equal(df.Y0, df.U0)
                np.testing.assert_array_equal(df.Y[df.D == 1], df.U1[df.D == 1])
                np.testing.assert_array_equal(df.Y[df.D == 0], df.U0[df.D == 0])
            elif i == 'TREATED':
                y_untreated = pd.DataFrame.sum(dict_['UNTREATED']['all'] * x, axis=1) + df.U0
                np.testing.assert_array_almost_equal(df.Y0, y_untreated, decimal=5)
                np.testing.assert_array_equal(df.Y1, df.U1)

            elif i == 'UNTREATED':
                y_treated = pd.DataFrame.sum(dict_['TREATED']['all'] * x, axis=1) + df.U1
                np.testing.assert_array_almost_equal(df.Y1, y_treated, decimal=5)
                np.testing.assert_array_equal(df.Y0, df.U0)
            else:
                y_treated = pd.DataFrame.sum(dict_['TREATED']['all'] * x, axis=1) + df.U1
                y_untreated = pd.DataFrame.sum(dict_['UNTREATED']['all'] * x, axis=1) + df.U0
                np.testing.assert_array_almost_equal(df.Y1, y_treated, decimal=5)
                np.testing.assert_array_almost_equal(df.Y0, y_untreated, decimal=5)

            np.testing.assert_array_equal(df.Y[df.D == 1], df.Y1[df.D == 1])
            np.testing.assert_array_equal(df.Y[df.D == 0], df.Y0[df.D == 0])
            np.testing.assert_array_almost_equal(df.V, (df.UC - df.U1 + df.U0))


def test3():
    """The fourth test checks whether the simulation process works if there are only treated or
    untreated Agents by setting the number of agents to one.
    """
    constr = constraints(probability=0.0, agents=1)
    for _ in range(10):
        generate_random_dict(constr)
        simulate('test.grmpy.ini')


def test4():
    """The fifth test tests the random init file generating process and the  import process. It
    generates an random init file, imports it again and compares the entries in the both dictio-
    naries.
    """
    for _ in range(10):
        gen_dict = generate_random_dict()
        init_file_name = gen_dict['SIMULATION']['source']
        print_dict(gen_dict, init_file_name)
        imp_dict = read(init_file_name + '.grmpy.ini')

        for key_ in ['TREATED', 'UNTREATED', 'COST', 'DIST']:
            np.testing.assert_array_almost_equal(gen_dict[key_]['all'], imp_dict[key_]['all'],
                                                 decimal=4)
            if key_ in ['TREATED', 'UNTREATED', 'COST']:
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
        x = df.filter(regex=r'^X\_', axis=1)
        q = [0.01] + list(np.arange(0.05, 1, 0.05)) + [0.99]
        mte = mte_information(coeffs_treated, coeffs_untreated, cov, q, x)

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
    if not dict_['TREATED']['types'] == dict_['UNTREATED']['types']:
        raise AssertionError()
    for key_ in ['TREATED', 'UNTREATED', 'COST']:
        if isinstance(dict_[key_]['types'][0], list):
            raise AssertionError()
    cleanup()


def test8():
    """We want to able to smoothly switch between generating and printing random initialization
    files."""
    for _ in range(10):
        generate_random_dict()
        read('test.grmpy.ini')
