"""The module provides unit tests for different aspects of the simulation process."""
import pandas as pd
import numpy as np

from grmpy.simulate.simulate_auxiliary import mte_information
from grmpy.test.resources.estimate_old import estimate_old
from grmpy.test.random_init import generate_random_dict
from grmpy.test.random_init import constraints
from grmpy.test.random_init import print_dict
from grmpy.estimate.estimate import estimate
from grmpy.simulate.simulate import simulate
from grmpy.test.auxiliary import cleanup
from grmpy.read.read import read



class TestClass:
    def test1(self):
        """The first test tests whether the relationships in the simulated datasets are appropriate
        in a deterministic and an undeterministic setting.<
        """
        for case in ['deterministic', 'nondeterministic']:
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

    def test2(self):
        """The third test  checks whether the relationships hold if the coefficients are zero in
        different setups.
        """
        for _ in range(10):
            for i in ['ALL', 'TREATED', 'UNTREATED', 'COST', 'TREATED & UNTREATED']:
                constr = constraints(probability=0.0)
                dict_ = generate_random_dict(constr)

                if i == 'ALL':
                    for key_ in ['TREATED', 'UNTREATED', 'COST']:
                        dict_[key_]['coeff'] = np.array([0.] * len(dict_[key_]['coeff']))
                elif i == 'TREATED & UNTREATED':
                    for key_ in ['TREATED', 'UNTREATED']:
                        dict_[key_]['coeff'] = np.array([0.] * len(dict_[key_]['coeff']))
                else:
                    dict_[i]['coeff'] = np.array([0.] * len(dict_[i]['coeff']))

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

    def test3(self):
        """The fourth test checks whether the simulation process works if there are only treated or
        untreated Agents by setting the number of agents to one.
        """
        constr = constraints(probability=0.0, agents=1)
        for _ in range(10):
            generate_random_dict(constr)
            simulate('test.grmpy.ini')

    def test4(self):
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
                np.testing.assert_array_almost_equal(gen_dict[key_]['coeff'], imp_dict[key_]['all'],
                                                     decimal=4)
                if key_ in ['TREATED', 'UNTREATED', 'COST']:
                    for i in range(len(gen_dict[key_]['types'])):
                        if isinstance(gen_dict[key_]['types'][i], str):
                            assert gen_dict[key_]['types'][i] == imp_dict[key_]['types'][i]
                        elif isinstance(gen_dict[key_]['types'][i], list):
                            assert gen_dict[key_]['types'][i][0] == imp_dict[key_]['types'][i][0]
                            np.testing.assert_array_almost_equal(
                                gen_dict[key_]['types'][i][1], imp_dict[key_]['types'][i][1], 4)

            for key_ in ['source', 'agents', 'seed']:
                assert gen_dict['SIMULATION'][key_] == imp_dict['SIMULATION'][key_]

    def test5(self):
        """The tests checks if the simulation process works even if the covariance between U1 and V
        and U0 and V is equal. Further the test ensures that the mte_information function returns
        the same value for each quantile.
        """
        for _ in range(10):
            dict_ = generate_random_dict()
            dict_['DIST']['coeff'][4] = dict_['DIST']['coeff'][5]
            print_dict(dict_)
            df = simulate('test.grmpy.ini')

            quantiles = [0.1] + np.arange(0.05, 1, 0.05).tolist() + [0.99]
            para = np.array([dict_['TREATED']['coeff'], dict_['UNTREATED']['coeff']])
            x = df.filter(regex=r'^X\_', axis=1)
            mte = mte_information(para, dict_['DIST']['coeff'][3:], quantiles, x)
            for i in mte:
                np.testing.assert_array_equal(i, mte[0])

    def test6(self):
        """The test ensures that the estimation process returns values that are approximately equal
        to the true values if the true values are set as start values for the estimation.
        """
        for i in range(10):
            constr = constraints(agents=1000, probability=0.0)
            generate_random_dict(constr)
            simulate('test.grmpy.ini')
            dict_ = read('test.grmpy.ini')
            true_dist = [dict_['DIST']['all'][0], dict_['DIST']['all'][3]]
            results = estimate('test.grmpy.ini', 'true_values')
            np.testing.assert_array_almost_equal(true_dist, results['DIST']['all'][:2])
            for key_ in ['TREATED', 'UNTREATED', 'COST']:
                np.testing.assert_array_almost_equal(results[key_]['all'], dict_[key_]['all'])


    def test7(self):
        """The test compares the estimation results from the old estimation process with the results
        of the new one.
        """
        constr = constraints(agents=100, probability=0.0)
        generate_random_dict(constr)
        simulate('test.grmpy.ini')
        results_old = estimate_old('test.grmpy.ini', 'true_values')
        results = estimate('test.grmpy.ini', 'true_values')
        for key_ in ['TREATED', 'UNTREATED', 'COST']:
            np.testing.assert_array_almost_equal(results[key_]['all'], results_old[key_]['all'])

        cleanup()