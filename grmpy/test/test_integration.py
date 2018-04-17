"""The module includes an integration and a regression test for the simulation process."""
import json

import numpy as np
import pytest

from grmpy.estimate.estimate_auxiliary import calculate_criteria
from grmpy.estimate.estimate_auxiliary import simulate_estimation
from grmpy.estimate.estimate_auxiliary import start_values
from grmpy.check.check import check_initialization_dict
from grmpy.test.random_init import generate_random_dict
from grmpy.check.custom_exceptions import UserError
from grmpy.grmpy_config import TEST_RESOURCES_DIR
from grmpy.test.random_init import constraints
from grmpy.check.check import check_init_file
from grmpy.test.random_init import print_dict
from grmpy.estimate.estimate import estimate
from grmpy.simulate.simulate import simulate
from grmpy.test.auxiliary import read_desc
from grmpy.test.auxiliary import cleanup
from grmpy.read.read import read


def test1():
    """The test runs a loop to check the consistency of the random init file generating process
    and the following simulation.
    """
    for _ in range(10):
        dict_ = generate_random_dict()
        print_dict(dict_)
        simulate('test.grmpy.ini')


def test2():
    """This test runs a random selection of five regression tests from the package's
    regression test vault.
    """
    fname = TEST_RESOURCES_DIR + '/regression_vault.grmpy.json'
    tests = json.load(open(fname))

    for i in np.random.choice(range(len(tests)), size=5):
        stat, dict_, criteria = tests[i]
        print_dict(dict_)
        df = simulate('test.grmpy.ini')
        init_dict = read('test.grmpy.ini')
        start = start_values(init_dict, df, 'init')
        criteria_ = calculate_criteria(init_dict, df, start)
        np.testing.assert_almost_equal(np.sum(df.sum()), stat)
        np.testing.assert_array_almost_equal(criteria, criteria_)


def test3():
    """The test checks if the criteria function value of the simulated and the 'estimated'
    sample is equal if both samples include an identical number of individuals.
    """
    for _ in range(5):
        constr = constraints(probability=0.0, agents=10000, start='init',
                             optimizer='SCIPY-BFGS', same_size=True)
        generate_random_dict(constr)

        df1 = simulate('test.grmpy.ini')
        rslt = estimate('test.grmpy.ini')
        init_dict = read('test.grmpy.ini')
        df2 = simulate_estimation(init_dict, rslt)
        start = start_values(init_dict, df1, 'init')

        criteria = []
        for data in [df1, df2]:
            criteria += [calculate_criteria(init_dict, data, start)]
        np.testing.assert_allclose(criteria[1], criteria[0], rtol=0.1)


def test4():
    """The test checks if the estimation process works if the Powell algorithm is specified as
    the optimizer option.
    """
    for _ in range(5):
        constr = constraints(probability=0.0, agents=10000, start='init',
                             optimizer='SCIPY-POWELL')
        generate_random_dict(constr)

        simulate('test.grmpy.ini')
        estimate('test.grmpy.ini')


def test5():
    """The test checks if the estimation process works properly when maxiter is set to zero."""
    for _ in range(10):
        constr = constraints(probability=0.0, maxiter=0)
        generate_random_dict(constr)
        simulate('test.grmpy.ini')
        estimate('test.grmpy.ini')


def test6():
    """Additionally to test5 this test checks if the descriptives file provides the expected
    output when maxiter is set to zero and the estimation process uses the initialization file
    values as start values.
    """
    for _ in range(5):
        constr = constraints(probability=0.0, maxiter=0, agents=10000, start='init', same_size=True)
        generate_random_dict(constr)
        simulate('test.grmpy.ini')
        estimate('test.grmpy.ini')
        dict_ = read_desc('comparison.grmpy.txt')
        for key_ in ['All', 'Treated', 'Untreated']:
            np.testing.assert_equal(len(set(dict_[key_]['Number'])), 1)
            np.testing.assert_array_equal(dict_[key_]['Observed Sample'],
                                          dict_[key_]['Simulated Sample (finish)'])
            np.testing.assert_array_equal(dict_[key_]['Simulated Sample (finish)'],
                                          dict_[key_]['Simulated Sample (start)'])


def test7():
    """This test ensures that the estimation process returns an UserError if one tries to execute an
    estimation process with initialization file values as start values for an deterministic setting.
    """
    fname_diff = TEST_RESOURCES_DIR + '/test_binary_diff.grmpy.ini'
    fname_vzero = TEST_RESOURCES_DIR + '/test_vzero.grmpy.ini'
    fname_possd = TEST_RESOURCES_DIR + '/test_npsd.grmpy.ini'
    fname_zero = TEST_RESOURCES_DIR + '/test_zero.grmpy.ini'

    for i in range(10):
        constr = constraints(agents=1000, probability=1.0)
        generate_random_dict(constr)
        dict_ = read('test.grmpy.ini')
        pytest.raises(UserError, check_init_file, dict_)
        pytest.raises(UserError, estimate, 'test.grmpy.ini')

        constr = constraints(agents=0, probability=.0, sample=100)
        generate_random_dict(constr)
        dict_ = read('test.grmpy.ini')
        pytest.raises(UserError, check_initialization_dict, dict_)
        pytest.raises(UserError, simulate, 'test.grmpy.ini')

        constr = constraints(agents=1000, probability=.0, sample=100)
        generate_random_dict(constr)
        dict_ = read('test.grmpy.ini')
        if len(dict_['COST']['order']) == 1:
            dict_['COST']['all'] = list(dict_['COST']['all'])
            dict_['COST']['all'] += [1.000]
            dict_['COST']['order'] += [2]
            dict_['COST']['types'] += ['nonbinary']

        dict_['COST']['order'][1] = 1
        print_dict(dict_)
        pytest.raises(UserError, check_initialization_dict, dict_)
        pytest.raises(UserError, simulate, 'test.grmpy.ini')
        pytest.raises(UserError, estimate, 'test.grmpy.ini')

        tests = []
        tests += [['TREATED','UNTREATED'], ['TREATED', 'COST'], ['UNTREATED', 'COST']]
        tests += [['TREATED', 'UNTREATED', 'COST']]

        for combi in tests:
            constr = constraints(0.0, agents=1000, state_diff=True, overlap=True)
            generate_random_dict(constr)
            dict_ = read('test.grmpy.ini')
            for j in combi:

                if len(dict_[j]['order']) == 1:
                    dict_[j]['all'] = list(dict_[j]['all'])
                    dict_[j]['all'] += [1.000]
                    dict_[j]['order'] += [2]
                    dict_[j]['types'] += ['nonbinary']
                else:
                    pass
                dict_[j]['order'][1] = len(dict_['AUX']['types']) +1

                frac = np.random.uniform(0.1, 0.8)
                dict_[j]['types'][1] = ['binary', frac]

            print_dict(dict_)

            pytest.raises(UserError, read, 'test.grmpy.ini')

    dict_ = read(fname_possd)
    pytest.raises(UserError, check_initialization_dict, dict_)
    pytest.raises(UserError, simulate, fname_possd)

    dict_ = read(fname_zero)
    pytest.raises(UserError, check_init_file, dict_)
    pytest.raises(UserError, estimate, fname_zero)

    dict_ = read(fname_vzero)
    pytest.raises(UserError, check_init_file, dict_)
    pytest.raises(UserError, estimate, fname_vzero)
    
    dict_ = read(fname_diff)
    pytest.raises(UserError, check_initialization_dict, dict_)
    pytest.raises(UserError, estimate, fname_diff)
    

def test8():
    """The test checks if an UserError occurs if wrong inputs are specified for a different
    functions/methods.
    """
    constr = constraints(agents=1000, probability=.0)
    generate_random_dict(constr)
    df = simulate('test.grmpy.ini')
    a = []
    dict_ = read('test.grmpy.ini')
    dict_['ESTIMATION']['file'] = 'data.grmpy.ini'
    print_dict(dict_, 'false_data')
    pytest.raises(UserError, estimate, 'tast.grmpy.ini')
    pytest.raises(UserError, estimate, 'false_data.grmpy.ini')
    pytest.raises(UserError, simulate, 'tast.grmpy.ini')
    pytest.raises(UserError, read, 'tast.grmpy.ini')
    pytest.raises(UserError, start_values, a, df, 'init')
    pytest.raises(UserError, generate_random_dict, a)

    cleanup()


def test9():
    """This test ensures that the random initialization file generating process, the read in process
    and the simulation process works if the constraints function allows for different number of co-
    variates for each treatment state and the occurence of cost-benefit shifters."""
    for _ in range(10):
        constr = constraints(0.0, agents=1000, state_diff=True, overlap=True)
        generate_random_dict(constr)
        read('test.grmpy.ini')
        simulate('test.grmpy.ini')
        estimate('test.grmpy.ini')




