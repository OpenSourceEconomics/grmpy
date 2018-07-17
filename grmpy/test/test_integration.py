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
    random_choice = np.random.choice(range(len(tests)), 5)
    print(random_choice)
    tests = [tests[i] for i in random_choice]

    for test in tests:
        stat, dict_, criteria = test
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
        constr = dict()
        constr['DETERMINISTIC'], constr['AGENTS'], constr['START'] = False, 1000, 'init'
        constr['OPTIMIZER'], constr['SAME_SIZE'] = 'SCIPY-BFGS', True
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
        constr = dict()
        constr['DETERMINISTIC'], constr['AGENTS'], constr['start'] = False, 10000, 'init'
        constr['optimizer'] = 'SCIPY-Powell'
        generate_random_dict(constr)

        simulate('test.grmpy.ini')
        estimate('test.grmpy.ini')


def test5():
    """The test checks if the estimation process works properly when maxiter is set to zero."""
    for _ in range(10):
        constr = dict()
        constr['DETERMINISTIC'], constr['MAXITER'] = False, 0
        generate_random_dict(constr)
        simulate('test.grmpy.ini')
        estimate('test.grmpy.ini')


def test6():
    """Additionally to test5 this test checks if the descriptives file provides the expected
    output when maxiter is set to zero and the estimation process uses the initialization file
    values as start values.
    """
    for _ in range(5):
        constr = dict()
        constr['DETERMINISTIC'], constr['MAXITER'], constr['AGENTS'] = False, 0, 10000
        constr['START'], constr['SAME_SIZE'] = 'init', True
        generate_random_dict(constr)
        simulate('test.grmpy.ini')
        estimate('test.grmpy.ini')
        dict_ = read_desc('comparison.grmpy.txt')
        for key_ in ['All', 'Treated', 'Untreated']:
            np.testing.assert_equal(len(set(dict_[key_]['Number'])), 1)
            np.testing.assert_almost_equal(
                dict_[key_]['Observed Sample'], dict_[key_]['Simulated Sample (finish)'], 0.001)
            np.testing.assert_array_almost_equal(
                dict_[key_]['Simulated Sample (finish)'],
                dict_[key_]['Simulated Sample (start)'], 0.001)


def test7():
    """This test ensures that the estimation process returns an UserError if one tries to execute an
    estimation process with initialization file values as start values for an deterministic setting.
    """
    fname_diff_categorical = TEST_RESOURCES_DIR + '/test_categorical_diff.grmpy.ini'
    fname_categorical = TEST_RESOURCES_DIR + '/test_categorical.grmpy.ini'
    fname_diff_binary = TEST_RESOURCES_DIR + '/test_binary_diff.grmpy.ini'
    fname_vzero = TEST_RESOURCES_DIR + '/test_vzero.grmpy.ini'
    fname_possd = TEST_RESOURCES_DIR + '/test_npsd.grmpy.ini'
    fname_zero = TEST_RESOURCES_DIR + '/test_zero.grmpy.ini'

    for _ in range(5):
        constr = dict()
        constr['AGENTS'], constr['DETERMINISTIC'] = 1000, True
        generate_random_dict(constr)
        dict_ = read('test.grmpy.ini')
        pytest.raises(UserError, check_init_file, dict_)
        pytest.raises(UserError, estimate, 'test.grmpy.ini')

        generate_random_dict(constr)
        dict_ = read('test.grmpy.ini')
        if len(dict_['CHOICE']['order']) == 1:
            dict_['CHOICE']['all'] = list(dict_['CHOICE']['all'])
            dict_['CHOICE']['all'] += [1.000]
            dict_['CHOICE']['order'] += [2]
            dict_['CHOICE']['types'] += ['nonbinary']

        dict_['CHOICE']['order'][1] = 1
        print_dict(dict_)
        pytest.raises(UserError, check_initialization_dict, dict_)
        pytest.raises(UserError, simulate, 'test.grmpy.ini')
        pytest.raises(UserError, estimate, 'test.grmpy.ini')

        constr['AGENTS'] = 0
        generate_random_dict(constr)
        dict_ = read('test.grmpy.ini')
        pytest.raises(UserError, check_initialization_dict, dict_)
        pytest.raises(UserError, simulate, 'test.grmpy.ini')

        tests = []
        tests += [['TREATED', 'UNTREATED'], ['TREATED', 'CHOICE'], ['UNTREATED', 'CHOICE']]
        tests += [['TREATED', 'UNTREATED', 'CHOICE']]

        for combi in tests:
            constr['STATE_DIFF'], constr['OVERLAP'] = True, True
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
                dict_[j]['order'][1] = len(dict_['AUX']['types']) + 1

                frac = np.random.uniform(0.1, 0.8)
                dict_[j]['types'][1] = ['binary', frac]

            print_dict(dict_)

            pytest.raises(UserError, read, 'test.grmpy.ini')

    dict_ = read(fname_possd)
    pytest.raises(UserError, check_initialization_dict, dict_)
    pytest.raises(UserError, simulate, fname_possd)

    dict_ = read(fname_categorical)
    pytest.raises(UserError, check_initialization_dict, dict_)
    pytest.raises(UserError, simulate, fname_categorical)

    dict_ = read(fname_zero)
    pytest.raises(UserError, check_init_file, dict_)
    pytest.raises(UserError, estimate, fname_zero)

    dict_ = read(fname_vzero)
    pytest.raises(UserError, check_init_file, dict_)
    pytest.raises(UserError, estimate, fname_vzero)

    dict_ = read(fname_diff_binary)
    pytest.raises(UserError, check_initialization_dict, dict_)
    pytest.raises(UserError, estimate, fname_diff_binary)

    dict_ = read(fname_diff_categorical)
    pytest.raises(UserError, check_initialization_dict, dict_)
    pytest.raises(UserError, estimate, fname_diff_categorical)


def test8():
    """The test checks if an UserError occurs if wrong inputs are specified for a different
    functions/methods.
    """
    constr = dict()
    constr['DETERMINISTIC'], constr['AGENTS'] = False, 1000
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


def test9():
    """This test ensures that the random initialization file generating process, the read in process
    and the simulation process works if the constraints function allows for different number of co-
    variates for each treatment state and the occurence of cost-benefit shifters."""
    for _ in range(5):
        constr = dict()
        constr['DETERMINISTIC'], constr['AGENT'], constr['STATE_DIFF'] = False, 1000, True
        constr['OVERLAP'] = True
        generate_random_dict(constr)
        read('test.grmpy.ini')
        simulate('test.grmpy.ini')
        estimate('test.grmpy.ini')

    cleanup()
