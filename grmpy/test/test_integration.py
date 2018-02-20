"""The module includes an integration and a regression test for the simulation process."""
import json
import os

import numpy as np
import pytest

from grmpy.estimate.estimate_auxiliary import calculate_criteria
from grmpy.estimate.estimate_auxiliary import simulate_estimation
from grmpy.estimate.estimate_auxiliary import start_values
from grmpy.check.check import check_initialization_dict
from grmpy.test.random_init import generate_random_dict
from grmpy.check.custom_exceptions import UserError
from grmpy.test.random_init import constraints
from grmpy.check.check import check_init_file
from grmpy.test.random_init import print_dict
from grmpy.estimate.estimate import estimate
from grmpy.simulate.simulate import simulate
from grmpy.test.auxiliary import read_desc
from grmpy.test.auxiliary import cleanup
from grmpy.read.read import read
import grmpy


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
    fname = os.path.dirname(grmpy.__file__) + '/test/resources/regression_vault.grmpy.json'
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
        df2 = simulate_estimation(init_dict, rslt, df1)
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
    """The test checks if the estimation process works properly when maxiter is set to
    zero.
    """
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
    fname_num = os.path.dirname(grmpy.__file__) + '/test/resources/test_num.grmpy.ini'
    fname_zero = os.path.dirname(grmpy.__file__) + '/test/resources/test_zero.grmpy.ini'
    fname_vzero = os.path.dirname(grmpy.__file__) + '/test/resources/test_vzero.grmpy.ini'
    fname_possd = os.path.dirname(grmpy.__file__) + '/test/resources/test_npsd.grmpy.ini'
    fname_order =  os.path.dirname(grmpy.__file__) + '/test/resources/test_order.grmpy.ini'

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
    dict_['COST']['order'][1] = 1   
    print_dict(dict_)
    pytest.raises(UserError, check_initialization_dict, dict_)
    pytest.raises(UserError, simulate, fname_order)
    pytest.raises(UserError, estimate, fname_order)


    dict_ = read(fname_num)
    pytest.raises(UserError, check_initialization_dict, dict_)
    pytest.raises(UserError, simulate, fname_num)

    dict_ = read(fname_possd)
    pytest.raises(UserError, check_initialization_dict, dict_)
    pytest.raises(UserError, simulate, fname_possd)


    dict_ = read(fname_zero)
    pytest.raises(UserError, check_init_file, dict_)
    pytest.raises(UserError, estimate, fname_zero)

    dict_ = read(fname_vzero)
    pytest.raises(UserError, check_init_file, dict_)
    pytest.raises(UserError, estimate, fname_vzero)

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
