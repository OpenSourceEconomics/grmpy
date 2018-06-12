#!/usr/bin/env python
"""The test provides the basic capabilities to work with the regression tests of the package."""
import argparse
import json
import os

import numpy as np
import grmpy

from grmpy.test.random_init import generate_random_dict
from grmpy.test.random_init import print_dict
from grmpy.simulate.simulate import simulate
from grmpy.test.auxiliary import cleanup


def process_arguments(parser):
    """This function parses the input arguments."""
    args = parser.parse_args()

    # Distribute input arguments
    request = args.request

    # Test validity of input arguments
    assert request in ['check', 'create']

    return request


def create_vault(num_tests=100, seed=123):
    """This function creates a new regression vault."""
    np.random.seed(seed)

    tests = []
    for _ in range(num_tests):
        dict_ = generate_random_dict()
        df = simulate('test.grmpy.ini')
        stat = np.sum(df.sum())
        tests += [(stat, dict_)]
        cleanup()

    json.dump(tests, open('regression_vault.grmpy.json', 'w'))


def check_vault():
    """This function checks the complete regression vault that is distributed as part of the
    package.
    """
    fname = os.path.dirname(grmpy.__file__) + '/test/resources/regression_vault.grmpy.json'
    tests = json.load(open(fname))

    for test in tests:
        stat, dict_, criteria = test
        print_dict(dict_)
        df = simulate('test.grmpy.ini')
        np.testing.assert_almost_equal(np.sum(df.sum()), stat)
        cleanup('regression')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Work with regression tests for package.')

    parser.add_argument('--request', action='store', dest='request', required=True,
                        choices=['check', 'create'], help='request')

    request = process_arguments(parser)

    if request == 'check':
        check_vault()
    elif request == 'create':
        create_vault()
