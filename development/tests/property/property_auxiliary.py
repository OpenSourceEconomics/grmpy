"""This module contains some auxiliary functions for the property testing."""
from datetime import datetime
import importlib
import argparse
import shutil
import string
import glob
import os

import numpy as np

from grmpy.grmpy_config import PACKAGE_DIR


def collect_tests():
    """This function collects all available tests."""

    current_wd = os.getcwd()
    os.chdir(PACKAGE_DIR + '/test')
    test_modules = glob.glob('test_*.py')
    os.chdir(current_wd)
    test_dict = dict()
    for module in sorted(test_modules):
        test_dict[module] = []
        mod = importlib.import_module('grmpy.test.' + module.replace('.py', ''))
        for candidate in sorted(dir(mod)):
            if 'test' in candidate and 'py' not in candidate:
                test_dict[module] += [candidate]
    return test_dict


def run_property_test(module, test, dirname=None):
    """This function runs a single replication test."""
    mod = importlib.import_module('grmpy.test.' + module.replace('.py', ''))
    test_fun = getattr(mod, test)

    # We do not switch directories if we are investigating a failed test case.
    if dirname is not None:
        if os.path.exists(dirname):
            shutil.rmtree(dirname)

        os.mkdir(dirname)
        os.chdir(dirname)

    test_fun()


def print_rslt_ext(start, timeout, rslt, err_msg):
    """This function print out the current state of the property tests."""

    start_time = start.strftime("%Y-%m-%d %H:%M:%S")
    end_time = (start + timeout).strftime("%Y-%m-%d %H:%M:%S")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open('property.grmpy.info', 'w') as outfile:

        # Write out some header information.
        outfile.write('\n\n')
        str_ = '\t{0[0]:<15}{0[1]:<20}\n\n'
        outfile.write(str_.format(['START', start_time]))
        outfile.write(str_.format(['FINISH', end_time]))
        outfile.write(str_.format(['UPDATE', current_time]))

        modules = sorted(rslt.keys())

        for module in modules:

            outfile.write('\n ' + module.replace('.py', '') + '\n\n')

            string = '{:>18}{:>15}{:>15}\n\n'
            outfile.write(string.format('Test', 'Success', 'Failure'))

            for test in sorted(rslt[module].keys()):

                stat = rslt[module][test]

                string = '{:>18}{:>15}{:>15}\n'
                outfile.write(string.format(*[test] + stat))

            outfile.write('\n')
        outfile.write('\n' + '-' * 79 + '\n\n')

        for err in err_msg:

            module, test, seed, msg = err

            string = 'MODULE {:<25} TEST {:<15} SEED {:<15}\n\n'
            outfile.write(string.format(*[module, test, seed]))
            outfile.write(msg)
            outfile.write('\n' + '-' * 79 + '\n\n')


def finish(rslt):
    """This function simply finalizes the logging."""
    # We want to record the overall performance.
    num_tests_total, num_success = 0, 0
    for module in sorted(rslt.keys()):
        for test in sorted(rslt[module].keys()):
            num_tests_total += np.sum(rslt[module][test])
            num_success += rslt[module][test][0]

    with open('property.grmpy.info', 'a') as outfile:
        string = '{:>18}{:>15}\n'
        outfile.write(string.format(*['Success', num_tests_total]))
        outfile.write(string.format(*['Total', num_success]))

        outfile.write('\n TERMINATED')


def distribute_command_line_arguments(args):
    """This function distributes the command line arguments."""
    rslt = dict()
    try:
        rslt['num_tests'] = args.num_tests
    except AttributeError:
        pass

    try:
        rslt['request'] = args.request
    except AttributeError:
        pass

    try:
        rslt['hours'] = args.hours
    except AttributeError:
        pass

    try:
        rslt['seed'] = args.seed
    except AttributeError:
        pass

    try:
        rslt['is_update'] = args.is_update
    except AttributeError:
        pass

    rslt['is_check'] = rslt['request'] in ['check', 'investigate']

    return rslt


def process_command_line_arguments(which):
    """This function processes the command line arguments for the test battery."""
    is_request, is_hours, is_seed, is_test, is_update = False, False, False, False, False

    if which == 'replication':
        msg = 'Test replication of package'
        is_request, is_hours, is_seed = True, True, True
    elif which == 'regression':
        msg = 'Test package for regressions'
        is_request, is_test, is_update = True, True, True
    elif which == 'property':
        msg = 'Property testing of package'
        is_request, is_seed, is_hours = True, True, True
    else:
        raise NotImplementedError

    parser = argparse.ArgumentParser(msg)

    if is_request:
        if which == 'regression':
            parser.add_argument('--request', action='store', dest='request', help='task to perform',
                                required=True, choices=['check', 'create'])
        else:
            parser.add_argument('--request', action='store', dest='request', help='task to perform',
                                required=True, choices=['run', 'investigate'])

    if is_hours:
        parser.add_argument('--hours', action='store', dest='hours', type=float, help='hours')

    if is_seed:
        parser.add_argument('--seed', action='store', dest='seed', type=int, help='seed')

    if is_test:
        parser.add_argument('--tests', action='store', dest='num_tests', required=True, type=int,
                            help='number of tests')

    if is_update:
        parser.add_argument('--update', action='store_true', dest='is_update', required=False,
                            help='update regression vault')

    return parser.parse_args()


def get_random_string(size=6):
    """This function samples a random string of varying size."""
    chars = list(string.ascii_lowercase)
    str_ = ''.join(np.random.choice(chars) for _ in range(size))
    return str_
