"""The test provides the basic capabilities to run numerous property tests."""
from datetime import timedelta
from datetime import datetime
import functools
import traceback
import shutil
import random
import os

import numpy as np

from grmpy.test.auxiliary import cleanup
from development.tests.property.property_auxiliary import distribute_command_line_arguments
from development.tests.property.property_auxiliary import process_command_line_arguments
from development.tests.property.property_auxiliary import get_random_string
from development.tests.property.property_auxiliary import run_property_test
from development.tests.property.property_auxiliary import print_rslt_ext
from development.tests.property.property_auxiliary import collect_tests
from development.tests.property.property_auxiliary import finish


def choose_module(inp_dict):
    """Chooses a module with probability proportional to number of stored tests."""
    prob_dist = np.array([])
    for module in inp_dict.keys():
        prob_dist = np.append(prob_dist, len(inp_dict[module]))
    prob_dist = prob_dist / np.sum(prob_dist)
    return np.random.choice(list(inp_dict.keys()), p=prob_dist)


def run(args):
    """This function runs the property test battery."""
    args = distribute_command_line_arguments(args)

    test_dict = collect_tests()

    rslt = dict()
    for module in test_dict.keys():
        rslt[module] = dict()
        for test in test_dict[module]:
            rslt[module][test] = [0, 0]

    cleanup()

    if args['is_check']:
        np.random.seed(args['seed'])
        module = choose_module(test_dict)
        test = np.random.choice(test_dict[module])
        run_property_test(module, test)

    else:
        err_msg = []

        start, timeout = datetime.now(), timedelta(hours=args['hours'])

        print_rslt = functools.partial(print_rslt_ext, start, timeout)
        print_rslt(rslt, err_msg)

        while True:

            seed = random.randrange(1, 100000)
            dirname = get_random_string()
            np.random.seed(seed)
            module = choose_module(test_dict)
            test = np.random.choice(test_dict[module])

            try:
                run_property_test(module, test, dirname)
                rslt[module][test][0] += 1
            except Exception:
                rslt[module][test][1] += 1
                msg = traceback.format_exc()
                err_msg += [(module, test, seed, msg)]

            os.chdir('../')

            shutil.rmtree(dirname)

            print_rslt(rslt, err_msg)

            if timeout < datetime.now() - start:
                break

        finish(rslt)


if __name__ == '__main__':

    args = process_command_line_arguments('property')

    run(args)
