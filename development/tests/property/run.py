"""The test provides the basic capabilities to run numerous property tests."""
import functools
import os
import random
import shutil
import traceback
from datetime import datetime, timedelta

import numpy as np

from development.tests.property.property_auxiliary import (
    collect_tests,
    distribute_command_line_arguments,
    finish,
    get_random_string,
    print_rslt_ext,
    process_command_line_arguments,
    run_property_test,
)
from grmpy.test.auxiliary import cleanup


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

    if args["is_check"]:
        np.random.seed(args["seed"])
        module = choose_module(test_dict)
        test = np.random.choice(test_dict[module])
        run_property_test(module, test)

    else:
        err_msg = []

        start, timeout = datetime.now(), timedelta(hours=args["hours"])

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

            os.chdir("../")

            shutil.rmtree(dirname)

            print_rslt(rslt, err_msg)

            if timeout < datetime.now() - start:
                break

        finish(rslt)


if __name__ == "__main__":

    args = process_command_line_arguments("property")

    run(args)
