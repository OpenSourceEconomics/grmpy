#!/usr/bin/env python
"""The test provides the basic capabilities to run numerous property tests."""
import datetime

import statsmodels
import subprocess


from grmpy.test.random_init import generate_random_dict
from grmpy.test.random_init import print_dict
from grmpy.test.random_init import constraints
import grmpy

# We simply specify a minimum number of minutes for our package to run with different requests.
MINUTES = 1

end_time = datetime.datetime.now() + datetime.timedelta(minutes=MINUTES)
counter = 1
while True:
    if datetime.datetime.now() >= end_time:
        break

    print('\n Iteration ', counter)

    constr = constraints()

    constr['DETERMINISTIC'] = False
    dict_ = generate_random_dict(constr)
    print_dict(dict_)

    grmpy.simulate('test.grmpy.ini')

    # This is a temporary fix so that the determination of starting values by PROBIT does
    # not work if we have a perfect separation.
    grmpy.estimate('test.grmpy.ini')
    subprocess.check_call(['git', 'clean', '-d', '-f'])

    counter += 1
