import copy

import numpy as np

from grmpy.test.random_init import generate_random_dict
from grmpy.test.random_init import constraints
from grmpy.test.random_init import print_dict
from grmpy.estimate.estimate import estimate
from grmpy.test.auxiliary import save_output
from grmpy.simulate.simulate import simulate
from grmpy.test.auxiliary import cleanup

np.random.seed(1234)
# Clean directory
cleanup()

# Construct a random dictionary
constr = constraints(probability=0.0, agents=1000, optimizer='SCIPY-BFGS', start='init_values')
init_dict = generate_random_dict(constr)

# Convert original init file
dict_ = copy.deepcopy(init_dict)
for key_ in ['TREATED', 'UNTREATED', 'COST']:
    dict_[key_]['coeff'][1:] = [0.0] * (len(dict_[key_]['coeff']) - 1)
print_dict(dict_, 'test_intercepts')

auto_dict = copy.deepcopy(init_dict)
auto_dict['ESTIMATION']['start'] = 'auto'
print_dict(auto_dict, 'test_auto')

# Simulate the data set
simulate('test.grmpy.ini')

# BFGS


# 1. Estimation with true values as start values
estimate('test.grmpy.ini')
save_output('est.grmpy.info', 'OUT_BFGS_true_values.info')

# 2. All COST/TREATED/UNTREATED coefficients are zero except intercepts
estimate('test_intercepts.grmpy.ini')
save_output('est.grmpy.info', 'OUT_BFGS_intercept_zero.info')

# 3. AUTO
estimate('test_auto.grmpy.ini')
save_output('est.grmpy.info', 'OUT_BFGS-auto.info')

# POWELL


# Adjust init files for POWELL estimation

for init in [init_dict, dict_, auto_dict]:
    init['ESTIMATION']['optimizer'] = 'SCIPY-POWELL'
print_dict(init_dict)
print_dict(dict_, 'test_intercepts')
print_dict(auto_dict, 'test_auto')

# 1. Estimation with true values as start values
estimate('test.grmpy.ini')
save_output('est.grmpy.info', 'OUT_POWELL_true_values')

# 2. All COST/TREATED/UNTREATED coefficients are zero except intercepts
estimate('test_intercepts.grmpy.ini')
save_output('est.grmpy.info', 'OUT_POWELL_intercept_zero.info')

# 3. AUTO
estimate('test_auto.grmpy.ini')
save_output('est.grmpy.info', 'OUT_POWELL-auto.info')
