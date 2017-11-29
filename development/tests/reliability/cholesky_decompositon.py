"""The module ensures that the cholesky decomposition and recomposition works appropriately.
For this purpose the module provides a loop over test6 from test_unit.py.
"""
import json
import os

from scipy.stats import wishart
import numpy as np

from grmpy.estimate.estimate_auxiliary import backward_cholesky_transformation
from grmpy.simulate.simulate_auxiliary import construct_covariance_matrix
from grmpy.estimate.estimate_auxiliary import provide_cholesky_decom
from grmpy.test.auxiliary import adjust_output_cholesky

NUM_TESTS = 10000
np.random.seed(1234235)
seeds = np.random.randint(0, 1000, size=NUM_TESTS)
df = np.random.randint(10,100, size=NUM_TESTS).tolist()
directory = os.path.dirname(__file__)
file_dir = os.path.join(directory, 'cholesky_decomposition.grmpy.json')


if True:
    pseudo_dict_list = []
    for i in range(NUM_TESTS):
        np.random.seed(seeds[i])
        pseudo_dict = {'WISHART': {'df': [df[i]]}, 'DIST': {'all': []}, 'AUX': {'init_values': []}}
        b = wishart.rvs(df=df[i], scale=np.identity(3), size=1)
        parameter = b[np.triu_indices(3)]
        for i in [0, 3, 5]:
            parameter[i] **= 0.5
        pseudo_dict['DIST']['all'] = parameter
        pseudo_dict['AUX']['init_values'] = parameter.tolist()
        pseudo_dict_list += [pseudo_dict]
        cov_1 = construct_covariance_matrix(pseudo_dict)
        x0, start = provide_cholesky_decom(pseudo_dict, [], 'init')
        output = backward_cholesky_transformation(x0, test=True)
        output = adjust_output_cholesky(output)
        pseudo_dict['DIST']['all'] = output
        cov_2 = construct_covariance_matrix(pseudo_dict)
        try:
            np.testing.assert_array_almost_equal(cov_1, cov_2)
            pseudo_dict['STATUS'] = True
        except AssertionError:
            print('The cholesky de- and recomposition functions failed.')
            pseudo_dict['STATUS'] = False
    json.dump(pseudo_dict_list, open(file_dir, 'w'))





