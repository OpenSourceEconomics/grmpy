"""The file provides a setup to check files specific to the #82 GitHub issue."""
import numpy as np


from grmpy.test.resources.estimate_old import estimate_old
from grmpy.simulate.simulate import simulate
from grmpy.estimate.estimate import estimate


simulate('test.grmpy.txt')
results_old = estimate_old('test.grmpy.txt')
results = estimate('test.grmpy.txt')
for key_ in ['TREATED', 'UNTREATED', 'COST']:
    np.testing.assert_array_almost_equal(results[key_]['all'], results_old[key_]['all'],
                                         decimal=3)

