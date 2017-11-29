"""The file provides a loop over test 7 from test_unit.py for generating init files that induce
different estimation results  using the old and the new estimation process.."""
import numpy as np

from grmpy.test.resources.estimate_old import estimate_old
from grmpy.test.random_init import generate_random_dict
from grmpy.test.random_init import constraints
from grmpy.test.auxiliary import save_output
from grmpy.estimate.estimate import estimate
from grmpy.simulate.simulate import simulate
from grmpy.test.auxiliary import cleanup

for i in range(100):
    print(i)
    constr = constraints(agents=100, probability=0.0, optimizer='SCIPY-BFGS',
                         start='init')
    generate_random_dict(constr)
    simulate('test.grmpy.ini')
    results_old = estimate_old('test.grmpy.ini')
    results = estimate('test.grmpy.ini')
    save_output('test.grmpy.ini', str(i))
    for key_ in ['TREATED', 'UNTREATED', 'COST']:
        np.testing.assert_array_almost_equal(results[key_]['all'], results_old[key_]['all'],
                                             decimal=3)
    cleanup()
