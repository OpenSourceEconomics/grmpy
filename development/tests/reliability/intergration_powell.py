from grmpy.test.random_init import generate_random_dict
from grmpy.test.random_init import constraints
from grmpy.estimate.estimate import estimate
from grmpy.simulate.simulate import simulate
from grmpy.test.auxiliary import cleanup


for _ in range(500):
    constr = constraints(probability=0.0, agents=1000, optimizer='SCIPY-POWELL', start='init')

    generate_random_dict(constr)

    simulate('test.grmpy.ini')

    estimate('test.grmpy.ini')

    cleanup()
