
import numpy as np

AGENTS = 1000
SOURCE = 'dataset'


def generate_random_dict():
    """generates a random initialization dictionary"""

    dict_ = {}

    # Coefficients
    for key_ in ['UNTREATED', 'TREATED', 'COST']:

        dict_[key_] = {}

        for i in range(4):
            dict_[key_]['coeff'] = np.random.uniform(0., 2., [4])

    # Simulation parameters
    dict_['SIMULATION'] = {}
    for key_ in ['agents', 'source', 'seed']:
        if key_ == 'seed':
            dict_['SIMULATION'][key_] = np.random.randint(1, 10000)
        elif key_ == 'agents':
            dict_['SIMULATION'][key_] = AGENTS
        else:
            dict_['SIMULATION'][key_] = SOURCE

    dict_['DIST'] = {}

    # Variance and covariance parameters
    A = np.random.rand(3, 3)
    B = np.dot(A, A.transpose())

    for i in [0, 1, 2]:
        dict_['DIST']['sigma' + str(i + 1)] = B[i, i]

    for i in [2, 3, ]:
        if i == 2:
            dict_['DIST']['sigma21'] = B[1, 0]
        if i == 3:
            dict_['DIST']['sigma31'] = B[2, 0]
            dict_['DIST']['sigma32'] = B[2, 1]
    return dict_
