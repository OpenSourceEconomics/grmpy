
import numpy as np
import random
import string

AGENTS = 1000


def generate_random_dict():
    """generates a random initialization dictionary"""
    SOURCE = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(8)])

    dict_ = {}
    treated_num = np.random.randint(4, 10)
    cost_num = np.random.randint(4, 10)

    # Coefficients
    for key_ in ['UNTREATED', 'TREATED', 'COST']:

        dict_[key_] = {}

        if key_ in ['UNTREATED', 'TREATED']:

            dict_[key_]['coeff'] = np.random.normal(0.0, 2., [treated_num])

        else:
            dict_[key_]['coeff'] = np.random.normal(0., 2., [cost_num])

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
    print(B)

    dict_['DIST']['coeff'] = []

    for i in range(3):
        dict_['DIST']['coeff'].append(B[i, i])

    dict_['DIST']['coeff'].append(B[1, 0])
    dict_['DIST']['coeff'].append(B[2, 0])
    dict_['DIST']['coeff'].append(B[2, 1])

    dict_['DIST']['coeff'] = np.asarray(dict_['DIST']['coeff'])


    return dict_

