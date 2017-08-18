

import numpy as np
import random
import string


def constraints(probability=0.1, is_zero=False, Agents=1000):
    constraints_dict = {}
    constraints_dict['DETERMINISTIC'] = random.random() < probability
    constraints_dict['IS_ZERO'] = is_zero
    constraints_dict['AGENTS'] = Agents
    return constraints_dict


def generate_random_dict(constraints_dict=None):
    """generates a random initialization dictionary"""

    if constraints_dict is not None:
        assert isinstance(constraints_dict, dict)
    else:
        constraints_dict = {}
        # adjust !!!!

    if 'DETERMINISTIC' in constraints_dict.keys():
        is_deterministic = constraints_dict['DETERMINISTIC']
    else:
        is_deterministic = False

    if 'IS_ZERO' in constraints_dict.keys():
        is_zerocoeff = constraints_dict['IS_ZERO']
    else:
        is_zerocoeff = False
    if 'AGENTS' in constraints_dict.keys():
        AGENTS = constraints_dict['AGENTS']
    else:
        AGENTS = 1000

    assert is_zerocoeff != is_deterministic or is_zerocoeff == is_deterministic == False

    SOURCE = ''.join(
        [random.choice(string.ascii_letters + string.digits) for n in range(8)])

    dict_ = {}
    treated_num = np.random.randint(1, 10)
    cost_num = np.random.randint(1, 10)
    # Coefficients
    for key_ in ['UNTREATED', 'TREATED', 'COST']:

        dict_[key_] = {}

        if key_ in ['UNTREATED', 'TREATED']:
            if not is_deterministic:

                if not is_zerocoeff:
                    dict_[key_]['coeff'] = np.random.normal(
                        0.0, 2., [treated_num])
                else:
                    dict_[key_]['coeff'] = np.array([0] * treated_num)
            else:
                dict_[key_]['coeff'] = np.array([])

        else:
            if not is_deterministic:
                if not is_zerocoeff:
                    dict_[key_]['coeff'] = np.random.normal(0., 2., [cost_num])
                else:
                    dict_[key_]['coeff'] = np.array([0] * cost_num)
            else:
                dict_[key_]['coeff'] = np.array([])

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

    dict_['DIST']['coeff'] = []

    for i in range(3):
        dict_['DIST']['coeff'].append(B[i, i])

    dict_['DIST']['coeff'].append(B[1, 0])
    dict_['DIST']['coeff'].append(B[2, 0])
    dict_['DIST']['coeff'].append(B[2, 1])

    dict_['DIST']['coeff'] = np.asarray(dict_['DIST']['coeff'])

    return dict_
