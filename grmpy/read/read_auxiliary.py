import numpy as np


def process(list_, dict_, keyword):
    """processes keyword parameters"""
    name, val = list_[0], list_[1]

    if name not in dict_[keyword].keys() and name in ['coeff']:
        dict_[keyword][name] = []
    # Type conversion
    if name in ['agents', 'seed']:
        val = int(val)
    elif name in ['source']:
        val = str(val)
    else:
        val = float(val)

    if name in ['coeff']:
        dict_[keyword][name] += [val]
    else:
        dict_[keyword][name] = val

    # Finishing.
    return dict_


def auxiliary(dict_):
    """
    """
    dict_['AUX'] = {}
    if dict_['DIST']['coeff'] == [0.0] * len(dict_['DIST']['coeff']):
        is_deterministic = True
    else:
        is_deterministic = False

    for key_ in ['UNTREATED', 'TREATED', 'COST', 'DIST']:
        if key_ in ['UNTREATED', 'TREATED', 'COST']:
                dict_[key_]['all'] = dict_[key_]['coeff']
                dict_[key_]['all'] = np.array(dict_[key_]['all'])
        else:
            dict_[key_]['all'] = dict_[key_]['coeff']
            dict_[key_]['all'] = np.array(dict_[key_]['all'])

    # Create keys that contain all standard deviation and covariance parameters

    # Number of covariates
    num_covars_out = len(dict_['TREATED']['all'])
    num_covars_cost = len(dict_['COST']['all'])

    dict_['AUX']['num_covars_out'] = num_covars_out
    dict_['AUX']['num_covars_cost'] = num_covars_cost

    # Number of parameters
    dict_['AUX']['num_paras'] = 2 * num_covars_out + num_covars_cost + 2 + 2

    # Starting values
    dict_['AUX']['init_values'] = []

    for key_ in ['TREATED', 'UNTREATED', 'COST', 'DIST']:
        dict_['AUX']['init_values'] += dict_[key_]['all'].tolist()

        for j in sorted(dict_[key_].keys()):
            if j in ['all']:
                pass
            else:
                del dict_[key_][j]
    dict_['DETERMINISTIC'] = is_deterministic

    return dict_
