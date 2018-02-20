"""This module provides auxiliary functions for the import process of the init file."""
import numpy as np

from grmpy.check.custom_exceptions import UserError

def process(list_, dict_, keyword):
    """The function processes keyword parameters and creates dictionary elements."""
    if len(list_) == 5:
        name, order,  val, type_, frac_ = list_[0], list_[1], list_[2], list_[3], list_[4]
    elif len(list_) in [3,4]:
        name, order, val  = list_[0], list_[1], list_[2]

    elif list_[0] == 'direc':
        name, val = list_[0], [list_[i] for i in range(len(list_)) if i > 0]

    else:
        name, val = list_[0], list_[1]


    if name not in dict_[keyword].keys() and name in ['coeff']:
        dict_[keyword][name] = []
    if keyword in ['TREATED', 'UNTREATED', 'COST'] and 'types' not in dict_[keyword].keys():
        dict_[keyword]['types'] = []
    if keyword in ['TREATED', 'UNTREATED', 'COST'] and 'order' not in dict_[keyword].keys():
        dict_[keyword]['order'] = []

    if keyword in ['TREATED', 'UNTREATED', 'COST']:
        if len(list_) == 5:
            dict_[keyword]['order'] += [int(order)]
            dict_[keyword]['types'] += [[type_, float(frac_)]]
        else:
            dict_[keyword]['order'] += [int(order)]
            dict_[keyword]['types'] += ['nonbinary']

    # Type conversion
    if name in ['agents', 'seed', 'maxiter', 'disp']:
        val = int(val)
    elif name in ['source', 'file', 'optimizer', 'start']:
        val = str(val)
    elif name in ['direc']:
        val = list(val)
    else:
        val = float(val)
    if name in ['coeff']:
        dict_[keyword][name] += [val]
    else:
        dict_[keyword][name] = val
    # Finishing.
    return dict_


def auxiliary(dict_):
    """The function creates an new dictionary entry 'AUX' that includes starting values of each
    parameter and the number of covariates.
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
            if j in ['all', 'types', 'order']:
                pass
            else:
                del dict_[key_][j]
    dict_['DETERMINISTIC'] = is_deterministic
    dict_ = check_types(dict_)

    return dict_


def check_types(dict_):
    """This function ensures that the variable types agree across the two treatment states and the
    costs.
    """
    list_ = []
    covars = set(dict_['TREATED']['order'] +  dict_['UNTREATED']['order'] +  dict_['COST']['order'])
    for i in covars:
        if i in dict_['TREATED']['order'] and i in dict_['UNTREATED']['order'] and \
                        i in dict_['COST']['order']:
            keys = ['TREATED', 'UNTREATED', 'COST']
            for key_ in keys:
                index = dict_[key_]['order'].index(i)
                if isinstance(dict_[key_]['types'][index], list):
                    other_keys = [j for j in keys if j != key_]
                    for other in other_keys:
                        index_other = dict_[other]['order'].index(i)
                        if not isinstance(dict_[other]['types'][index_other], list):
                            dict_[other]['types'][index_other] = dict_[key_]['types'][index]
                        elif dict_[other]['types'][index_other] == dict_[key_]['types'][index]:
                            pass
                        else:
                            msg = 'Your initilaization file has two different binary specification ' \
                                  'for the same covariate.'
                            raise UserError(msg)
            list_ += [dict_['COST']['types'][index]]

        elif (i in dict_['TREATED']['order'] and i in dict_['UNTREATED']['order']):
            keys = ['TREATED', 'UNTREATED']
            for key_ in keys:
                index = dict_[key_]['order'].index(i)
                if isinstance(dict_[key_]['types'][index], list):
                    other_keys = [j for j in keys if j != key_]

                    for other in other_keys:
                        index_other = dict_[other]['order'].index(i)
                        if not isinstance(dict_[other]['types'][index_other], list):
                            dict_[other]['types'][index_other] = dict_[key_]['types'][index]

                        elif dict_[other]['types'][index_other] == dict_[key_]['types'][index]:
                            pass
                        else:
                            msg = 'Your initilaization file has two different binary specification ' \
                                  'for the same covariate.'
                            raise UserError(msg)
            list_ += [dict_['UNTREATED']['types'][index]]

        elif (i in dict_['UNTREATED']['order'] and i in dict_['COST']['order']):
            keys = ['UNTREATED', 'COST']
            for key_ in keys:
                index = dict_[key_]['order'].index(i)
                if isinstance(dict_[key_]['types'][index], list):
                    other_keys = [j for j in keys if j != key_]
                    for other in other_keys:
                        index_other = dict_[other]['order'].index(i)
                        if not isinstance(dict_[other]['types'][index_other], list):
                            dict_[other]['types'][index_other] = dict_[key_]['types'][index]
                        elif dict_[other]['types'][index_other] == dict_[key_]['types'][index]:
                            pass
                        else:
                            msg = 'Your initilaization file has two different binary specification ' \
                                  'for the same covariate.'
                            raise UserError(msg)
            list_ += [dict_['COST']['types'][index]]

        elif (i in dict_['TREATED']['order'] and i in dict_['COST']['order']):
            keys = ['TREATED', 'COST']
            for key_ in keys:
                index = dict_[key_]['order'].index(i)
                if isinstance(dict_[key_]['types'][index], list):
                    other_keys = [j for j in keys if j != key_]
                    for other in other_keys:
                        index_other = dict_[other]['order'].index(i)
                        if not isinstance(dict_[other]['types'][index_other], list):
                            dict_[other]['types'][index_other] = dict_[key_]['types'][index]
                        elif dict_[other]['types'][index_other] == dict_[key_]['types'][index]:
                            pass
                        else:
                            msg = 'Your initilaization file has two different binary specification ' \
                                  'for the same covariate.'
                            raise UserError(msg)
            list_ += [dict_['COST']['types'][index]]

        else:
            if i in dict_['TREATED']['order']:
                index = dict_['TREATED']['order'].index(i)
                list_ += [dict_['TREATED']['types'][index]]
            elif i in dict_['UNTREATED']['order']:
                index = dict_['UNTREATED']['order'].index(i)
                list_ += [dict_['UNTREATED']['types'][index]]
            else:
                index = dict_['COST']['order'].index(i)
                list_ += [dict_['COST']['types'][index]]
    dict_['AUX']['types'] = list_


    return dict_
