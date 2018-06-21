"""This module provides auxiliary functions for the import process of the init file."""
import numpy as np

from grmpy.check.custom_exceptions import UserError


def process(list_, dict_, keyword):
    """The function processes keyword parameters and creates dictionary elements."""
    if len(list_) == 5:
        name, order,  val, type_, frac_ = list_[0], list_[1], list_[2], list_[3], list_[4]
    elif len(list_) in [3,4]:
        name, order, val  = list_[0], list_[1], list_[2]
    else:
        name, val = list_[0], list_[1]

    if name not in dict_[keyword].keys() and name in ['coeff']:
        dict_[keyword][name] = []
    if keyword in ['TREATED', 'UNTREATED', 'CHOICE'] and 'types' not in dict_[keyword].keys():
        dict_[keyword]['types'] = []
    if keyword in ['TREATED', 'UNTREATED', 'CHOICE'] and 'order' not in dict_[keyword].keys():
        dict_[keyword]['order'] = []

    if keyword in ['TREATED', 'UNTREATED', 'CHOICE']:
        if order not in dict_['varnames']:
            dict_['varnames'] += [order]
        if len(list_) == 5:
            dict_[keyword]['types'] += [[type_, float(frac_)]]
            dict_[keyword]['order'] += [dict_['varnames'].index(order)+1]
        else:
            dict_[keyword]['order'] += [dict_['varnames'].index(order)+1]
            dict_[keyword]['types'] += ['nonbinary']

    # Type conversion
    if name in ['agents', 'seed', 'maxiter', 'disp']:
        val = int(val)
    elif name in ['source', 'file', 'optimizer', 'start', 'dependent', 'indicator']:
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

    for key_ in ['UNTREATED', 'TREATED', 'CHOICE', 'DIST']:
        if key_ in ['UNTREATED', 'TREATED', 'CHOICE']:
            dict_[key_]['all'] = dict_[key_]['coeff']
            dict_[key_]['all'] = np.array(dict_[key_]['all'])
        else:
            dict_[key_]['all'] = dict_[key_]['coeff']
            dict_[key_]['all'] = np.array(dict_[key_]['all'])

    # Ensure that the Estimation section contains information about the indicator and the dependent
    # variable labels
    if 'indicator' not in dict_['ESTIMATION'].keys():
        dict_['ESTIMATION']['indicator'] = 'D'
    if 'dependent' not in dict_['ESTIMATION'].keys():
        dict_['ESTIMATION']['dependent'] = 'Y'

    # Number of covariates
    num_covars_treated = len(dict_['TREATED']['all'])
    num_covars_untreated = len(dict_['UNTREATED']['all'])
    num_covars_cost = len(dict_['CHOICE']['all'])

    dict_['AUX']['num_covars_treated'] = num_covars_treated
    dict_['AUX']['num_covars_untreated'] = num_covars_untreated
    dict_['AUX']['num_covars_cost'] = num_covars_cost

    # Number of parameters
    dict_['AUX']['num_paras'] = num_covars_treated +  num_covars_untreated + num_covars_cost + 2 + 2

    # Starting values
    dict_['AUX']['init_values'] = []

    for key_ in ['TREATED', 'UNTREATED', 'CHOICE', 'DIST']:
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
    covars = set(dict_['TREATED']['order'] +  dict_['UNTREATED']['order'] +  dict_['CHOICE']['order'])
    for i in covars:
        if i in dict_['TREATED']['order'] and i in dict_['UNTREATED']['order'] and \
                        i in dict_['CHOICE']['order']:
            if i == 1:
                keys = ['TREATED', 'UNTREATED', 'CHOICE']
                for key_ in keys:
                    index = dict_[key_]['order'].index(i)
                    dict_[key_]['types'][index] = 'nonbinary'
            else:
                keys = ['TREATED', 'UNTREATED', 'CHOICE']
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
                                msg = 'Your initilaization file has two different binary ' \
                                      'specification for the same covariate.'
                                raise UserError(msg)
            list_ += [dict_['CHOICE']['types'][index]]

        elif i in dict_['TREATED']['order'] and i in dict_['UNTREATED']['order'] and\
                        i not in dict_['CHOICE']['order']:
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
                            msg = 'Your initilaization file has two different binary ' \
                                  'specification for the same covariate.'
                            raise UserError(msg)
            list_ += [dict_['UNTREATED']['types'][index]]

        elif i not in dict_['TREATED']['order'] and i in dict_['UNTREATED']['order'] and\
                        i in dict_['CHOICE']['order']:
            keys = ['UNTREATED', 'CHOICE']
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
                            msg = 'Your initilaization file has two different binary ' \
                                  'specification for the same covariate.'
                            raise UserError(msg)
            list_ += [dict_['CHOICE']['types'][index]]

        elif i in dict_['TREATED']['order'] and i not in dict_['UNTREATED']['order'] and\
                        i in dict_['CHOICE']['order']:
            keys = ['TREATED', 'CHOICE']
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
                            msg = 'Your initilaization file has two different binary ' \
                                  'specification for the same covariate.'
                            raise UserError(msg)
            list_ += [dict_['CHOICE']['types'][index]]

        else:
            if i in dict_['TREATED']['order']:
                index = dict_['TREATED']['order'].index(i)
                list_ += [dict_['TREATED']['types'][index]]
            elif i in dict_['UNTREATED']['order']:
                index = dict_['UNTREATED']['order'].index(i)
                list_ += [dict_['UNTREATED']['types'][index]]
            else:
                index = dict_['CHOICE']['order'].index(i)
                list_ += [dict_['CHOICE']['types'][index]]
    dict_['AUX']['types'] = list_


    return dict_


