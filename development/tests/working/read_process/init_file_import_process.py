"""This module contains the draft for the adjusted init file import process."""
import yaml

import numpy as np


def read(file):
    """This function processes the initialization file so that it can be used for simulation as well
     as estimation purposes.
     """
    # Load the initialization file
    with open(file) as y:
        init_dict = yaml.load(y)

    # Process the initialization file
    attr_dict = init_dict_to_attr_dict(init_dict)

    return attr_dict


def init_dict_to_attr_dict(init):
    """This function processes the imported initialization file so that it fulfills the
    requirements for the following simulation and estimation process.
    """
    attr = {}
    for key in ['TREATED', 'UNTREATED', 'CHOICE']:
        attr[key] = {'all': np.array(init[key]['params']),
                     'order': []}

    attr['DIST'] = {'all': np.array(init['DIST']['params'])}
    attr['DETERMINISTIC'] = (attr['DIST']['all'] == 0).all()

    for key in ['ESTIMATION', 'SCIPY-BFGS', 'SCIPY-POWELL', 'SIMULATION']:
        attr[key] = init[key]

    varnames = []
    for key in ['TREATED', 'UNTREATED', 'CHOICE']:
        attr[key]['types'] = []
        for name in init[key]['order']:
            if name not in varnames:
                varnames.append(name)
            attr[key]['types'] += [init['VARTYPES'][name]]

    attr['varnames'] = varnames

    for key in ['TREATED', 'UNTREATED', 'CHOICE']:
        for name in init[key]['order']:
            index = attr['varnames'].index(name)
            attr[key]['order'] += [index + 1]

    attr['AUX'] = {'init_values'}

    init_values = []
    for key in ['TREATED', 'UNTREATED', 'CHOICE']:
        init_values += init[key]['params']
    init_values += [init['DIST']['params'][0]] + init['DIST']['params'][2:5]

    # Generate the AUX section that include some additional auxiliary information
    attr['AUX'] = {'init_values': init_values,
                   'num_covars_choice': len(attr['CHOICE']['all']),
                   'num_covars_treated': len(attr['TREATED']['all']),
                   'num_covars_untreated': len(attr['UNTREATED']['all']),
                   'num_paras': len(init_values) + 1}

    attr['AUX']['types'] = []
    for name in attr['varnames']:
        attr['AUX']['types'] += [init['VARTYPES'][name]]

    return attr
