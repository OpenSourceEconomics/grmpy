"""This module provides auxiliary functions for the import process of the init file."""
import numpy as np


def init_dict_to_attr_dict(init):
    """This function processes the imported initialization file so that it fulfills the requirements
     for the following simulation and estimation process.
    """
    attr = {}
    for key in ['TREATED', 'UNTREATED', 'CHOICE']:
        attr[key] = {'all': np.array(init[key]['params']),
                     'order': [], 'types': [] }

    attr['DIST'] = {'all': np.array(init['DIST']['params'])}
    attr['DETERMINISTIC'] = (attr['DIST']['all'] == 0).all()

    for key in ['ESTIMATION', 'SCIPY-BFGS', 'SCIPY-POWELL', 'SIMULATION']:
        attr[key] = init[key]

    vartypes, varnames = [], []

    for key in ['TREATED', 'UNTREATED', 'CHOICE']:
        for name in init[key]['order']:
            if name not in varnames:
                varnames.append(name)
                vartypes.append(init['VARTYPES'][name])

    attr['varnames'] = varnames

    for key in ['TREATED', 'UNTREATED', 'CHOICE']:
        for name in init[key]['order']:
            attr[key]['types'] += [init['VARTYPES'][name]]
            index = attr['varnames'].index(name)
            attr[key]['order'] += [index + 1]

    attr['AUX'] = {'init_values'}

    init_values = []
    for key in ['TREATED', 'UNTREATED', 'CHOICE', 'DIST']:
        init_values += list(init[key]['params'])

    # Generate the AUX section that include some additional auxiliary information
    attr['AUX'] = {'init_values': np.array(init_values),
                   'num_covars_choice': len(attr['CHOICE']['all']),
                   'num_covars_treated': len(attr['TREATED']['all']),
                   'num_covars_untreated': len(attr['UNTREATED']['all']),
                   'num_paras': len(init_values) + 1}

    attr['AUX']['types'] = []
    attr['AUX']['types'] += vartypes

    return attr
