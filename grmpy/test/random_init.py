"""The module provides a random dictionary generating process for test purposes."""
import uuid

from scipy.stats import wishart
import numpy as np

from grmpy.check.check import UserError


def generate_random_dict(constr=None):
    """The function generates a random initialization dictionary."""

    if constr is not None:
        if not isinstance(constr, dict):
            msg = '{} is not a dictionary.'.format(constr)
            raise UserError(msg)
    else:
        constr = dict()

    if 'DETERMINISTIC' in constr.keys():
        is_deterministic = constr['DETERMINISTIC']
    else:
        is_deterministic = np.random.random_sample() < 0.1

    if 'STATE_DIFF' in constr.keys():
        state_diff = constr['STATE_DIFF']
    else:
        state_diff = np.random.random_sample() < 0.5

    if 'OPTIMIZER' in constr.keys():
        optimizer = constr['OPTIMIZER']
    else:
        optimizer = np.random.choice(a=['SCIPY-BFGS', 'SCIPY-POWELL'], p=[0.5, 0.5])

    if 'SAME_SIZE' in constr.keys():
        same_size = constr['SAME_SIZE']
    else:
        same_size = False

    if 'IS_ZERO' in constr.keys() and not is_deterministic:
        is_zero = np.random.random_sample() < 0.1 / (1 - 0.1)
    else:
        is_zero = False

    if 'OVERLAP' in constr.keys():
        overlap = constr['OVERLAP']
    else:
        overlap = np.random.random_sample() < 0.5

    if 'MAXITER' in constr.keys():
        maxiter = constr['MAXITER']
    else:
        maxiter = np.random.randint(0, 10000)

    if 'AGENTS' in constr.keys():
        agents = constr['AGENTS']

    else:
        agents = np.random.randint(1, 1000)

    if 'START' in constr.keys():
        start = constr['START']
    else:
        start = np.random.choice(a=['init', 'auto'])
    if 'SEED' in constr.keys():
        seed = constr['SEED']
    else:
        seed = np.random.randint(1, 10000)
    if 'CATEGORICAL' in constr.keys():
        categorical = constr['CATEGORICAL']
    else:
        categorical = np.random.choice([True, False])

    source = my_random_string(8)

    dict_ = {}
    treated_num = np.random.randint(1, 10)
    if state_diff:
        untreated_num = np.random.randint(1, 10)
    else:
        pass
    cost_num = np.random.randint(1, 10)
    # Coefficients
    for key_ in ['UNTREATED', 'TREATED', 'CHOICE']:

        dict_[key_] = {}

        if key_ in ['UNTREATED', 'TREATED']:
            if state_diff:
                if key_ == 'TREATED':
                    x = treated_num
                else:
                    x = untreated_num
                dict_[key_]['all'], dict_[key_]['types'] = generate_coeff(x, key_, is_zero)

            else:
                dict_[key_]['all'], dict_[key_]['types'] = generate_coeff(treated_num, key_,
                                                                          is_zero)
        else:
            dict_[key_]['all'], dict_[key_]['types'] = generate_coeff(cost_num, key_, is_zero)

    if not state_diff:
        dict_ = overlap_treat_cost(dict_, treated_num, cost_num, overlap)
    else:
        dict_ = overlap_treat_untreat(dict_, treated_num, untreated_num)
        dict_ = overlap_treat_untreat_cost(dict_, cost_num, overlap)
    dict_ = types(dict_, categorical)
    # Simulation parameters
    dict_['SIMULATION'] = {}
    for key_ in ['agents', 'source', 'seed']:
        if key_ == 'seed':
            dict_['SIMULATION'][key_] = seed
        elif key_ == 'agents':
            dict_['SIMULATION'][key_] = agents
        else:
            dict_['SIMULATION'][key_] = source
    # Estimation parameters
    dict_['ESTIMATION'] = {}
    if same_size:
        dict_['ESTIMATION']['agents'] = agents
    else:
        dict_['ESTIMATION']['agents'] = np.random.randint(1, 1000)
    dict_['ESTIMATION']['file'] = source + '.grmpy.txt'
    dict_['ESTIMATION']['optimizer'] = optimizer
    dict_['ESTIMATION']['start'] = start
    dict_['ESTIMATION']['maxiter'] = maxiter
    dict_['ESTIMATION']['dependent'] = 'Y'
    dict_['ESTIMATION']['indicator'] = 'D'
    dict_['ESTIMATION']['output_file'] = 'est.grmpy.info'
    dict_['ESTIMATION']['comparison'] = '1'

    for key_ in ['SCIPY-BFGS', 'SCIPY-POWELL']:
        dict_[key_] = {}
        if key_ == 'SCIPY-BFGS':
            dict_[key_]['gtol'] = np.random.uniform(1.5e-05, 0.8e-05)
            dict_[key_]['eps'] = np.random.uniform(1.4901161193847655e-08, 1.4901161193847657e-08)
        else:
            dict_[key_]['xtol'] = np.random.uniform(0.00009, 0.00011)
            dict_[key_]['ftol'] = np.random.uniform(0.00009, 0.00011)

    # Variance and covariance parameters
    dict_['DIST'] = {}
    if not is_deterministic:
        scale_matrix = np.identity(3) * 0.1
        b = wishart.rvs(df=10, scale=scale_matrix, size=1)
    else:
        b = np.zeros((3, 3))
    dict_['DIST']['all'] = []
    dict_['DIST']['all'].append(b[0, 0] ** 0.5)
    dict_['DIST']['all'].append(b[0, 1])
    dict_['DIST']['all'].append(b[0, 2])
    dict_['DIST']['all'].append(b[1, 1] ** 0.5)
    dict_['DIST']['all'].append(b[1, 2])
    dict_['DIST']['all'].append(b[2, 2] ** 0.5)
    print_dict(dict_)
    return dict_


def print_dict(dict_, file_name='test'):
    """The function creates an init file from a given dictionary."""
    labels = ['SIMULATION', 'ESTIMATION', 'TREATED', 'UNTREATED', 'CHOICE', 'DIST', 'SCIPY-BFGS',
              'SCIPY-POWELL']
    write_nonbinary = np.random.random_sample() < 0.5

    with open(file_name + '.grmpy.ini', 'w') as file_:

        for label in labels:
            file_.write('   {}'.format(label) + '\n\n')

            if label in ['SIMULATION', 'ESTIMATION', 'SCIPY-BFGS', 'SCIPY-POWELL']:
                if label == 'SIMULATION':
                    structure = ['seed', 'agents', 'source']
                elif label == 'ESTIMATION':
                    structure = ['file', 'start', 'agents', 'optimizer', 'maxiter', 'dependent',
                                 'indicator', 'output_file', 'comparison']
                elif label == 'SCIPY-BFGS':
                    structure = ['gtol', 'eps']
                else:
                    structure = ['xtol', 'ftol']
                for key_ in structure:
                    if key_ in ['source', 'file', 'norm', 'optimizer', 'start']:
                        str_ = '        {0:<25} {1:>20}\n'
                        file_.write(str_.format(key_, dict_[label][key_]))
                    elif key_ in ['gtol', 'xtol', 'ftol', 'norm', 'eps']:
                        str_ = '        {0:<13} {1:>32}\n'
                        file_.write(str_.format(key_, dict_[label][key_]))
                    else:
                        if key_ in ['indicator', 'dependent']:
                            if key_ not in dict_['ESTIMATION'].keys():
                                continue
                            else:
                                str_ = '        {0:<10} {1:>35}\n'
                                file_.write(str_.format(key_, dict_[label][key_]))
                        else:
                            str_ = '        {0:<10} {1:>35}\n'
                            file_.write(str_.format(key_, dict_[label][key_]))

            elif label in ['TREATED', 'UNTREATED', 'CHOICE', 'DIST']:
                for i, _ in enumerate(dict_[label]['all']):
                    if 'order' in dict_[label].keys():
                        if 'types' in dict_[label].keys():
                            if isinstance(dict_[label]['types'][i], list):
                                if dict_[label]['types'][i][0] == 'binary':
                                    str_ = '        {0:<10} {1:>14} {2:>20.4f} {3:>14} {4:>5.4f}\n'
                                    file_.write(
                                        str_.format(
                                            'coeff', dict_[label]['order'][i],
                                            dict_[label]['all'][i], dict_[label]['types'][i][0],
                                            dict_[label]['types'][i][1])
                                    )
                                elif dict_[label]['types'][i][0] == 'categorical':
                                    str_ = '        {0:<10} {1:>14} {2:>20.4f} {3:>19} '
                                    for j in [1, 2]:
                                        str_ += ' ('
                                        for counter, k in enumerate(dict_[label]['types'][i][j]):
                                            if counter < len(dict_[label]['types'][i][j]) - 1:
                                                str_ += '{:>1}'.format(str(k)) + ','
                                            else:
                                                str_ += '{}'.format(str(k)) + ')'
                                    str_ += '\n'
                                    file_.write(
                                        str_.format(
                                            'coeff', dict_[label]['order'][i],
                                            dict_[label]['all'][i], dict_[label]['types'][i][0])
                                    )

                            else:
                                if write_nonbinary:
                                    str_ = '        {0:<10} {1:>14} {2:>20.4f} {3:>17}\n'
                                    file_.write(str_.format('coeff', dict_[label]['order'][i],
                                                            dict_[label]['all'][i],
                                                            dict_[label]['types'][i]))
                                else:
                                    str_ = '        {0:<10} {1:>14} {2:>20.4f}\n'
                                    file_.write(str_.format('coeff', dict_[label]['order'][i],
                                                            dict_[label]['all'][i]))

                    else:
                        str_ = '        {0:<10} {1:>35.4f}\n'
                        file_.write(str_.format('coeff', dict_[label]['all'][i]))

            file_.write('\n')


def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()).upper().replace('-', '')
    return random[0:string_length]


def generate_coeff(num, key_, is_zero):
    """The function generates random coefficients for creating the random init dictionary."""
    keys = ['UNTREATED', 'CHOICE', 'TREATED']
    if not is_zero:
        list_ = np.random.normal(0., 2., [num]).tolist()
    else:
        list_ = np.array([0] * num).tolist()

    if key_ in keys:
        binary_list = ['nonbinary'] * num
    else:
        binary_list = []

    return list_, binary_list


def types(dict_, categorical):
    """This function determines if a specified covariate is a binary or a non-binary variable.
    Additionally it """
    all_ = []
    for key_ in ['TREATED', 'UNTREATED', 'CHOICE']:
        all_ += dict_[key_]['order']
    all_ = [k for k in all_ if k != 1]
    for i in list(set(all_)):
        if np.random.random_sample() < 0.2:
            if np.random.random_sample() < 0.5:
                frac = np.random.uniform(0, 0.8)
                for section in ['TREATED', 'UNTREATED', 'CHOICE']:
                    if i in dict_[section]['order']:
                        index = dict_[section]['order'].index(i)
                        dict_[section]['types'][index] = ['binary', frac]
            else:
                if categorical:

                    num = np.random.choice([3, 4, 5, 6, 7], size=1)
                    cat = list(range(1, int(num) + 1))
                    prob = prob_weights(num)
                    for section in ['TREATED', 'UNTREATED', 'CHOICE']:
                        if i in dict_[section]['order']:
                            index = dict_[section]['order'].index(i)
                            dict_[section]['types'][index] = ['categorical', cat, prob]
                else:
                    pass
        else:
            pass

    return dict_


def overlap_treat_cost(dict_, treated_num, cost_num, overlap):
    """This function determines the variables that affect the output when selecting into treatment
    and the costs.
    """
    if overlap:
        treated_ord = list(range(1, treated_num + 1))
        x = list(range(2, treated_num + 1))
        cost_ord = []
        y = 1
        for i in list(range(cost_num)):
            if i == 0:
                cost_ord += [1]
            else:
                if np.random.random_sample() < 0.2:
                    if len(x) == 0:
                        cost_ord += [treated_ord[treated_num - 1] + y]
                        y += 1
                    else:
                        a = np.random.choice(x)
                        cost_ord += [int(a)]
                        x = [j for j in x if j != a]
                else:
                    cost_ord += [treated_ord[treated_num - 1] + y]
                    y += 1
    else:
        treated_ord = list(range(1, treated_num + 1))
        cost_ord = list(range(treated_num + 1, treated_num + cost_num))
        cost_ord = [1] + cost_ord

    dict_['TREATED']['order'] = treated_ord
    dict_['UNTREATED']['order'] = treated_ord
    dict_['CHOICE']['order'] = cost_ord

    return dict_


def overlap_treat_untreat(dict_, treated_num, untreated_num):
    """This function determines the variables that affect the output independent of the decision
    of an individual.
    """
    treated_ord = list(range(1, treated_num + 1))
    x = list(range(2, treated_num + 1))
    untreated_ord = []
    y = 1
    for i in list(range(untreated_num)):
        if i == 0:
            untreated_ord += [1]
        else:
            if np.random.random_sample() < 0.3:
                if len(x) == 0:
                    untreated_ord += [treated_ord[treated_num - 1] + y]
                    y += 1
                else:
                    a = np.random.choice(x)
                    untreated_ord += [int(a)]
                    x = [j for j in x if j != a]
            else:
                untreated_ord += [treated_ord[treated_num - 1] + y]
                y += 1
    dict_['TREATED']['order'] = treated_ord
    dict_['UNTREATED']['order'] = untreated_ord

    return dict_


def overlap_treat_untreat_cost(dict_, cost_num, overlap):
    """This function determines the variables that affect the output of both treatment states as
    well as the costs.
    """
    num_var = len(set(dict_['TREATED']['order'] + dict_['UNTREATED']['order']))
    if overlap:
        treated_ord = list(range(1, num_var + 1))
        x = list(range(2, num_var + 1))
        cost_ord = []
        y = 1
        for i in list(range(cost_num)):
            if i == 0:
                cost_ord += [1]
            else:
                if np.random.random_sample() < .2:
                    if len(x) == 0:
                        cost_ord += [treated_ord[num_var - 1] + y]
                        y += 1
                    else:
                        a = np.random.choice(x)
                        cost_ord += [int(a)]
                        x = [j for j in x if j != a]
                else:
                    cost_ord += [treated_ord[num_var - 1] + y]
                    y += 1
    else:
        cost_ord = list(range(num_var + 1, num_var + cost_num))
        cost_ord = [1] + cost_ord

    dict_['CHOICE']['order'] = cost_ord

    return dict_


def prob_weights(n):
    """This function creates the probabilities for categorical variables given the number of
    different categories"""
    x = 0.95
    weights = []
    for i in range(int(n)):
        if i == 0:
            prob = np.random.choice(np.arange(0.1, 0.5, 0.05))
        else:
            if i == n - 1:
                prob = 1 - sum(weights)
            else:
                if x / 2 == 0.05:
                    prob = 0.05
                else:
                    prob = np.random.choice(np.arange(0.05, x / 2, 0.05))

        x = round(x - prob, 4)
        weights += [prob]
    return list(np.around(weights, 2))
