"""The module provides a random dictionary generating process for test purposes."""
import uuid

from scipy.stats import wishart
import numpy as np

from grmpy.check.check import UserError


def constraints(probability=0.1, is_zero=True, agents=None, seed=None, sample=None,
                optimizer=None, start=None, maxiter=None, same_size=False, overlap=None,
                state_diff=None):
    """The constraints function returns an dictionary that provides specific characteristics for the
    random dictionary generating process.
    """
    constraints_dict = dict()
    constraints_dict['DETERMINISTIC'] = np.random.random_sample() < probability
    if not constraints_dict['DETERMINISTIC'] and is_zero:
        constraints_dict['IS_ZERO'] = np.random.random_sample() < probability / (1 - probability)
    else:
        constraints_dict['IS_ZERO'] = False
    if agents is None:
        constraints_dict['AGENTS'] = np.random.randint(1, 1000)
    else:
        constraints_dict['AGENTS'] = agents
    if seed is None:
        constraints_dict['SEED'] = np.random.randint(1, 10000)
    else:
        constraints_dict['SEED'] = seed
    if sample is None:
        if constraints_dict['AGENTS'] != 1:
            constraints_dict['SAMPLE_SIZE'] = np.random.randint(1, constraints_dict['AGENTS'])
        else:
            constraints_dict['SAMPLE_SIZE'] = 1
    else:
        constraints_dict['SAMPLE_SIZE'] = sample
    if optimizer is None:
        constraints_dict['OPTIMIZER'] = np.random.choice(a=['SCIPY-BFGS', 'SCIPY-POWELL'],
                                                         p=[0.5, 0.5])
    else:
        constraints_dict['OPTIMIZER'] = optimizer
    if start is None:
        constraints_dict['START'] = np.random.choice(a=['init', 'auto'])
    else:
        constraints_dict['START'] = start
    if maxiter is None:
        constraints_dict['MAXITER'] = np.random.randint(0, 10000)
    else:
        constraints_dict['MAXITER'] = maxiter

    constraints_dict['SAME_SIZE'] = same_size

    if overlap is None:
        constraints_dict['OVERLAP'] = np.random.random_sample() < 0.5
    else:
        constraints_dict['OVERLAP'] = overlap

    if state_diff is None:
        constraints_dict['STATE_DIFF'] = np.random.random_sample() < 0.5
    else:
        constraints_dict['STATE_DIFF'] = state_diff

    return constraints_dict


def generate_random_dict(constraints_dict=None):
    """The function generates a random initialization dictionary."""

    if constraints_dict is not None:
        if not isinstance(constraints_dict, dict):
            msg = '{} is not a dictionary.'.format(constraints_dict)
            raise UserError(msg)
        if len(constraints_dict.keys()) < 9:
            help_dict = constraints()
            for key_ in help_dict.keys():
                if key_ not in constraints_dict.keys():
                    constraints_dict[key_] = help_dict[key_]
    else:
        constraints_dict = constraints()

    is_deterministic = constraints_dict['DETERMINISTIC']

    state_diff = constraints_dict['STATE_DIFF']

    optimizer = constraints_dict['OPTIMIZER']

    same_size = constraints_dict['SAME_SIZE']

    is_zero = constraints_dict['IS_ZERO']

    overlap = constraints_dict['OVERLAP']

    maxiter = constraints_dict['MAXITER']

    agents = constraints_dict['AGENTS']

    start = constraints_dict['START']

    seed = constraints_dict['SEED']

    source = my_random_string(8)

    dict_ = {}
    treated_num = np.random.randint(1, 10)
    if state_diff is True:
        untreated_num = np.random.randint(1, 10)
    else:
        pass
    cost_num = np.random.randint(1, 10)
    # Coefficients
    for key_ in ['UNTREATED', 'TREATED', 'COST']:

        dict_[key_] = {}

        if key_ in ['UNTREATED', 'TREATED']:
            if state_diff is True:
                if key_ == 'TREATED':
                    x = treated_num
                else:
                    x= untreated_num
                dict_[key_]['all'], dict_[key_]['types'] = generate_coeff(x, key_,
                                                                              is_zero, state_diff)

            else:
                dict_[key_]['all'], dict_[key_]['types'] = generate_coeff(treated_num, key_,
                                                                              is_zero)
        else:
            dict_[key_]['all'], dict_[key_]['types'] = generate_coeff(cost_num, key_, is_zero)

    if state_diff is False:
        dict_ = overlap_treat_cost(dict_, treated_num, cost_num, overlap)
    else:
        dict_ = overlap_treat_untreat(dict_, treated_num, untreated_num)
        dict_ = overlap_treat_untreat_cost(dict_, cost_num, overlap)
    dict_ = types(dict_)
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
    if same_size is True:
        dict_['ESTIMATION']['agents'] = agents
    else:
        dict_['ESTIMATION']['agents'] = np.random.randint(1, 1000)
    dict_['ESTIMATION']['file'] = source + '.grmpy.txt'
    dict_['ESTIMATION']['optimizer'] = optimizer
    dict_['ESTIMATION']['start'] = start
    dict_['ESTIMATION']['maxiter'] = maxiter
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
        b = wishart.rvs(df=10, scale=np.identity(3), size=1)
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
    labels = ['SIMULATION', 'ESTIMATION', 'TREATED', 'UNTREATED', 'COST', 'DIST', 'SCIPY-BFGS',
              'SCIPY-POWELL']
    write_nonbinary = np.random.random_sample() < 0.5


    with open(file_name + '.grmpy.ini', 'w') as file_:

        for label in labels:

            file_.write('   {}'.format(label) + '\n\n')

            if label in ['SIMULATION', 'ESTIMATION', 'SCIPY-BFGS', 'SCIPY-POWELL']:
                if label == 'SIMULATION':
                    structure = ['seed', 'agents', 'source']
                elif label == 'ESTIMATION':
                    structure = ['file', 'start', 'agents', 'optimizer', 'maxiter']
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
                        str_ = '        {0:<10} {1:>35}\n'
                        file_.write(str_.format(key_, dict_[label][key_]))


            elif label in ['TREATED', 'UNTREATED', 'COST', 'DIST']:
                for i, _ in enumerate(dict_[label]['all']):
                    if 'order' in dict_[label].keys():
                        if 'types' in dict_[label].keys():
                            if isinstance(dict_[label]['types'][i], list):
                                str_ = '        {0:<10} {1:>14} {2:>20.4f} {3:>10} {4:>5.4f}\n'
                                file_.write(
                                    str_.format(
                                        'coeff', dict_[label]['order'][i], dict_[label]['all'][i],
                                        dict_[label]['types'][i][0],dict_[label]['types'][i][1])
                                )
                            else:
                                if write_nonbinary:
                                    str_ = '        {0:<10} {1:>14} {2:>20.4f} {3:>17}\n'
                                    file_.write(str_.format('coeff',dict_[label]['order'][i],
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
    random = str(uuid.uuid4()).upper().replace("-", "")
    return random[0:string_length]


def generate_coeff(num, key_, is_zero, state_diff=False):
    """The function generates random coefficients for creating the random init dictionary."""
    keys = ['UNTREATED', 'COST', 'TREATED']
    if not is_zero:
        list_ = np.random.normal(0., 2., [num]).tolist()
        if key_ in keys:
            binary_list = ['nonbinary'] * num
    else:
        list_ = np.array([0] * num).tolist()

    if key_ in keys:
        binary_list = ['nonbinary'] * num
    else:
        binary_list = []


    return list_, binary_list

def types(dict_):
    all = []
    for key_ in ['TREATED', 'UNTREATED', 'COST']:
        all += dict_[key_]['order']
    all = [k for k in all if k != 1]
    for i in list(set(all)):
        if np.random.random_sample() < 0.1:
            frac = np.random.uniform(0, 1)
            for section in ['TREATED', 'UNTREATED', 'COST']:
                if i in dict_[section]['order']:
                    index = dict_[section]['order'].index(i)
                    dict_[section]['types'][index] = ['binary', frac]
        else:
            pass

    return dict_


def overlap_treat_cost(dict_, treated_num, cost_num, overlap):
    if overlap is True:
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
    dict_['COST']['order'] = cost_ord

    return dict_

def overlap_treat_untreat(dict_, treated_num, untreated_num):
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
    num_var = len(set(dict_['TREATED']['order'] + dict_['UNTREATED']['order']))
    if overlap is True:
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

    dict_['COST']['order'] = cost_ord

    return dict_
