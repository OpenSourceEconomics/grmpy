""" This module provides auxiliary functions for the simulate.py module. It includes simulation
processes of the unobservable and endogenous variables of the model as well as functions regarding
the info file output.
"""
from scipy.stats import norm
import pandas as pd
import numpy as np


def simulate_covariates(init_dict):
    """The function simulates the covariates for the choice and the output functions."""
    # Distribute information
    num_agents = init_dict['SIMULATION']['agents']

    # Construct auxiliary information

    num_covars = len(init_dict['AUX']['types'])
    types = init_dict['AUX']['types']

    # As our baseline we simulate covariates from a standard normal distribution.
    means = np.tile(0.0, num_covars)
    covs = np.identity(num_covars)
    X = np.random.multivariate_normal(means, covs, num_agents)

    # We now perform some selective replacements.
    X[:, 0] = 1.0
    for i in list(range(num_covars)):
        if isinstance(types[i], list):
            if types[i][0] == 'binary':
                if i != 0:
                    frac = types[i][1]
                    binary = np.random.binomial(1, frac, size=num_agents)
                    X[:, i] = binary
                else:
                    pass
            elif types[i][0] == 'categorical':
                prob = types[i][2]
                cat = types[i][1]
                rand = np.random.choice(cat, size=num_agents, p=prob)
                X[:,i] = rand

    return X


def simulate_unobservables(init_dict):
    """The function simulates the unobservable error terms."""
    num_agents = init_dict['SIMULATION']['agents']
    cov = construct_covariance_matrix(init_dict)

    U = np.random.multivariate_normal(np.zeros(3), cov, num_agents)
    V = np.array(U[:, 2])

    return U, V


def simulate_outcomes(init_dict, X, U, V):
    """The function simulates the potential outcomes Y0 and Y1, the resulting treatment dummy D and
    the realized outcome Y.
    """
    X = pd.DataFrame(X)
    Z = X[[i - 1 for i in init_dict['CHOICE']['order']]].as_matrix()
    X_treated = X[[i - 1 for i in init_dict['TREATED']['order']]].as_matrix()
    X_untreated = X[[i - 1 for i in init_dict['UNTREATED']['order']]].as_matrix()
    # Distribute information
    coeffs_untreated = init_dict['UNTREATED']['all']
    coeffs_treated = init_dict['TREATED']['all']
    coeffs_choice = init_dict['CHOICE']['all']

    # Calculate potential outcomes and choice
    Y_1 = np.dot(coeffs_treated, X_treated.T) + U[:, 0]
    Y_0 = np.dot(coeffs_untreated, X_untreated.T) + U[:, 1]
    C = np.dot(coeffs_choice, Z.T) - V

    # Calculate expected benefit and the resulting treatment dummy
    D = np.array((C > 0).astype(int))

    # Observed outcomes
    Y = D * Y_1 + (1 - D) * Y_0

    return Y, D, Y_1, Y_0


def write_output(init_dict, Y, D, X, Y_1, Y_0, U, V):
    """The function converts the simulated variables to a panda data frame and saves the data in a
    txt and a pickle file.
    """
    # Distribute information
    source = init_dict['SIMULATION']['source']

    # Stack arrays
    data = np.column_stack((Y, D, X, Y_1, Y_0, U[:, 0], U[:, 1], V))

    # Construct list of column labels
    dep, indicator = init_dict['ESTIMATION']['dependent'], init_dict['ESTIMATION']['indicator']
    column = [dep, indicator]

    for i in list(range(X.shape[1])):
        str_ = 'X' + str(i)
        column.append(str_)
    column += [dep + '1', dep + '0', 'U1', 'U0', 'V']

    # Generate data frame, save it with pickle and create a txt file
    df = pd.DataFrame(data=data, columns=column)
    df[indicator] = df[indicator].apply(np.int64)
    df2=pd.DataFrame()
    for i in df.columns.values:
        if "X" in i:
            df2[init_dict['varnames'][int(i[1:5])]]=df[i]
        else:
            df2[i]=df[i]
    df2.to_pickle(source + '.grmpy.pkl')
    with open(source + '.grmpy.txt', 'w') as file_:
        df2.to_string(file_, index=False, na_rep='.', col_space=15, justify='left')
    return df2


def construct_all_coefficients(init_dict):
    """This function constructs all coefficients from the initialization dictionary."""
    coeffs_all = []
    for label in ['TREATED', 'UNTREATED', 'CHOICE', 'DIST']:
        coeffs_all += init_dict[label]['all'].tolist()

    return coeffs_all


def print_info(init_dict, data_frame):
    """The function writes an info file for the specific data frame."""
    # Distribute information
    coeffs_untreated = init_dict['UNTREATED']['all']
    coeffs_treated = init_dict['TREATED']['all']
    source = init_dict['SIMULATION']['source']
    dep, indicator = init_dict['ESTIMATION']['dependent'], init_dict['ESTIMATION']['indicator']

    # Construct auxiliary information
    coeffs_all = construct_all_coefficients(init_dict)
    cov = construct_covariance_matrix(init_dict)

    with open(source + '.grmpy.info', 'w') as file_:

        # First we note some basic information ab out the dataset.
        header = '\n\n Number of Observations \n\n'
        file_.write(header)

        info_ = [data_frame.shape[0], (data_frame[indicator] == 1).sum(), (data_frame[indicator] == 0).sum()]

        fmt = '  {:<10}' + ' {:>20}' * 1 + '\n\n'
        file_.write(fmt.format(*['', 'Count']))

        for i, label in enumerate(['All', 'Treated', 'Untreated']):
            str_ = '  {:<10} {:20}\n'
            file_.write(str_.format(*[label, info_[i]]))

        # Second, we describe the distribution of outcomes and effects.
        for label in ['Outcomes', 'Effects']:

            header = '\n\n Distribution of ' + label + '\n\n'
            file_.write(header)

            fmt = '  {:<10}' + ' {:>20}' * 5 + '\n\n'
            args = ['', 'Mean', 'Std-Dev.', '25%', '50%', '75%']
            file_.write(fmt.format(*args))

            for group in ['All', 'Treated', 'Untreated']:

                if label == 'Outcomes':
                    data = data_frame[dep]
                elif label == 'Effects':
                    data = data_frame[dep + '1'] - data_frame[dep + '0']
                else:
                    raise AssertionError

                if group == 'Treated':
                    data = data[data_frame[indicator] == 1]
                elif group == 'Untreated':
                    data = data[data_frame[indicator] == 0]
                else:
                    pass
                fmt = '  {:<10}' + ' {:>20.4f}' * 5 + '\n'
                info = list(data.describe().tolist()[i] for i in [1, 2, 4, 5, 6])
                if pd.isnull(info).all():
                    fmt = '  {:<10}' + ' {:>20}' * 5 + '\n'
                    info = ['---'] * 5
                elif pd.isnull(info[1]):
                    info[1] = '---'
                    fmt = '  {:<10}' ' {:>20.4f}' ' {:>20}' + ' {:>20.4f}' * 3 + '\n'

                file_.write(fmt.format(*[group] + info))

        # Implement the criteria function value , the MTE and parameterization
        header = '\n\n {} \n\n'.format('Criterion Function')
        file_.write(header)
        if 'criteria_value' in init_dict['AUX'].keys():
            str_ = '  {0:<10}      {1:<20.12f}\n\n'.format('Value',
                                                           init_dict['AUX']['criteria_value'])
        else:
            str_ = '  {0:>10} {1:>20}\n\n'.format('Value', '---')
        file_.write(str_)

        header = '\n\n {} \n\n'.format('MTE Information')
        file_.write(header)
        quantiles = [1] + np.arange(5, 100, 5).tolist() + [99]
        args = [str(i) + '%' for i in quantiles]
        quantiles = [i * 0.01 for i in quantiles]

        help_ = list(set(init_dict['TREATED']['order'] + init_dict['UNTREATED']['order']))
        x = data_frame[[init_dict['varnames'][i-1] for i in help_]]
        value = mte_information(coeffs_treated, coeffs_untreated, cov, quantiles, x, init_dict)
        str_ = '  {0:>10} {1:>20}\n\n'.format('Quantile', 'Value')
        file_.write(str_)
        len_ = len(value) - 1
        for i in range(len_):
            if isinstance(value[i], float):
                file_.write('  {0:>10} {1:>20.4f}\n'.format(str(args[i]), value[i]))
            else:
                file_.write('  {0:>10} {1:>20.4}\n'.format(str(args[i]), value[i]))

        # Write out parameterization of the model.
        file_.write('\n\n {} \n\n'.format('Parameterization'))
        file_.write('  {:>10} {:>20}\n\n'.format('Identifier', 'Value'))
        for i, coeff in enumerate(coeffs_all):
            file_.write('  {0:>10} {1:>20.4f}\n'.format(i, coeff))


def mte_information(coeffs_treated, coeffs_untreated, cov, quantiles, x, dict_):
    """The function calculates the marginal treatment effect for pre specified quantiles of the
    collected unobservable variables.
    """
    # Construct auxiliary information
    if dict_['TREATED']['order'] != dict_['UNTREATED']['order']:
        para_diff = []
        for i in set(dict_['TREATED']['order'] + dict_['UNTREATED']['order']):
            if i in dict_['TREATED']['order'] and i in dict_['UNTREATED']['order']:
                index_treated = dict_['TREATED']['order'].index(i)
                index_untreated = dict_['UNTREATED']['order'].index(i)
                diff = dict_['TREATED']['all'][index_treated] - dict_['UNTREATED']['all'][index_untreated]
            elif i in dict_['TREATED']['order'] and i not in dict_['UNTREATED']['order']:
                index = dict_['TREATED']['order'].index(i)
                diff = dict_['TREATED']['all'][index]

            elif i not in dict_['TREATED']['order'] and i in dict_['UNTREATED']['order']:
                index = dict_['UNTREATED']['order'].index(i)
                diff = - dict_['UNTREATED']['all'][index]
            para_diff += [diff]
    else:
        para_diff = coeffs_treated - coeffs_untreated
    MTE = []
    for i in quantiles:
        if cov[2, 2] == 0.00:
            MTE += ['---']
        else:
            MTE += [
                np.mean(np.dot(x, para_diff)) + (cov[2, 0] - cov[2, 1]) * norm.ppf(i)
            ]

    return MTE


def construct_covariance_matrix(init_dict):
    """This function constructs the covariance matrix based on the user's initialization file."""
    cov = np.zeros((3, 3))
    cov[np.triu_indices(3)] = init_dict['DIST']['all']
    cov[np.tril_indices(3, k=-1)] = cov[np.triu_indices(3, k=1)]
    cov[np.diag_indices(3)] **= 2
    return cov
