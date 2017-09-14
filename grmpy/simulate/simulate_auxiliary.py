""" This module provides auxiliary functions for the simulate.py module. It includes simulation
processes of the unobservable and endogenous variables of the model as well as functions regarding
the info file output.
"""
from scipy.stats import norm
import pandas as pd
import numpy as np


def simulate_covariates(init_dict, cov_type):
    """The function simulates the covariates for the cost and the output functions."""
    # Distribute information
    num_agents = init_dict['SIMULATION']['agents']

    # Construct auxiliary information
    num_covars = init_dict[cov_type]['all'].shape[0]

    # As our baseline we simulate covariates from a standard normal distribution.
    means = np.tile(0.0, num_covars)
    covs = np.identity(num_covars)
    X = np.random.multivariate_normal(means, covs, num_agents)

    # We now perform some selective replacements.
    X[:, 0] = 1.0
    for i in range(num_covars):
        if isinstance(init_dict[cov_type]['types'][i], list):
            if i != 0:
                frac = init_dict[cov_type]['types'][i][1]
                binary = np.random.binomial(1, frac, size=num_agents)
                X[:, i] = binary

    return X


def simulate_unobservables(init_dict):
    """The function simulates the unobservable error terms."""
    num_agents = init_dict['SIMULATION']['agents']
    cov = construct_covariance_matrix(init_dict)

    U = np.random.multivariate_normal(np.zeros(3), cov, num_agents)
    V = np.array(U[0:, 2])

    # Here we keep track of the implied value for U_C.
    U[:, 2] = V - U[:, 0] + U[:, 1]

    return U, V


def simulate_outcomes(init_dict, X, Z, U):
    """The function simulates the potential outcomes Y0 and Y1, the resulting treatment dummy D and
    the realized outcome Y.
    """
    # Distribute information
    coeffs_untreated = init_dict['UNTREATED']['all']
    coeffs_treated = init_dict['TREATED']['all']
    coeffs_cost = init_dict['COST']['all']

    # Calculate potential outcomes and costs
    Y_1 = np.dot(coeffs_treated, X.T) + U[:, 1]
    Y_0 = np.dot(coeffs_untreated, X.T) + U[:, 0]
    C = np.dot(coeffs_cost, Z.T) + U[:, 2]

    # Calculate expected benefit and the resulting treatment dummy
    D = np.array((Y_1 - Y_0 - C > 0).astype(int))

    # Observed outcomes
    Y = D * Y_1 + (1 - D) * Y_0

    return Y, D, Y_1, Y_0


def write_output(init_dict, Y, D, X, Z, Y_1, Y_0, U, V):
    """The function converts the simulated variables to a panda data frame and saves the data in a
    txt and a pickle file.
    """
    # Distribute information
    source = init_dict['SIMULATION']['source']

    # Stack arrays
    data = np.column_stack((Y, D, X, Z, Y_1, Y_0, U, V))

    # Construct list of column labels
    column = ['Y', 'D']
    for i in range(X.shape[1]):
        str_ = 'X_' + str(i)
        column.append(str_)
    for i in range(Z.shape[1]):
        str_ = 'Z_' + str(i)
        column.append(str_)
    column += ['Y1', 'Y0', 'U0', 'U1', 'UC', 'V']

    # Generate data frame, save it with pickle and create a txt file
    df = pd.DataFrame(data=data, columns=column)
    df['D'] = df['D'].apply(np.int64)
    df.to_pickle(source + '.grmpy.pkl')

    with open(source + '.grmpy.txt', 'w') as file_:
        df.to_string(file_, index=False, na_rep='.', col_space=15)

    return df


def construct_all_coefficients(init_dict):
    """This function constructs all coefficients from the initialization dictionary."""
    coeffs_all = []
    for label in ['TREATED', 'UNTREATED', 'COST', 'DIST']:
        coeffs_all += init_dict[label]['all'].tolist()

    return coeffs_all


def print_info(init_dict, data_frame):
    """The function writes an info file for the specific data frame."""
    # Distribute information
    coeffs_untreated = init_dict['TREATED']['all']
    coeffs_treated = init_dict['TREATED']['all']
    source = init_dict['SIMULATION']['source']

    # Construct auxiliary information
    coeffs_all = construct_all_coefficients(init_dict)
    cov = construct_covariance_matrix(init_dict)

    with open(source + '.grmpy.info', 'w') as file_:

        # First we note some basic information ab out the dataset.
        header = '\n\n Number of Observations \n\n'
        file_.write(header)

        info_ = [data_frame.shape[0], (data_frame['D'] == 1).sum(), (data_frame['D'] == 0).sum()]

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
                    object = data_frame['Y']
                elif label == 'Effects':
                    object = data_frame['Y1'] - data_frame['Y0']
                else:
                    raise AssertionError

                if group == 'Treated':
                    object = object[data_frame['D'] == 1]
                elif group == 'Untreated':
                    object = object[data_frame['D'] == 0]
                else:
                    pass
                fmt = '  {:<10}' + ' {:>20.4f}' * 5 + '\n'
                info = list(object.describe().tolist()[i] for i in [1, 2, 4, 5, 6])
                if pd.isnull(info).all():
                    fmt = '  {:<10}' + ' {:>20}' * 5 + '\n'
                    info = ['---'] * 5
                elif pd.isnull(info[1]):
                    info[1] = '---'
                    fmt = '  {:<10}' ' {:>20.4f}' ' {:>20}' + ' {:>20.4f}' * 3 + '\n'

                file_.write(fmt.format(*[group] + info))

        # Implement MTE and Parameterization
        header = '\n\n {} \n\n'.format('MTE Information')
        file_.write(header)
        quantiles = [1] + np.arange(5, 100, 5).tolist() + [99]
        args = [str(i) + '%' for i in quantiles]
        quantiles = [i * 0.01 for i in quantiles]
        x = data_frame.filter(regex=r'^X\_', axis=1)
        value = mte_information(coeffs_treated, coeffs_untreated, cov, quantiles, x)
        str_ = '  {0:>10} {1:>20}\n\n'.format('Quantile', 'Value')
        file_.write(str_)
        len_ = len(value) - 1
        for i in range(len_):
            file_.write('  {0:>10} {1:>20.4f}\n'.format(str(args[i]), value[i]))

        # Write out parameterization of the model.
        file_.write('\n\n {} \n\n'.format('Parameterization'))
        file_.write('  {:>10} {:>20}\n\n'.format('Identifier', 'Value'))
        for i, coeff in enumerate(coeffs_all):
            file_.write('  {0:>10} {1:>20.4f}\n'.format(i, coeff))


def mte_information(coeffs_treated, coeffs_untreated, cov, quantiles, x):
    """The function calculates the marginal treatment effect for pre specified quantiles of the
    collected unobservable variables.
    """
    # Construct auxiliary information
    para_diff = coeffs_treated - coeffs_untreated

    MTE = []
    for i in quantiles:
        MTE += [np.mean(np.dot(para_diff, x.T)) - (cov[1, 2] - cov[0, 2]) * norm.ppf(i)]

    return MTE


def construct_covariance_matrix(init_dict):
    """This function constructs the covariance matrix based on the user's initialization file"""
    cov = np.zeros((3, 3))
    cov[np.triu_indices(3)] = init_dict['DIST']['all']
    cov[np.tril_indices(3, k=-1)] = cov[np.triu_indices(3, k=1)]
    cov[np.diag_indices(3)] **= 2

    return cov
