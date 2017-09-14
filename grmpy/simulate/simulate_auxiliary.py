""" This module provides auxiliary functions for the simulate.py module. It includes simulation
processes of the unobservable and endogenous variables of the model as well as functions regarding
the info file output.
"""
from scipy.stats import norm
import pandas as pd
import numpy as np


def simulate_covariates(init_dict, cov_type, num_agents):
    """The function simulates the covariates for the cost and the output functions."""
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


def simulate_unobservables(cov, num_agents):
    """The function simulates the unobservable error terms."""
    U = np.random.multivariate_normal(np.zeros(3), cov, num_agents)
    V = np.array(U[0:, 2])

    # Here we keep track of the implied value for U_C.
    U[0:, 2] = V - U[0:, 0] + U[0:, 1]

    return U, V


def simulate_outcomes(exog, err, coeff):
    """The function simulates the potential outcomes Y0 and Y1, the resulting treatment dummy D and
    the realized outcome Y.
    """
    # individual outcomes
    Y_0, Y_1 = np.add(
        np.dot(coeff[0], exog[0].T), err[0:, 0]), np.add(np.dot(coeff[1], exog[0].T), err[0:, 1])
    cost = np.add(np.dot(coeff[2], exog[1].T), err[0:, 2])

    # Calculate expected benefit and the resulting treatment dummy
    benefits = np.subtract(Y_1, Y_0)
    D = np.array((benefits - cost > 0).astype(int))

    # Observed outcomes
    Y = D * Y_1.T + (1 - D) * Y_0.T

    return Y, D, Y_1, Y_0


def write_output(end, exog, err, source):
    """The function converts the simulated variables to a panda data frame and saves the data in a
    txt and a pickle file.
    """
    column = ['Y', 'D']

    # Stack arrays
    data = np.column_stack((end[0], end[1], exog[0], exog[1], end[2], end[3]))
    data = np.column_stack((data, err[0][0:, 0], err[0][0:, 1], err[0][0:, 2], err[1]))

    # List of column names
    for i in range(exog[0].shape[1]):
        str_ = 'X_' + str(i)
        column.append(str_)
    for i in range(exog[1].shape[1]):
        str_ = 'Z_' + str(i)
        column.append(str_)
    column += ['Y1', 'Y0', 'U0', 'U1', 'UC', 'V']

    # Generate data frame, save it with pickle and create a txt file
    df = pd.DataFrame(data=data, columns=column)
    df['D'] = df['D'].apply(np.int64)
    df.to_pickle(source + '.grmpy.pkl')

    with open(source + '.grmpy.txt', 'w') as file_:
        df.to_string(file_, index=False, header=True, na_rep='.', col_space=15)

    return df


def print_info(data_frame, coeffs, file_name):
    """The function writes an info file for the specific data frame."""
    with open(file_name + '.grmpy.info', 'w') as file_:

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
        for label in ['MTE Information', 'Parametrization']:
            header = '\n\n {} \n\n'.format(label)
            file_.write(header)
            if label == 'MTE Information':
                quantiles = [1] + np.arange(5, 100, 5).tolist() + [99]
                args = [str(i) + '%' for i in quantiles]
                quantiles = [i * 0.01 for i in quantiles]
                x = data_frame.filter(regex=r'^X\_', axis=1)
                value = mte_information(coeffs[:2], coeffs[3][:3], quantiles, x)
                str_ = '  {0:>10} {1:>20}\n\n'.format('Quantile', 'Value')

            else:
                value = np.append(np.append(coeffs[0], coeffs[1]), np.append(coeffs[2], coeffs[3]))
                str_ = '  {0:>10} {1:>20}\n\n'.format('Identifier', 'Value')
                args = list(range(len(value) - 1))
            file_.write(str_)
            len_ = len(value) - 1
            for i in range(len_):
                file_.write('  {0:>10} {1:>20.4f}\n'.format(str(args[i]), value[i]))


def mte_information(para, cov, quantiles, x):
    """The function calculates the marginal treatment effect for pre specified quantiles of the
    collected unobservable variables.
    """
    MTE = []
    para_diff = para[1] - para[0]
    for i in quantiles:
        MTE += [np.mean(np.dot(para_diff, x.T)) - (cov[2] - cov[1]) * norm.ppf(i)]

    return MTE


def construct_covariance_matrix(init_dict):
    """This function constructs the covariance matrix based on the user's initialization file"""
    cov = np.zeros((3, 3))
    cov[np.triu_indices(3)] = init_dict['DIST']['all']
    cov[np.tril_indices(3, k=-1)] = cov[np.triu_indices(3, k=1)]
    cov[np.diag_indices(3)] **= 2

    return cov
