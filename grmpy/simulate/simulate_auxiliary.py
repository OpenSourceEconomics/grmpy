""" This module provides auxiliary functions for the simulate.py module. It includes simulation
processes of the unobservable and endogeneous variables of the model as well as functions regarding
the info file output.
"""
import pandas as pd
import numpy as np


def simulate_covariates(init_dict, cov_type, num_agents):
    """The function simulates the covariate variables for the cost and the output."""
    num_covar = init_dict[cov_type]['all'].shape[0]

    means = np.tile(0.0, num_covar)
    covs = np.identity(num_covar)
    X = np.random.multivariate_normal(means, covs, num_agents)
    X[:, 0] = 1.0
    for i in range(num_covar):
        if isinstance(init_dict[cov_type]['types'][i], list):
            if i != 0:
                frac = init_dict[cov_type]['types'][i][1]
                binary = np.random.binomial(1, frac, size=num_agents)
                X[:, i] = binary
    return X


def simulate_unobservables(covar, vars_, num_agents):
    """The function simulates the unobservable error terms."""
    # Create a Covariance matrix
    cov_ = np.diag(vars_)

    cov_[0, 1], cov_[1, 0] = covar[0], covar[0]
    cov_[0, 2], cov_[2, 0] = covar[1], covar[1]
    cov_[1, 2], cov_[2, 1] = covar[2], covar[2]

    assert np.all(np.linalg.eigvals(cov_) >= 0)
    U = np.random.multivariate_normal([0.0, 0.0, 0.0], cov_, num_agents)

    V = U[0:, 2] - U[0:, 1] + U[0:, 0]
    return U, V


def simulate_outcomes(exog, err, coeff):
    """The function simulates the potential outcomes Y0 and Y1, the resulting treatment dummy D and
    the realized outcome Y.
    """
    # Expected values for individuals
    exp_y0, exp_y1 = np.dot(coeff[0], exog[0].T), np.dot(coeff[1], exog[0].T)
    cost_exp = np.dot(coeff[2], exog[1].T)

    # Calculate expected benefit and the resulting treatment dummy
    expected_benefits = np.subtract(exp_y1, exp_y0)

    cost = np.add(cost_exp, err[0:, 2])

    D = np.array((expected_benefits - cost > 0).astype(int))

    # Realized outcome in both cases for each individual
    Y_0, Y_1 = np.add(exp_y0, err[0:, 0]), np.add(exp_y1, err[0:, 1])

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

        info = [data_frame.shape[0], (data_frame['D'] == 1).sum(), (data_frame['D'] == 0).sum()]

        fmt = '  {:<10}' + ' {:>20}' * 1 + '\n\n'
        file_.write(fmt.format(*['', 'Count']))

        for i, label in enumerate(['All', 'Treated', 'Untreated']):
            str_ = '  {:<10} {:20}\n'
            file_.write(str_.format(*[label, info[i]]))

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
                file_.write(fmt.format(* [group] + info))

        # Third we write out the parametrization of the model.
        header = '\n\n Parametrization \n\n'
        file_.write(header)
        str_ = '  {0:>10} {1:>20}\n\n'.format('Identifier', 'Value')
        file_.write(str_)

        value = np.append(np.append(coeffs[0], coeffs[1]), np.append(coeffs[2], coeffs[3]))
        len_ = len(value) - 1
        for i in range(len_):
            file_.write('  {0:>10} {1:>20.4f}\n'.format(str(i), value[i]))

