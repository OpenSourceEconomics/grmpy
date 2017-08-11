

import numpy as np
import pandas as pd
import pickle


def simulation(init_dict):
    """Main function, defines variables by using the init_dict.
    It creates the endogeneous variables X and Z and relies
    on the _simulate_outcome and _simulate_unobservables functions
    to simulate the model. Finally it writes an output file by using
    the _write_output function."""

    # Distribute information
    num_agents = init_dict['SIMULATION']['agents']
    source = init_dict['SIMULATION']['source']

    Y1_coeffs = init_dict['TREATED']['all']
    Y0_coeffs = init_dict['UNTREATED']['all']
    C_coeffs = init_dict['COST']['all']
    coeffs = [Y0_coeffs, Y1_coeffs, C_coeffs]

    U0_sd, U1_sd, V_sd = init_dict['DIST']['all'][:3]
    vars_ = [U0_sd ** 2, U1_sd ** 2, V_sd ** 2]
    U01, U0_V, U1_V = init_dict['DIST']['all'][3:]
    covar_ = [U01**2, U0_V**2, U1_V**2]
    print(U01, U0_V, U1_V)

    num_covars_out = Y1_coeffs.shape[0]
    num_covars_cost = C_coeffs.shape[0]

    # Simulate observables

    means = np.tile(0.0, num_covars_out)
    covs = np.identity(num_covars_out)
    X = np.random.multivariate_normal(means, covs, num_agents)

    means = np.tile(0.0, num_covars_cost)
    covs = np.identity(num_covars_cost)
    Z = np.random.multivariate_normal(means, covs, num_agents)

    Z[:, 0], X[:, 0] = 1.0, 1.0

    # Simulate unobservables
    # Read information about the distribution and the specific means from the init dic

    U = _simulate_unobservables(covar_, vars_, num_agents)

    # Simulate endogeneous variables

    Y, D, Y_1, Y_0 = _simulate_outcomes([X, Z], U, coeffs)

    # Write output file
    df = _write_output([Y, D, Y_1, Y_0], [X, Z], source)

    return df, Y, Y_1, Y_0, D


def _simulate_unobservables(covar, vars_, num_agents):
    """Creates the error term values for each type of error term variable
    """
    # Create a Covariance matrix
    cov_ = np.diag(vars_)

    cov_[0, 1], cov_[1, 0] = covar[0], covar[0]
    cov_[0, 2], cov_[2, 0] = covar[1], covar[1]
    cov_[1, 2], cov_[2, 1] = covar[2], covar[2]
    print()

    # Option to integrate case specifications for different distributions

    U = np.random.multivariate_normal([0.0, 0.0, 0.0], cov_, num_agents)

    return U


def _simulate_outcomes(exog, err, coeff):
    """ Simulates the potential outcomes Y0 and Y1, the resulting
        treatment dummy D and the realized outcome Y """

    # Expected values for individuals

    exp_y0, exp_y1 = np.dot(coeff[0], exog[0].T), np.dot(coeff[1], exog[0].T)

    cost_exp = np.dot(coeff[2], exog[1].T)

    # Calculate expected benefit and the resulting treatment dummy

    expected_benefits = exp_y1 - exp_y0

    cost = cost_exp + err[0:, 2]

    D = np.array((expected_benefits - cost > 0).astype(int))

    # Realized outcome in both cases for each individual

    Y_0, Y_1 = exp_y0 + err[0:, 0], exp_y0 + err[0:, 1]

    # Observed outcomes

    Y = D * Y_1 + (1 - D) * Y_0

    return Y, D, Y_1, Y_0


def _write_output(end, exog, source, unobserved=False):
    '''Converts simulated data to a panda data frame
    and saves the data in an html file/pickle'''

    # Stack arrays
    data = np.column_stack((end[0], end[1], exog[0], exog[1],end[2], end[3]))

    # List of column names
    column = ['Y', 'D']

    for i in range(exog[0].shape[1]):
        str_ = 'X_' + str(i)
        column.append(str_)
    for i in range(exog[1].shape[1]):
        str_ = 'Z_' + str(i)
        column.append(str_)
    column.append('Y1')
    column.append('Y0')
    header={}
    for i in range(len(column)):
        header[str(i)] = column[i]
    print(header)
    # Generate data frame, save it with pickle and create a html file


    df = pd.DataFrame(data=data, columns=column)

    df.to_pickle(source + '.pkl')

    pd.DataFrame.to_html(df, source + '.html',
                         float_format='%.3f', header=column)
    df.to_stata('./' + source + '.dat')


    return df
