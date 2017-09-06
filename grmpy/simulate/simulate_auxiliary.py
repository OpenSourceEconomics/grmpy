""" This module provides auxiliary functions for the simulate.py module. It includes simulation pro-
cesses of the unobservable and endogeneous variables of the model as well as functions regarding the
info file output.
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
            else:
                pass
    return X


def simulate_unobservables(covar, vars_, num_agents):
    """The function simulates the unobservable error terms."""
    # Create a Covariance matrix
    cov_ = np.diag(vars_)

    cov_[0, 1], cov_[1, 0] = covar[0], covar[0]
    cov_[0, 2], cov_[2, 0] = covar[1], covar[1]
    cov_[1, 2], cov_[2, 1] = covar[2], covar[2]

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


def collect_information(data_frame):
    """The function collects the required information for the info file."""
    # Number of individuals:
    indiv = data_frame.shape[0]

    # Counts by treatment status
    treated_num = data_frame[data_frame.D == 1].shape[0]
    untreated_num = indiv - treated_num

    no_treatment, all_treatment = _adjust_collecting(treated_num, untreated_num)
    # Average Treatment Effect

    ate, tt, tut, mean_over, sd_over, quant_over, mean_treat, sd_treat, quant_treat, mean_untreat, \
    sd_untreat, quant_untreat = _calc_parameters(data_frame, no_treatment, all_treatment)

    data = {
        'Number of Agents': indiv, 'Treated Agents': treated_num, 'Untreated Agents': untreated_num,
        'Mean Untreated': mean_untreat, 'Mean Treated': mean_treat, 'Mean Overall': mean_over,
        'Quantiles Untreated': quant_untreat, 'Quantiles Treated': quant_treat, 'Quantiles Overall':
            quant_over, 'Std Untreated': sd_untreat, 'Std Treated': sd_treat,
        'Std Overall': sd_over,
        'ate': ate, 'tut': tut, 'tt': tt,
    }

    return data


def print_info(data_frame, coeffs, file_name):
    """The function writes an info file for the specififc data frame."""
    data_ = collect_information(data_frame)
    labels = ['Simulation', 'Additional Information', 'Effects', 'Model Parameterization']

    with open(file_name + '.grmpy.info', 'w') as file_:

        for label in labels:
            file_.write('\n' + label + '\n\n')

            if label == 'Simulation':
                structure = ['Number of Agents', 'Treated Agents', 'Untreated Agents']

                for s in structure:
                    str_ = '{0:<20} {1:20}\n'

                    file_.write(str_.format(s + ':', data_[s]))

            elif label == 'Additional Information':
                _print_dist(data_, file_)

            elif label == 'Effects':
                _print_effects(data_, file_)

            else:
                str_ = '{0:>10} {1:>18}\n\n'.format('Identifier', 'Value')
                file_.write(str_)

                value = np.append(np.append(coeffs[0], coeffs[1]), np.append(coeffs[2], coeffs[3]))
                len_ = len(value) - 1
                for i in range(len_):
                    file_.write('{0:>10} {1:>20.4f}\n'.format(str(i), value[i]))


def _print_effects(data_, file_name):
    """The function writes the effect information to the init file."""
    structure = ['ate', 'tt', 'tut']
    sub_structure = ['', '_0.1', '_0.25', '_0.5', '_0.75', '_0.9']
    str_ = '{0:>10} {1:>18} {2:>18} {3:>18} {4:>18} {5:>18} {6:>18}\n\n'.format(
        'Effect', 'Overall', '0.1', '0.25', '0.5', '0.75', '0.9')
    file_name.write(str_)

    for label in structure:
        str_ = '{0:>10}'.format(label.upper())
        for s in sub_structure:
            entry = label + s
            if isinstance(data_[label], str):
                str_ += '{0:>19}'.format(data_[label])
            else:
                str_ += '{0:>19.4f}'.format(data_[label][entry])
        file_name.write(str_ + '\n\n')


def _print_dist(data_, file_name):
    """The function writes the distributional information to the init file."""
    str_ = '{0:>10} {1:>20} {2:>20} {3:>20} {4:>20} {5:>20}\n\n'.format \
        ('Case', 'Mean', 'Std-Dev.', '2.Decile', '5.Decile', '8.Decile')
    file_name.write(str_)

    labels = ['Overall', 'Treated', 'Untreated']
    for label in labels:
        str_ = '{0:>10}'.format(label)

        structure = ['Mean ', 'Std ', 'Quantiles ']

        for s in structure:
            s += label
            if s.startswith('Quantiles'):
                if isinstance(data_[s], str):
                    q02 = data_[s]
                    q05 = data_[s]
                    q08 = data_[s]
                    str_ += '{0:>20} {1:>20} {2:>20}'.format(q02, q05, q08)
                else:
                    q02 = data_[s].loc[0.2]
                    q05 = data_[s].loc[0.5]
                    q08 = data_[s].loc[0.8]
                    str_ += '{0:>20.4f} {1:>20.4f} {2:>20.4f}'.format(q02, q05, q08)
            else:
                if isinstance(data_[s], float):
                    str_ += '{0:>21.4f}'.format(data_[s])
                else:
                    str_ += '{0:>20}'.format(data_[s])

        file_name.write(str_ + '\n\n')


def _calc_parameters(data_frame, no_treatment, all_treatment):
    """The function calculates the distributional information and effects for the info file. The
    calculation depends on the specific case that occurs (no treated agents, no untreated agents or
    treated and untreated agents).
    """
    out_quant = data_frame.Y.quantile([0.1, 0.25, 0.5, 0.75, 0.9])

    # Average Treatment effect
    ate_ = np.mean(data_frame.Y1 - data_frame.Y0)
    ate = {'ate': ate_}
    for i in [0.1, 0.25, 0.5, 0.75, 0.9]:
        ate['ate_' + str(i)] = calculate_ate(data_frame, out_quant, i)

    # Treatment on the Treated
    if no_treatment:
        tt = '---'
        mean_treat = "---"
        sd_treat = "---"
        quant_treat = "---"
    else:
        tt_ = np.mean(data_frame.Y1[data_frame.D == 1]) - np.mean(data_frame.Y0[data_frame.D == 1])
        tt = {'tt': tt_}
        for i in [0.1, 0.25, 0.5, 0.75, 0.9]:
            tt['tt_' + str(i)] = calculate_tt(data_frame, out_quant, i)
        mean_treat = np.mean(data_frame.Y[data_frame.D == 1])
        sd_treat = np.std(data_frame.Y[data_frame.D == 1])
        quant_treat = data_frame.Y[data_frame.D == 1].quantile([0.2, 0.5, 0.8])

    # Treatment on Untreated
    if all_treatment:
        tut = "---"
        mean_untreat = "---"
        quant_untreat = "---"
        sd_untreat = "---"
    else:
        tut_ = np.mean(data_frame.Y1[data_frame.D == 0]) - np.mean(data_frame.Y0[data_frame.D == 0])
        tut = {'tut': tut_}
        for i in [0.1, 0.25, 0.5, 0.75, 0.9]:
            tut['tut_' + str(i)] = calculate_tt(data_frame, out_quant, i)
        mean_untreat = np.mean(data_frame.Y[data_frame.D == 0])
        quant_untreat = data_frame.Y[data_frame.D == 0].quantile([
            0.2, 0.5, 0.8])
        sd_untreat = np.std(data_frame.Y[data_frame.D == 0])

    # Average observed wage overall and by treatment status
    mean_over = np.mean(data_frame.Y)
    sd_over = np.std(data_frame.Y)
    quant_over = data_frame.Y.quantile([0.2, 0.5, 0.8])

    return ate, tt, tut, mean_over, sd_over, quant_over, mean_treat, sd_treat, quant_treat, \
           mean_untreat, sd_untreat, quant_untreat


def _adjust_collecting(treated_num, untreated_num):
    """The function determines if there are only treated, only untreated or treated and untreated
    individuals.
    """
    if treated_num == 0:
        no_treatment = True
    else:
        no_treatment = False

    if untreated_num == 0:
        all_treatment = True
    else:
        all_treatment = False

    assert no_treatment != all_treatment or no_treatment == all_treatment is False

    return no_treatment, all_treatment


def calculate_ate(df, quant, q):
    """The function calculates the average treatment effect for a given quantile."""
    ate = np.mean(df.Y1[df.Y <= quant.loc[q]] - df.Y0[df.Y <= quant.loc[q]])
    return ate


def calculate_tt(df, quant, q):
    """The function calculates the treatment on treated effect for a given quantile."""
    tt = np.mean(df.Y1[(df.D == 1) & (df.Y <= quant.loc[q])]) \
         - np.mean(df.Y0[(df.D == 1) & (df.Y <= quant.loc[q])])
    return tt


def calculate_tut(df, quant, q):
    """The function calculates the treatment on untreated effect for a given quantile."""
    tut = np.mean(df.Y1[(df.D == 0) & (df.Y <= quant.loc[q])]) \
          - np.mean(df.Y0[(df.D == 0) & (df.Y <= quant.loc[q])])
    return tut
