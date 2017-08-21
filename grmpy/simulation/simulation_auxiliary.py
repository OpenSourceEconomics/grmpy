import pandas as pd
import numpy as np


def simulate_unobservables(covar, vars_, num_agents):
    """Creates the error term values for each type of error term variable
    """
    # Create a Covariance matrix
    cov_ = np.diag(vars_)

    cov_[0, 1], cov_[1, 0] = covar[0], covar[0]
    cov_[0, 2], cov_[2, 0] = covar[1], covar[1]
    cov_[1, 2], cov_[2, 1] = covar[2], covar[2]

    # Option to integrate case specifications for different distributions

    U = np.random.multivariate_normal([0.0, 0.0, 0.0], cov_, num_agents)

    V = U[0:, 2] - U[0:, 1] + U[0:, 0]
    return U, V


def simulate_outcomes(exog, err, coeff):
    """ Simulates the potential outcomes Y0 and Y1, the resulting
        treatment dummy D and the realized outcome Y """

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


def write_output(end, exog, err, source, is_deterministic):
    """Converts simulated data to a panda data frame
    and saves the data in an html file/pickle"""
    column = ['Y', 'D']

    # Stack arrays
    if not is_deterministic:
        data = np.column_stack(
            (end[0], end[1], exog[0], exog[1], end[2], end[3]))
        data = np.column_stack(
            (data, err[0][0:, 0], err[0][0:, 1], err[0][0:, 2], err[1]))
        # List of column names

        for i in range(exog[0].shape[1]):
            str_ = 'X_' + str(i)
            column.append(str_)
        for i in range(exog[1].shape[1]):
            str_ = 'Z_' + str(i)
            column.append(str_)
    else:
        data = np.column_stack(
            (end[0], end[1], end[2], end[3], err[0][0:, 0], err[0][0:, 1], err[0][0:, 2], err[1]))
    column = column + ['Y1', 'Y0', 'U0', 'U1', 'UC', 'V']
    # Generate data frame, save it with pickle and create a html file

    df = pd.DataFrame(data=data, columns=column)
    df['D'] = df['D'].apply(np.int64)
    df.to_pickle(source + '.grmpy.pkl')

    with open(source + '.grmpy.txt', 'w') as file_:
        df.to_string(file_, index=False, header=True, na_rep='.', col_space=15)

    return df


def collect_information(data_frame):
    """Calculates the required information for the info file"""

    # Number of individuals:
    Indiv = len(data_frame)

    # Counts by treatment status
    treated_num = data_frame[data_frame.D == 1].count()['D']
    untreated_num = Indiv - treated_num

    no_treatment, all_treatment = _adjust_collecting(
        treated_num, untreated_num)
    # Average Treatment Effect
    ATE = np.mean(data_frame.Y1 - data_frame.Y0)
    Mean = np.mean(data_frame.Y)

    TT, TUT, Mean_treat, SD_treat, Quant_treat, Mean_untreat, SD_untreat, Quant_untreat = \
        _calc_parameters(data_frame, no_treatment, all_treatment)

    data = {
        'Number of Agents': Indiv, 'Treated Agents': treated_num,
        'Untreated Agents': untreated_num, 'Average Treatment Effect': ATE,
        'Treatment on Treated': TT, 'Treatment on Untreated': TUT,
        'Mean': Mean, 'Mean Treated': Mean_treat, 'Mean Untreated': Mean_untreat,
        'Std Treated': SD_treat, 'Std Untreated': SD_untreat, 'Quantiles Treated': Quant_treat,
        'Quantiles Untreated': Quant_untreat
    }

    return data


def print_info(data_frame, coeffs, file_name):
    data_ = collect_information(data_frame)
    no_treatment, all_treatment = _adjust_collecting(
        data_['Treated Agents'], data_['Untreated Agents'])
    '''Prints an info file for the specififc dataset'''
    labels = [
        'Simulation',
        'Additional Information',
        'Effects',
        'Model Paramerization'
    ]

    with open(file_name + '.grmpy.info', 'w') as file_:

        for label in labels:
            file_.write('\n' + label + '\n\n')

            if label == 'Simulation':
                structure = [
                    'Number of Agents',
                    'Treated Agents',
                    'Untreated Agents'
                ]

                for s in structure:
                    str_ = '{0:<25} {1:20}\n'

                    file_.write(str_.format(s + ':', data_[s]))

            elif label == 'Additional Information':
                if no_treatment:
                    _print_notreated(data_, file_)
                elif all_treatment:
                    _print_alltreated(data_, file_)
                else:
                    _print_normal(data_, file_)

            elif label == 'Effects':
                structure = [
                    'Average Treatment Effect',
                    'Treatment on Treated',
                    'Treatment on Untreated'
                ]

                for s in structure:
                    if s == 'Treatment on Treated':
                        if no_treatment:
                            str_ = '{0:<25} {1:>20}\n'
                            file_.write(str_.format(s + ':', data_[s]))
                        else:
                            str_ = '{0:<25} {1:20.4f}\n'
                            file_.write(str_.format(s + ':', data_[s]))
                    elif s == 'Treatment on Untreated':
                        if all_treatment:
                            str_ = '{0:<25} {1:>20}\n'
                            file_.write(str_.format(s + ':', data_[s]))
                        else:
                            str_ = '{0:<25} {1:20.4f}\n'
                            file_.write(str_.format(s + ':', data_[s]))
                    else:
                        str_ = '{0:<25} {1:20.4f}\n'
                        file_.write(str_.format(s + ':', data_[s]))

            else:
                structure = [
                    'TREATED', 'UNTREATED', 'COST', 'DIST'
                ]

                for s in structure:
                    file_.write('\n' + s + '\n\n')

                    if s == 'TREATED':
                        str_ = '{0:<25}'
                        str_coeff = ''
                        for i in range(len(coeffs[1])):
                            str_coeff = str_coeff + \
                                ' {0:10.4f}'.format(coeffs[1][i])
                        file_.write(str_.format('coeff') + str_coeff + '\n')
                    elif s == 'UNTREATED':
                        str_ = '{0:<25}'
                        str_coeff = ''
                        for i in range(len(coeffs[0])):
                            str_coeff = str_coeff + \
                                ' {0:10.4f}'.format(coeffs[0][i])
                        file_.write(str_.format('coeff') + str_coeff + '\n')
                    elif s == 'COST':
                        str_ = '{0:<25}'
                        str_coeff = ''
                        for i in range(len(coeffs[2])):
                            str_coeff = str_coeff + \
                                ' {0:10.4f}'.format(coeffs[2][i])
                        file_.write(str_.format('coeff') + str_coeff + '\n')
                    else:
                        str_ = '{0:<25}'
                        str_coeff = ''
                        for i in range(len(coeffs[3])):
                            str_coeff = str_coeff + \
                                ' {0:10.4f}'.format(coeffs[3][i])
                        file_.write(str_.format('coeff') + str_coeff + '\n')


def _print_notreated(data_, file_name):
    str_ = '{0:<25} {1:20.4f}\n'
    file_name.write(str_.format('Mean' + ':', data_['Mean']))

    labels = ['Treated', 'Untreated']
    for label in labels:
        str_ = '{0:<25}\n\n'
        file_name.write(str_.format(label))

        str_ = '{0:<20} {1:>20} {2:>20} {3:>20} {4:>20}\n\n'.format(
            'Mean', 'Std-Dev.', '2.Decile', '5.Decile', '8.Decile')
        file_name.write(str_)

        structure = [
            'Mean ',
            'Std ',
            'Quantiles ',
        ]
        str_ = ''
        for s in structure:
            s = s + label
            if s == 'Quantiles Untreated':
                q02 = data_[s].loc[0.2]
                q05 = data_[s].loc[0.5]
                q08 = data_[s].loc[0.8]
                str_ = str_ + '{0:20.4f} {1:20.4f} {2:20.4f}'.format(
                    q02, q05, q08)
            elif s == 'Quantiles Treated':
                q02 = data_[s]
                q05 = data_[s]
                q08 = data_[s]
                str_ = str_ + '{0:>20} {1:>20} {2:>20}'.format(q02, q05, q08)
            else:
                if isinstance(data_[s], str):
                    str_ = str_ + '{0:>20}'.format(data_[s])
                else:
                    str_ = str_ + '{0:20.4f}'.format(data_[s])
        file_name.write(str_ + '\n\n')


def _print_alltreated(data_, file_name):
    str_ = '{0:<25} {1:20.4f}\n'
    file_name.write(str_.format('Mean' + ':', data_['Mean']))

    labels = ['Treated', 'Untreated']
    for label in labels:
        str_ = '{0:<25}\n\n'
        file_name.write(str_.format(label))

        str_ = '{0:>20} {1:>20} {2:>20} {3:>20} {4:>20}\n\n'.format(
            'Mean', 'Std-Dev.', '2.Decile', '5.Decile', '8.Decile')
        file_name.write(str_)

        structure = [
            'Mean ',
            'Std ',
            'Quantiles ',
        ]
        str_ = ''
        for s in structure:
            s = s + label
            if s == 'Quantiles Treated':
                q02 = data_[s].loc[0.2]
                q05 = data_[s].loc[0.5]
                q08 = data_[s].loc[0.8]
                str_ = str_ + '{0:20.4f} {1:20.4f} {2:20.4f}'.format(
                    q02, q05, q08)
            elif s == 'Quantiles Untreated':
                q02 = data_[s]
                q05 = data_[s]
                q08 = data_[s]
                str_ = str_ + '{0:>20} {1:>20} {2:>20}'.format(q02, q05, q08)
            else:
                if isinstance(data_[s], str):
                    str_ = str_ + '{0:>20}'.format(data_[s])
                else:
                    str_ = str_ + '{0:20.4f}'.format(data_[s])
        file_name.write(str_ + '\n\n')


def _print_normal(data_, file_name):
    str_ = '{0:<25} {1:20.4f}\n\n'
    file_name.write(str_.format('Mean' + ':', data_['Mean']))

    labels = ['Treated', 'Untreated']
    for label in labels:
        str_ = '{0:<25}\n\n'
        file_name.write(str_.format(label))

        str_ = '{0:>20} {1:>20} {2:>20} {3:>20} {4:>20}\n\n'.format(
            'Mean', 'Std-Dev.', '2.Decile', '5.Decile', '8.Decile')
        file_name.write(str_)

        structure = [
            'Mean ',
            'Std ',
            'Quantiles ',
        ]
        str_ = ''

        for s in structure:
            s = s + label
            if s.startswith('Quantiles'):
                q02 = data_[s].loc[0.2]
                q05 = data_[s].loc[0.5]
                q08 = data_[s].loc[0.8]
                str_ = str_ + '{0:20.4f} {1:20.4f} {2:20.4f}'.format(
                    q02, q05, q08
                )
            else:
                str_ = str_ + '{0:20.4f}'.format(data_[s])
        file_name.write(str_ + '\n\n')


def _calc_parameters(data_frame, no_treatment, all_treatment):
    if not no_treatment and not all_treatment:

        # Treatment on the Treated
        TT = np.mean(data_frame.Y1[data_frame.D == 1]) - \
            np.mean(data_frame.Y0[data_frame.D == 1])
        # Treatment on Untreated
        TUT = np.mean(data_frame.Y1[data_frame.D == 0]) - \
            np.mean(data_frame.Y0[data_frame.D == 0])
        # Average observed wage overall and by treatment status
        Mean_treat = np.mean(data_frame.Y[data_frame.D == 1])
        SD_treat = np.std(data_frame.Y[data_frame.D == 1])
        Quant_treat = data_frame.Y[data_frame.D == 1].quantile([0.2, 0.5, 0.8])
        Mean_untreat = np.mean(data_frame.Y[data_frame.D == 0])
        Quant_untreat = data_frame.Y[data_frame.D == 0].quantile([
            0.2, 0.5, 0.8])
        SD_untreat = np.std(data_frame.Y[data_frame.D == 0])
        # Print out model parameterization

    elif no_treatment:
        TT = "---"
        TUT = np.mean(data_frame.Y1[data_frame.D == 0]) - \
            np.mean(data_frame.Y0[data_frame.D == 0])
        Mean_treat = "---"
        SD_treat = "---"
        Quant_treat = "---"
        Mean_untreat = np.mean(data_frame.Y[data_frame.D == 0])
        Quant_untreat = data_frame.Y[data_frame.D == 0].quantile([
                                                                 0.2, 0.5, 0.8])
        SD_untreat = np.std(data_frame.Y[data_frame.D == 0])

    elif all_treatment:
        TT = np.mean(data_frame.Y1[data_frame.D == 1]) - \
            np.mean(data_frame.Y0[data_frame.D == 1])
        TUT = "---"
        Mean_untreat = "---"
        Quant_untreat = "---"
        SD_untreat = "---"
        Mean_treat = np.mean(data_frame.Y[data_frame.D == 1])
        SD_treat = np.std(data_frame.Y[data_frame.D == 1])
        Quant_treat = data_frame.Y[data_frame.D == 1].quantile([0.2, 0.5, 0.8])

    else:
        print('Error,')
    return TT, TUT, Mean_treat, SD_treat, Quant_treat, Mean_untreat, SD_untreat, Quant_untreat


def _adjust_collecting(treated_num, untreated_num):
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
