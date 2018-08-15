import pandas as pd
import numpy as np
import math


from grmpy.simulate.simulate_auxiliary import construct_covariance_matrix
from grmpy.simulate.simulate_auxiliary import simulate_unobservables
from grmpy.simulate.simulate_auxiliary import simulate_covariates
from grmpy.simulate.simulate_auxiliary import simulate_outcomes
from grmpy.simulate.simulate_auxiliary import mte_information


def print_logfile(init_dict, rslt):
    """The function writes the log file for the estimation process."""
    # Adjust output
    auxiliary = process_se_log(rslt, init_dict)
    with open('est.grmpy.info', 'w') as file_:

        for label in ['Optimization Information', 'Criterion Function', 'Economic Parameters']:
            header = '\n \n  {:<10}\n\n'.format(label)
            file_.write(header)
            if label == 'Optimization Information':
                for section in ['Optimizer', 'Start values', 'Success', 'Status',
                                'Number of Evaluations',
                                'Criterion', 'Message', 'Warning']:
                    fmt = '  {:<10}' + ' {:<20}'
                    if section == 'Number of Evaluations':
                        if len(str(rslt['nfev'])) == 4:
                            fmt += '  {:>21}\n'
                        else:
                            fmt += '  {:>20}\n'
                        file_.write(fmt.format('', section + ':', rslt['nfev']))
                    elif section == 'Start values':
                        fmt += '  {:>23}\n'
                        file_.write(fmt.format('', section + ':',
                                               init_dict['ESTIMATION']['start']))
                    elif section == 'Optimizer':
                        if init_dict['ESTIMATION']['optimizer'] == 'SCIPY-POWELL':
                            fmt += '  {:>31}\n'
                        else:
                            fmt += '  {:>29}\n'
                        file_.write(fmt.format('', section + ':',
                                               init_dict['ESTIMATION']['optimizer']))
                    elif section == 'Criterion':
                        fmt += '       {:>20.4f}\n'
                        file_.write(fmt.format('', section + ':', rslt['crit']))
                    elif section in ['Message', 'Warning']:
                        fmt += '                     {:>20}\n'
                        file_.write(fmt.format('', section + ':', rslt[section.lower()]) + '\n')
                        if section == 'Warning':
                            if 'warning' in init_dict['ESTIMATION'].keys():
                                file_.write(fmt.format('', '', init_dict['ESTIMATION']['warning']))
                    else:
                        fmt += '  {:>20}\n'
                        file_.write(fmt.format('', section + ':', rslt[section.lower()]))
            elif label == 'Criterion Function':
                fmt = '  {:<10}' * 2 + ' {:>20}' * 2 + '\n\n'
                file_.write(fmt.format('', '', 'Start', 'Finish'))
                file_.write('\n' + fmt.format('', '', init_dict['AUX']['criteria'], rslt['crit']))

            else:
                file_.write(fmt.format(*['', 'Identifier', 'Start', 'Finish']) + '\n\n')
                fmt = '  {:>10}' * 2 + ' {:>20.4f}' * 2 + '{:>10}' + '{:>10.4f}'* 2
                for i in range(len(rslt['AUX']['x_internal'])):
                    file_.write('{0}\n'.format(
                        fmt.format('', str(i), init_dict['AUX']['starting_values'][i],
                                   rslt['AUX']['x_internal'][i], auxiliary[i],
                                   rslt['AUX']['confidence_intervals'][i][0],
                                   rslt['AUX']['confidence_intervals'][i][1])))


def write_comparison(init_dict, df1, rslt):
    """The function writes the info file including the descriptives of the original and the
    estimated sample.
    """
    indicator = init_dict['ESTIMATION']['indicator']
    dep = init_dict['ESTIMATION']['dependent']
    df3, df2 = simulate_estimation(init_dict, rslt, True)
    with open('comparison.grmpy.txt', 'w') as file_:
        # First we note some basic information ab out the dataset.
        header = '\n\n Number of Observations \n\n'
        file_.write(header)
        info_ = []
        for i, label in enumerate([df1, df2, df3]):
            info_ += [[label.shape[0], (label[indicator] == 1).sum(),
                       (label[indicator] == 0).sum()]]

        fmt = '    {:<25}' + ' {:>20}' * 3 + '\n\n\n'
        file_.write(fmt.format(*['Sample', 'Observed', 'Simulated (finish)',
                                 'Simulated (start)']))

        for i, label in enumerate(['All', 'Treated', 'Untreated']):
            str_ = '    {:<25}' + ' {:>20}' * 3 + '\n'
            file_.write(str_.format(label, info_[0][i], info_[1][i], info_[2][i]))

        header = '\n\n Distribution of Outcomes\n\n'
        file_.write(header)
        for group in ['All', 'Treated', 'Untreated']:
            header = '\n\n ' '  {:<10}'.format(group) + '\n\n'
            file_.write(header)
            fmt = '    {:<25}' + ' {:>20}' * 5 + '\n\n'
            args = ['', 'Mean', 'Std-Dev.', '25%', '50%', '75%']
            file_.write(fmt.format(*args))

            for sample in ['Observed Sample', 'Simulated Sample (finish)',
                           'Simulated Sample (start)']:

                if sample == 'Observed Sample':
                    data_frame = df1
                elif sample == 'Simulated Sample (finish)':
                    data_frame = df2
                else:
                    data_frame = df3

                data = data_frame[dep]

                if group == 'Treated':
                    data = data[data_frame[indicator] == 1]
                elif group == 'Untreated':
                    data = data[data_frame[indicator] == 0]
                else:
                    pass
                fmt = '    {:<25}' + ' {:>20.4f}' * 5 + '\n'
                info = list(data.describe().tolist()[i] for i in [1, 2, 4, 5, 6])
                if pd.isnull(info).all():
                    fmt = '    {:<10}' + ' {:>20}' * 5 + '\n'
                    info = ['---'] * 5
                elif pd.isnull(info[1]):
                    info[1] = '---'
                    fmt = '    {:<25}' ' {:>20.4f}' ' {:>20}' + ' {:>20.4f}' * 3 + '\n'

                file_.write(fmt.format(*[sample] + info))

        header = '\n\n {} \n\n'.format('MTE Information')
        file_.write(header)
        value, args = calculate_mte(rslt, df1)
        str_ = '  {0:>10} {1:>20}\n\n'.format('Quantile', 'Value')
        file_.write(str_)
        len_ = len(value)
        for i in range(len_):
            if isinstance(value[i], float):
                file_.write('  {0:>10} {1:>20.4f}\n'.format(str(args[i]), value[i]))


def write_output_estimation(labels, Y, D, X, Y_1, Y_0, init_dict):
    """The function converts the simulated variables to a panda data frame."""
    indicator = init_dict['ESTIMATION']['indicator']
    dep = init_dict['ESTIMATION']['dependent']
    # Stack arrays
    data = np.column_stack((Y, D, X, Y_1, Y_0))

    # Construct list of column labels
    column = [dep, indicator] + labels

    column += [dep + '1', dep + '0']

    # Generate data frame
    df = pd.DataFrame(data=data, columns=column)
    df[indicator] = df[indicator].apply(np.int64)
    return df


def simulate_estimation(init_dict, rslt, start=False):
    """The function simulates a new sample based on the estimated coefficients."""

    # Distribute information
    seed = init_dict['SIMULATION']['seed']
    labels = init_dict['varnames']
    # Determine parametrization and read in /simulate observables
    if start:
        start_dict, rslt_dict = process_results(init_dict, rslt, start)
        dicts = [start_dict, rslt_dict]
    else:
        rslt_dict = process_results(init_dict, rslt, start)
        dicts = [rslt_dict]
    data_frames = []
    for dict_ in dicts:

        # Set seed value
        np.random.seed(seed)
        # Simulate unobservables
        U, V = simulate_unobservables(dict_)
        X = simulate_covariates(rslt_dict)

        # Simulate endogeneous variables
        Y, D, Y_1, Y_0 = simulate_outcomes(dict_, X, U, V)

        df = write_output_estimation(labels, Y, D, X, Y_1, Y_0, init_dict)
        data_frames += [df]

    if start:
        return data_frames[0], data_frames[1]
    else:
        return data_frames[0]



def process_results(init_dict, rslt, start=False):
    """The function processes the results dictionary for the following simulation."""
    rslt_dict = dict(rslt)
    start_dict = dict(rslt)
    num_treated = init_dict['AUX']['num_covars_treated']
    num_untreated = num_treated + init_dict['AUX']['num_covars_untreated']

    dicts = [rslt_dict, start_dict]
    for dict_ in dicts:
        dict_['SIMULATION'] = {}
        dict_['SIMULATION'] = dict(init_dict['SIMULATION'])
        if dict_ == rslt_dict:
                dict_ = transform_rslt_DIST(rslt['AUX']['x_internal'], dict_)
        else:
            if start:
                dict_['TREATED']['all'] = init_dict['AUX']['starting_values'][:num_treated]
                dict_['UNTREATED']['all'] = \
                    init_dict['AUX']['starting_values'][num_treated:num_untreated]
                dict_['CHOICE']['all'] = \
                    init_dict['AUX']['starting_values'][num_untreated:-4]
                dict_ = transform_rslt_DIST(init_dict['AUX']['starting_values'], dict_)
                return start_dict, rslt_dict
            else:
                return rslt_dict


def transform_rslt_DIST(rslt, dict_):
    """The function converts the correlation parameters from the estimation outcome to
    covariances for the simulation of the estimation sample.
    """
    dict_['DIST'] = {}
    aux = rslt[-4:]
    cov01 = 0.0
    cov0V = aux[1] * aux[0]
    cov1V = aux[3] * aux[2]

    dict_['DIST']['all'] = [aux[0], cov01, cov0V, aux[2], cov1V, 1.0]

    for i, element in enumerate(dict_['DIST']['all']):
        dict_['DIST']['all'][i] = round(element, 4)

    return dict_


def calculate_mte(rslt, init_dict, data_frame, quant=None):
    coeffs_treated = rslt['TREATED']['all']
    coeffs_untreated = rslt['UNTREATED']['all']

    if quant is None:
        quantiles = [1] + np.arange(2.5, 100, 2.5).tolist() + [99]
        args = [str(i) + '%' for i in quantiles]
        quantiles = [i * 0.01 for i in quantiles]
    else:
        quantiles = quant

    cov = np.zeros((3,3))
    cov[2, 0] = rslt['AUX']['x_internal'][-3] * rslt['AUX']['x_internal'][-4]
    cov[2, 1] = rslt['AUX']['x_internal'][-1] * rslt['AUX']['x_internal'][-2]
    cov[2,2] = 1.0
    print(cov)
    help_ = list(set(init_dict['TREATED']['order'] + init_dict['UNTREATED']['order']))
    x = data_frame[[init_dict['varnames'][i - 1] for i in help_]]

    value = mte_information(coeffs_treated, coeffs_untreated, cov, quantiles, x, rslt)
    if quant is None:
        return value, args
    else:
        return value


def process_se_log(rslt, init_dict):
    """This function processes the standard error values for the log file."""
    se = ['------' if math.isnan(i) else str(i) for i in np.round(rslt['AUX']['standard_errors'], 4)
          ]
    list_ = ['({})'.format(i) if len(i) == 6 else '({}0)'.format(i) for i in se]
    return list_
