"""The module provides auxiliary functions for the estimation process"""
from scipy.stats import norm
import statsmodels.api as sm
import pandas as pd
import numpy as np

from grmpy.simulate.simulate_auxiliary import simulate_unobservables
from grmpy.simulate.simulate_auxiliary import simulate_covariates
from grmpy.simulate.simulate_auxiliary import simulate_outcomes


def log_likelihood(init_dict, data_frame, rslt, dict_=None):
    """The function provides the log-likelihood function for the minimization process."""
    beta1, beta0, gamma, sd1, sd0, sdv, rho1v, rho0v, choice = \
        _prepare_arguments(init_dict, rslt)
    likl = []
    for i in [0.0, 1.0]:
        if i == 1.00:
            beta, gamma, rho, sd, sdv = beta1, gamma, rho1v, sd1, sdv
        else:
            beta, gamma, rho, sd, sdv = beta0, gamma, rho0v, sd0, sdv
        data = data_frame[data_frame['D'] == i]
        X = data.filter(regex=r'^X\_')
        Z = data.filter(regex=r'^Z\_')
        g = pd.concat((X, Z), axis=1)
        choice_ = pd.DataFrame.sum(choice * g, axis=1)
        part1 = (data['Y'] - pd.DataFrame.sum(beta * X, axis=1)) / sd
        part2 = (choice_ - rho * sdv * part1) / (np.sqrt((1 - rho ** 2) * sdv ** 2))
        dist_1, dist_2 = norm.pdf(part1), norm.cdf(part2)
        if i == 1.00:
            contrib = (1.0 / sd) * dist_1 * dist_2
        else:
            contrib = (1.0 / sd) * dist_1 * (1.0 - dist_2)
        likl.append(contrib)
    likl = np.append(likl[0], likl[1])
    likl = - np.mean(np.log(np.clip(likl, 1e-20, np.inf)))
    if dict_ is None:
        pass
    else:
        dict_['crit'][str(len(dict_['crit']))] = likl
    return likl


def _prepare_arguments(init_dict, rslt):
    """The function prepares the coefficients for the log-liklihood function."""
    beta1 = np.array(rslt['TREATED']['all'])
    beta0 = np.array(rslt['UNTREATED']['all'])
    gamma = np.array(rslt['COST']['all'])
    sd1 = rslt['DIST']['all'][1]
    sd0 = rslt['DIST']['all'][0]
    sdv = init_dict['DIST']['all'][5]
    rho1, rho0 = rslt['DIST']['all'][3], rslt['DIST']['all'][2]
    choice = np.concatenate(((np.subtract(beta1, beta0)), -gamma))

    return beta1, beta0, gamma, sd1, sd0, sdv, rho1, rho0, choice


def start_values(init_dict, data_frame, option):
    """The function selects the start values for the minimization process."""

    assert isinstance(init_dict, dict)
    numbers = [init_dict['AUX']['num_covars_out'], init_dict['AUX']['num_covars_cost']]

    if option == 'true_values':
        # Set coefficients equal the true init file values
        x0 = init_dict['AUX']['init_values'][:2 * numbers[0] + numbers[1]]
        x0 += [init_dict['AUX']['init_values'][2 * numbers[0] + numbers[1]]]
        x0 += [init_dict['AUX']['init_values'][2 * numbers[0] + numbers[1] + 3]]
        rho0v = init_dict['AUX']['init_values'][2 * numbers[0] + numbers[1] + 2] / (
            init_dict['AUX']['init_values'][2 * numbers[0] + numbers[1]] *
            init_dict['AUX']['init_values'][2 * numbers[0] + numbers[1] + 5])
        rho1v = init_dict['AUX']['init_values'][2 * numbers[0] + numbers[1] + 4] / (
            init_dict['AUX']['init_values'][2 * numbers[0] + numbers[1] + 3] *
            init_dict['AUX']['init_values'][2 * numbers[0] + numbers[1] + 5])
        x0 += [rho0v, rho1v]
    elif option == 'auto':

        # Estimate beta1 and beta0:
        beta = []
        sd_ = []
        for i in [0.0, 1.0]:
            Y, X = data_frame.Y[data_frame.D == i], data_frame.filter(regex=r'^X\_')[
                data_frame.D == i]
            ols_results = sm.OLS(Y, X).fit()
            beta += [ols_results.params]
            sd_ += [np.sqrt(ols_results.scale)]

        # Estimate gamma via probit
        X = data_frame.filter(regex=r'^X\_')
        Z = (data_frame.filter(regex=r'^Z\_')).drop('Z_0', axis=1)
        XZ = np.concatenate((X, Z), axis=1)
        probitRslt = sm.Probit(data_frame.D, XZ).fit(disp=0)
        sd = init_dict['DIST']['all'][5]
        gamma = probitRslt.params * sd
        gamma_const = np.subtract(np.subtract(beta[1][0], beta[0][0]), gamma[0])
        if len(init_dict['COST']['all']) == 1:
            gamma = [gamma_const]
        else:
            gamma = np.concatenate(([gamma_const], gamma[-(numbers[1] - 1):]))
        rho = [0.00, 0.00]

        # Arange starting values
        x0 = np.concatenate((beta[1], beta[0]))
        x0 = np.concatenate((x0, gamma))
        x0 = np.concatenate((x0, sd_))
        x0 = np.concatenate((x0, rho))
    init_dict['AUX']['starting_values'] = x0
    x0 = _transform_start(x0)
    x0 = np.array(x0)

    return x0


def distribute_parameters(init_dict, start_values, dict_=None):
    """The function generates a dictionary for the representation of the optimization output."""
    if dict_ is None:
        pass
    else:
        dict_['parameter'][str(len(dict_['parameter']))] = start_values

    num_covars_out = init_dict['AUX']['num_covars_out']
    rslt = dict()

    rslt['TREATED'] = dict()
    rslt['UNTREATED'] = dict()
    rslt['COST'] = dict()
    rslt['DIST'] = dict()

    # Distribute parameters
    rslt['TREATED']['all'] = start_values[:num_covars_out]
    rslt['UNTREATED']['all'] = start_values[num_covars_out:(2 * num_covars_out)]
    rslt['COST']['all'] = start_values[(2 * num_covars_out):(-4)]

    rslt['DIST']['all'] = start_values[-4:]
    rslt['DIST']['all'][0] = start_values[-4]
    rslt['DIST']['all'][1] = start_values[-3]
    rslt['DIST']['all'][2] = -1.0 + 2 / (1.0 + np.exp(-start_values[-2]))
    rslt['DIST']['all'][3] = -1.0 + 2 / (1.0 + np.exp(-start_values[-1]))

    # Update auxiliary versions
    rslt['AUX'] = dict()
    rslt['AUX']['x_internal'] = start_values[:]
    rslt['AUX']['x_internal'][-4] = start_values[(-4)]
    rslt['AUX']['x_internal'][-3] = start_values[(-3)]
    rslt['AUX']['x_internal'][-2] = -1.0 + 2 / (1.0 + np.exp(-start_values[-2]))
    rslt['AUX']['x_internal'][-1] = -1.0 + 2 / (1.0 + np.exp(-start_values[-1]))
    rslt['AUX']['init_values'] = init_dict['AUX']['init_values']

    return rslt


def minimizing_interface(start_values, init_dict, data_frame, dict_):
    """The function provides the minimization interface for the estimation process."""
    # Collect arguments
    rslt = distribute_parameters(init_dict, start_values, dict_)

    # Calculate liklihood for pre specified arguments
    likl = log_likelihood(init_dict, data_frame, rslt, dict_)

    return likl


def _transform_start(x):
    """The function transforms the starting values to cover the whole real line."""
    # Coefficients
    x[:(-4)] = x[:(-4)]

    # Variances
    x[(-4)] = x[(-4)]
    x[(-3)] = x[(-3)]

    # Correlations
    transform = (x[(-2)] + 1) / 2
    x[(-2)] = np.log(transform / (1.0 - transform))

    transform = (x[(-1)] + 1) / 2
    x[(-1)] = np.log(transform / (1.0 - transform))

    # Finishing
    return x


def calculate_criteria(init_dict, data_frame, start_values):
    """The function calculates the criteria function value."""
    rslt = distribute_parameters(init_dict, start_values)
    criteria = log_likelihood(init_dict, data_frame, rslt)
    return criteria


def print_logfile(init_dict, rslt):
    """The function writes the log file for the estimation process."""

    # Adjust output
    init_dict, rslt = adjust_print_output(init_dict, rslt)

    with open('est.grmpy.info', 'w') as file_:

        for label in ['Optimization Information', 'Criterion Function', 'Economic Parameters']:
            header = '\n \n  {:<10}\n\n'.format(label)
            file_.write(header)
            if label == 'Optimization Information':
                for section in ['Optimizer', 'Success', 'Status', 'Number of Evaluations',
                                'Criteria', 'Message', 'Warning']:
                    fmt = '  {:<10}' + ' {:<20}' + '  {:>20}\n\n'
                    if section == 'Number of Evaluations':
                        file_.write(fmt.format('', section + ':', rslt['nfev']))
                    elif section == 'Optimizer':
                        file_.write(fmt.format('', section + ':',
                                               init_dict['ESTIMATION']['optimizer']))
                    elif section == 'Criteria':
                        fmt = '  {:<10}' + ' {:<20}' + '       {:>20.4f}\n\n'
                        file_.write(fmt.format('', section + ':', rslt['crit']))
                    else:
                        file_.write(fmt.format('', section + ':', rslt[section.lower()]))
            elif label == 'Criterion Function':
                fmt = '  {:<10}' * 2 + ' {:>20}' * 2 + '\n\n'
                file_.write(fmt.format('', '', 'Start', 'Current'))
                file_.write('\n' + fmt.format('', '', init_dict['AUX']['criteria'], rslt['crit']))

            else:
                file_.write(fmt.format(*['', 'Identifier', 'Start', 'Current']) + '\n\n')
                fmt = '  {:>10}' * 2 + ' {:>20.4f}' * 2
                for i in range(len(rslt['AUX']['x_internal'])):
                    file_.write('{0}\n'.format(
                        fmt.format('', str(i), init_dict['AUX']['starting_values'][i],
                                   rslt['AUX']['x_internal'][i])))


def optimizer_options(init_dict_):
    """The function provides the optimizer options given the initialization dictionary."""
    method = init_dict_['ESTIMATION']['optimizer'].split('-')[1]
    opt_dict = init_dict_['SCIPY-' + method]

    return opt_dict, method


def simulate_estimation(init_dict, rslt):
    """The function simulates a new sample based on the estimated coefficients."""

    dict_ = process_results(init_dict, rslt)

    # Distribute information
    seed = dict_['SIMULATION']['seed']

    # Simulate unobservables
    U, V = simulate_unobservables(dict_)

    # Simulate observables
    X = simulate_covariates(dict_, 'TREATED')
    Z = simulate_covariates(dict_, 'COST')

    # Simulate endogeneous variables
    Y, D, Y_1, Y_0 = simulate_outcomes(dict_, X, Z, U)

    df = write_output_estimation(Y, D, X, Z, Y_1, Y_0)

    return df


def process_results(init_dict, rslt):
    """The function processes the results dictionary for the following simulation."""
    dict_ = {}

    for key_ in ['TREATED', 'UNTREATED', 'COST']:
        dict_[key_] = {}
        dict_[key_]['types'] = init_dict[key_]['types']
        dict_[key_]['all'] = rslt[key_]['all']
    dict_['SIMULATION'] = {}
    dict_['SIMULATION']['agents'] = init_dict['ESTIMATION']['agents']
    dict_['SIMULATION']['seed'] = init_dict['SIMULATION']['seed']
    dict_ = transform_rslt_DIST(rslt, dict_)

    return dict_


def write_descriptives(init_dict, df1, rslt):
    """The function writes the info file including the descriptives of the original and the
    estimated sample.
    """
    df2 = simulate_estimation(init_dict, rslt)
    with open('descriptives.grmpy.info', 'w') as file_:
        # First we note some basic information ab out the dataset.
        header = '\n\n Number of Observations \n\n'
        file_.write(header)
        info_ = []
        for i, label in enumerate([df1, df2]):
            info_ += [[label.shape[0], (label['D'] == 1).sum(), (label['D'] == 0).sum()]]

        fmt = '  {:<10}' + ' {:>20}' * 2 + '\n\n'
        file_.write(fmt.format(*['', 'Sample', 'Estimated Sample']))

        for i, label in enumerate(['All', 'Treated', 'Untreated']):
            str_ = '  {:<10}' + ' {:20}' * 2 + '\n'
            file_.write(str_.format(label, info_[0][i], info_[1][i]))

        header = '\n\n Distribution of Outcomes\n\n'
        file_.write(header)

        for data in ['Sample', 'Estimated Sample']:
            header = '\n\n ' '  {:<10}'.format(data) + '\n\n'
            file_.write(header)

            fmt = '    {:<10}' + ' {:>20}' * 5 + '\n\n'
            args = ['', 'Mean', 'Std-Dev.', '25%', '50%', '75%']
            file_.write(fmt.format(*args))

            if data == 'Sample':
                data_frame = df1
            else:
                data_frame = df2
            for group in ['All', 'Treated', 'Untreated']:

                object = data_frame['Y']

                if group == 'Treated':
                    object = object[data_frame['D'] == 1]
                elif group == 'Untreated':
                    object = object[data_frame['D'] == 0]
                else:
                    pass
                fmt = '    {:<10}' + ' {:>20.4f}' * 5 + '\n'
                info = list(object.describe().tolist()[i] for i in [1, 2, 4, 5, 6])
                if pd.isnull(info).all():
                    fmt = '    {:<10}' + ' {:>20}' * 5 + '\n'
                    info = ['---'] * 5
                elif pd.isnull(info[1]):
                    info[1] = '---'
                    fmt = '    {:<10}' ' {:>20.4f}' ' {:>20}' + ' {:>20.4f}' * 3 + '\n'

                file_.write(fmt.format(*[group] + info))


def simulate_outcomes_estimation(init_dict, X, Z):
    """The function simulates the outcome Y, the resulting treatment dummy."""
    # Distribute information
    coeffs_untreated = init_dict['UNTREATED']['all']
    coeffs_treated = init_dict['TREATED']['all']
    coeffs_cost = init_dict['COST']['all']

    # Calculate potential outcomes and costs
    Y_1 = np.dot(coeffs_treated, X.T)
    Y_0 = np.dot(coeffs_untreated, X.T)
    C = np.dot(coeffs_cost, Z.T)

    # Calculate expected benefit and the resulting treatment dummy
    D = np.array((Y_1 - Y_0 - C > 0).astype(int))

    # Observed outcomes
    Y = D * Y_1 + (1 - D) * Y_0

    return Y, D, Y_1, Y_0


def write_output_estimation(Y, D, X, Z, Y_1, Y_0):
    """The function converts the simulated variables to a panda data frame."""

    # Stack arrays
    data = np.column_stack((Y, D, X, Z, Y_1, Y_0))

    # Construct list of column labels
    column = ['Y', 'D']
    for i in range(X.shape[1]):
        str_ = 'X_' + str(i)
        column.append(str_)
    for i in range(Z.shape[1]):
        str_ = 'Z_' + str(i)
        column.append(str_)
    column += ['Y1', 'Y0']

    # Generate data frame
    df = pd.DataFrame(data=data, columns=column)
    df['D'] = df['D'].apply(np.int64)
    return df


def process_rslt(init_dict, dict_, rslt, start_values):
    """The function checks if the criteria function value is smaller for the optimization output as
    for the start values.  """

    x = min(dict_['crit'], key=dict_['crit'].get)
    if init_dict['AUX']['criteria'] <= rslt['crit']:
        warning = 'The optimization algorithm has failed to provide the parametrization that ' \
                  'leads to the minimal criteria function value. \n                           ' \
                  '        The estimation output is automatically adjusted.'

        rslt['warning'] = warning

        if dict_['crit'][str(x)] < init_dict['AUX']['criteria']:
            rslt['AUX']['x_internal'] = dict_['parameter'][str(x)].tolist()
            rslt['crit'] = dict_['crit'][str(x)]
        else:
            rslt['AUX']['x_internal'] = init_dict['AUX']['starting_values']
            rslt['crit'] = init_dict['AUX']['criteria']
    else:
        rslt['warning'] = '---'


def bfgs_dict(optimizer):
    """The function provides a dictionary for tracking the criteria function values and the
    associated parametrization."""
    rslt_dict = {'parameter': {}, 'crit': {}}
    return rslt_dict


def adjust_output(opt_rslt, init_dict, start_values, optimizer, dict_=None):
    """The function adds different information of the minimization process to the estimation
    output."""
    rslt = distribute_parameters(init_dict, start_values)
    rslt['success'], rslt['status'] = opt_rslt['success'], opt_rslt['status']
    rslt['message'], rslt['nfev'], rslt['crit'] = opt_rslt['message'], opt_rslt['nfev'], \
                                                  opt_rslt['fun']

    process_rslt(init_dict, dict_, rslt, start_values)

    return rslt


def adjust_output_maxiter_zero(init_dict, start_values):
    """The function returns a result dictionary if the maximum number of evaluations is zero."""
    num_covars_out = init_dict['AUX']['num_covars_out']
    rslt = dict()

    rslt['TREATED'] = dict()
    rslt['UNTREATED'] = dict()
    rslt['COST'] = dict()
    rslt['DIST'] = dict()

    # Distribute parameters
    rslt['TREATED']['all'] = start_values[:num_covars_out]
    rslt['UNTREATED']['all'] = start_values[num_covars_out:(2 * num_covars_out)]
    rslt['COST']['all'] = start_values[(2 * num_covars_out):(-4)]

    rslt['DIST']['all'] = start_values[-4:]
    rslt['DIST']['all'][2] = start_values[-2]
    rslt['DIST']['all'][3] = start_values[-1]

    # Update auxiliary versions
    rslt['AUX'] = dict()
    rslt['AUX']['x_internal'] = start_values[:]
    rslt['AUX']['x_internal'][-4] = start_values[(-4)]
    rslt['AUX']['x_internal'][-3] = start_values[(-3)]
    rslt['AUX']['x_internal'][-2] = start_values[(-2)]
    rslt['AUX']['x_internal'][-1] = start_values[(-1)]
    rslt['AUX']['init_values'] = init_dict['AUX']['init_values']

    rslt['success'], rslt['status'] = False, 2
    rslt['message'], rslt['nfev'], rslt['crit'] = '---', 0, init_dict['AUX']['criteria']

    return rslt


def adjust_print_output(init_dict, rslt):
    """The function arranges the distributional parameters."""

    rho10 = init_dict['DIST']['all'][1] / (
    init_dict['DIST']['all'][0] * init_dict['DIST']['all'][3])
    sdv = init_dict['DIST']['all'][5]

    for dict_ in [init_dict, rslt]:
        if dict_ == init_dict:
            key_ = 'starting_values'
        else:
            key_ = 'x_internal'
        if not isinstance(dict_['AUX'][key_], list):
            dict_['AUX'][key_] = dict_['AUX'][key_].tolist()
        place_holder = dict_['AUX'][key_][:]
        dict_['AUX'][key_] = place_holder[:-4] + [place_holder[-4], rho10, place_holder[-2],
                                                  place_holder[-3], place_holder[-1], sdv]
        dict_['AUX'][key_] = np.array(dict_['AUX'][key_])

    return init_dict, rslt


def transform_rslt_DIST(rslt, dict_):
    """The function converts the correlation parameters from the estimation outcome to
    covariances for the simulation of the estimation sample.
    """
    dict_['DIST'] = {}
    place_holder = rslt['AUX']['x_internal'][-6:]
    cov01 = place_holder[1] * place_holder[0] * place_holder[3]
    cov0V = place_holder[2] * place_holder[0] * place_holder[5]
    cov1V = place_holder[4] * place_holder[3] * place_holder[5]

    dict_['DIST']['all'] = [place_holder[0], cov01, cov0V, place_holder[3], cov1V, place_holder[5]]

    return dict_