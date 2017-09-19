"""The module provides auxiliary functions for the estimation process"""
from scipy.stats import norm
import statsmodels.api as sm
import pandas as pd
import numpy as np


def log_likelihood(data_frame, init_dict, rslt):
    """The function provides the loglikelihood function for the minimization process."""
    beta1, beta0, gamma, sd0, sd1, sdv, rho1v, rho0v, choice = \
        _prepare_arguments(rslt, init_dict)
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
        choice_ = pd.DataFrame.sum(choice * g, axis =1)
        part1 =  (data['Y'] - pd.DataFrame.sum(beta * X, axis=1))/sd
        part2 = (choice_ - rho * sdv * part1) / (np.sqrt((1 - rho ** 2) * sdv ** 2))
        dist_1, dist_2 = norm.pdf(part1), norm.cdf(part2)
        if i == 1.00:
            contrib = (1.0 / sd) * dist_1 * dist_2
        else:
            contrib = (1.0 / sd) * dist_1 * (1.0 - dist_2)
        likl.append(contrib)
    likl = np.append(likl[0], likl[1])
    likl = - np.mean(np.log(np.clip(likl, 1e-20, np.inf)))
    return likl


def _prepare_arguments(rslt, init_dict):
    """The function prepares the coefficients for the logliklihood function."""
    beta1 = np.array(rslt['TREATED']['all'])
    beta0 = np.array(rslt['UNTREATED']['all'])
    gamma = np.array(rslt['COST']['all'])
    sd1 = rslt['DIST']['all'][1]
    sd0 = rslt['DIST']['all'][0]
    sdv = init_dict['DIST']['all'][5]
    rho1 = rslt['DIST']['all'][3]
    rho0 = rslt['DIST']['all'][2]
    choice = np.concatenate(((beta1 - beta0), -gamma))

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
        gamma_const = beta[1][0] - beta[0][0] - gamma[0]
        gamma = np.concatenate(([gamma_const], gamma[-(numbers[1] - 1):]))

        # Arange starting values
        x0 = np.concatenate((beta[1], beta[0]))
        x0 = np.concatenate((x0, gamma))
        x0 = np.concatenate((x0, sd_))
        x0 = np.concatenate((x0, [0.00, 0.00]))
        x0 = _transform_start(x0)
        x0 = np.array(x0)
    init_dict['AUX']['starting_values'] = x0

    return x0


def distribute_parameters(start_values, init_dict):
    """The function generates a dictionary for the representation of the optimization output."""
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
    rslt['DIST']['all'][2] = -1.0 + 2.0 / (1.0 + float(np.exp(-start_values[-2])))
    rslt['DIST']['all'][3] = -1.0 + 2.0 / (1.0 + float(np.exp(-start_values[-1])))

    # Update auxiliary versions
    rslt['AUX'] = dict()
    rslt['AUX']['x_internal'] = start_values.copy()
    rslt['AUX']['x_internal'][-4] = start_values[(-4)]
    rslt['AUX']['x_internal'][-3] = start_values[(-3)]
    rslt['AUX']['x_internal'][-2] = -1.0 + 2.0 / (1.0 + float(np.exp(-start_values[-2])))
    rslt['AUX']['x_internal'][-1] = -1.0 + 2.0 / (1.0 + float(np.exp(-start_values[-1])))
    rslt['AUX']['init_values'] = init_dict['AUX']['init_values']

    return rslt


def minimizing_interface(start_values, data_frame, init_dict):
    """The function provides the minimization interface for the estimation process."""
    # Collect arguments
    rslt = distribute_parameters(start_values, init_dict)

    # Calculate liklihood for pre specified arguments
    likl = log_likelihood(data_frame, init_dict, rslt)

    return likl


def _transform_start(x):
    """ Transform starting values to cover the whole real line."""

    # Coefficients
    x[:(-4)] = x[:(-4)]

    # Variances
    x[(-4)] = np.log(x[(-4)])
    x[(-3)] = np.log(x[(-3)])

    # Correlations
    transform = (x[(-2)] + 1) / 2
    x[(-2)] = np.log(transform / (1.0 - transform))

    transform = (x[(-1)] + 1) / 2
    x[(-1)] = np.log(transform / (1.0 - transform))

    # Finishing
    return x

def calculate_criteria(start_values, init_dict, data_frame):
    rslt = distribute_parameters(start_values, init_dict)
    criteria = log_likelihood(data_frame, init_dict, rslt)
    return criteria



def print_logfile(rslt, init_dict):
    with open('est.grmpy.info', 'w') as file_:

        for label in ['Optimization Information', 'Criterion Function', 'Economic Parameters']:
            header = '\n \n  {:<10}\n\n'.format(label)
            file_.write(header)
            if label == 'Optimization Information':
                for section in ['Success', 'Status', 'Number of Evaluations', 'F-value', 'Message']:
                    fmt = '  {:<10}' + ' {:<20}' + '  {:>20}\n\n'
                    if section == 'Number of Evaluations':
                        file_.write(fmt.format('', section + ':', rslt['nfev']))
                    elif section == 'F-value':
                        fmt = '  {:<10}' + ' {:<20}' + '       {:>20.4f}\n\n'
                        file_.write(fmt.format('', section + ':', rslt['fval']))
                    else:
                        file_.write(fmt.format('', section + ':', rslt[section.lower()]))
            elif label == 'Criterion Function':
                fmt = '  {:<10}' * 2 + ' {:>20}' * 2 + '\n\n'
                file_.write(fmt.format('', '', 'Start', 'Current') )
                file_.write('\n' + fmt.format('', '', init_dict['AUX']['criteria'], rslt['crit']))

            else:
                file_.write(fmt.format(*['', 'Identifier', 'Start', 'Current'])+ '\n\n')
                fmt = '  {:>10}' * 2 + ' {:>20.4f}' * 2
                for i in range(len(rslt['AUX']['x_internal'])):
                    file_.write('{0}\n'.format(
                        fmt.format('', str(i), init_dict['AUX']['starting_values'][i],
                                   rslt['AUX']['x_internal'][i])))




