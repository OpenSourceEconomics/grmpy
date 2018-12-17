"""The module provides auxiliary functions for the estimation process"""
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from statsmodels.tools.numdiff import approx_hess_cs
from numpy.linalg import LinAlgError
from scipy.stats import norm, t
import statsmodels.api as sm
import pandas as pd
import numpy as np

from grmpy.check.check import check_start_values
from grmpy.check.check import UserError


def start_values(init_dict, data_frame, option):
    """The function selects the start values for the minimization process."""
    if not isinstance(init_dict, dict):
        msg = 'The input object ({})for specifing the start values isn`t a dictionary.' \
            .format(init_dict)
        raise UserError(msg)
    indicator = init_dict['ESTIMATION']['indicator']
    dep = init_dict['ESTIMATION']['dependent']

    if option == 'init':
        # Set coefficients equal the true init file values
        x0 = init_dict['AUX']['init_values'][:-6]
    elif option == 'auto':

        try:

            # Estimate beta1 and beta0:
            beta = []
            sd_ = []

            for i in [1.0, 0.0]:
                Y = data_frame[dep][data_frame[indicator] == i]
                if i == 1:
                    order = init_dict['TREATED']['order']
                else:
                    order = init_dict['UNTREATED']['order']
                X = data_frame[order][
                    i == data_frame[indicator]]

                ols_results = sm.OLS(Y, X).fit()
                beta += [ols_results.params]
                sd_ += [np.sqrt(ols_results.scale), 0.0]
            # Estimate gamma via Probit
            Z = data_frame[init_dict['CHOICE']['order']]
            probitRslt = sm.Probit(data_frame[indicator], Z).fit(disp=0)
            gamma = probitRslt.params
            # Adjust estimated cost-benefit shifter and intercept coefficients
            # Arrange starting values
            x0 = np.concatenate((beta[0], beta[1], gamma, sd_))
            check_start_values(x0)

        except (PerfectSeparationError, ValueError, UserError):
            msg = 'The estimation process wasn`t able to provide automatic start values due to ' \
                  'perfect seperation. \n                                                     ' \
                  ' The intialization specifications are used as start ' \
                  'values during the further process.'

            # Set coefficients equal the true init file values
            x0 = init_dict['AUX']['init_values'][:-6]
            init_dict['ESTIMATION']['warning'] = msg
            option = 'init'

    x0 = start_value_adjustment(x0, init_dict, option)
    x0 = np.array(x0)
    return x0


def start_value_adjustment(x, init_dict, option):
    """This function transforms the rho values so that they are always between zero and 1.
    Additionally it transforms the sigma values so that all values are always positive.
    """

    # if option = init the estimation process takes its distributional arguments from the
    # inititialization dict
    if option == 'init':
        rho1 = init_dict['DIST']['params'][2] / init_dict['DIST']['params'][0]
        rho0 = init_dict['DIST']['params'][4] / init_dict['DIST']['params'][3]
        dist = [init_dict['DIST']['params'][0], rho1, init_dict['DIST']['params'][3], rho0]
        x = np.concatenate((x, dist))

    # transform the distributional characteristics s.t. r = log((1-rho)/(1+rho))/2
    for k in [-4, -3, -2, -1]:
        if k in [-3, -1]:
            x[k] = np.log((1 + x[k]) / (1 - x[k])) / 2
        else:
            x[k] = np.log(x[k])
    return x


def backward_transformation(x0, dict_=None):
    """The function generates a dictionary for the representation of the optimization output."""
    x = x0.copy()
    for k in [-4, -3, -2, -1]:
        if k in [-3, -1]:
            x[k] = (np.exp(2 * x[k]) - 1) / (np.exp(2 * x[k]) + 1)
        else:
            x[k] = np.exp(x[k])
    if dict_ is None:
        pass
    else:
        dict_['parameter'][str(len(dict_['parameter']))] = x
    return x


def log_likelihood(x0, init_dict, data_frame, dict_=None):
    """The function provides the log-likelihood function for the minimization process."""
    # Distribute parameter
    num_treated = init_dict['AUX']['num_covars_treated']
    num_untreated = num_treated + init_dict['AUX']['num_covars_untreated']

    beta1, beta0, gamma = x0[:num_treated], x0[num_treated:num_untreated], x0[num_untreated:-4]
    sd1, sd0, rho1v, rho0v, sdv = x0[-4], x0[-2], x0[-3], x0[-1], 1.0
    # Set labels for indicator and the dependent variable
    indicator = init_dict['ESTIMATION']['indicator']
    dep = init_dict['ESTIMATION']['dependent']
    # Provide parameterization for D=1 and D=0 and provide auxiliary list likl
    likl = []
    for i in [0.0, 1.0]:
        if i == 1.0:
            beta, gamma, rho, sd, sdv = beta1, gamma, rho1v, sd1, sdv
            key_ = 'TREATED'
        else:
            beta, gamma, rho, sd, sdv = beta0, gamma, rho0v, sd0, sdv
            key_ = 'UNTREATED'
        # Prepare data
        data = data_frame[data_frame[indicator] == i]
        Z = data[init_dict['CHOICE']['order']]
        X = data[init_dict[key_]['order']]

        choice_ = pd.DataFrame.sum(gamma * Z, axis=1)
        part1 = (data[dep] - pd.DataFrame.sum(beta * X, axis=1)) / sd
        part2 = (choice_ - rho * sdv * part1) / (np.sqrt((1 - (rho ** 2)) * (sdv ** 2)))
        dist_1, dist_2 = norm.pdf(part1), norm.cdf(part2)

        if i == 1.0:
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


def calculate_criteria(init_dict, data_frame, x0):
    """The function calculates the criteria function value."""
    x = backward_transformation(x0)
    criteria = log_likelihood(x, init_dict, data_frame)
    return criteria


def optimizer_options(init_dict_):
    """The function provides the optimizer options given the initialization dictionary."""
    method = init_dict_['ESTIMATION']['optimizer'].split('-')[1:]
    if isinstance(method, list):
        method = '-'.join(method)
    opt_dict = init_dict_['SCIPY-' + method]
    opt_dict['maxiter'] = init_dict_['ESTIMATION']['maxiter']

    return opt_dict, method


def minimizing_interface(x0, init_dict, data_frame, dict_):
    """The function provides the minimization interface for the estimation process."""
    # Collect arguments
    x0 = backward_transformation(x0, dict_)
    # Calculate likelihood for pre-specified arguments
    likl = log_likelihood(x0, init_dict, data_frame, dict_)

    return likl


def process_output(init_dict, dict_, x0, flag):
    """The function checks if the criteria function value is smaller for the optimization output as
    for the start values.
    """

    x = min(dict_['crit'], key=dict_['crit'].get)
    if flag == 'adjustment':
        if dict_['crit'][str(x)] < init_dict['AUX']['criteria']:
            x0 = dict_['parameter'][str(x)].tolist()
            crit = dict_['crit'][str(x)]
            warning = 'The optimization algorithm has failed to provide the parametrization that ' \
                      'leads to the minimal criterion function value. \n                         ' \
                      '                             The estimation output is automatically ' \
                      'adjusted and provides the parameterization with the smallest criterion ' \
                      'function value that was reached during the optimization.'
    if flag == 'notfinite':
        x0 = init_dict['AUX']['starting_values']
        crit = init_dict['AUX']['criteria']
        warning = 'Tho optimization process is not able to provide finite values. This is ' \
                  'probably due to perfect separation.'
    return x0, crit, warning


def check_rslt_parameters(init_dict, data_frame, dict_, x0):
    """This function checks if the algorithms has provided a parameterization with a lower criterium
     function value.
     """
    crit = calculate_criteria(init_dict, data_frame, x0)
    x = min(dict_['crit'], key=dict_['crit'].get)
    if False in np.isfinite(x0).tolist():
        check, flag = True, 'notfinite'

    elif dict_['crit'][str(x)] <= crit:
        check, flag = True, 'adjustment'
    else:
        check, flag = False, None
    return check, flag


def bfgs_dict():
    """The function provides a dictionary for tracking the criteria function values and the
    associated parametrization.
    """
    rslt_dict = {'parameter': {}, 'crit': {}}
    return rslt_dict


def adjust_output(opt_rslt, init_dict, x0, data_frame, dict_=None):
    """The function adds different information of the minimization process to the estimation
    output.
    """
    num_treated = init_dict['AUX']['num_covars_treated']
    num_untreated = num_treated + init_dict['AUX']['num_covars_untreated']

    rslt = {'AUX': {}}
    # Adjust output if
    if init_dict['ESTIMATION']['maxiter'] == 0:
        x = backward_transformation(x0)
        rslt['success'], rslt['status'] = False, 2
        rslt['message'], rslt['nfev'], rslt['crit'] = '---', 0, init_dict['AUX']['criteria']
        rslt['warning'] = ['---']

    else:
        # Check if the algorithm has returned the values with the lowest criterium function value
        check, flag = check_rslt_parameters(init_dict, data_frame, dict_, x0)
        # Adjust values if necessary
        if check:
            x, crit, warning = process_output(init_dict, dict_, x0, flag)
            rslt['crit'] = crit
            rslt['warning'] = [warning]

        else:
            x = backward_transformation(x0)
            rslt['crit'] = opt_rslt['fun']
            rslt['warning'] = ['---']

        rslt['success'], rslt['status'] = opt_rslt['success'], opt_rslt['status']
        rslt['message'], rslt['nfev'] = opt_rslt['message'], opt_rslt['nfev']

    # Adjust Result dict
    rslt['AUX']['x_internal'] = x
    rslt['ESTIMATION'] = init_dict['ESTIMATION']
    for key_ in ['TREATED', 'UNTREATED', 'CHOICE', 'AUX']:
        if key_ == 'AUX':
            rslt['AUX']['labels'] = init_dict['AUX']['labels']
        else:
            rslt[key_] = {}
            rslt[key_]['order'] = init_dict[key_]['order']
    rslt['VARTYPES'] = init_dict['VARTYPES']

    rslt['TREATED']['params'] = np.array(x[:num_treated])
    rslt['UNTREATED']['params'] = np.array(x[num_treated:num_untreated])
    rslt['CHOICE']['params'] = np.array(x[num_untreated:-4])
    rslt = calculate_se(rslt, init_dict, data_frame)
    return rslt


def calculate_se(rslt, init_dict, data_frame):
    """This function calculates the standard errors for given parameterization via an approximation
    of the hessian matrix."""

    x0 = rslt['AUX']['x_internal']

    if init_dict['ESTIMATION']['maxiter'] == 0:
        rslt['AUX']['standard_errors'] = [np.nan] * len(x0)
        rslt['AUX']['hess_inv'] = '---'
        rslt['AUX']['confidence_intervals'] = [[np.nan, np.nan]] * len(x0)
    else:
        # Calculate the hessian matrix, check if it is p
        hess = approx_hess_cs(x0, log_likelihood, args=(init_dict, data_frame))
        try:
            hess_inv = np.linalg.inv(hess)
            se = np.sqrt(np.diag(hess_inv) / data_frame.shape[0])
            rslt['AUX']['standard_errors'] = se
            rslt['AUX']['hess_inv'] = hess_inv
            rslt['AUX']['confidence_intervals'] = []
            for counter, param in enumerate(x0):
                upper = param + norm.ppf(0.975) * se[counter]
                lower = param - norm.ppf(0.975) * se[counter]
                rslt['AUX']['confidence_intervals'] += [[lower, upper]]

        except LinAlgError:
            rslt['AUX']['standard_errors'] = [np.nan] * len(x0)
            rslt['AUX']['hess_inv'] = '---'
            rslt['AUX']['confidence_intervals'] = [[np.nan, np.nan]] * len(x0)

        # Check if standard errors are defined, if not add warning message
        rslt['AUX']['p_values'], rslt['AUX']['t_values'] = \
            calculate_p_values(rslt['AUX']['standard_errors'], x0, data_frame)

        if False in np.isfinite(rslt['AUX']['standard_errors']):
            rslt['warning'] += ['The estimation process was not able to provide standard errors for'
                                ' the estimation results, because the approximation \n            '
                                '                                          of the hessian matrix '
                                'leads to a singular Matrix']

    return rslt

def calculate_p_values(se, x0, dataframe):
    """This function calculates the p values, given the estimation results and the standard errors.
    """
    p_values = []
    t_values = []
    df = dataframe.shape[0] - len(x0)
    for counter, value in enumerate(x0):
        if isinstance(value, float):
            p_values += [1 - t.cdf(np.abs(value/se[counter]), df=df)]
            t_values += [value/se[counter]]
        else:
            p_values += [np.nan]
            t_values += [np.nan]
    return p_values, t_values
