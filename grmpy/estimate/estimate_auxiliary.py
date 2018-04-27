"""The module provides auxiliary functions for the estimation process"""
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from scipy.stats import norm
import statsmodels.api as sm
import pandas as pd
import numpy as np

from grmpy.simulate.simulate_auxiliary import construct_covariance_matrix
from grmpy.simulate.simulate_auxiliary import simulate_unobservables
from grmpy.simulate.simulate_auxiliary import simulate_covariates
from grmpy.simulate.simulate_auxiliary import simulate_outcomes
from grmpy.check.check import UserError


def log_likelihood(init_dict, data_frame, rslt, dict_=None):
    """The function provides the log-likelihood function for the minimization process."""
    beta1, beta0, gamma, sd1, sd0, sdv, rho1v, rho0v, choice = \
        _prepare_arguments(init_dict, rslt)
    likl = []
    order_outcome = list(set(init_dict['TREATED']['order'] + init_dict['UNTREATED']['order']))
    for i in [0.0, 1.0]:
        if i == 1.0:
            beta, gamma, rho, sd, sdv = beta1, gamma, rho1v, sd1, sdv
            key_ = 'TREATED'
        else:
            beta, gamma, rho, sd, sdv = beta0, gamma, rho0v, sd0, sdv
            key_ = 'UNTREATED'
        data = data_frame[data_frame['D'] == i]

        Z = data[['X{}'.format(j - 1) for j in init_dict['COST']['order']]]
        X = data[['X{}'.format(j - 1) for j in init_dict[key_]['order']]]
        XX = data[['X{}'.format(j - 1) for j in order_outcome]]
        g = pd.concat((XX, Z), axis=1)

        choice_ = pd.DataFrame.sum(choice * g, axis=1)
        part1 = (data['Y'] - pd.DataFrame.sum(beta * X, axis=1)) / sd
        part2 = (choice_ - rho * sdv * part1) / (np.sqrt((1 - rho ** 2) * sdv ** 2))
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


def _prepare_arguments(init_dict, rslt):
    """The function prepares the coefficients for the log-liklihood function."""
    beta1 = np.array(rslt['TREATED']['all'])
    beta0 = np.array(rslt['UNTREATED']['all'])
    gamma = np.array(rslt['COST']['all'])
    sd0 = rslt['DIST']['all'][0]
    sd1 = rslt['DIST']['all'][1]
    sdv = init_dict['DIST']['all'][5]
    rho1, rho0 = rslt['DIST']['all'][3], rslt['DIST']['all'][2]
    choice = []
    for i in set(init_dict['TREATED']['order'] + init_dict['UNTREATED']['order']):
        if i in init_dict['TREATED']['order'] and i in init_dict['UNTREATED']['order']:
            index_treated = init_dict['TREATED']['order'].index(i)
            index_untreated = init_dict['UNTREATED']['order'].index(i)
            choice += [beta1[index_treated] - beta0[index_untreated]]
        elif i in init_dict['TREATED']['order'] and i not in init_dict['UNTREATED']['order']:
            index = init_dict['TREATED']['order'].index(i)
            choice += [beta1[index]]
        elif i not in init_dict['TREATED']['order'] and i in init_dict['UNTREATED']['order']:
            index = init_dict['UNTREATED']['order'].index(i)
            choice += [-beta0[index]]
    choice = np.concatenate((np.array(choice), -gamma))
    return beta1, beta0, gamma, sd1, sd0, sdv, rho1, rho0, choice


def start_values(init_dict, data_frame, option):
    """The function selects the start values for the minimization process."""

    if not isinstance(init_dict, dict):
        msg = 'The input object ({})for specifing the start values isn`t a dictionary.' \
            .format(init_dict)
        raise UserError(msg)
    numbers = [init_dict['AUX']['num_covars_treated'], init_dict['AUX']['num_covars_untreated'],
               init_dict['AUX']['num_covars_cost']]

    if option == 'init':
        # Set coefficients equal the true init file values
        x0 = init_dict['AUX']['init_values'][:numbers[0] + numbers[1] + numbers[2]]
        sd_ = None
    elif option == 'auto':

        try:

            # Estimate beta1 and beta0:
            beta = []
            sd_ = []

            for i in [1.0, 0.0]:
                Y = data_frame.Y[data_frame.D == i]
                if i == 1:
                    order = init_dict['TREATED']['order']
                else:
                    order = init_dict['UNTREATED']['order']
                X = data_frame[['X{}'.format(j - 1) for j in order]][data_frame.D == i]

                ols_results = sm.OLS(Y, X).fit()
                beta += [ols_results.params]
                sd_ += [np.sqrt(ols_results.scale)]

            # Estimate gamma via Probit
            XZ = data_frame[[j for j in data_frame.columns.values if j.startswith('X')]]
            probitRslt = sm.Probit(data_frame.D, XZ).fit(disp=0)
            help_gamma = probitRslt.params
            
            # Adjust estimated cost-benefit shifter and intercept coefficients
            help_ = init_dict['TREATED']['order'] + init_dict['UNTREATED']['order']
            adj = [i - 1 for i in init_dict['COST']['order'] if i in help_]
            for j in adj:
                if i in init_dict['TREATED']['order'] and i in init_dict['UNTREATED']['order']:
                    help_gamma[j] = np.subtract(np.subtract(beta[0][j], beta[1][j]), help_gamma[j])
                elif i in init_dict['TREATED']['order'] and i not in init_dict['UNTREATED']['order']:
                    help_gamma[j] = np.subtract(beta[0][j], help_gamma[j])
                elif i not in init_dict['TREATED']['order'] and i in init_dict['UNTREATED']['order']:
                    help_gamma[j] = np.subtract((- beta[1][j]), help_gamma[j])

            gamma = []
            for i in init_dict['COST']['order']:
                gamma += [help_gamma[i - 1]]

            # Arrange starting values
            x0 = np.concatenate((beta[0], beta[1]))
            x0 = np.concatenate((x0, gamma))

        except (PerfectSeparationError, ValueError):
            msg = 'The estimation process wasn`t able to provide automatic start values due to ' \
                  'perfect seperation. \n                                                     ' \
                  ' The intialization specifications are used as start ' \
                  'values during the further process.'
            # Set coefficients equal the true init file values
            x0 = init_dict['AUX']['init_values'][: numbers[0] + numbers[1] +numbers[2]]
            sd_ = None
            init_dict['ESTIMATION']['warning'] = msg
            option = 'init'

    x0, start = provide_cholesky_decom(init_dict, x0, option, sd_)
    init_dict['AUX']['starting_values'] = x0[:]
    init_dict['AUX']['start_values'] = start
    x0 = np.array(x0)

    return x0


def distribute_parameters(init_dict, start_values, dict_=None):
    """The function generates a dictionary for the representation of the optimization output."""
    if dict_ is None:
        pass
    else:
        dict_['parameter'][str(len(dict_['parameter']))] = start_values

    num_covars_treated = init_dict['AUX']['num_covars_treated']
    num_covars_untreated = init_dict['AUX']['num_covars_untreated']
    rslt = dict()

    rslt['TREATED'] = dict()
    rslt['UNTREATED'] = dict()
    rslt['COST'] = dict()
    rslt['DIST'] = dict()

    # Distribute parameters
    rslt['TREATED']['all'] = start_values[:num_covars_treated]
    rslt['UNTREATED']['all'] = start_values[num_covars_treated:num_covars_treated + num_covars_untreated]
    rslt['COST']['all'] = start_values[num_covars_treated + num_covars_untreated:(-6)]
    for key_ in ['TREATED', 'UNTREATED', 'COST']:
        rslt[key_]['order'] = init_dict[key_]['order']
        rslt[key_]['types'] = init_dict[key_]['types']

    rslt['DIST']['all'] = backward_cholesky_transformation(start_values, True)

    # Update auxiliary versions
    rslt['AUX'] = dict()
    rslt['AUX']['x_internal'] = start_values[:]
    rslt['AUX']['init_values'] = init_dict['AUX']['init_values']
    return rslt


def minimizing_interface(start_values, init_dict, data_frame, dict_):
    """The function provides the minimization interface for the estimation process."""
    # Collect arguments
    rslt = distribute_parameters(init_dict, start_values, dict_)

    # Calculate likelihood for pre-specified arguments
    likl = log_likelihood(init_dict, data_frame, rslt, dict_)

    return likl


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
                fmt = '  {:>10}' * 2 + ' {:>20.4f}' * 2
                for i in range(len(rslt['AUX']['x_internal'])):
                    file_.write('{0}\n'.format(
                        fmt.format('', str(i), init_dict['AUX']['starting_values'][i],
                                   rslt['AUX']['x_internal'][i])))


def optimizer_options(init_dict_):
    """The function provides the optimizer options given the initialization dictionary."""
    method = init_dict_['ESTIMATION']['optimizer'].split('-')[1]
    opt_dict = init_dict_['SCIPY-' + method]
    opt_dict['maxiter'] = init_dict_['ESTIMATION']['maxiter']

    return opt_dict, method


def simulate_estimation(init_dict, rslt, start=False):
    """The function simulates a new sample based on the estimated coefficients."""

    # Distribute information
    seed = init_dict['SIMULATION']['seed']
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
        U, _ = simulate_unobservables(dict_)
        X = simulate_covariates(rslt_dict)

        # Simulate endogeneous variables
        Y, D, Y_1, Y_0 = simulate_outcomes(dict_, X, U)

        df = write_output_estimation(Y, D, X, Y_1, Y_0)
        data_frames += [df]

    if start:
        return data_frames[0], data_frames[1]
    else:
        return data_frames[0]


def process_results(init_dict, rslt, start=False):
    """The function processes the results dictionary for the following simulation."""
    rslt_dict = {}
    start_dict = {}
    num_treated = len(init_dict['TREATED']['all'])
    num_untreated = len(init_dict['UNTREATED']['all'])
    dicts = [rslt_dict, start_dict]
    for dict_ in dicts:
        dict_['SIMULATION'] = {}
        dict_['SIMULATION']['agents'] = init_dict['ESTIMATION']['agents']
        dict_['SIMULATION']['seed'] = init_dict['SIMULATION']['seed']
        dict_['AUX'] = {}
        dict_['AUX']['types'] = init_dict['AUX']['types']
        if dict_ == rslt_dict:
            for key_ in ['TREATED', 'UNTREATED', 'COST']:
                dict_[key_] = {}
                dict_[key_]['types'] = init_dict[key_]['types']
                dict_[key_]['order'] = init_dict[key_]['order']
                dict_[key_]['all'] = rslt[key_]['all']
                dict_ = transform_rslt_DIST(rslt['AUX']['x_internal'], dict_)
        else:
            if start:
                for key_ in ['TREATED', 'UNTREATED', 'COST']:
                    dict_[key_] = {}
                    dict_[key_]['types'] = init_dict[key_]['types']
                    dict_[key_]['order'] = init_dict[key_]['order']
                dict_['TREATED']['all'] = init_dict['AUX']['starting_values'][:num_treated]
                dict_['UNTREATED']['all'] = init_dict['AUX']['starting_values'][
                                            num_treated: num_treated + num_untreated]
                dict_['COST']['all'] = init_dict['AUX']['starting_values'][num_treated + num_untreated:-6]
                dict_ = transform_rslt_DIST(init_dict['AUX']['starting_values'][-6:], dict_)
                return start_dict, rslt_dict
            else:
                return rslt_dict


def write_comparison(init_dict, df1, rslt):
    """The function writes the info file including the descriptives of the original and the
    estimated sample.
    """
    df3, df2 = simulate_estimation(init_dict, rslt, True)
    with open('comparison.grmpy.txt', 'w') as file_:
        # First we note some basic information ab out the dataset.
        header = '\n\n Number of Observations \n\n'
        file_.write(header)
        info_ = []
        for i, label in enumerate([df1, df2, df3]):
            info_ += [[label.shape[0], (label['D'] == 1).sum(), (label['D'] == 0).sum()]]

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

                data = data_frame['Y']

                if group == 'Treated':
                    data = data[data_frame['D'] == 1]
                elif group == 'Untreated':
                    data = data[data_frame['D'] == 0]
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


def write_output_estimation(Y, D, X, Y_1, Y_0):
    """The function converts the simulated variables to a panda data frame."""

    # Stack arrays
    data = np.column_stack((Y, D, X, Y_1, Y_0))

    # Construct list of column labels
    column = ['Y', 'D']

    for i in list(range(X.shape[1])):
        str_ = 'X' + str(i)
        column.append(str_)

    column += ['Y1', 'Y0']

    # Generate data frame
    df = pd.DataFrame(data=data, columns=column)
    df['D'] = df['D'].apply(np.int64)
    return df


def process_rslt(init_dict, dict_, rslt):
    """The function checks if the criteria function value is smaller for the optimization output as
    for the start values.
    """
    x = min(dict_['crit'], key=dict_['crit'].get)
    if dict_['crit'][str(x)] <= rslt['crit']:
        warning = 'The optimization algorithm has failed to provide the parametrization that ' \
                  'leads to the minimal criterion function value. \n                           ' \
                  '                           The estimation output is automatically adjusted.'

        rslt['warning'] = warning

        if dict_['crit'][str(x)] < init_dict['AUX']['criteria']:
            rslt['AUX']['x_internal'] = dict_['parameter'][str(x)].tolist()
            rslt['crit'] = dict_['crit'][str(x)]
        else:
            rslt['AUX']['x_internal'] = init_dict['AUX']['starting_values']
            rslt['crit'] = init_dict['AUX']['criteria']
    else:
        rslt['warning'] = '---'


def bfgs_dict():
    """The function provides a dictionary for tracking the criteria function values and the
    associated parametrization.
    """
    rslt_dict = {'parameter': {}, 'crit': {}}
    return rslt_dict


def adjust_output(opt_rslt, init_dict, start_values, dict_=None):
    """The function adds different information of the minimization process to the estimation
    output.
    """
    rslt = distribute_parameters(init_dict, start_values)
    rslt['success'], rslt['status'] = opt_rslt['success'], opt_rslt['status']
    rslt['message'], rslt['nfev'] = opt_rslt['message'], opt_rslt['nfev']
    rslt['crit'] = opt_rslt['fun']

    process_rslt(init_dict, dict_, rslt)

    return rslt


def adjust_output_maxiter_zero(init_dict, start_values):
    """The function returns a result dictionary if the maximum number of evaluations is zero."""
    num_covars_treated = init_dict['AUX']['num_covars_treated']
    num_covars_untreated = init_dict['AUX']['num_covars_untreated']
    rslt = dict()
    rslt['TREATED'] = dict()
    rslt['UNTREATED'] = dict()
    rslt['COST'] = dict()
    rslt['DIST'] = dict()

    # Distribute parameters
    rslt['TREATED']['all'] = start_values[:num_covars_treated]
    rslt['UNTREATED']['all'] = start_values[num_covars_treated:num_covars_treated + num_covars_untreated]
    rslt['COST']['all'] = start_values[num_covars_treated + num_covars_untreated:(-6)]

    rslt['DIST']['all'] = start_values[-6:]

    # Update auxiliary versions
    rslt['AUX'] = dict()
    rslt['AUX']['x_internal'] = start_values[:]

    rslt['AUX']['init_values'] = init_dict['AUX']['init_values'][:]
    rslt['success'], rslt['status'] = False, 2
    rslt['message'], rslt['nfev'], rslt['crit'] = '---', 0, init_dict['AUX']['criteria']
    rslt['warning'] = '---'

    return rslt


def adjust_print_output(init_dict, rslt):
    """The function arranges the distributional parameters."""

    for dict_ in [init_dict, rslt]:
        if dict_ == init_dict:
            key_ = 'starting_values'
        else:
            key_ = 'x_internal'
        if not isinstance(dict_['AUX'][key_], list):
            dict_['AUX'][key_] = dict_['AUX'][key_].tolist()
        dict_['AUX'][key_] = backward_cholesky_transformation(dict_['AUX'][key_])
        dict_['AUX'][key_] = np.array(dict_['AUX'][key_])

    return init_dict, rslt


def transform_rslt_DIST(rslt, dict_):
    """The function converts the correlation parameters from the estimation outcome to
    covariances for the simulation of the estimation sample.
    """
    dict_['DIST'] = {}
    place_holder = rslt[-6:]
    cov01 = place_holder[1] * place_holder[0] * place_holder[3]
    cov0V = place_holder[2] * place_holder[0] * place_holder[5]
    cov1V = place_holder[4] * place_holder[3] * place_holder[5]

    dict_['DIST']['all'] = [place_holder[0], cov01, cov0V, place_holder[3], cov1V, place_holder[5]]

    for i, element in enumerate(dict_['DIST']['all']):
        dict_['DIST']['all'][i] = round(element, 4)

    return dict_


def provide_cholesky_decom(init_dict, x0, option, sd_=None):
    """The function transforms the start covariance matrix into its cholesky decomposition."""
    if option == 'init':
        cov = construct_covariance_matrix(init_dict)
        L = np.linalg.cholesky(cov)
        L = L[np.tril_indices(3)]
        distribution_characteristics = init_dict['AUX']['init_values'][-6:]
        x0 = np.concatenate((x0, L))

    elif option == 'auto':
        distribution_characteristics = [sd_[0], 0, 0, sd_[1], 0, 1]
        cov = np.zeros((3, 3))
        cov[np.triu_indices(3)] = [distribution_characteristics]
        cov[np.tril_indices(3, k=-1)] = cov[np.triu_indices(3, k=1)]
        cov[np.diag_indices(3)] **= 2
        L = np.linalg.cholesky(cov)
        L = L[np.tril_indices(3)]
        x0 = np.concatenate((x0, L))
    init_dict['AUX']['cholesky_decomposition'] = L.tolist()
    start = [i for i in x0] + distribution_characteristics

    return x0, start


def backward_cholesky_transformation(x0, dist=False, test=False):
    """The function creates a positive semi definite covariance matrix from the given cholesky
    decomposition elements.
    """
    start_cholesky = x0[-6:]

    cholesky = np.zeros((3, 3))
    cholesky[np.tril_indices(3)] = start_cholesky
    cov = np.dot(cholesky, cholesky.T)
    sdv = cov[2, 2] ** 0.5

    if dist:
        sd1 = cov[0, 0] ** 0.5
        sd0 = cov[1, 1] ** 0.5
        rho0 = cov[1, 2] / (sd0 * sdv)
        rho1 = cov[0, 2] / (sd1 * sdv)

        dist_parameter = [sd0, sd1, rho0, rho1]
        return dist_parameter
    else:
        dist_para = cov[np.triu_indices(3)]
        sd0, sd1, sdv = dist_para[3] ** 0.5, dist_para[0] ** 0.5, dist_para[5] ** 0.5
        rho1, rho0 = dist_para[2] / (sd1 * sdv), dist_para[4] / (sd0 * sdv)
        rho01 = dist_para[1] / (sd0 * sd1)
        if not test:
            output = x0[:-6] + [sd1, rho01, rho1, sd0, rho0, sdv]
        else:
            output = [sd1, rho01, rho1, sd0, rho0, sdv]
        return output
