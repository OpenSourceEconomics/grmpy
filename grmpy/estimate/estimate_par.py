"""
The module provides auxiliary functions for the estimation process.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.linalg import LinAlgError
from random import randint
from scipy.optimize import minimize
from scipy.stats import norm, t
from statsmodels.tools.numdiff import approx_fprime_cs
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from grmpy.check.check import UserError, check_start_values
from grmpy.estimate.estimate_output import print_logfile


def par_fit(dict_, data):
    """This function runs the parametric estimation of the marginal treatment effect.

    Parameters
    ----------
    dict_: dict
        Estimation dictionary. Returned by grmpy.read(init_file)).
    data: pandas.DataFrame
        Data set to perform the estimation on. Specified
        under dict_["ESTIMATION"]["file"].

    Returns
    ------
    rslt: dict
        Result dictionary containing
        - opt_info: dict
            Dictionary that contains information regarding the conducted optimization
        - opt_rslt: pandas.DataFrame
            Dataframe that provides the results of the optimization process
        - mte
        - mte_x
        - mte_u
        - mte_min
        - mte_max
        - X
        - b1
        - b0
    """

    # process the data set
    D, X1, X0, Z1, Z0, Y1, Y0 = process_data(data, dict_)

    # process optimization options
    opt_dict, method, grad_opt, start_option, print_output, seed_ = process_inputs(
        dict_
    )
    num_treated = X1.shape[1]
    num_untreated = num_treated + X0.shape[1]

    # set seed
    np.random.seed(seed_)

    # set up rslt dataframe object
    rslt_cont = create_rslt_df(dict_)

    # determine start values
    x0 = start_values(dict_, D, X1, X0, Z1, Z0, Y1, Y0, start_option)
    dict_["AUX"]["criteria"] = calculate_criteria(x0, X1, X0, Z1, Z0, Y1, Y0)
    rslt_cont["start_values"] = backward_transformation(x0)
    bfgs_dict = {"parameter": {}, "crit": {}, "grad": {}}

    opt_rslt = minimize(
        minimizing_interface,
        x0,
        args=(X1, X0, Z1, Z0, Y1, Y0, num_treated, num_untreated, bfgs_dict, grad_opt),
        method=method,
        options=opt_dict,
        jac=grad_opt,
    )
    rslt = adjust_output(
        opt_rslt,
        dict_,
        rslt_cont,
        x0,
        method,
        start_option,
        X1,
        X0,
        Z1,
        Z0,
        Y1,
        Y0,
        bfgs_dict,
    )
    # Print Output files
    print_logfile(dict_, rslt, print_output)

    quantiles, cov, X, b1_b0, b1, b0 = prepare_mte_calc(rslt["opt_rslt"], data)

    mte_x = np.dot(X, b1_b0)

    mte_u = (cov[2, 0] - cov[2, 1]) * norm.ppf(quantiles)

    # Put the MTE together
    mte = mte_x.mean(axis=0) + mte_u

    # Account for variation in X
    mte_min = np.min(mte_x) + mte_u
    mte_max = np.max(mte_x) + mte_u

    rslt.update(
        {
            "quantiles": quantiles,
            "mte": mte,
            "mte_x": mte_x,
            "mte_u": mte_u,
            "mte_min": mte_min,
            "mte_max": mte_max,
            "X": X,
            "b1": b1,
            "b0": b0,
        }
    )

    return rslt


def process_data(data, dict_):
    """This function process the data for the optimization process and returns the
    different arrays for the upcoming optimization.

    Parameters
    ----------
    data: pandas.DataFrame
        Data set to perform the estimation on. Specified
        under dict_["ESTIMATION"]["file"].

    dict_: dict
        Estimation dictionary. Returned by grmpy.read(init_file)).

    Returns
    ------
    D: numpy.array
        Treatment indicator
    X1: numpy.array
        Outcome related regressors of the treated individuals
    X0: numpy.array
        Outcome related regressors of the untreated individuals
    Z1: numpy.array
        Choice related regressors of the treated individuals
    Z0: numpy.array
        Choice related regressors of the untreated individuals
    Y1: numpy.array
        Outcomes of the treated individuals
    Y0: numpy.array
        Outcomes of the untreated individuals
    """
    indicator = dict_["ESTIMATION"]["indicator"]
    outcome = dict_["ESTIMATION"]["dependent"]
    D = data[indicator].values

    data1 = data[data[indicator] == 1]
    data2 = data[data[indicator] == 0]

    X1 = data1[dict_["TREATED"]["order"]].values
    X0 = data2[dict_["UNTREATED"]["order"]].values
    Z1 = data1[dict_["CHOICE"]["order"]].values
    Z0 = data2[dict_["CHOICE"]["order"]].values

    Y1 = data1[outcome].values
    Y0 = data2[outcome].values

    return D, X1, X0, Z1, Z0, Y1, Y0


def process_inputs(dict_):
    """This function processes the specifications related to the optimzation routine.

    Parameters
    ----------
    dict_: dict_
        Estimation dictionary. Returned by grmpy.read(init_file)).

    Returns
    ------
    opt_dict: dict
        Solver options
    method: str
        Algorithm that is used for the minimization
    grad_opt: bool
        Boolean that determines whether the chosen algorithm is derivative based
    start_option: str
        Denotes which start value routine should be used. Options are
        either "init" or "auto".
    print_output: bool
        If True the estimation output is printed
    seed_: int
        Seed value for drawing the random start values for the rho1 and rho0
    """
    try:
        method = dict_["ESTIMATION"]["optimizer"]
    except KeyError:
        method = "BFGS"

    grad_opt = method == "BFGS"

    try:
        opt_dict = dict_["SCIPY-" + method]
    except KeyError:
        opt_dict = {}

    try:
        start_option = dict_["ESTIMATION"]["start"]
    except KeyError:
        start_option = "auto"

    try:
        opt_dict["maxiter"] = dict_["ESTIMATION"]["maxiter"]
        if opt_dict["maxiter"] == 0:
            start_option = "init"
    except KeyError:
        pass

    try:
        print_output = dict_["ESTIMATION"]["print_output"]
    except KeyError:
        print_output = True

    try:
        seed_ = dict_["SIMULATION"]["seed"]
    except KeyError:
        seed_ = randint(0, 9999)

    return opt_dict, method, grad_opt, start_option, print_output, seed_


def create_rslt_df(dict_):
    """This function creates the pandas dataframe container in which the estimation
    rslts are stored.

    Parameters
    ----------
    dict_: dict
        Estimation dictionary. Returned by grmpy.read(init_file)).

    Returns
    ------
    rslt_cont: pandas.DataFrame
        Container for stroing the upcoming estimation results.
    """

    index = []
    # set up multiindex
    for section in ["TREATED", "UNTREATED", "CHOICE"]:
        index += [(section, i) for i in dict_[section]["order"]]
    for subsection in ["sigma1", "rho1", "sigma0", "rho0"]:
        index += [("DIST", subsection)]

    column_names = [
        "params",
        "start_values",
        "std",
        "t_values",
        "p_values",
        "conf_int_low",
        "conf_int_up",
    ]

    return pd.DataFrame(
        index=pd.MultiIndex.from_tuples(index, names=["section", "name"]),
        columns=column_names,
    )


def start_values(dict_, D, X1, X0, Z1, Z0, Y1, Y0, start_option):
    """The function selects the start values for the minimization process. If option is
    set to init the function returns the values that are specified in the initialization
    file. Otherwise the function conducts a Probit estimation for determining the choice
    related parameters as well as two OLS estimations for the outcome related parameters
    associated with the different treatment states. In this case the sigma values are set
    to the sum of residual squares of the particular OLS regression, whereas the rho
    values are drawn randomly. Finally the sigma and rho values are converted by applying
    a method based on Lokshin and Sajaia (2004) independent on the chosen start value
    option.

    Parameters
    ----------
    dict_: dict
        Estimation dictionary. Returned by grmpy.read(init_file)).
    D: numpy.array
        Treatment indicator
    X1: numpy.array
        Outcome related regressors of the treated individuals.
    X0: numpy.array
        Outcome related regressors of the untreated individuals.
    Z1: numpy.array
        Choice related regressors of the treated individuals.
    Z0: numpy.array
        Choice related regressors of the untreated individuals.
    Y1: numpy.array
        Outcomes of the treated individuals.
    Y0: numpy.array
        Outcomes of the untreated individuals.
    start_option: str
        Denotes which start value routine should be used. Options are
        either "init" or "auto".

    Returns
    ------
    x0: numpy.array
        Start values for the estimation routine
    """
    if not isinstance(dict_, dict):
        msg = (
            "The input object ({})for specifing the start values isn`t a "
            "dictionary.".format(dict_)
        )
        raise UserError(msg)

    if start_option == "init":
        # Set coefficients equal the true init file values
        rho1 = dict_["DIST"]["params"][2] / dict_["DIST"]["params"][0]
        rho0 = dict_["DIST"]["params"][4] / dict_["DIST"]["params"][3]
        dist = [dict_["DIST"]["params"][0], rho1, dict_["DIST"]["params"][3], rho0]
        x0 = np.concatenate((dict_["AUX"]["init_values"][:-6], dist))
    elif start_option == "auto":
        try:
            if D.shape[0] == sum(D):
                raise PerfectSeparationError
            # Estimate beta1 and beta0:
            beta = []
            sd_ = []

            for data_out in [(Y1, X1), (Y0, X0)]:
                ols_results = sm.OLS(data_out[0], data_out[1]).fit()
                beta += [ols_results.params]
                sd = np.sqrt(ols_results.scale)
                rho = np.random.uniform(-sd, sd, 1) / sd
                sd_ += [sd, rho[0]]

            # Estimate gamma via Probit
            Z = np.vstack((Z0, Z1))
            probitRslt = sm.Probit(np.sort(D), Z).fit(disp=0)
            gamma = probitRslt.params
            # Adjust estimated cost-benefit shifter and intercept coefficients
            # Arrange starting values
            x0 = np.concatenate((beta[0], beta[1], gamma, sd_))
            check_start_values(x0)

        except (PerfectSeparationError, ValueError, UserError):
            msg = (
                "The estimation process wasn`t able to provide automatic"
                " start values due to perfect seperation. \n"
                " The intialization specifications are used as start "
                "values during the further process."
            )

            # Set coefficients equal the true init file values
            rho1 = dict_["DIST"]["params"][2] / dict_["DIST"]["params"][0]
            rho0 = dict_["DIST"]["params"][4] / dict_["DIST"]["params"][3]
            dist = [dict_["DIST"]["params"][0], rho1, dict_["DIST"]["params"][3], rho0]
            x0 = np.concatenate((dict_["AUX"]["init_values"][:-6], dist))
            dict_["ESTIMATION"]["warning"] = msg
            start_option = "init"

    x0 = start_value_adjustment(x0)
    return np.array(x0)


def start_value_adjustment(x):
    """This function conducts an approach by Lokshin and Sajaia (2004) and takes the
    logarithm of the sigma values as well as the inverse hyperbolic tangent of the speci-
    fied start values for the rhovalues. The transformation will be inversed right within
    the minimization interface function. Through this we ensure that the estimated values
    for sigma are always larger than 0 and that the rho values are bounded between -1 and
    1.

    Parameters
    ----------
    x: numpy.array
        Start values for the estimation routine

    Returns
    ------
    x: numpy.array
        Transformed start values for the estimation routine
    """

    # transform the distributional characteristics s.t. r = log((1-rho)/(1+rho))/2
    x[-4:] = [
        np.log(x[-4]),
        np.log((1 + x[-3]) / (1 - x[-3])) / 2,
        np.log(x[-2]),
        np.log((1 + x[-1]) / (1 - x[-1])) / 2,
    ]

    return x


def backward_transformation(x_trans, bfgs_dict=None):
    """This function reverses the transformation of the sigma and rho values.

    Parameters
    ----------
    x_trans: numpy.array
        Transformed parameter values
    bfgs_dict: dict
        Dictionary that logs the different parameterizations that are
        evaluated during the minimization.
    Returns
    ------
    x_rev: numpy.array
        Reversed parameter values
    """
    x_rev = x_trans.copy()
    x_rev[-4:] = [
        np.exp(x_rev[-4]),
        (np.exp(2 * x_rev[-3]) - 1) / (np.exp(2 * x_rev[-3]) + 1),
        np.exp(x_rev[-2]),
        (np.exp(2 * x_rev[-1]) - 1) / (np.exp(2 * x_rev[-1]) + 1),
    ]
    if bfgs_dict is not None:
        bfgs_dict["parameter"][str(len(bfgs_dict["parameter"]))] = x_rev
    return x_rev


def log_likelihood(
    x0,
    X1,
    X0,
    Z1,
    Z0,
    Y1,
    Y0,
    num_treated,
    num_untreated,
    bfgs_dict=None,
    grad_opt=True,
):
    """This is the the log-likelihood function of our minimization problem.

    Parameters
    ----------
    x0: numpy.array
        Parameter values
    X1: numpy.array
        Outcome related regressors of the treated individuals
    X0: numpy.array
        Outcome related regressors of the untreated individuals
    Z1: numpy.array
        Choice related regressors of the treated individuals
    Z0: numpy.array
        Choice related regressors of the untreated individuals
    Y1: numpy.array
        Outcomes of the treated individuals
    Y0: numpy.array
        Outcomes of the untreated individuals
    bfgs_dict: dict
        Dictionary that logs the different parameterizations that are
        evaluated during the minimization.
    grad_opt: bool
        If True, the function returns not only the likelihood value
        but also the gradient

    Returns
    ------
    likl: float
        Negative log-likelihood value
    llh_grad: numpy.array
        Gradient of the minimization interface, only returned if grad_opt==True
    """

    # assign parameter values
    beta1, beta0, gamma = (
        x0[:num_treated],
        x0[num_treated:num_untreated],
        x0[num_untreated:-4],
    )
    sd1, sd0, rho1v, rho0v = x0[-4], x0[-2], x0[-3], x0[-1]

    nu1 = (Y1 - np.dot(beta1, X1.T)) / sd1
    lambda1 = (np.dot(gamma, Z1.T) - rho1v * nu1) / (np.sqrt(1 - rho1v ** 2))

    nu0 = (Y0 - np.dot(beta0, X0.T)) / sd0
    lambda0 = (np.dot(gamma, Z0.T) - rho0v * nu0) / (np.sqrt(1 - rho0v ** 2))

    treated = (1 / sd1) * norm.pdf(nu1) * norm.cdf(lambda1)
    untreated = (1 / sd0) * norm.pdf(nu0) * (1 - norm.cdf(lambda0))

    likl = -np.mean(np.log(np.append(treated, untreated)))

    if bfgs_dict is not None:
        bfgs_dict["crit"][str(len(bfgs_dict["crit"]))] = likl

    if grad_opt is True:
        llh_grad = gradient(
            X1, X0, Z1, Z0, nu1, nu0, lambda1, lambda0, gamma, sd1, sd0, rho1v, rho0v
        )
        return likl, llh_grad
    else:
        return likl


def calculate_criteria(x0, X1, X0, Z1, Z0, Y1, Y0):
    """The function computes the criterion function value for a given parameter specifi-
    cation.

    Parameters
    ----------
    x0: numpy.array
        Parameter values
    X1: numpy.array
        Outcome related regressors of the treated individuals
    X0: numpy.array
        Outcome related regressors of the untreated individuals
    Z1: numpy.array
        Choice related regressors of the treated individuals
    Z0: numpy.array
        Choice related regressors of the untreated individuals
    Y1: numpy.array
        Outcomes of the treated individuals
    Y0: numpy.array
        Outcomes of the untreated individuals

    Returns
    ------
    crit_value: float
        criterion function value
    """
    x = backward_transformation(x0)
    num_treated = X1.shape[1]
    num_untreated = num_treated + X0.shape[1]

    return log_likelihood(
        x, X1, X0, Z1, Z0, Y1, Y0, num_treated, num_untreated, None, False
    )


def minimizing_interface(
    x0, X1, X0, Z1, Z0, Y1, Y0, num_treated, num_untreated, bfgs_dict, grad_opt
):
    """This function is the objective for the minimization routine. It transforms the
    provided parameters according to the and returns the associated log-likelihood value.

    Parameters
    ----------
    x0: numpy.array
        Parameter values
    X1: numpy.array
        Outcome related regressors of the treated individuals
    X0: numpy.array
        Outcome related regressors of the untreated individuals
    Z1: numpy.array
        Choice related regressors of the treated individuals
    Z0: numpy.array
        Choice related regressors of the untreated individuals
    Y1: numpy.array
        Outcomes of the treated individuals
    Y0: numpy.array
        Outcomes of the untreated individuals
    num_treated: float
        number of regressors of the outcome equation for the treated individuals
    num_untreated: numpy.array
        number of regressors of the outcome equation for the treated
        and untreated individuals
    bfgs_dict: dict
        Dictionary that logs the different parameterizations that are
        evaluated during the minimization.
    grad_opt: bool
        If True, the function returns not only the likelihood value
        but also the gradient
    Returns
    ------
    likl: float
        Negative log-likelihood value
    llh_grad: numpy.array
        Jacobian of the minimization interface, only returned if grad_opt==True
    """

    # transform input parameter vector
    x0 = backward_transformation(x0, bfgs_dict)

    # Calculate likelihood
    return log_likelihood(
        x0, X1, X0, Z1, Z0, Y1, Y0, num_treated, num_untreated, bfgs_dict, grad_opt
    )


def adjust_output(
    opt_rslt,
    dict_,
    rslt_cont,
    start_values,
    method,
    start_option,
    X1,
    X0,
    Z1,
    Z0,
    Y1,
    Y0,
    bfgs_dict=None,
):
    """The function adds different information of the minimization process to the
    estimation output.
    """

    rslt = {
        "opt_info": {
            "optimizer": method,
            "start": start_option,
            "indicator": dict_["ESTIMATION"]["indicator"],
            "dependent": dict_["ESTIMATION"]["dependent"],
            "observations": Y1.shape[0] + Y0.shape[0],
        }
    }
    # Adjust output if
    if opt_rslt["nit"] == 0:
        x = backward_transformation(opt_rslt["x"])
        rslt["opt_info"]["success"], rslt["opt_info"]["status"] = False, 2
        (
            rslt["opt_info"]["message"],
            rslt["opt_info"]["nit"],
            rslt["opt_info"]["crit"],
        ) = ("---", 0, dict_["AUX"]["criteria"])
        rslt["opt_info"]["warning"] = ["---"]

    else:
        # Check if the algorithm has returned the values with the lowest criterium
        # function value
        check, flag = check_rslt_parameters(
            opt_rslt["x"], X1, X0, Z1, Z0, Y1, Y0, bfgs_dict
        )
        # Adjust values if necessary
        if check:
            x, crit, warning = process_output(dict_, bfgs_dict, opt_rslt["x"], flag)
            rslt["opt_info"]["crit"] = crit
            rslt["opt_info"]["warning"] = [warning]

        else:
            x = backward_transformation(opt_rslt["x"])
            rslt["opt_info"]["crit"] = opt_rslt["fun"]
            rslt["opt_info"]["warning"] = ["---"]

        (
            rslt["opt_info"]["success"],
            rslt["opt_info"]["status"],
            rslt["opt_info"]["message"],
            rslt["opt_info"]["nit"],
        ) = map(opt_rslt.get, ["success", "status", "message", "nit"])

    # Adjust Result dict
    rslt_cont["params"] = x

    (
        rslt_cont["std"],
        rslt["hessian_inv"],
        rslt_cont["conf_int_low"],
        rslt_cont["conf_int_up"],
        rslt_cont["p_values"],
        rslt_cont["t_values"],
        warning_se,
    ) = calculate_se(x, dict_, X1, X0, Z1, Z0, Y1, Y0)

    rslt["opt_rslt"] = rslt_cont

    if warning_se is not None:
        rslt["opt_info"]["warning"] += warning_se
    return rslt


def check_rslt_parameters(x0, X1, X0, Z1, Z0, Y1, Y0, bfgs_dict):
    """This function checks if the algorithm has not touched a parameterization during
    the optimization process that leads to a lower criterion function value than the one
    that the minimization routine returned.

    Parameters
    ----------
    x0: numpy.array
        Parameter values
    X1: numpy.array
        Outcome related regressors of the treated individuals
    X0: numpy.array
        Outcome related regressors of the untreated individuals
    Z1: numpy.array
        Choice related regressors of the treated individuals
    Z0: numpy.array
        Choice related regressors of the untreated individuals
    Y1: numpy.array
        Outcomes of the treated individuals
    Y0: numpy.array
        Outcomes of the untreated individuals
    bfgs_dict: dict
        Dictionary that logs the different parameterizations that are
        evaluated during the minimization.

    Returns
    ------
    check: bool
        True if the bfgs_dict contains a parameterization that leads to a smaller
        log-likelihood value or if the array contains nan/nonfinite values
    flag: str
        Flag that indicates whether the array contained nan/nonfinite values or if there
        the algorithm reached a parameterization that leads to smaller likelihood values
        than the one that the algorithm converged to.
    """

    crit = calculate_criteria(x0, X1, X0, Z1, Z0, Y1, Y0)
    x = min(bfgs_dict["crit"], key=bfgs_dict["crit"].get)
    if False in np.isfinite(x0).tolist():
        check, flag = True, "notfinite"

    elif bfgs_dict["crit"][str(x)] < crit:
        check, flag = True, "adjustment"

    else:
        check, flag = False, None
    return check, flag


def process_output(init_dict, bfgs_dict, x0, flag):
    """The function checks if the criteria function value is smaller for the
    optimization output as for the start values.

    Parameters
    ----------
    x0: numpy.array
        Parameter values
    X1: numpy.array
        Outcome related regressors of the treated individuals
    X0: numpy.array
        Outcome related regressors of the untreated individuals
    Z1: numpy.array
        Choice related regressors of the treated individuals
    Z0: numpy.array
        Choice related regressors of the untreated individuals
    Y1: numpy.array
        Outcomes of the treated individuals
    Y0: numpy.array
        Outcomes of the untreated individuals
    bfgs_dict: dict
        Dictionary that logs the different parameterizations that are
        evaluated during the minimization.

    Returns
    ------
    check: bool
        True if the bfgs_dict contains a parameterization that leads to a smaller
        log-likelihood value or if the array contains nan/nonfinite values
    flag: str
        Flag that indicates whether the array contained nan/nonfinite values or if there
        the algorithm reached a parameterization that leads to smaller likelihood values
        than the one that the algorithm converged to.
    """

    x = min(bfgs_dict["crit"], key=bfgs_dict["crit"].get)
    if flag == "adjustment":
        if bfgs_dict["crit"][str(x)] < init_dict["AUX"]["criteria"]:
            x0 = bfgs_dict["parameter"][str(x)].tolist()
            crit = bfgs_dict["crit"][str(x)]
            warning = (
                "The optimization algorithm has failed to provide the "
                "parametrization that leads to the minimal criterion function "
                "value.The estimation output is automatically adjusted and "
                "provides the parameterization with the smallest criterion "
                "function value that was reached during the optimization."
            )
        else:
            x0 = x0
            crit = bfgs_dict["crit"][str(x)]
            warning = "NONE"

    elif flag == "notfinite":
        x0 = init_dict["AUX"]["starting_values"]
        crit = init_dict["AUX"]["criteria"]
        warning = (
            "The optimization process is not able to provide finite values. This is "
            "probably due to perfect separation."
        )
    else:
        crit = x
    return x0, crit, warning


def calculate_se(x, maxiter, X1, X0, Z1, Z0, Y1, Y0):
    """This function computes the standard errors of the parameters by approximating the
    Jacobian of the gradient function. Based on that it computes the confidence

    Parameters
    ----------
    x: numpy.array
        Parameter values
    maxiter: float
        maximum number of iterations
    X1: numpy.array
        Outcome related regressors of the treated individuals
    X0: numpy.array
        Outcome related regressors of the untreated individuals
    Z1: numpy.array
        Choice related regressors of the treated individuals
    Z0: numpy.array
        Choice related regressors of the untreated individuals
    Y1: numpy.array
        Outcomes of the treated individuals
    Y0: numpy.array
        Outcomes of the untreated individuals

    Returns
    ------
    se: numpy.array
        Standard errors of the parameters
    hess_inv: numpy.array
        Inverse hessian matrix evaluated at the parameter vector
    conf_interval: numpy.array
        Confidence intervals of the parameters
    p_values: numpy.array
        p-values of the parameters
    t_values: numpy.array
        t-values of the parameters
    warning: str
        Warning message if the approximated hessian matrix is not invertible

    """
    num_ind = Y1.shape[0] + Y0.shape[0]
    x0 = x.copy()
    warning = None

    if maxiter == 0:
        se = [np.nan] * len(x0)
        hess_inv = np.full((len(x0), len(x0)), np.nan)
        conf_interval = [[np.nan, np.nan]] * len(x0)
        p_values, t_values = len(x0) * [np.nan], len(x0) * [np.nan]
    else:
        norm_value = norm.ppf(0.975)
        # Calculate the hessian matrix, check if it is p
        hess = approx_fprime_cs(x0, gradient_hessian, args=(X1, X0, Z1, Z0, Y1, Y0))
        try:
            hess_inv = np.linalg.inv(hess)
            se = np.sqrt(np.diag(hess_inv) / num_ind)
            aux = norm_value * se
            hess_inv = hess_inv
            conf_interval = np.vstack((np.subtract(x0, aux), np.add(x0, aux))).T
            t_values = np.divide(x0, se)
            p_values = 2 * (1 - t.cdf(np.abs(t_values), df=num_ind - len(x0)))

        except LinAlgError:
            se = np.full(len(x0), np.nan)
            hess_inv = np.full((len(x0), len(x0)), np.nan)
            conf_interval = np.full((len(x0), 2), np.nan)
            t_values = np.full(len(x0), np.nan)
            p_values = np.full(len(x0), np.nan)

        # Check if standard errors are defined, if not add warning message

        if False in np.isfinite(se):
            warning = [
                "The estimation process was not able to provide standard errors for"
                " the estimation results, because the approximation of the hessian "
                "matrix leads to a singular Matrix."
            ]

    return (
        se,
        hess_inv,
        conf_interval[:, 0],
        conf_interval[:, 1],
        p_values,
        t_values,
        warning,
    )


def gradient(X1, X0, Z1, Z0, nu1, nu0, lambda1, lambda0, gamma, sd1, sd0, rho1v, rho0v):
    """This function returns the gradient of our minimization interface.

    Parameters
    ----------
    X1: numpy.array
        Outcome related regressors of the treated individuals
    X0: numpy.array
        Outcome related regressors of the untreated individuals
    Z1: numpy.array
        Choice related regressors of the treated individuals
    Z0: numpy.array
        Choice related regressors of the untreated individuals
    Y1: numpy.array
        Outcomes of the treated individuals
    Y0: numpy.array
        Outcomes of the untreated individuals
    nu1: numpy.array
        residual of the outcome equation of the treated individuals divided by sigma 1
    nu0: numpy.array
        residual of the outcome equation of the untreated individuals divided by sigma 1
    lambda1: numpy.array
        (gamma * Z / rho1 * nu1) / sqrt(1 -  rho1 ** 2)
    lambda0: numpy.array
        (gamma * Z / rho0 * nu0) / sqrt(1 -  rho0 ** 2)
    gamma: numpy.array
        Choice related parameters
    sd1: float
        Sigma 1
    sd0: float
        Sigma 0
    rho1v: float
        rho1
    rho0v: float
        rho0

    Returns
    ------
    grad: numpy.array
        Gradient of the minimization interface
    """
    n_obs = X1.shape[0] + X0.shape[0]

    # compute gradient coef for beta 1

    grad_beta1 = np.sum(
        np.einsum(
            "ij, i ->ij",
            X1,
            -(norm.pdf(lambda1) / norm.cdf(lambda1))
            * (rho1v / (np.sqrt(1 - rho1v ** 2) * sd1))
            - nu1 / sd1,
        ),
        0,
    )

    # compute coef for beta 0
    grad_beta0 = np.sum(
        np.einsum(
            "ij, i ->ij",
            X0,
            norm.pdf(lambda0)
            / (1 - norm.cdf(lambda0))
            * (rho0v / (np.sqrt(1 - rho0v ** 2) * sd0))
            - nu0 / sd0,
        ),
        0,
    )
    grad_sd1 = np.sum(
        sd1
        * (
            +1 / sd1
            - (norm.pdf(lambda1) / norm.cdf(lambda1))
            * (rho1v * nu1 / (np.sqrt(1 - rho1v ** 2) * sd1))
            - nu1 ** 2 / sd1
        ),
        keepdims=True,
    )
    grad_sd0 = np.sum(
        sd0
        * (
            +1 / sd0
            + (norm.pdf(lambda0) / (1 - norm.cdf(lambda0)))
            * (rho0v * nu0 / (np.sqrt(1 - rho0v ** 2) * sd0))
            - nu0 ** 2 / sd0
        ),
        keepdims=True,
    )
    grad_rho1v = np.sum(
        (
            -(norm.pdf(lambda1) / norm.cdf(lambda1))
            * ((np.dot(gamma, Z1.T) * rho1v) - nu1)
            / (1 - rho1v ** 2) ** (1 / 2)
        ),
        keepdims=True,
    )

    grad_rho0v = np.sum(
        (
            (norm.pdf(lambda0) / (1 - norm.cdf(lambda0)))
            * ((np.dot(gamma, Z0.T) * rho0v) - nu0)
            / (1 - rho0v ** 2) ** (1 / 2)
        ),
        keepdims=True,
    )

    grad_gamma = +sum(
        np.einsum(
            "ij, i ->ij",
            Z1,
            (norm.pdf(lambda1) / norm.cdf(lambda1)) * 1 / np.sqrt(1 - rho1v ** 2),
        )
    ) - sum(
        np.einsum(
            "ij, i ->ij",
            Z0,
            (norm.pdf(lambda0) / (1 - norm.cdf(lambda0)))
            * (1 / np.sqrt(1 - rho0v ** 2)),
        )
    )

    grad = np.concatenate(
        (
            grad_beta1,
            grad_beta0,
            -grad_gamma,
            grad_sd1,
            grad_rho1v,
            grad_sd0,
            grad_rho0v,
        )
    )

    return grad / n_obs


def gradient_hessian(x0, X1, X0, Z1, Z0, Y1, Y0):
    """This function computes the gradient of our log-likelihood function at a given
    paramterization x0. The function is used to approximate the hessian matrix for the
    calculation of the standard errors.

    Parameters
    ----------
    x0: numpy.array
        Parameter values
    X1: numpy.array
        Outcome related regressors of the treated individuals
    X0: numpy.array
        Outcome related regressors of the untreated individuals
    Z1: numpy.array
        Choice related regressors of the treated individuals
    Z0: numpy.array
        Choice related regressors of the untreated individuals
    Y1: numpy.array
        Outcomes of the treated individuals
    Y0: numpy.array
        Outcomes of the untreated individuals

    Returns
    ------
    grad: numpy.array
        Gradient of the log-likelihood function.
    """
    num_treated = X1.shape[1]
    num_untreated = num_treated + X0.shape[1]

    beta1, beta0, gamma = (
        x0[:num_treated],
        x0[num_treated:num_untreated],
        x0[num_untreated:-4],
    )
    sd1, sd0, rho1v, rho0v = x0[-4], x0[-2], x0[-3], x0[-1]

    # compute gradient for beta 1

    nu1 = (Y1 - np.dot(beta1, X1.T)) / sd1
    lambda1 = (np.dot(gamma, Z1.T) - rho1v * nu1) / (np.sqrt(1 - rho1v ** 2))

    nu0 = (Y0 - np.dot(beta0, X0.T)) / sd0
    lambda0 = (np.dot(gamma, Z0.T) - rho0v * nu0) / (np.sqrt(1 - rho0v ** 2))

    grad = gradient(
        X1, X0, Z1, Z0, nu1, nu0, lambda1, lambda0, gamma, sd1, sd0, rho1v, rho0v
    )

    multiplier = np.concatenate(
        (
            np.ones(len(grad[:-4])),
            np.array([1 / sd1, 1 / (1 - rho1v ** 2), 1 / sd0, 1 / (1 - rho0v ** 2)]),
        )
    )

    return multiplier * grad


def prepare_mte_calc(opt_rslt, data):
    """This function construct the marginal treatment effect
    given the optimization results.

    Parameters
    ----------
    opt_rslt: pandas.DataFrame
        Dataframe that contains the optimization results.
    data: pandas.DataFrame
        Data set to perform the estimation on.

    Returns
    ------
    quantiles: numpy.array
        Gradient of the log-likelihood function.
    cov: numpy.array
        Covariance matrix based on the estimation results.
    X: numpy.array
        Array object that contains all covariates that affect the treated
        as well as the untreated outcome.
    b1_b0: numpy.array
        Difference of the coefficients of the treated and the untreated outcome equation.
    """

    quantiles = [0.0001] + np.arange(0.01, 1.0, 0.01).tolist() + [0.9999]

    # create a proper covariance matrix from the estimation results
    dist_params = opt_rslt.loc["DIST", "params"]
    cov = np.zeros((3, 3))
    np.fill_diagonal(cov, np.array([dist_params[0], dist_params[2], 1.0]) ** 2)
    cov[2, 0] = dist_params[0] * dist_params[1]
    cov[2, 1] = dist_params[2] * dist_params[3]

    treated_labels = opt_rslt.loc["TREATED", :].index.values
    untreated_labels = opt_rslt.loc["UNTREATED", :].index.values

    common = np.intersect1d(treated_labels, untreated_labels)
    only_treated = treated_labels[np.where(treated_labels != untreated_labels)[0]]
    only_untreated = untreated_labels[np.where(treated_labels != untreated_labels)[0]]

    # correct the parameter vectors if the outcome equations contain different covariates
    if (len(only_treated) != 0) | (len(only_untreated) != 0):
        beta1 = np.append(
            np.append(
                opt_rslt.loc[("TREATED", common), "params"],
                opt_rslt.loc[("TREATED", only_treated), "params"],
            ),
            np.zeros(len(only_untreated)),
        )

        beta0 = np.append(
            np.append(
                opt_rslt.loc[("UNTREATED", common), "params"],
                np.zeros(len(only_treated)),
            ),
            opt_rslt.loc[("UNTREATED", only_untreated), "params"],
        )

    else:
        beta1 = opt_rslt.loc[("TREATED", common), "params"].values
        beta0 = opt_rslt.loc[("UNTREATED", common), "params"].values

    all_labels = (
        [i for i in treated_labels if i in common]
        + only_treated.tolist()
        + only_untreated.tolist()
    )

    X = data[all_labels]
    b1_b0 = beta1 - beta0

    return quantiles, cov, X, b1_b0, beta1, beta0
