"""
The module provides auxiliary functions for the estimation process.
"""

import copy
from random import randint

import numpy as np
import statsmodels.api as sm
from numpy.linalg import LinAlgError
from scipy.optimize import minimize
from scipy.stats import norm, t
from statsmodels.tools.numdiff import approx_fprime_cs
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from grmpy.check.check import UserError, check_start_values
from grmpy.estimate.estimate_output import print_logfile, write_comparison


def par_fit(dict_, data):
    """The function estimates the coefficients of the simulated data set."""
    # Set seed
    if "SIMULATION" not in dict_ or "seed" not in dict_["SIMULATION"]:
        seed_ = randint(0, 9999)
        np.random.seed(seed_)
    else:
        np.random.seed(dict_["SIMULATION"]["seed"])

    #
    _, X1, X0, Z1, Z0, Y1, Y0 = process_data(data, dict_)

    num_treated = X1.shape[1]
    num_untreated = num_treated + X0.shape[1]

    if dict_["ESTIMATION"]["maxiter"] == 0:
        option = "init"
    else:
        option = dict_["ESTIMATION"]["start"]

    # define starting values
    x0 = start_values(dict_, data, option)
    opts, method, grad_opt = optimizer_options(dict_)
    dict_["AUX"]["criteria"] = calculate_criteria(dict_, X1, X0, Z1, Z0, Y1, Y0, x0)
    dict_["AUX"]["starting_values"] = backward_transformation(x0)
    rslt_dict = bfgs_dict()
    if opts["maxiter"] == 0:
        rslt = adjust_output(None, dict_, x0, X1, X0, Z1, Z0, Y1, Y0, rslt_dict)
    else:
        opt_rslt = minimize(
            minimizing_interface,
            x0,
            args=(
                X1,
                X0,
                Z1,
                Z0,
                Y1,
                Y0,
                num_treated,
                num_untreated,
                rslt_dict,
                grad_opt,
            ),
            method=method,
            options=opts,
            jac=grad_opt,
        )
        rslt = adjust_output(
            opt_rslt, dict_, opt_rslt["x"], X1, X0, Z1, Z0, Y1, Y0, rslt_dict
        )
    # Print Output files
    print_logfile(dict_, rslt)

    if "SIMULATION" in dict_:
        if "comparison" in dict_["ESTIMATION"].keys():
            if dict_["ESTIMATION"]["comparison"] == 0:
                pass
            else:
                write_comparison(data, rslt)
        else:
            write_comparison(data, rslt)
    else:
        rslt.update({"ESTIMATION": {"seed": seed_}})

    return rslt


def process_data(data, dict_):
    """This function process the data for the optimization process"""
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


def start_values(init_dict, data_frame, option):
    """The function selects the start values for the minimization process."""
    if not isinstance(init_dict, dict):
        msg = (
            "The input object ({})for specifing the start values isn`t a "
            "dictionary.".format(init_dict)
        )
        raise UserError(msg)
    indicator = init_dict["ESTIMATION"]["indicator"]
    dep = init_dict["ESTIMATION"]["dependent"]

    if option == "init":
        # Set coefficients equal the true init file values
        x0 = init_dict["AUX"]["init_values"][:-6]

    elif option in ["auto", "random"]:

        try:
            if data_frame[indicator].shape[0] == sum(data_frame[indicator]):
                raise PerfectSeparationError
            # Estimate beta1 and beta0:
            beta = []
            sd_ = []

            for i in [1.0, 0.0]:
                Y = data_frame[dep][data_frame[indicator] == i]
                if i == 1:
                    order = init_dict["TREATED"]["order"]
                else:
                    order = init_dict["UNTREATED"]["order"]
                X = data_frame[order][i == data_frame[indicator]]
                ols_results = sm.OLS(Y, X).fit()
                beta += [ols_results.params]
                if option == "random":
                    rho = np.random.uniform(
                        -np.sqrt(ols_results.scale), np.sqrt(ols_results.scale), 1
                    ) / np.sqrt(ols_results.scale)
                    sd_ += [np.sqrt(ols_results.scale), rho[0]]
                elif option == "auto":
                    sd_ += [np.sqrt(ols_results.scale), 0]

            # Estimate gamma via Probit
            Z = data_frame[init_dict["CHOICE"]["order"]]
            probitRslt = sm.Probit(data_frame[indicator], Z).fit(disp=0)
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
            x0 = init_dict["AUX"]["init_values"][:-6]
            init_dict["ESTIMATION"]["warning"] = msg
            option = "init"

    x0 = start_value_adjustment(x0, init_dict, option)
    x0 = np.array(x0)
    return x0


def start_value_adjustment(x, init_dict, option):
    """This function transforms the rho values so that they are always between zero and 1
    Additionally it transforms the sigma values so that all values are always positive.
    """

    # if option = init the estimation process takes its distributional arguments from the
    # inititialization dict
    if option == "init":
        rho1 = init_dict["DIST"]["params"][2] / init_dict["DIST"]["params"][0]
        rho0 = init_dict["DIST"]["params"][4] / init_dict["DIST"]["params"][3]
        dist = [
            init_dict["DIST"]["params"][0],
            rho1,
            init_dict["DIST"]["params"][3],
            rho0,
        ]
        x = np.concatenate((x, dist))

    elif option == "random":
        sds = np.random.uniform(0, 0.5, 2)
        rho1 = np.random.uniform(-sds[0], sds[0], 1) / sds[0]
        rho0 = np.random.uniform(-sds[1], sds[1], 1) / sds[1]
        x = np.concatenate((x, [sds[0], rho1[0], sds[1], rho0[0]]))

    # transform the distributional characteristics s.t. r = log((1-rho)/(1+rho))/2
    x[-4:] = [
        np.log(x[-4]),
        np.log((1 + x[-3]) / (1 - x[-3])) / 2,
        np.log(x[-2]),
        np.log((1 + x[-1]) / (1 - x[-1])) / 2,
    ]

    return x


def backward_transformation(x0, dict_=None):
    """The function generates a dictionary for the representation of the optimization
     output.
     """
    x = x0.copy()
    x[-4:] = [
        np.exp(x[-4]),
        (np.exp(2 * x[-3]) - 1) / (np.exp(2 * x[-3]) + 1),
        np.exp(x[-2]),
        (np.exp(2 * x[-1]) - 1) / (np.exp(2 * x[-1]) + 1),
    ]
    if dict_ is None:
        pass
    else:
        dict_["parameter"][str(len(dict_["parameter"]))] = x
    return x


def log_likelihood(
    x0, X1, X0, Z1, Z0, Y1, Y0, num_treated, num_untreated, dict_=None, grad_opt=True
):
    """The function provides the log-likelihood function for the minimization process."""

    beta1, beta0, gamma = (
        x0[:num_treated],
        x0[num_treated:num_untreated],
        x0[num_untreated:-4],
    )
    sd1, sd0, rho1v, rho0v = x0[-4], x0[-2], x0[-3], x0[-1]
    # Provide parameterization for D=1 and D=0 and provide auxiliary list likl

    nu1 = (Y1 - np.dot(beta1, X1.T)) / sd1
    lambda1 = (np.dot(gamma, Z1.T) - rho1v * nu1) / (np.sqrt((1 - rho1v ** 2)))

    nu0 = (Y0 - np.dot(beta0, X0.T)) / sd0
    lambda0 = (np.dot(gamma, Z0.T) - rho0v * nu0) / (np.sqrt((1 - rho0v ** 2)))

    treated = (1 / sd1) * norm.pdf(nu1) * norm.cdf(lambda1)
    untreated = (1 / sd0) * norm.pdf(nu0) * (1 - norm.cdf(lambda0))

    likl = -np.mean(np.log(np.append(treated, untreated)))
    if dict_ is None:
        pass
    else:
        dict_["crit"][str(len(dict_["crit"]))] = likl

    if grad_opt is True:
        llh_grad = gradient(
            X1, X0, Z1, Z0, nu1, nu0, lambda1, lambda0, gamma, sd1, sd0, rho1v, rho0v
        )
        return likl, llh_grad
    else:
        return likl


def calculate_criteria(init_dict, X1, X0, Z1, Z0, Y1, Y0, x0):
    """The function calculates the criteria function value."""
    x = backward_transformation(x0)
    num_treated = init_dict["AUX"]["num_covars_treated"]
    num_untreated = num_treated + init_dict["AUX"]["num_covars_untreated"]
    criteria = log_likelihood(
        x, X1, X0, Z1, Z0, Y1, Y0, num_treated, num_untreated, None, False
    )
    return criteria


def optimizer_options(init_dict_):
    """The function provides the optimizer options given the initialization
    dictionary.
    """
    method = init_dict_["ESTIMATION"]["optimizer"]
    opt_dict = init_dict_["SCIPY-" + method]
    opt_dict["maxiter"] = init_dict_["ESTIMATION"]["maxiter"]
    grad_opt = method == "BFGS"
    return opt_dict, method, grad_opt


def minimizing_interface(
    x0, X1, X0, Z1, Z0, Y1, Y0, num_treated, num_untreated, dict_, grad_opt
):
    """The function provides the minimization interface for the estimation process."""
    # Collect arguments
    x0 = backward_transformation(x0, dict_)
    # Calculate likelihood for pre-specified arguments

    return log_likelihood(
        x0, X1, X0, Z1, Z0, Y1, Y0, num_treated, num_untreated, dict_, grad_opt
    )


def adjust_output(opt_rslt, init_dict, x0, X1, X0, Z1, Z0, Y1, Y0, dict_=None):
    """The function adds different information of the minimization process to the
    estimation output.
    """
    num_treated = init_dict["AUX"]["num_covars_treated"]
    num_untreated = num_treated + init_dict["AUX"]["num_covars_untreated"]
    rslt = copy.deepcopy(init_dict)
    rslt["ESTIMATION"]["start values"] = init_dict["ESTIMATION"]["start"]
    rslt["AUX"] = {}
    rslt["observations"] = Y1.shape[0] + Y0.shape[0]
    # Adjust output if
    if init_dict["ESTIMATION"]["maxiter"] == 0:
        x = backward_transformation(x0)
        rslt["success"], rslt["status"] = False, 2
        rslt["message"], rslt["nfev"], rslt["crit"] = (
            "---",
            0,
            init_dict["AUX"]["criteria"],
        )
        rslt["warning"] = ["---"]

    else:
        # Check if the algorithm has returned the values with the lowest criterium
        # function value
        check, flag = check_rslt_parameters(
            init_dict, X1, X0, Z1, Z0, Y1, Y0, dict_, x0
        )
        # Adjust values if necessary
        if check:
            x, crit, warning = process_output(init_dict, dict_, x0, flag)
            rslt["crit"] = crit
            rslt["warning"] = [warning]

        else:
            x = backward_transformation(x0)
            rslt["crit"] = opt_rslt["fun"]
            rslt["warning"] = ["---"]

        rslt["success"], rslt["status"] = opt_rslt["success"], opt_rslt["status"]
        rslt["message"], rslt["nfev"] = opt_rslt["message"], opt_rslt["nfev"]

    # Adjust Result dict
    rslt["AUX"]["x_internal"] = x
    rslt["AUX"]["init_values"] = init_dict["AUX"]["init_values"]

    rslt["AUX"]["standard_errors"], rslt["AUX"]["hess_inv"], rslt["AUX"][
        "confidence_intervals"
    ], rslt["AUX"]["p_values"], rslt["AUX"]["t_values"], warning_se = calculate_se(
        x, init_dict, X1, X0, Z1, Z0, Y1, Y0, num_treated, num_untreated
    )

    rslt["TREATED"]["params"] = np.array(x[:num_treated])
    rslt["TREATED"]["starting_values"] = np.array(
        init_dict["AUX"]["starting_values"][:num_treated]
    )

    rslt["TREATED"]["standard_errors"] = np.array(
        rslt["AUX"]["standard_errors"][:num_treated]
    )
    rslt["TREATED"]["confidence_intervals"] = np.array(
        rslt["AUX"]["confidence_intervals"][:num_treated]
    )
    rslt["TREATED"]["p_values"] = np.array(rslt["AUX"]["p_values"][:num_treated])
    rslt["TREATED"]["t_values"] = np.array(rslt["AUX"]["t_values"][:num_treated])

    rslt["UNTREATED"]["params"] = np.array(x[num_treated:num_untreated])
    rslt["UNTREATED"]["starting_values"] = np.array(
        init_dict["AUX"]["starting_values"][num_treated:num_untreated]
    )

    rslt["UNTREATED"]["standard_errors"] = np.array(
        rslt["AUX"]["standard_errors"][num_treated:num_untreated]
    )
    rslt["UNTREATED"]["confidence_intervals"] = np.array(
        rslt["AUX"]["confidence_intervals"][num_treated:num_untreated]
    )
    rslt["UNTREATED"]["p_values"] = np.array(
        rslt["AUX"]["p_values"][num_treated:num_untreated]
    )
    rslt["UNTREATED"]["t_values"] = np.array(
        rslt["AUX"]["t_values"][num_treated:num_untreated]
    )

    rslt["CHOICE"]["params"] = np.array(x[num_untreated:-4])
    rslt["CHOICE"]["starting_values"] = np.array(
        init_dict["AUX"]["starting_values"][num_untreated:-4]
    )

    rslt["CHOICE"]["standard_errors"] = np.array(
        rslt["AUX"]["standard_errors"][num_untreated:-4]
    )
    rslt["CHOICE"]["confidence_intervals"] = np.array(
        rslt["AUX"]["confidence_intervals"][num_untreated:-4]
    )
    rslt["CHOICE"]["p_values"] = np.array(rslt["AUX"]["p_values"][num_untreated:-4])
    rslt["CHOICE"]["t_values"] = np.array(rslt["AUX"]["t_values"][num_untreated:-4])

    rslt["DIST"]["params"] = np.array(x[-4:])
    rslt["DIST"]["starting_values"] = np.array(init_dict["AUX"]["starting_values"][-4:])

    rslt["DIST"]["order"] = ["sigma1", "rho1", "sigma0", "rho0"]
    rslt["DIST"]["standard_errors"] = np.array(rslt["AUX"]["standard_errors"][-4:])
    rslt["DIST"]["confidence_intervals"] = np.array(
        rslt["AUX"]["confidence_intervals"][-4:]
    )
    rslt["DIST"]["p_values"] = np.array(rslt["AUX"]["p_values"][-4:])
    rslt["DIST"]["t_values"] = np.array(rslt["AUX"]["t_values"][-4:])
    for subkey in [
        "num_covars_choice",
        "num_covars_treated",
        "num_covars_untreated",
        "num_paras",
        "num_covars",
        "labels",
    ]:
        rslt["AUX"][subkey] = init_dict["AUX"][subkey]
    if warning_se is not None:
        rslt["warning"] += warning_se
    return rslt


def check_rslt_parameters(init_dict, X1, X0, Z1, Z0, Y1, Y0, dict_, x0):
    """This function checks if the algorithms has provided a parameterization with a
    lower criterium function value.
     """
    crit = calculate_criteria(init_dict, X1, X0, Z1, Z0, Y1, Y0, x0)
    x = min(dict_["crit"], key=dict_["crit"].get)
    if False in np.isfinite(x0).tolist():
        check, flag = True, "notfinite"

    elif dict_["crit"][str(x)] < crit:
        check, flag = True, "adjustment"

    else:
        check, flag = False, None
    return check, flag


def process_output(init_dict, dict_, x0, flag):
    """The function checks if the criteria function value is smaller for the
    optimization output as for the start values.
    """

    x = min(dict_["crit"], key=dict_["crit"].get)
    if flag == "adjustment":
        if dict_["crit"][str(x)] < init_dict["AUX"]["criteria"]:
            x0 = dict_["parameter"][str(x)].tolist()
            crit = dict_["crit"][str(x)]
            warning = (
                "The optimization algorithm has failed to provide the parametrization "
                "that leads to the minimal criterion function value. \n"
                "                         "
                "                  The estimation output is automatically "
                "adjusted and provides the parameterization with the smallest "
                "criterion function value \n                         "
                "                  that was reached during the optimization.\n"
            )
        else:
            x0 = x0
            crit = dict_["crit"][str(x)]
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


def bfgs_dict():
    """The function provides a dictionary for tracking the criteria function values and
    the associated parametrization.
    """
    rslt_dict = {"parameter": {}, "crit": {}, "grad": {}}

    return rslt_dict


def calculate_se(x, init_dict, X1, X0, Z1, Z0, Y1, Y0, num_treated, num_untreated):
    """This function calculates the standard errors for given parameterization via an
    approximation of the hessian matrix.
    """
    num_ind = Y1.shape[0] + Y0.shape[0]
    x0 = x.copy()
    warning = None

    if init_dict["ESTIMATION"]["maxiter"] == 0:
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

            upper = np.add(x0, aux)
            lower = np.subtract(x0, aux)

            se = se.copy()
            hess_inv = hess_inv
            conf_interval = [[lower[i], upper[i]] for i in range(len(lower))]
            p_values, t_values = calculate_p_values(se, x0, num_ind)

        except LinAlgError:
            se = [np.nan] * len(x0)
            hess_inv = np.full((len(x0), len(x0)), np.nan)
            conf_interval = [[np.nan, np.nan]] * len(x0)
            t_values = len(se) * [np.nan]
            p_values = len(se) * [np.nan]

        # Check if standard errors are defined, if not add warning message

        if False in np.isfinite(se):
            warning = [
                "The estimation process was not able to provide standard errors for"
                " the estimation results, because the approximation \n            "
                "                               of the hessian matrix "
                "leads to a singular Matrix.\n"
            ]

    return se, hess_inv, conf_interval, p_values, t_values, warning


def calculate_p_values(se, x0, num_ind):
    """This function calculates the p values, given the estimation results and the
    standard errors.
    """
    df = num_ind - len(x0)
    t_values = np.divide(x0, se)
    p_values = 2 * (1 - t.cdf(np.abs(t_values), df=df))

    return p_values, t_values


def gradient(X1, X0, Z1, Z0, nu1, nu0, lambda1, lambda0, gamma, sd1, sd0, rho1v, rho0v):
    """This function returns the jacobian of our Maximum Likelihood function."""
    n = X1.shape[0] + X0.shape[0]

    # compute gradient coef for beta 1

    grad_beta1 = (norm.pdf(lambda1) / norm.cdf(lambda1)) * (
        rho1v / (np.sqrt(1 - rho1v ** 2) * sd1)
    ) + nu1 / sd1

    # compute coef for beta 0
    grad_beta0 = (
        -norm.pdf(lambda0)
        / (1 - norm.cdf(lambda0))
        * (rho0v / (np.sqrt(1 - rho0v ** 2) * sd0))
        + nu0 / sd0
    )

    grad_sd1 = sd1 * (
        +1 / sd1
        - (norm.pdf(lambda1) / norm.cdf(lambda1))
        * (rho1v * nu1 / (np.sqrt(1 - rho1v ** 2) * sd1))
        - nu1 ** 2 / sd1
    )
    grad_sd0 = sd0 * (
        +1 / sd0
        + (norm.pdf(lambda0) / (1 - norm.cdf(lambda0)))
        * (rho0v * nu0 / (np.sqrt(1 - rho0v ** 2) * sd0))
        - nu0 ** 2 / sd0
    )
    grad_rho1v = (
        -(norm.pdf(lambda1) / norm.cdf(lambda1))
        * ((np.dot(gamma, Z1.T) * rho1v) - nu1)
        / (1 - rho1v ** 2) ** (1 / 2)
    )
    grad_rho0v = (
        (norm.pdf(lambda0) / (1 - norm.cdf(lambda0)))
        * ((np.dot(gamma, Z0.T) * rho0v) - nu0)
        / (1 - rho0v ** 2) ** (1 / 2)
    )

    grad_gamma = sum(
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

    grad = np.sum(np.einsum("ij, i ->ij", X1, -grad_beta1), 0)
    grad = np.append(grad, np.sum(np.einsum("ij, i ->ij", X0, -grad_beta0), 0))
    grad = np.append(grad, -grad_gamma)
    grad = np.append(
        grad,
        np.append(np.sum([grad_sd1, grad_rho1v], 1), np.sum([grad_sd0, grad_rho0v], 1)),
    )

    return grad / n


def gradient_hessian(x0, X1, X0, Z1, Z0, Y1, Y0):

    num_treated = X1.shape[1]
    num_untreated = num_treated + X0.shape[1]

    beta1, beta0, gamma = (
        x0[:num_treated],
        x0[num_treated:num_untreated],
        x0[num_untreated:-4],
    )
    sd1, sd0, rho1v, rho0v = x0[-4], x0[-2], x0[-3], x0[-1]

    n_obs = Y1.shape[0] + Y0.shape[0]
    # compute gradient for beta 1

    nu1 = (Y1 - np.dot(beta1, X1.T)) / sd1
    lambda1 = (np.dot(gamma, Z1.T) - rho1v * nu1) / (np.sqrt((1 - rho1v ** 2)))

    nu0 = (Y0 - np.dot(beta0, X0.T)) / sd0
    lambda0 = (np.dot(gamma, Z0.T) - rho0v * nu0) / (np.sqrt((1 - rho0v ** 2)))

    grad_beta1 = (norm.pdf(lambda1) / norm.cdf(lambda1)) * (
        rho1v / (np.sqrt(1 - rho1v ** 2) * sd1)
    ) + nu1 / sd1

    grad_beta0 = (
        -norm.pdf(lambda0)
        / (1 - norm.cdf(lambda0))
        * (rho0v / (np.sqrt(1 - rho0v ** 2) * sd0))
        + nu0 / sd0
    )

    grad_sd1 = (
        +1 / sd1
        - (norm.pdf(lambda1) / norm.cdf(lambda1))
        * (rho1v * nu1 / (np.sqrt(1 - rho1v ** 2) * sd1))
        - nu1 ** 2 / sd1
    )
    grad_sd0 = (
        +1 / sd0
        + (norm.pdf(lambda0) / (1 - norm.cdf(lambda0)))
        * (rho0v * nu0 / (np.sqrt(1 - rho0v ** 2) * sd0))
        - nu0 ** 2 / sd0
    )
    grad_rho1v = (
        -(norm.pdf(lambda1) / norm.cdf(lambda1))
        * ((np.dot(gamma, Z1.T) * rho1v) - nu1)
        / (1 - rho1v ** 2) ** (3 / 2)
    )
    grad_rho0v = (
        (norm.pdf(lambda0) / (1 - norm.cdf(lambda0)))
        * ((np.dot(gamma, Z0.T) * rho0v) - nu0)
        / (1 - rho0v ** 2) ** (3 / 2)
    )

    grad_gamma = np.sum(
        np.einsum(
            "ij, i ->ij",
            Z1,
            (norm.pdf(lambda1) / norm.cdf(lambda1)) * 1 / np.sqrt(1 - rho1v ** 2),
        ),
        0,
    ) - sum(
        np.einsum(
            "ij, i ->ij",
            Z0[:, :],
            (norm.pdf(lambda0) / (1 - norm.cdf(lambda0)))
            * (1 / np.sqrt(1 - rho0v ** 2)),
        )
    )

    grad = np.sum(np.einsum("ij, i ->ij", X1, -grad_beta1), 0)
    grad = np.append(grad, np.sum(np.einsum("ij, i ->ij", X0, -grad_beta0), 0))
    grad = np.append(grad, -grad_gamma)
    grad = np.append(
        grad,
        np.append(np.sum([grad_sd1, grad_rho1v], 1), np.sum([grad_sd0, grad_rho0v], 1)),
    )

    return grad / n_obs
