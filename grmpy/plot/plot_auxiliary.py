"""
This module provides auxiliary functions for the plot_mte function.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.utils import resample

from grmpy.check.auxiliary import read_data
from grmpy.check.check import check_presence_init
from grmpy.estimate.estimate_semipar import (
    double_residual_reg,
    estimate_treatment_propensity,
    mte_observed,
    mte_unobserved_semipar,
    process_primary_inputs,
    process_secondary_inputs,
    trim_support,
)
from grmpy.read.read import read

# surpress pandas warning
pd.options.mode.chained_assignment = None


def plot_curve(mte, quantiles, con_u, con_d, font_size, label_size, color, save_plot):
    """
    This function plots the MTE curve (based on either the
    parametric or semiparmaetric model) along with its
    90 percent confidence bands.

    Parameters
    ----------
    mte: np.ndarray
        Estimate of the parametric MTE.
    quantiles: np.ndarray
        Quantiles of the u_D, along which the *mte* has been estimated.
    con_u: list
        Upper bound of the 90 percent confidence interval.
    con_d: list
        Lower bound of the 90 percent confidence interval.
    font_size: int, default is 22
        Font size of the MTE graph.
    label_size: int, default is 16
        Label size of the MTE graph
    color: str, default is "blue"
        Color of the MTE curve.
    save_plot: bool or str or PathLike or file-like object, default is False
        If False, the resulting plot is shown but not saved.
        If True, the MTE plot is saved as 'MTE_plot.png'.
        Else, if a str or Pathlike or file-like object is specified,
        the plot is saved according to *save_plot*.
        The output format is inferred from the extension ('png', 'pdf', 'svg'... etc.)
        By default, '.png' is assumed.
    """
    ax = plt.figure(figsize=(17.5, 10)).add_subplot(111)

    ax.set_ylabel(r"$MTE$", fontsize=font_size)
    ax.set_xlabel("$u_D$", fontsize=font_size)
    ax.tick_params(
        axis="both",
        direction="in",
        length=5,
        width=1,
        grid_alpha=0.25,
        labelsize=label_size,
    )
    # TODO: trim the y axis based on the results
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")

    ax.plot(quantiles, mte, color=color, linewidth=4)
    ax.plot(quantiles, con_u, color=color, linestyle=":", linewidth=3)
    ax.plot(quantiles, con_d, color=color, linestyle=":", linewidth=3)

    if save_plot is False:
        pass
    elif save_plot is True:
        plt.savefig("MTE_plot.png", dpi=300)
    else:
        plt.savefig(save_plot, dpi=300)

    plt.show()


def mte_and_cof_int_semipar(rslt, init_file, college_years, nboot):
    """
    This function returns the semiparametric MTE divided by the number
    of college years, which represents the returns per YEAR of
    post-secondary schooling.
    The corresponding 90 percent confidence intervals are bootstrapped
    based on *nboot* iterations.

    Parameters
    ----------
    rslt: dict
        Result dictionary containing parameters for the estimation
        process.
    init_file: yaml
        Initialization file containing parameters for the estimation
        process.
    college_years: int, default is 4
        Average duration of college degree. The MTE plotted will thus
        refer to the returns per one year of college education.
    nboot: int
        Number of bootstrap iterations, i.e. number of times
        the MTE is computed via bootstrap.

    Returns
    -------
    quantiles: np.ndarray
        Quantiles of the u_D, along which the *mte* has been estimated.
    mte: np.ndarray
        Estimate of the parametric MTE.
    con_u: list
        Upper bound of the 90 percent confidence interval.
    con_d: list
        Lower bound of the 90 percent confidence interval.
    """
    # Define quantiles of u_D (unobserved resistance to treatment)
    quantiles = rslt["quantiles"]

    # MTE per year of post-secondary education
    mte = rslt["mte"] / college_years

    # bootstrap 90 percent confidence bands
    mte_boot = bootstrap(init_file, nboot)
    mte_boot = mte_boot / college_years

    # Get standard error of MTE at each gridpoint u_D
    mte_boot_std = np.std(mte_boot, axis=1)

    # Compute 90 percent confidence intervals
    con_u = mte + norm.ppf(0.95) * mte_boot_std
    con_d = mte - norm.ppf(0.95) * mte_boot_std

    return quantiles, mte, con_u, con_d


def mte_and_cof_int_par(rslt, data, college_years):
    """
    This function returns the parametric MTE divided by the number
    of college years, which represents the returns per YEAR of
    post-secondary schooling.
    90 percent confidence intervals are computed analytically.

    Parameters
    ----------
    rslt: dict
        Result dictionary containing parameters for the estimation
        process.
    data: pandas.DataFrame
        Data set containing the observables (explanatory and outcome variables)
        analyzed in the generalized Roy framework.
    college_years: int, default is 4
        Average duration of college degree. The MTE plotted will thus
        refer to the returns per one year of college education.
    """
    # Define quantiles of u_D (unobserved resistance to treatment)
    quantiles = rslt["quantiles"]

    # MTE per year of post-secondary education
    mte = rslt["mte"] / college_years

    con_u, con_d = calculate_cof_int(rslt, data, quantiles)

    return quantiles, mte, mte + con_u / college_years, mte - con_d / college_years


def calculate_cof_int(rslt, data, quantiles):
    """
    This function calculates the analytical confidence intervals of
    the parametric marginal treatment effect.

    Parameters
    ----------
    rslt: dict
        Result dictionary containing parameters for the estimation
        process.
    data: pandas.DataFrame
        Data set containing the observables (explanatory and outcome variables)
        analyzed in the generalized Roy framework.
    quantiles: np.ndarray
        Quantiles of the u_D, along which the *mte* has been estimated.

    Returns
    ------
    con_u: list
        Upper bound of the 90 percent confidence interval.
    con_d: list
        Lower bound of the 90 percent confidence interval.
    """

    # Import parameters and inverse hessian matrix
    hess_inv = rslt["hessian_inv"] / data.shape[0]
    params = rslt["opt_rslt"]["params"].values
    numx = (
        rslt["opt_rslt"].loc["TREATED"].shape[0]
        + rslt["opt_rslt"].loc["UNTREATED"].shape[0]
    )

    # Distribute parameters
    dist_cov = hess_inv[-4:, -4:]
    param_cov = hess_inv[:numx, :numx]
    dist_gradients = params[-4:]

    # Process data

    # goal should be to take into account that the treated and the
    # untreated section can contain different covariates
    x_treated = np.mean(data[rslt["opt_rslt"].loc["TREATED"].index.values]).values
    x_untreated = np.mean(data[rslt["opt_rslt"].loc["UNTREATED"].index.values]).values
    x = np.append(x_treated, -x_untreated)

    # Create auxiliary parameters
    part1 = np.dot(x, np.dot(param_cov, x))
    part2 = np.dot(dist_gradients, np.dot(dist_cov, dist_gradients))

    # Prepare two lists for storing the values

    # Combine all auxiliary parameters and calculate the confidence intervals
    value = part2 * (norm.ppf(quantiles)) ** 2
    aux = np.sqrt(part1 + value)
    con_u = norm.ppf(0.95) * aux
    con_d = norm.ppf(0.95) * aux

    return con_u, con_d


def bootstrap(init_file, nboot):
    """
    This function generates bootsrapped standard errors
    given an init_file and the number of bootstraps to be drawn.

    Parameters
    ----------
    init_file: yaml
        Initialization file containing parameters for the estimation
        process.
    nboot: int
        Number of bootstrap iterations, i.e. number of times
        the MTE is computed via bootstrap.

    Returns
    -------
    mte_boot: np.ndarray
        Array containing *nbootstrap* estimates of the MTE.
    """
    check_presence_init(init_file)
    dict_ = read(init_file, semipar=True)

    # Process the information specified in the initialization file
    bins, logit, bandwidth, gridsize, startgrid, endgrid = process_primary_inputs(dict_)
    trim, rbandwidth, reestimate_p, show_output = process_secondary_inputs(dict_)

    # Suppress output
    show_output = False

    # Prepare empty array to store output values
    mte_boot = np.zeros([gridsize, nboot])

    # Load the baseline data
    data = read_data(dict_["ESTIMATION"]["file"])

    counter = 0
    while counter < nboot:
        boot_data = resample(data, replace=True, n_samples=len(data), random_state=None)

        # Estimate propensity score P(z)
        boot_data = estimate_treatment_propensity(dict_, boot_data, logit, show_output)
        prop_score = boot_data["prop_score"]

        if isinstance(prop_score, pd.Series):
            # Define common support and trim the data (if trim=True)
            X, Y, prop_score = trim_support(
                dict_, data, logit, bins, trim, reestimate_p, show_output=False
            )

            b0, b1_b0 = double_residual_reg(X, Y, prop_score)

            # # Construct the MTE
            mte_x = mte_observed(X, b1_b0)
            mte_u = mte_unobserved_semipar(
                X, Y, b0, b1_b0, prop_score, bandwidth, gridsize, startgrid, endgrid
            )

            # Put the MTE together
            mte = mte_x + mte_u
            mte_boot[:, counter] = mte

            counter += 1

        else:
            continue

    return mte_boot
