"""This module provides auxiliary functions for the plot_mte function."""

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.utils import resample

from grmpy.read.read import read
from grmpy.check.auxiliary import read_data
from grmpy.check.check import check_presence_init
from grmpy.estimate.estimate_output import calculate_mte

from grmpy.estimate.estimate_semipar import (
    estimate_treatment_propensity,
    process_choice_data,
    mte_components,
    process_default_input,
    process_user_input,
    trim_support,
)

# surpress pandas warning
pd.options.mode.chained_assignment = None


def plot_curve(mte, quantiles, con_u, con_d, font_size, label_size, color, save_plot):
    """This function plots the MTE curve along with the
    90 percent confidence bands.
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
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")

    ax.plot(quantiles, mte, color=color, linewidth=4)
    ax.plot(quantiles, con_u, color=color, linestyle=":", linewidth=3)
    ax.plot(quantiles, con_d, color=color, linestyle=":", linewidth=3)

    if save_plot is not False:
        plt.savefig(save_plot, dpi=300)

    plt.show()


def mte_and_cof_int_semipar(rslt, init_file, college_years, nboot):
    """This function returns the semiparametric MTE divided by the number
     of college years, which represents the returns per YEAR of
     post-secondary schooling.
     The corresponding 90 percent confidence intervals are bootstrapped
     based on 'nboot' iterations.
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


def mte_and_cof_int_par(rslt, init_dict, data, college_years):
    """This function returns the parametric MTE divided by the number
     of college years, which represents the returns per YEAR of
     post-secondary schooling.
     90 percent confidence intervals are computed analytically.
    """
    # Define quantiles of u_D (unobserved resistance to treatment)
    quantiles = [0.0001] + np.arange(0.01, 1.0, 0.01).tolist() + [0.9999]

    # Calculate the MTE and confidence intervals
    mte = calculate_mte(rslt, data, quantiles)
    mte = [i / college_years for i in mte]
    con_u, con_d = calculate_cof_int(rslt, init_dict, data, mte, quantiles)

    return quantiles, mte, con_u, con_d


def calculate_cof_int(rslt, init_dict, data_frame, mte, quantiles):
    """This function calculates the confidence intervals of
    the parametric marginal treatment effect.
    """
    # Import parameters and inverse hessian matrix
    hess_inv = rslt["AUX"]["hess_inv"] / data_frame.shape[0]
    params = rslt["AUX"]["x_internal"]
    numx = len(init_dict["TREATED"]["order"]) + len(init_dict["UNTREATED"]["order"])

    # Distribute parameters
    dist_cov = hess_inv[-4:, -4:]
    param_cov = hess_inv[:numx, :numx]
    dist_gradients = np.array([params[-4], params[-3], params[-2], params[-1]])

    # Process data
    covariates = init_dict["TREATED"]["order"]
    x = np.mean(data_frame[covariates]).tolist()
    x_neg = [-i for i in x]
    x += x_neg
    x = np.array(x)

    # Create auxiliary parameters
    part1 = np.dot(x, np.dot(param_cov, x))
    part2 = np.dot(dist_gradients, np.dot(dist_cov, dist_gradients))

    # Prepare two lists for storing the values
    con_u = []
    con_d = []

    # Combine all auxiliary parameters and calculate the confidence intervals
    for counter, i in enumerate(quantiles):
        value = part2 * (norm.ppf(i)) ** 2
        aux = np.sqrt(part1 + value)
        con_u += [mte[counter] + norm.ppf(0.95) * aux]
        con_d += [mte[counter] - norm.ppf(0.95) * aux]

    return con_u, con_d


def bootstrap(init_file, nbootstraps):
    """
    This function generates bootsrapped standard errors
    given an init_file and the number of bootsraps to be drawn.
    """
    check_presence_init(init_file)
    dict_ = read(init_file, semipar=True)

    # Process the information specified in the initialization file
    nbins, logit, bandwidth, gridsize, a, b = process_user_input(dict_)
    trim, rbandwidth, reestimate_p = process_default_input(dict_)

    # Suppress output
    show_output = False

    # Prepare empty array to store output values
    mte_boot = np.zeros([gridsize, nbootstraps])

    # Load the baseline data
    data = read_data(dict_["ESTIMATION"]["file"])

    counter = 0
    while counter < nbootstraps:
        boot_data = resample(data, replace=True, n_samples=len(data), random_state=None)

        # Process the inputs for the decision equation
        indicator, D, Z = process_choice_data(dict_, boot_data)

        # Estimate propensity score P(z)
        ps = estimate_treatment_propensity(D, Z, logit, show_output)

        if isinstance(ps, np.ndarray):
            # Define common support and trim the data, if trim=True
            boot_data, ps = trim_support(
                dict_,
                boot_data,
                logit,
                ps,
                indicator,
                nbins,
                trim,
                reestimate_p,
                show_output,
            )

            # Estimate the observed and unobserved component of the MTE
            X, b1_b0, b0, mte_u = mte_components(
                dict_, boot_data, ps, rbandwidth, bandwidth, gridsize, a, b, show_output
            )

            # Calculate the MTE component that depends on X
            mte_x = np.dot(X, b1_b0).mean(axis=0)

            # Put the MTE together
            mte = mte_x + mte_u
            mte_boot[:, counter] = mte

            counter += 1

        else:
            continue

    return mte_boot
