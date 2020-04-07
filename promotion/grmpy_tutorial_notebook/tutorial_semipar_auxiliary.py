"""This module provides auxiliary functions for the semiparametric tutorial"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.utils import resample

from grmpy.check.auxiliary import read_data
from grmpy.check.check import check_presence_init
from grmpy.estimate.estimate_semipar import (
    estimate_treatment_propensity,
    process_choice_data,
    mte_components,
    process_default_input,
    process_user_input,
    trim_support,
)
from grmpy.read.read import read

# surpress pandas warning
pd.options.mode.chained_assignment = None


def plot_semipar_mte(rslt, init_file, nbootstraps=250):
    """This function plots the original and the replicated MTE
    from Carneiro et al. (2011)"""
    # mte per year of university education
    mte = rslt["mte"] / 4
    quantiles = rslt["quantiles"]

    # bootstrap 90 percent confidence bands
    mte_boot = bootstrap(init_file, nbootstraps)

    # mte per year of university education
    mte_boot = mte_boot / 4

    # Get standard error of MTE at each gridpoint u_D
    mte_boot_std = np.std(mte_boot, axis=1)

    # Compute 90 percent confidence intervals
    con_u = mte + norm.ppf(0.95) * mte_boot_std
    con_d = mte - norm.ppf(0.95) * mte_boot_std

    # Load original data
    mte_ = pd.read_csv("data/mte_semipar_original.csv")

    # Plot
    ax = plt.figure(figsize=(17.5, 10)).add_subplot(111)

    ax.set_ylabel(r"$MTE$", fontsize=20)
    ax.set_xlabel("$u_D$", fontsize=20)
    ax.tick_params(
        axis="both", direction="in", length=5, width=1, grid_alpha=0.25, labelsize=14
    )
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    ax.xaxis.set_ticks(np.arange(0, 1.1, step=0.1))
    ax.yaxis.set_ticks(np.arange(-1.8, 0.9, step=0.2))

    ax.set_ylim([-0.77, 0.86])
    ax.set_xlim([0, 1])
    ax.margins(x=0.003)
    ax.margins(y=0.03)

    # Plot replicated curves
    ax.plot(quantiles, mte, label="replicated $MTE$", color="orange", linewidth=4)
    ax.plot(quantiles, con_u, color="orange", linestyle=":", linewidth=3)
    ax.plot(quantiles, con_d, color="orange", linestyle=":", linewidth=3)

    # Plot original curve
    ax.plot(
        mte_["quantiles"],
        mte_["mte"],
        label="$original MTE$",
        color="blue",
        linewidth=4,
    )
    ax.plot(mte_["quantiles"], mte_["con_u"], color="blue", linestyle=":", linewidth=3)
    ax.plot(mte_["quantiles"], mte_["con_d"], color="blue", linestyle=":", linewidth=3)

    blue_patch = mpatches.Patch(color="orange", label="replicated $MTE$")
    orange_patch = mpatches.Patch(color="blue", label="original $MTE$")
    plt.legend(handles=[blue_patch, orange_patch], prop={"size": 16})

    plt.show()

    return mte, quantiles


def bootstrap(init_file, nbootstraps):
    """
    This function generates bootsrapped standard errors
    given an init_file and the number of bootsraps to be drawn.
    """
    check_presence_init(init_file)
    dict_ = read(init_file)

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

        if isinstance(ps, np.ndarray):  # & (np.min(ps) <= 0.3) & (np.max(ps) >= 0.7):
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
