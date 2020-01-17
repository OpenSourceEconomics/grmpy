"""This module provides auxiliary functions for the semiparametric tutorial"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from grmpy.KernReg.locpoly import locpoly
from sklearn.utils import resample
from scipy.stats import norm

from grmpy.estimate.estimate_semipar import estimate_treatment_propensity
from grmpy.estimate.estimate_semipar import define_common_support
from grmpy.estimate.estimate_semipar import double_residual_reg
from grmpy.estimate.estimate_semipar import construct_Xp
from grmpy.estimate.estimate_semipar import trim_data

from grmpy.check.check import check_presence_init
from grmpy.check.auxiliary import read_data
from grmpy.read.read import read


def plot_common_support(init_file, nbins, fs=24, output=False):
    """This function plots histograms of the treated and untreated population
    to assess the common support of the propensity score"""
    dict_ = read(init_file)

    # Distribute initialization information.
    data = read_data(dict_["ESTIMATION"]["file"])

    # Process data for the semiparametric estimation.
    indicator = dict_["ESTIMATION"]["indicator"]
    D = data[indicator].values
    Z = data[dict_["CHOICE"]["order"]]
    logit = dict_["ESTIMATION"]["logit"]

    # estimate propensity score
    ps = estimate_treatment_propensity(D, Z, logit, show_output=False)

    data["ps"] = ps

    treated = data[[indicator, "ps"]][data[indicator] == 1].values
    untreated = data[[indicator, "ps"]][data[indicator] == 0].values

    treated = treated[:, 1].tolist()
    untreated = untreated[:, 1].tolist()

    # Make the histogram using a list of lists
    fig = plt.figure(figsize=(17.5, 10))
    plt.hist(
        [treated, untreated],
        bins=nbins,
        weights=[
            np.ones(len(treated)) / len(treated),
            np.ones(len(untreated)) / len(untreated),
        ],
        density=0,
        alpha=0.55,
        label=["Treated", "Untreated"],
    )

    # Plot formatting
    plt.tick_params(axis="both", labelsize=14)
    plt.legend(loc="upper right", prop={"size": 14})
    plt.xticks(np.arange(0, 1.1, step=0.1))
    plt.grid(axis="y", alpha=0.25)
    plt.xlabel("$P$", fontsize=fs)
    plt.ylabel("$f(P)$", fontsize=fs)
    # plt.title('Support of $P(\hat{Z})$ for $D=1$ and $D=0$', fontsize=fs)

    if output is not False:
        plt.savefig(output, dpi=300)

    fig.show()


def plot_semipar_mte(rslt, init_file, nbootstraps=250):
    """This function plots the original and the replicated MTE from Carneiro et al. (2011)"""
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
    mte_ = pd.read_csv(
        "promotion/grmpy_tutorial_notebook/data/mte_semipar_original.csv"
    )

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

    ax.set_ylim([-0.77, 0.86])
    ax.set_xlim([0, 1])

    blue_patch = mpatches.Patch(color="orange", label="replicated $MTE$")
    orange_patch = mpatches.Patch(color="blue", label="original $MTE$")
    plt.legend(handles=[blue_patch, orange_patch], prop={"size": 16})

    plt.show()

    return mte, quantiles


def bootstrap(init_file, nbootstraps, show_output=False):
    """
    This function generates bootsrapped standard errors
    given an init_file and the number of bootsraps to be drawn.
    """
    check_presence_init(init_file)
    dict_ = read(init_file)

    nbins = dict_["ESTIMATION"]["nbins"]
    trim = dict_["ESTIMATION"]["trim_support"]
    rbandwidth = dict_["ESTIMATION"]["rbandwidth"]
    bandwidth = dict_["ESTIMATION"]["bandwidth"]
    gridsize = dict_["ESTIMATION"]["gridsize"]
    a = dict_["ESTIMATION"]["ps_range"][0]
    b = dict_["ESTIMATION"]["ps_range"][1]

    logit = dict_["ESTIMATION"]["logit"]

    # Distribute initialization information.
    data = read_data(dict_["ESTIMATION"]["file"])

    # Prepare empty arrays to store output values
    mte_boot = np.zeros([gridsize, nbootstraps])

    counter = 0
    while counter < nbootstraps:
        boot = resample(data, replace=True, n_samples=len(data), random_state=None)

        # Process data for the semiparametric estimation.
        indicator = dict_["ESTIMATION"]["indicator"]
        D = boot[indicator].values
        Z = boot[dict_["CHOICE"]["order"]]

        # The Local Instrumental Variables (LIV) approach

        # 1. Estimate propensity score P(z)
        ps = estimate_treatment_propensity(D, Z, logit, show_output)

        if isinstance(ps, np.ndarray):  # & (np.min(ps) <= 0.3) & (np.max(ps) >= 0.7):

            # 2a. Find common support
            treated, untreated, common_support = define_common_support(
                ps, indicator, boot, nbins, show_output
            )

            # 2b. Trim the data
            if trim is True:
                boot, ps = trim_data(ps, common_support, boot)

            # 3. Double Residual Regression
            # Sort data by ps
            boot = boot.sort_values(by="ps", ascending=True)
            ps = np.sort(ps)

            X = boot[dict_["TREATED"]["order"]]
            Xp = construct_Xp(X, ps)
            Y = boot[[dict_["ESTIMATION"]["dependent"]]]

            b0, b1_b0 = double_residual_reg(ps, X, Xp, Y, rbandwidth, show_output)

            # Turn the X, Xp, and Y DataFrames into np.ndarrays
            X_arr = np.array(X)
            Xp_arr = np.array(Xp)
            Y_arr = np.array(Y).ravel()

            # 4. Compute the unobserved part of Y
            Y_tilde = Y_arr - np.dot(X_arr, b0) - np.dot(Xp_arr, b1_b0)

            # 5. Estimate mte_u, the unobserved component of the MTE,
            # through a locally quadratic regression
            quantiles, mte_u = locpoly(ps, Y_tilde, 1, 2, bandwidth, gridsize, a, b)

            # 6. construct MTE
            # Calculate the MTE component that depends on X
            mte_x = np.dot(X, b1_b0).mean(axis=0)

            # Put the MTE together
            mte = mte_x + mte_u

            mte_boot[:, counter] = mte

            counter += 1

        else:
            continue

    return mte_boot
