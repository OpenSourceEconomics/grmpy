"""
This module provides auxiliary functions for the semiparametric tutorial.
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from grmpy.plot.plot_auxiliary import bootstrap

# surpress pandas warning
pd.options.mode.chained_assignment = None


def plot_semipar_mte(rslt, init_file, nbootstraps):
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
