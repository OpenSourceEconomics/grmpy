"""This script replicates the results of Caneiro 2011 via using a
 mock data set and plots the original as well as the estimated mar-
 ginal treatment effect"""
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


from fig_config import OUTPUT_DIR, RESOURCE_DIR
from grmpy.estimate.estimate import fit
from grmpy.read.read import read
from grmpy.plot.plot_auxiliary import mte_and_cof_int_par


plt.style.use("resources/grmpy.mplstyle")


def plot_rslts(rslt, file):
    init_dict = read(file)
    data_frame = pd.read_pickle(init_dict["ESTIMATION"]["file"])

    # Define the Quantiles and read in the original results
    mte_ = json.load(open("resources/mte_original.json"))
    mte_original = mte_[1]
    mte_original_d = mte_[0]
    mte_original_u = mte_[2]

    # Calculate the MTE and confidence intervals
    quantiles, mte, mte_up, mte_d = mte_and_cof_int_par(rslt, data_frame, 4)

    # Plot both curves
    ax = plt.figure().add_subplot(111)

    ax.set_ylabel(r"$MTE$")
    ax.set_xlabel("$u_D$")
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.plot(quantiles, mte, label="grmpy MTE", color="blue", linewidth=4)
    ax.plot(quantiles, mte_up, color="blue", linestyle=":", linewidth=3)
    ax.plot(quantiles, mte_d, color="blue", linestyle=":", linewidth=3)
    ax.plot(
        quantiles, mte_original, label="original${MTE}$", color="orange", linewidth=4
    )
    ax.plot(quantiles, mte_original_d, color="orange", linestyle=":", linewidth=3)
    ax.plot(quantiles, mte_original_u, color="orange", linestyle=":", linewidth=3)
    ax.xaxis.set_ticks(np.arange(0, 1.1, step=0.1))
    ax.yaxis.set_ticks(np.arange(-0.5, 0.5, step=0.1))

    ax.set_ylim([-0.37, 0.47])
    ax.set_xlim([0, 1])
    ax.margins(x=0.003)
    ax.margins(y=0.03)

    blue_patch = mpatches.Patch(color="blue", label="original $MTE$")
    orange_patch = mpatches.Patch(color="orange", label="replicated $MTE$")
    plt.legend(handles=[blue_patch, orange_patch], prop={"size": 16})
    plt.savefig(
        OUTPUT_DIR + "/fig-marginal-benefit-parametric-replication.png", dpi=300
    )


if __name__ == "__main__":

    rslt_dict = fit(RESOURCE_DIR + "/replication.grmpy.yml")
    plot_rslts(rslt_dict, RESOURCE_DIR + "/replication.grmpy.yml")
