"""This module contains the code for a local average treatment graph.
"""
import matplotlib.pyplot as plt

from fig_config import OUTPUT_DIR, RESOURCE_DIR
from grmpy.read.read import read
from grmpy.simulate.simulate_auxiliary import (
    mte_information,
    simulate_covariates,
    construct_covariance_matrix,
)

filename = "/tutorial.grmpy.yml"
plt.style.use("resources/grmpy.mplstyle")

GRID = [i / 100 for i in range(1, 100, 1)]
init_dict = read(RESOURCE_DIR + filename)


def plot_local_average_treatment(mte):
    ax = plt.figure().add_subplot(111)

    # Plot the mte
    ax.plot(GRID, mte)

    # Plot vertical lines and dots
    for xtick in [0.2, 0.3, 0.4, 0.6, 0.7, 0.8]:
        index = GRID.index(xtick)
        height = mte[index]
        ax.plot((xtick, xtick), (0, height), color="grey", alpha=0.7)
        endpoints = [(xtick, xtick), (0, height)]
        if xtick == 0.3:
            ax.scatter(*endpoints, color="black")
            ax.annotate(
                "$LATE(p_2, p_1)$",
                xy=(xtick - 0.005, height + 0.1),
                xytext=(xtick - 0.1, height + 0.5),
                arrowprops=dict(facecolor="black"),
            )
            ax.annotate(
                "$u_S(p_2, p_1)$",
                xy=(xtick, 0),
                xytext=(xtick - 0.05, -0.15),
                bbox=dict(facecolor="white"),
            )
        elif xtick == 0.7:
            ax.scatter(*endpoints, color="black")
            ax.annotate(
                "$LATE(p_4, p_3)$",
                xy=(xtick - 0.005, height + 0.1),
                xytext=(xtick - 0.1, height + 0.5),
                arrowprops=dict(facecolor="black"),
            )
            ax.annotate(
                "$u_S(p_4, p_3)$",
                xy=(xtick, 0),
                xytext=(xtick - 0.05, -0.15),
                bbox=dict(facecolor="white"),
            )

    # Set squared braces
    for xtick in [0.195, 0.594]:
        index = GRID.index(round(xtick, 2))
        height = mte[index]
        ax.text(x=xtick - 0.002, y=height - 0.1, s="[", fontsize=30)
    for xtick in [0.39, 0.79]:
        index = GRID.index(round(xtick, 2))
        height = mte[index]
        ax.text(x=xtick - 0.01, y=height - 0.1, s="]", fontsize=30)

    ax.set_xlabel("$u_S$")
    ax.set_ylabel(r"$MTE$")

    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xticklabels([0, "$p_1$", "$p_2$", "$p_3$", "$p_4$", 1])
    ax.tick_params(axis="x", which="major")
    ax.set_ylim([1.5, 4.5])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "/fig-local-average-treatment.png", dpi=300)


if __name__ == "__main__":
    coeffs_untreated = init_dict["UNTREATED"]["params"]
    coeffs_treated = init_dict["TREATED"]["params"]
    cov = construct_covariance_matrix(init_dict)
    x = simulate_covariates(init_dict)

    mte = mte_information(coeffs_treated, coeffs_untreated, cov, GRID, x, init_dict)

    plot_local_average_treatment(mte)
