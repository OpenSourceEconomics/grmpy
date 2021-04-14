"""The script creates a figure to illustrate the appearance of the marginal treatment
effect in the abscence and presence of individual heterogeneity.
"""
import numpy as np
import matplotlib.pyplot as plt

from fig_config import OUTPUT_DIR, RESOURCE_DIR
from grmpy.read.read import read
from grmpy.test.auxiliary import cleanup
from grmpy.simulate.simulate import simulate
from grmpy.simulate.simulate_auxiliary import (
    mte_information,
    construct_covariance_matrix,
)

filename = "/tutorial.grmpy.yml"

GRID = np.linspace(0.01, 0.99, num=99, endpoint=True)
init_dict = read(RESOURCE_DIR + filename)
plt.style.use("resources/grmpy.mplstyle")


def plot_marginal_treatment_effect(pres, abs_):
    ax = plt.figure().add_subplot(111)

    ax.set_ylabel(r"$B^{MTE}$")
    ax.set_xlabel("$u_S$")
    ax.plot(GRID, pres, label="Presence")
    ax.plot(GRID, abs_, label="Absence", linestyle="--")

    ax.set_ylim([1.5, 4.5])

    plt.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "/fig-eh-marginal-effect.png", dpi=300)


if __name__ == "__main__":
    coeffs_untreated = init_dict["UNTREATED"]["params"]
    coeffs_treated = init_dict["TREATED"]["params"]
    cov = construct_covariance_matrix(init_dict)
    df = simulate(RESOURCE_DIR + filename)
    x = df[init_dict["TREATED"]["order"]]
    MTE_pres = mte_information(
        coeffs_treated, coeffs_untreated, cov, GRID, x, init_dict
    )

    para_diff = coeffs_treated - coeffs_untreated

    MTE_abs = []
    for i in GRID:
        if cov[2, 2] == 0.00:
            MTE_abs += ["---"]
        else:
            MTE_abs += [np.mean(np.dot(para_diff, x.T))]

    plot_marginal_treatment_effect(MTE_pres, MTE_abs)
    cleanup()
