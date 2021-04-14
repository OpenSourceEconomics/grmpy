""" This script creates a figure to illustrate how the usual treatment effects can be
constructed by using differen weights on the marginal treatment effect.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad

from fig_config import OUTPUT_DIR, RESOURCE_DIR
from grmpy.read.read import read
from grmpy.simulate.simulate_auxiliary import (
    mte_information,
    simulate_covariates,
    construct_covariance_matrix,
)

filename = "/tutorial.grmpy.yml"

init_dict = read(RESOURCE_DIR + filename)
GRID = np.linspace(0.01, 0.99, num=99, endpoint=True)

plt.style.use("resources/grmpy.mplstyle")


def weights_treatment_parameters(init_dict, GRID):
    """This function calculates the weights for the special case in
    Heckman & Vytlacil (2005) Figure 1B.

    """
    GRID = np.linspace(0.01, 0.99, num=99, endpoint=True)

    coeffs_untreated = init_dict["UNTREATED"]["params"]
    coeffs_treated = init_dict["TREATED"]["params"]
    cov = construct_covariance_matrix(init_dict)
    x = simulate_covariates(init_dict)

    # We take the specified distribution for the cost shifters from the paper.
    cost_mean, cost_sd = -0.0026, np.sqrt(0.270)
    v_mean, v_sd = 0.00, np.sqrt(cov[2, 2])

    eval_points = norm.ppf(GRID, loc=v_mean, scale=v_sd)

    ate_weights = np.tile(1.0, 99)
    tut_weights = norm.cdf(eval_points, loc=cost_mean, scale=cost_sd)

    tt_weights = 1 - tut_weights

    def tut_integrand(point):
        eval_point = norm.ppf(point, loc=v_mean, scale=v_sd)
        return norm.cdf(eval_point, loc=cost_mean, scale=cost_sd)

    def tt_integrand(point):
        eval_point = norm.ppf(point, loc=v_mean, scale=v_sd)
        return norm.cdf(eval_point, loc=cost_mean, scale=cost_sd)

    # Scaling so that the weights integrate to one.
    tut_scaling = quad(tut_integrand, 0.01, 0.99)[0]
    tut_weights /= tut_scaling

    tt_scaling = quad(tt_integrand, 0.01, 0.99)[0]
    tt_weights /= tt_scaling

    mte = mte_information(coeffs_treated, coeffs_untreated, cov, GRID, x, init_dict)

    return ate_weights, tt_weights, tut_weights, mte


def plot_weights_marginal_effect(ate, tt, tut, mte):
    ax = plt.figure().add_subplot(111)
    ax.set_xlabel(r"$u_S$")
    ax.set_ylabel(r"$\omega(u_S)$")
    ax.set_ylim(0, 4.5)
    ax.set_xlim(0.0, 1.0)
    ax.plot(GRID, ate, label=r"$\omega^{ATE}$", linestyle=":")
    ax.plot(GRID, tt, label=r"$\omega^{TT}$", linestyle="--")
    ax.plot(GRID, tut, label=r"$\omega^{TUT}$", linestyle="-.")
    ax.plot(GRID, mte, label="MTE")
    plt.legend()

    ax2 = ax.twinx()
    ax2.set_ylabel("MTE")
    ax2.set_ylim(0, 0.35)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + "/fig-weights-marginal-effect.png", dpi=300)


if __name__ == "__main__":
    ate_weights, tt_weights, tut_weights, mte = weights_treatment_parameters(
        init_dict, GRID
    )
    plot_weights_marginal_effect(ate_weights, tt_weights, tut_weights, mte)
