"""This module provides the analytical solution for computing the hessian matrix of our
loglikelihood function
"""
import numpy as np
from scipy.stats import norm


def compute_hessian(x0, X1, X0, Z1, Z0, Y1, Y0):
    """This function wraps all subroutines and returns the hessian matrix of our
    log-likelihood function
    """

    # def auxiliary parameters
    num_obs = X1.shape[0] + X0.shape[0]
    n_col_X1 = X1.shape[1]
    n_col_X0 = X0.shape[1]
    n_col_Z = Z1.shape[1]

    # parameters
    num_col_X1X0 = n_col_X1 + n_col_X0
    num_col_X1X0Z1 = num_col_X1X0 + n_col_Z

    beta1, beta0, gamma = (
        x0[:n_col_X1],
        x0[n_col_X1:num_col_X1X0],
        x0[num_col_X1X0:-4],
    )
    sd1, sd0, rho1v, rho0v = x0[-4], x0[-2], x0[-3], x0[-1]

    # aux_params
    nu1 = (Y1 - np.dot(beta1, X1.T)) / sd1
    lambda1 = (np.dot(gamma, Z1.T) - rho1v * nu1) / (np.sqrt(1 - rho1v**2))
    nu0 = (Y0 - np.dot(beta0, X0.T)) / sd0
    lambda0 = (np.dot(gamma, Z0.T) - rho0v * nu0) / (np.sqrt(1 - rho0v**2))

    eta1 = (
        -lambda1 * norm.pdf(lambda1) * norm.cdf(lambda1) - norm.pdf(lambda1) ** 2
    ) / (norm.cdf(lambda1) ** 2)
    eta0 = (
        lambda0 * norm.pdf(lambda0) * (1 - norm.cdf(lambda0)) - norm.pdf(lambda0) ** 2
    ) / (1 - norm.cdf(lambda0)) ** 2
    # combinations of obs
    X1X1 = np.einsum("ij, i ->ij", X1, eta1).T @ X1
    X1Z1 = np.einsum("ij, i ->ij", X1, eta1).T @ Z1

    X0X0 = np.einsum("ij, i ->ij", X0, eta0).T @ X0
    X0Z0 = np.einsum("ij, i ->ij", X0, eta0).T @ Z0

    Z1Z1 = np.einsum("ij, i ->ij", Z1, eta1).T @ Z1
    Z0Z0 = np.einsum("ij, i ->ij", Z0, eta0).T @ Z0

    # beginning with derivations of beta1
    derv_beta1 = calc_hess_beta1(
        X1X1, X1Z1, X1, sd1, rho1v, nu1, lambda1, eta1, n_col_X1, n_col_X0, num_obs
    )
    derv_beta0 = calc_hess_beta0(
        X0X0, X0Z0, X0, sd0, rho0v, nu0, lambda0, eta0, n_col_X1, n_col_X0, num_obs
    )
    derv_gamma = calc_hess_gamma(
        Z1Z1,
        Z0Z0,
        Z1,
        X1,
        Z0,
        X0,
        sd0,
        sd1,
        rho0v,
        rho1v,
        eta1,
        eta0,
        nu0,
        nu1,
        lambda0,
        lambda1,
        num_col_X1X0,
        num_obs,
    )
    derv_dist = calc_hess_dist(
        Z1,
        Z0,
        gamma,
        sd1,
        sd0,
        rho1v,
        rho0v,
        lambda1,
        lambda0,
        nu1,
        nu0,
        eta1,
        eta0,
        num_col_X1X0Z1,
        num_obs,
    )

    # convert results to a symmetric hessian matrix
    hessian_upper = np.triu(
        np.concatenate((derv_beta1, derv_beta0, derv_gamma, derv_dist), axis=0)
    )

    aux = hessian_upper.copy()
    for i in range(hessian_upper.shape[0]):
        hessian_upper[:, i][i + 1 :] = hessian_upper[i][i + 1 :]
    return hessian_upper, aux


def calc_hess_beta1(
    X1X1, X1Z1, X1, sd1, rho1v, nu1, lambda1, eta1, n_col_X1, n_col_X0, num_obs
):
    """This function computes the derivatives of the first order conditions of beta1 wrt
    all other parameters.
    """

    # define some auxiliary variables
    rho_aux1 = lambda1 * rho1v / (1 - rho1v**2) - nu1 / (1 - rho1v**2) ** 0.5
    rho_aux2 = rho1v**2 / ((1 - rho1v**2) ** (3 / 2)) + 1 / (1 - rho1v**2) ** 0.5
    sd_aux1 = rho1v**2 / (1 - rho1v**2)
    sd_aux2 = rho1v / np.sqrt(1 - rho1v**2)
    # derivation wrt beta1
    der_b1_beta1 = -(
        X1X1 * (rho1v**2 / (1 - rho1v**2)) * 1 / sd1**2 - X1.T @ X1 / sd1**2
    )
    # add zeros for derv beta 0
    der_b1_beta1 = np.concatenate(
        (der_b1_beta1, np.zeros((n_col_X1, n_col_X0))), axis=1
    )

    # derivation wrt gamma
    der_b1_gamma = -(X1Z1 * rho1v / (sd1 * (1 - rho1v**2)))
    der_b1_gamma = np.concatenate((der_b1_beta1, der_b1_gamma), axis=1)
    # derv wrt sigma 1
    der_b1_sd = (
        -1
        / sd1
        * (
            (
                (eta1 * sd_aux1 * nu1 - norm.pdf(lambda1) / norm.cdf(lambda1) * sd_aux2)
                - 2 * nu1
            )
            * 1
            / sd1
        )
    )

    # expand_dimensions and add
    der_b1_sd = np.expand_dims((der_b1_sd.T @ X1), 1)
    der_b1_sd = np.concatenate((der_b1_gamma, der_b1_sd), axis=1)

    # derv wrt rho1
    der_b1_rho = (
        -(
            eta1 * rho_aux1 * rho1v / ((1 - rho1v**2) ** 0.5)
            + norm.pdf(lambda1) / norm.cdf(lambda1) * rho_aux2
        )
        * 1
        / sd1
    )
    # expand_dimensions and add

    der_b1_rho = np.expand_dims((der_b1_rho.T @ X1), 1)
    der_b1_rho = np.concatenate((der_b1_sd, der_b1_rho), axis=1)

    # add zeros for sigma0 and rho0
    der_b1 = np.concatenate((der_b1_rho, np.zeros((n_col_X1, 2))), axis=1)
    der_beta1 = der_b1 / num_obs

    return der_beta1


def calc_hess_beta0(
    X0X0, X0Z0, X0, sd0, rho0v, nu0, lambda0, eta0, n_col_X1, n_col_X0, num_obs
):
    """This function computes the derivatives of the first order conditions of beta0 wrt
    all other parameters.
    """

    # define some aux_vars
    rho_aux1 = lambda0 * rho0v / (1 - rho0v**2) - nu0 / (1 - rho0v**2) ** 0.5
    rho_aux2 = rho0v**2 / ((1 - rho0v**2) ** (3 / 2)) + 1 / (1 - rho0v**2) ** 0.5
    sd_aux1 = rho0v**2 / (1 - rho0v**2)
    sd_aux2 = rho0v / (np.sqrt(1 - rho0v**2))

    # add zeros for beta0
    der_b0_beta1 = np.zeros((n_col_X1, n_col_X0))

    # beta0
    der_b0_beta0 = (
        -(X0X0 * (rho0v**2 / (1 - rho0v**2)) * 1 / sd0**2) + X0.T @ X0 / sd0**2
    )
    der_b0_beta0 = np.concatenate((der_b0_beta1, der_b0_beta0), axis=1)
    # gamma
    der_b0_gamma = -X0Z0 * rho0v / (1 - rho0v**2) * 1 / sd0
    der_b0_gamma = np.concatenate((der_b0_beta0, der_b0_gamma), axis=1)

    # add zeros for sigma1 and rho1
    der_b0_gamma = np.concatenate((der_b0_gamma, np.zeros((n_col_X0, 2))), axis=1)

    # sigma
    der_b0_sd = (
        -(
            eta0 * nu0 * sd_aux1
            + norm.pdf(lambda0) / (1 - norm.cdf(lambda0)) * sd_aux2
            - 2 * nu0
        )
        * 1
        / sd0**2
    )
    der_b0_sd = np.expand_dims((der_b0_sd.T @ X0), 1)
    der_b0_sd = np.concatenate((der_b0_gamma, der_b0_sd), axis=1)

    # rho
    der_b0_rho = (
        (
            eta0 * -rho_aux1 * (rho0v / ((1 - rho0v**2) ** 0.5))
            + norm.pdf(lambda0) / (1 - norm.cdf(lambda0)) * rho_aux2
        )
        * 1
        / sd0
    )
    der_b0_rho = np.expand_dims((der_b0_rho.T @ X0), 1)
    der_b0_rho = np.concatenate((der_b0_sd, der_b0_rho), axis=1)

    der_beta0 = der_b0_rho / num_obs

    return der_beta0


def calc_hess_gamma(
    Z1Z1,
    Z0Z0,
    Z1,
    X1,
    Z0,
    X0,
    sd0,
    sd1,
    rho0v,
    rho1v,
    eta1,
    eta0,
    nu0,
    nu1,
    lambda0,
    lambda1,
    num_col_X1X0,
    num_obs,
):
    """This function computes the derivatives of the first order conditions of gamma wrt
    all other parameters.
    """

    der_gamma_beta = np.zeros((Z1.shape[1], num_col_X1X0))

    der_g_gamma = -(1 / (1 - rho1v**2) * Z1Z1 + 1 / (1 - rho0v**2) * Z0Z0)
    der_g_gamma = np.concatenate((der_gamma_beta, der_g_gamma), axis=1)

    # sigma1
    der_g_sd1 = -(
        np.einsum("ij, i ->ij", Z1, eta1).T
        @ np.einsum("ij, i ->ij", X1, nu1)
        / sd1
        * rho1v
        / (1 - rho1v**2)
    )[:, 0]
    der_g_sd1 = np.expand_dims(der_g_sd1, 0).T
    der_g_sd1 = np.concatenate((der_g_gamma, der_g_sd1), axis=1)

    # rho1
    aux_rho11 = np.einsum("ij, i ->ij", Z1, eta1).T @ (
        lambda1 * rho1v / (1 - rho1v**2) - nu1 / np.sqrt(1 - rho1v**2)
    )
    aux_rho21 = Z1.T @ (norm.pdf(lambda1) / norm.cdf(lambda1))

    der_g_rho1 = -aux_rho11 * 1 / (np.sqrt(1 - rho1v**2)) - aux_rho21 * rho1v / (
        (1 - rho1v**2) ** (3 / 2)
    )
    der_g_rho1 = np.expand_dims(der_g_rho1, 0).T
    der_g_rho1 = np.concatenate((der_g_sd1, der_g_rho1), axis=1)

    # sigma0
    der_g_sd0 = (
        np.einsum("ij, i ->ij", Z0, eta0).T
        @ np.einsum("ij, i ->ij", X0, nu0)
        / sd0
        * rho0v
        / (1 - rho0v**2)
    )[:, 0]
    der_g_sd0 = np.expand_dims(der_g_sd0, 0).T
    der_g_sd0 = np.concatenate((der_g_rho1, -der_g_sd0), axis=1)

    # rho1
    aux_rho10 = np.einsum("ij, i ->ij", Z0, eta0).T @ (
        lambda0 * rho0v / (1 - rho0v**2) - nu0 / np.sqrt(1 - rho0v**2)
    )
    aux_rho20 = -Z0.T @ (norm.pdf(lambda0) / (1 - norm.cdf(lambda0)))

    der_g_rho0 = aux_rho10 * 1 / (np.sqrt(1 - rho0v**2)) + aux_rho20 * rho0v / (
        (1 - rho0v**2) ** (3 / 2)
    )
    der_g_rho0 = np.expand_dims(-der_g_rho0, 0).T
    der_g_rho0 = np.concatenate((der_g_sd0, der_g_rho0), axis=1)

    return der_g_rho0 / num_obs


def calc_hess_dist(
    Z1,
    Z0,
    gamma,
    sd1,
    sd0,
    rho1v,
    rho0v,
    lambda1,
    lambda0,
    nu1,
    nu0,
    eta1,
    eta0,
    num_col_X1X0Z1,
    num_obs,
):
    """This function computes the derivatives of the first order conditions of all
    distribution parameters wrt all other parameters.
    """
    # aux_vars
    Delta_sd1 = (
        +1 / sd1
        - (norm.pdf(lambda1) / norm.cdf(lambda1))
        * (rho1v * nu1 / (np.sqrt(1 - rho1v**2) * sd1))
        - nu1**2 / sd1
    )
    Delta_sd1_der = (
        nu1
        / sd1
        * (
            -eta1 * (rho1v**2 * nu1) / (1 - rho1v**2)
            + (norm.pdf(lambda1) / norm.cdf(lambda1)) * rho1v / np.sqrt(1 - rho1v**2)
            + 2 * nu1
        )
    )

    Delta_sd0 = (
        +1 / sd0
        + (norm.pdf(lambda0) / (1 - norm.cdf(lambda0)))
        * (rho0v * nu0 / (np.sqrt(1 - rho0v**2) * sd0))
        - nu0**2 / sd0
    )
    Delta_sd0_der = (
        nu0
        / sd0
        * (
            -eta0 * (rho0v**2 * nu0) / (1 - rho0v**2)
            - (norm.pdf(lambda0) / (1 - norm.cdf(lambda0)))
            * rho0v
            / np.sqrt(1 - rho0v**2)
            + 2 * nu0
        )
    )

    aux_rho11 = lambda1 * rho1v / (1 - rho1v**2) - nu1 / np.sqrt(1 - rho1v**2)
    aux_rho12 = 1 / (1 - rho1v**2) ** (3 / 2)

    aux_rho_rho11 = (np.dot(gamma, Z1.T) * rho1v - nu1) / (1 - rho1v**2) ** (3 / 2)
    aux_rho_rho12 = (
        2 * np.dot(gamma, Z1.T) * rho1v**2 + np.dot(gamma, Z1.T) - 3 * nu1 * rho1v
    ) / (1 - rho1v**2) ** (5 / 2)

    aux_rho01 = lambda0 * rho0v / (1 - rho0v**2) - nu0 / np.sqrt(1 - rho0v**2)
    aux_rho02 = 1 / (1 - rho0v**2) ** (3 / 2)

    aux_rho_rho01 = (np.dot(gamma, Z0.T) * rho0v - nu0) / (1 - rho0v**2) ** (3 / 2)
    aux_rho_rho02 = (
        2 * np.dot(gamma, Z0.T) * rho0v**2 + np.dot(gamma, Z0.T) - 3 * nu0 * rho0v
    ) / (1 - rho0v**2) ** (5 / 2)

    # for sigma1

    # wrt sd1
    derv_sd1_sd1 = 1 / sd1 * (-Delta_sd1 + Delta_sd1_der)

    # wrt rho1
    derv_sd1_rho1 = (
        1
        / sd1
        * (
            -eta1 * aux_rho11 * (rho1v * nu1) / (np.sqrt(1 - rho1v**2))
            - (norm.pdf(lambda1) / norm.cdf(lambda1)) * aux_rho12 * nu1
        )
    )

    # append values
    derv_sd1 = np.append(
        np.zeros(num_col_X1X0Z1), [sum(derv_sd1_sd1), sum(derv_sd1_rho1), 0, 0]
    )

    # for rho1
    # wrt to rho1
    derv_rho1v_rho1 = (
        -eta1 * aux_rho11 * aux_rho_rho11
        - (norm.pdf(lambda1) / norm.cdf(lambda1)) * aux_rho_rho12
    )
    derv_rho1 = np.append(np.zeros(num_col_X1X0Z1 + 1), [sum(derv_rho1v_rho1), 0, 0])

    # for sigma0
    # wrt sd0
    derv_sd0_sd0 = 1 / sd0 * (-Delta_sd0 + Delta_sd0_der)
    # wrt rho0
    derv_sd0_rho0 = (
        1
        / sd0
        * (
            -eta0 * aux_rho01 * (rho0v * nu0) / (np.sqrt(1 - rho0v**2))
            + (norm.pdf(lambda0) / (1 - norm.cdf(lambda0))) * aux_rho02 * nu0
        )
    )
    derv_sd0 = np.append(
        np.zeros(num_col_X1X0Z1 + 2), [sum(derv_sd0_sd0), sum(derv_sd0_rho0)]
    )

    # for rho0
    derv_rho0v_rho0 = -(
        eta0 * aux_rho01 * aux_rho_rho01
        - (norm.pdf(lambda0) / (1 - norm.cdf(lambda0))) * aux_rho_rho02
    )
    derv_rho0 = np.append(np.zeros(num_col_X1X0Z1 + 3), [sum(derv_rho0v_rho0)])

    derv_dist = np.stack([derv_sd1, derv_rho1, derv_sd0, derv_rho0]) / num_obs

    return derv_dist
