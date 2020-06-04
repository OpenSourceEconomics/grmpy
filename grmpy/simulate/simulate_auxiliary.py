""" This module provides auxiliary functions for the simulate.py module. It includes
simulation processes of the unobservable and endogenous variables of the model as well
as functions regarding the info file output.
"""
from scipy.stats import norm
import pandas as pd
import numpy as np


def simulate_covariates(init_dict):
    """The function simulates the covariates for the choice and the output functions."""
    # Distribute information
    num_agents = init_dict["SIMULATION"]["agents"]

    # Construct auxiliary information

    num_covars = init_dict["AUX"]["num_covars"]
    labels = init_dict["AUX"]["labels"]

    # As our baseline we simulate covariates from a standard normal distribution.
    means = np.tile(0.0, num_covars)
    covs = np.identity(num_covars)
    X = pd.DataFrame(
        np.random.multivariate_normal(means, covs, num_agents), columns=labels
    )

    # We now perform some selective replacements.
    # Set intercept
    X[labels[0]] = 1.0

    # Include binary variables
    if "VARTYPES" in init_dict:
        for variable in init_dict["VARTYPES"]:
            if isinstance(init_dict["VARTYPES"][variable], list):
                X[variable] = np.random.binomial(
                    1, init_dict["VARTYPES"][variable][1], size=num_agents
                )
    else:
        pass

    return X


def simulate_unobservables(init_dict):
    """The function simulates the unobservable error terms."""
    num_agents = init_dict["SIMULATION"]["agents"]
    cov = construct_covariance_matrix(init_dict)

    U = pd.DataFrame(
        np.random.multivariate_normal(np.zeros(3), cov, num_agents),
        columns=["U1", "U0", "V"],
    )

    return U


def simulate_outcomes(init_dict, X, U):
    """The function simulates the potential outcomes Y0 and Y1, the resulting treatment
    dummy D and the realized outcome Y.
    """
    dep = init_dict["ESTIMATION"]["dependent"]
    indicator = init_dict["ESTIMATION"]["indicator"]

    df = X.join(U)
    Z = df[init_dict["CHOICE"]["order"]].values
    X_treated, X_untreated = (
        df[init_dict["TREATED"]["order"]],
        df[init_dict["UNTREATED"]["order"]],
    )

    # Distribute information
    coeffs_untreated = init_dict["UNTREATED"]["params"]
    coeffs_treated = init_dict["TREATED"]["params"]
    coeffs_choice = init_dict["CHOICE"]["params"]

    # Calculate potential outcomes and choice
    df[dep + "1"] = np.dot(coeffs_treated, X_treated.T) + df["U1"]
    df[dep + "0"] = np.dot(coeffs_untreated, X_untreated.T) + df["U0"]
    C = np.dot(coeffs_choice, Z.T) - df["V"]

    # Calculate expected benefit and the resulting treatment dummy
    df[indicator] = np.array((C > 0).astype(float))

    # Observed outcomes
    df[dep] = df[indicator] * df[dep + "1"] + (1 - df[indicator]) * df[dep + "0"]

    return df


def write_output(init_dict, df):
    """The function converts the simulated variables to a panda data frame and saves the
    data in a txt and a pickle file.
    """
    # Distribute information
    source = init_dict["SIMULATION"]["source"]

    df.to_pickle(source + ".grmpy.pkl")
    with open(source + ".grmpy.txt", "w") as file_:
        df.to_string(file_, index=False, na_rep=".", col_space=15, justify="left")
    return df


def print_info(init_dict, data_frame):
    """The function writes an info file for the specific data frame."""
    # Distribute information
    coeffs_untreated = init_dict["UNTREATED"]["params"]
    coeffs_treated = init_dict["TREATED"]["params"]
    source = init_dict["SIMULATION"]["source"]
    dep, indicator = (
        init_dict["ESTIMATION"]["dependent"],
        init_dict["ESTIMATION"]["indicator"],
    )

    # Construct auxiliary information
    cov = construct_covariance_matrix(init_dict)

    with open(source + ".grmpy.info", "w") as file_:

        # First we note some basic information ab out the dataset.
        header = "\n\n Number of Observations \n\n"
        file_.write(header)

        info_ = [
            data_frame.shape[0],
            (data_frame[indicator] == 1).sum(),
            (data_frame[indicator] == 0).sum(),
        ]

        fmt = "  {:<10}" + " {:>20}" * 1 + "\n\n"
        file_.write(fmt.format(*["", "Count"]))

        for i, label in enumerate(["All", "Treated", "Untreated"]):
            str_ = "  {:<10} {:20}\n"
            file_.write(str_.format(*[label, info_[i]]))

        # Second, we describe the distribution of outcomes and effects.
        for label in ["Outcomes", "Effects"]:

            header = "\n\n Distribution of " + label + "\n\n"
            file_.write(header)

            fmt = "  {:<10}" + " {:>20}" * 5 + "\n\n"
            args = ["", "Mean", "Std-Dev.", "25%", "50%", "75%"]
            file_.write(fmt.format(*args))

            for group in ["All", "Treated", "Untreated"]:

                if label == "Outcomes":
                    data = data_frame[dep]
                elif label == "Effects":
                    data = data_frame[dep + "1"] - data_frame[dep + "0"]

                if group == "Treated":
                    data = data[data_frame[indicator] == 1]
                elif group == "Untreated":
                    data = data[data_frame[indicator] == 0]
                else:
                    pass
                fmt = "  {:<10}" + " {:>20.4f}" * 5 + "\n"
                info = list(data.describe().tolist()[i] for i in [1, 2, 4, 5, 6])
                if pd.isnull(info).all():
                    fmt = "  {:<10}" + " {:>20}" * 5 + "\n"
                    info = ["---"] * 5
                elif pd.isnull(info[1]):
                    info[1] = "---"
                    fmt = "  {:<10}" " {:>20.4f}" " {:>20}" + " {:>20.4f}" * 3 + "\n"

                file_.write(fmt.format(*[group] + info))

        # Implement the criteria function value , the MTE and parameterization
        header = "\n\n {} \n\n".format("Criterion Function")
        file_.write(header)
        if "criteria_value" in init_dict["AUX"].keys():
            str_ = "  {0:<10}      {1:<21.12f}\n\n".format(
                "Value", init_dict["AUX"]["criteria_value"]
            )
        else:
            str_ = "  {0:>10} {1:>20}\n\n".format("Value", "---")
        file_.write(str_)

        header = "\n\n {} \n\n".format("Marginal Treatment Effect")
        file_.write(header)
        quantiles = [1] + np.arange(5, 100, 5).tolist() + [99]
        args = [str(i) + "%" for i in quantiles]
        quantiles = [i * 0.01 for i in quantiles]

        x = data_frame
        value = mte_information(
            coeffs_treated, coeffs_untreated, cov, quantiles, x, init_dict
        )
        str_ = "  {0:>10} {1:>20}\n\n".format("Quantile", "Value")
        file_.write(str_)
        len_ = len(value)
        for i in range(len_):
            if isinstance(value[i], float):
                file_.write("  {0:>10} {1:>20.4f}\n".format(str(args[i]), value[i]))
            else:
                file_.write("  {0:>10} {1:>20.4}\n".format(str(args[i]), value[i]))

        # Write out parameterization of the model.
        write_identifier_section_simulate(init_dict, file_)


def mte_information(coeffs_treated, coeffs_untreated, cov, quantiles, x, dict_):
    """The function calculates the marginal treatment effect for pre specified quantiles
    of the collected unobservable variables.
    """

    labels = [k for k in dict_["TREATED"]["order"]]
    labels += [j for j in dict_["UNTREATED"]["order"] if j not in labels]

    # Construct auxiliary information
    if dict_["TREATED"]["order"] != dict_["UNTREATED"]["order"]:

        para_diff = []
        for var in labels:
            if var in dict_["TREATED"]["order"] and var in dict_["UNTREATED"]["order"]:
                index_treated = dict_["TREATED"]["order"].index(var)
                index_untreated = dict_["UNTREATED"]["order"].index(var)
                diff = (
                    dict_["TREATED"]["params"][index_treated]
                    - dict_["UNTREATED"]["params"][index_untreated]
                )
            elif (
                var in dict_["TREATED"]["order"]
                and var not in dict_["UNTREATED"]["order"]
            ):
                index = dict_["TREATED"]["order"].index(var)
                diff = dict_["TREATED"]["params"][index]

            elif (
                var not in dict_["TREATED"]["order"]
                and var in dict_["UNTREATED"]["order"]
            ):
                index = dict_["UNTREATED"]["order"].index(var)
                diff = -dict_["UNTREATED"]["params"][index]
            para_diff += [diff]
    else:

        para_diff = coeffs_treated - coeffs_untreated
    x = x[labels]
    MTE = []
    for i in quantiles:
        if cov[2, 2] == 0.00:
            MTE += ["---"]
        else:
            MTE += [
                np.mean(np.dot(x, para_diff)) + (cov[2, 0] - cov[2, 1]) * norm.ppf(i)
            ]

    return MTE


def write_identifier_section_simulate(init_dict, file_):
    """This function prints the information about the estimation results in the output
     file.
     """

    file_.write("\n\n {} \n\n".format("Parameterization"))

    fmt_ = " {:<10}" + "    {:>10}" + "{:>15}"

    file_.write(fmt_.format(*["Section", "Identifier", "Coef"]) + "\n")

    num_treated = len(init_dict["TREATED"]["order"])
    num_untreated = num_treated + len(init_dict["UNTREATED"]["order"])
    num_choice = num_untreated + len(init_dict["CHOICE"]["order"])

    identifier_treated = init_dict["TREATED"]["order"]
    identifier_untreated = init_dict["UNTREATED"]["order"]
    identifier_choice = init_dict["CHOICE"]["order"]
    identifier_distribution = [
        "sigma1",
        "sigma10",
        "sigma1v",
        "sigma0",
        "sigma0v",
        "sigmaV",
    ]
    identifier = (
        identifier_treated
        + identifier_untreated
        + identifier_choice
        + identifier_distribution
    )
    coeffs = init_dict["AUX"]["init_values"].copy()
    fmt = "  {:>10}" + "   {:<15}" + " {:>11.4f}"
    for i in range(len(init_dict["AUX"]["init_values"])):
        if i == 0:
            file_.write("\n  {:<10} \n".format("TREATED"))
        elif i == num_treated:
            file_.write("\n  {:<10} \n".format("UNTREATED"))
        elif i == num_untreated:
            file_.write("\n  {:<10} \n".format("CHOICE"))
        elif i == num_choice:
            file_.write("\n  {:<10} \n".format("DIST"))

        file_.write("{0}\n".format(fmt.format("", identifier[i], coeffs[i])))


def construct_covariance_matrix(init_dict):
    """This function constructs the covariance matrix based on the user's initialization
     file.
     """
    cov = np.zeros((3, 3))
    cov[np.triu_indices(3)] = init_dict["DIST"]["params"]
    cov[np.tril_indices(3, k=-1)] = cov[np.triu_indices(3, k=1)]
    cov[np.diag_indices(3)] **= 2
    return cov
