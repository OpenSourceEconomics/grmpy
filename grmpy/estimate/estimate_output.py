"""This module contains methods for producing the estimation output files."""
import copy

import numpy as np

from grmpy.simulate.simulate_auxiliary import (
    mte_information,
    simulate_covariates,
    simulate_outcomes,
    simulate_unobservables,
)


def print_logfile(init_dict, rslt):
    """The function writes the log file for the estimation process."""
    # Adjust output

    if "output_file" in init_dict["ESTIMATION"].keys():
        file_name = init_dict["ESTIMATION"]["output_file"]
    else:
        file_name = "est.grmpy.info"

    file_input = ""
    for label in [
        "Optimization Information",
        "Criterion Function",
        "Economic Parameters",
    ]:
        header = "\n \n  {:<10}\n\n".format(label)
        file_input += header
        if label == "Optimization Information":
            for section in [
                "Optimizer",
                "Start Values",
                "Success",
                "Status",
                "Message",
                "Number of Evaluations",
                "Criterion",
                "Observations",
                "Warning",
            ]:
                fmt = "  {:<10}" + " {:<30}" + "{:<30} \n"

                if section == "Number of Evaluations":
                    file_input += fmt.format("", section + ":", rslt["nfev"])
                elif section in ["Start Values", "Optimizer"]:
                    file_input += fmt.format(
                        "",
                        section + ":",
                        rslt["ESTIMATION"][section.split(" ")[0].lower()],
                    )

                elif section == "Criterion":
                    fmt_float = "  {:<10}" + " {:<30}" + "{:<30.4f}\n"
                    file_input += fmt_float.format("", section + ":", rslt["crit"])
                elif section in ["Warning"]:

                    for counter, _ in enumerate(rslt[section.lower()]):
                        if counter == 0:
                            file_input += fmt.format(
                                "", section + ":", rslt[section.lower()][counter]
                            )
                        else:
                            file_input += fmt.format(
                                "", "", rslt[section.lower()][counter]
                            )

                    if section == "Warning":
                        if "warning" in init_dict["ESTIMATION"].keys():
                            file_input += fmt.format(
                                "", "", init_dict["ESTIMATION"]["warning"]
                            )

                else:
                    file_input += fmt.format("", section + ":", rslt[section.lower()])

        elif label == "Criterion Function":
            fmt = "  {:<10}" * 2 + " {:>20}" * 2 + "\n\n"
            file_input += fmt.format("", "", "Start", "Finish")
            file_input += "\n" + fmt.format(
                "", "", init_dict["AUX"]["criteria"], rslt["crit"]
            )

        else:

            file_input += write_identifier_section(rslt)
        if rslt["ESTIMATION"]["print_output"] == "1":
            print(file_input)
        with open(file_name, "w") as file_:
            file_.write(file_input)


def write_identifier_section(rslt):
    """This function prints the information about the estimation results in the output
     file.
     """
    file_ = ""
    fmt_ = "\n  {:<10}" + "{:>10}" + " {:>18}" + "{:>16}" + "\n\n"

    file_ += fmt_.format(*["", "", "Start", "Finish"])

    fmt_ = (
        " {:<10}"
        + "    {:>10}"
        + "{:>15}" * 2
        + "{:>18}"
        + "{:>9}"
        + "{:>19}"
        + "{:>24}"
    )

    file_ += (
        fmt_.format(
            *[
                "Section",
                "Identifier",
                "Coef",
                "Coef",
                "Std err",
                "t",
                "P>|t|",
                "95% Conf. Int.",
            ]
        )
        + "\n"
    )

    fmt = (
        "  {:>10}"
        + "   {:<15}"
        + " {:>11.4f}"
        + "{:>15.4f}" * 4
        + "{:>15.4f}"
        + "{:>10.4f}"
    )

    for category in ["TREATED", "UNTREATED", "CHOICE", "DIST"]:
        file_ += "\n  {:<10} \n".format(category)
        for counter, var in enumerate(rslt[category]["order"]):
            file_ += "{0}\n".format(
                fmt.format(
                    "",
                    var,
                    rslt[category]["starting_values"][counter],
                    rslt[category]["params"][counter],
                    rslt[category]["standard_errors"][counter],
                    rslt[category]["t_values"][counter],
                    rslt[category]["p_values"][counter],
                    rslt[category]["confidence_intervals"][counter][0],
                    rslt[category]["confidence_intervals"][counter][1],
                )
            )
    return file_


def write_comparison(df1, rslt):
    """The function writes the info file including the descriptives of the original and the
    estimated sample.
    """

    file_name = "comparison.grmpy.info"

    df3, df2 = simulate_estimation(rslt)
    file_input = ""
    # First we note some basic information ab out the dataset.
    header = "\n\n Number of Observations \n\n"
    file_input += header
    indicator = rslt["ESTIMATION"]["indicator"]
    dep = rslt["ESTIMATION"]["dependent"]

    datasets_ = [df1[dep].values, df2[dep].values, df3[dep].values]
    datasets1 = [
        df1[df1[indicator] == 1][dep].values,
        df2[df2[indicator] == 1][dep].values,
        df3[df3[indicator] == 1][dep].values,
    ]
    datasets0 = [
        df1[df1[indicator] == 0][dep].values,
        df2[df2[indicator] == 0][dep].values,
        df3[df3[indicator] == 0][dep].values,
    ]
    datasets = {"ALL": datasets_, "TREATED": datasets1, "UNTREATED": datasets0}

    fmt = "    {:<25}" + " {:>20}" * 3
    file_input += (
        fmt.format(*["Sample", "Observed", "Simulated (finish)", "Simulated (start)"])
        + "\n\n\n"
    )

    for cat in datasets.keys():

        info = [cat] + [data.shape[0] for data in datasets[cat]]
        file_input += fmt.format(*info) + "\n"

    header = "\n\n Distribution of Outcomes\n\n"
    file_input += header
    args = ["", "Mean", "Std-Dev.", "25%", "50%", "75%"]

    for group in datasets.keys():
        header = "\n\n " "  {:<10}".format(group) + "\n\n"
        fmt = "    {:<25}" + " {:>20}" * 5 + "\n\n"
        file_input += header
        file_input += fmt.format(*args)

        for counter, sample in enumerate(
            ["Observed Sample", "Simulated Sample (finish)", "Simulated Sample (start)"]
        ):
            data = datasets[group][counter]
            fmt = "    {:<25}" + " {:>20.4f}" * 5 + "\n"
            info = [
                np.mean(data),
                np.std(data),
                np.quantile(data, 0.25),
                np.quantile(data, 0.5),
                np.quantile(data, 0.75),
            ]
            if np.isnan(info).all():
                fmt = "    {:<10}" + " {:>20}" * 5 + "\n"
                info = ["---"] * 5
            elif np.isnan(info[1]):
                info[1] = "---"
                fmt = "    {:<25}" " {:>20.4f}" " {:>20}" + " {:>20.4f}" * 3 + "\n"

            file_input += fmt.format(*[sample] + info)

    header = "\n\n {} \n\n".format("MTE Information")
    file_input += header
    value, args = calculate_mte(rslt, df1)
    str_ = "  {0:>10} {1:>20}\n\n".format("Quantile", "Value")
    file_input += str_
    len_ = len(value)
    for i in range(len_):
        if isinstance(value[i], float):
            file_input += "  {0:>10} {1:>20.4f}\n".format(str(args[i]), value[i])

    with open(file_name, "w") as file_:
        file_.write(file_input)


def simulate_estimation(rslt):
    """The function simulates a new sample based on the estimated coefficients."""

    # Distribute information
    seed = rslt["SIMULATION"]["seed"]
    # Determine parametrization and read in /simulate observables
    start, finish = process_results(rslt)
    data_frames = []
    for dict_ in [start, finish]:

        # Set seed value
        np.random.seed(seed)
        # Simulate unobservables
        U = simulate_unobservables(dict_)
        X = simulate_covariates(dict_)

        # Simulate endogeneous variables
        df = simulate_outcomes(dict_, X, U)
        data_frames += [df]

    return data_frames[0], data_frames[1]


def process_results(rslt):
    """The function processes the results dictionary for the following simulation."""
    start, finish = copy.deepcopy(rslt), copy.deepcopy(rslt)
    maxiter = rslt["ESTIMATION"]["maxiter"]
    if maxiter != 0:
        for section in ["TREATED", "UNTREATED", "CHOICE", "DIST"]:
            start[section]["params"] = start[section]["starting_values"]
    start["DIST"]["params"] = transform_rslt_DIST(start, maxiter, start=True)
    finish["DIST"]["params"] = transform_rslt_DIST(finish, maxiter)
    return start, finish


def transform_rslt_DIST(dict_, maxiter, start=False):
    """The function converts the correlation parameters from the estimation outcome to
    covariances for the simulation of the estimation sample.
    """
    x0 = dict_["DIST"]["params"]
    aux = x0[-4:].copy()
    cov1V = aux[1] * aux[0]
    cov0V = aux[3] * aux[2]
    if maxiter != 0:
        if not start:
            list_ = np.round([aux[0], 0.0, cov1V, aux[2], cov0V, 1.0], 4)
        else:
            list_ = dict_["AUX"]["init_values"][-6:].copy()
    else:
        list_ = dict_["AUX"]["init_values"][-6:].copy()

    return list_


def calculate_mte(rslt, data_frame, quant=None):

    coeffs_treated = rslt["TREATED"]["params"]
    coeffs_untreated = rslt["UNTREATED"]["params"]

    if quant is None:
        quantiles = [1] + np.arange(5, 100, 5).tolist() + [99]
        args = [str(i) + "%" for i in quantiles]
        quantiles = [i * 0.01 for i in quantiles]
    else:
        quantiles = quant

    cov = np.zeros((3, 3))
    cov[2, 0] = rslt["AUX"]["x_internal"][-3] * rslt["AUX"]["x_internal"][-4]
    cov[2, 1] = rslt["AUX"]["x_internal"][-1] * rslt["AUX"]["x_internal"][-2]
    cov[2, 2] = 1.0

    value = mte_information(
        coeffs_treated, coeffs_untreated, cov, quantiles, data_frame, rslt
    )
    if quant is None:
        return value, args
    else:
        return value
