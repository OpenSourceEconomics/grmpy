"""This module provides auxiliary functions for the import process of the init file."""
import numpy as np


def create_attr_dict_est(init_dict, semipar=False, include_constant=False):
    """This function processes the imported initialization file so that it fulfills the
    requirements for the subsequent estimation process.
    """
    init_dict["AUX"] = {"init_values"}
    init_values = []

    if semipar is True:
        if include_constant is True:
            init_dict = add_constant(init_dict, semipar)
        else:
            pass

        init_dict = read_keys_semipar(init_dict, init_values)

    # semipar is False
    else:
        if include_constant is True:
            init_dict = add_constant(init_dict, semipar)
        else:
            pass

        init_dict = read_keys_par(init_dict, init_values)

    init_dict = provide_auxiliary_information(init_dict, init_values)

    return init_dict


def create_attr_dict_sim(init_dict):
    """This function processes the imported initialization file so that it fulfills the
    requirements for the following simulation and estimation process.
    """
    init_dict["AUX"] = {"init_values"}
    init_values = []

    init_dict = read_keys_par(init_dict, init_values)
    init_dict = provide_auxiliary_information(init_dict, init_values)

    return init_dict


def add_constant(init_dict, semipar=False):
    """The function checks if the user has provided a constant
    for the relevant subsections:
    ["TREATED", "UNTREATED", "CHOICE"] for the parametric, and
    ["CHOICE"] for the semiparamteric estimation, respectively.
    """

    if semipar is True:
        if "const" not in init_dict["CHOICE"]["order"]:
            init_dict["CHOICE"]["order"].insert(0, "const")
            init_dict["CHOICE"]["params"] = np.array([1.0])
        else:
            pass

    # semipar is False
    else:
        for key in ["TREATED", "UNTREATED", "CHOICE"]:
            if "const" not in init_dict[key]["order"]:
                init_dict[key]["order"].insert(0, "const")
                init_dict[key]["params"] = np.array([1.0])
            else:
                pass

    return init_dict


def read_keys_par(init_dict, init_values):
    """This function reads the information provided by the
    ["TREATED", "UNTREATED", "CHOICE", "DIST"] keys for
    the simulation and parametric estimation.
    """
    for key in ["TREATED", "UNTREATED", "CHOICE", "DIST"]:
        if "params" in init_dict[key].keys():
            init_dict[key]["params"] = np.array(init_dict[key]["params"])
            init_values += list(init_dict[key]["params"])
        else:
            init_values += [0.0] * len(init_dict[key]["order"])

    if np.all(init_dict["DIST"]["params"] == 0):
        init_dict["DETERMINISTIC"] = True
    else:
        init_dict["DETERMINISTIC"] = False

    return init_dict


def read_keys_semipar(init_dict, init_values):
    """This function reads the information provided by the
    ["TREATED", "UNTREATED", "CHOICE"] keys for
    semiparametric estimation.
    """
    for key in ["TREATED", "UNTREATED", "CHOICE"]:
        if "params" in init_dict[key].keys():
            init_dict[key]["params"] = np.array(init_dict[key]["params"])
            init_values += list(init_dict[key]["params"])
        else:
            init_values += [0.0] * len(init_dict[key]["order"])

    return init_dict


def provide_auxiliary_information(init_dict, init_values):
    """This function generates auxiliary information
    given the parameters in the initialization dictionary
    """
    num_covars = len(
        set(
            init_dict["TREATED"]["order"]
            + init_dict["UNTREATED"]["order"]
            + init_dict["CHOICE"]["order"]
        )
    )

    covar_label = []
    for section in ["TREATED", "UNTREATED", "CHOICE"]:
        covar_label += [i for i in init_dict[section]["order"] if i not in covar_label]

        # Generate the AUX section that include some additional auxiliary information
        init_dict["AUX"] = {
            "init_values": np.array(init_values),
            "num_covars_choice": len(init_dict["CHOICE"]["order"]),
            "num_covars_treated": len(init_dict["TREATED"]["order"]),
            "num_covars_untreated": len(init_dict["UNTREATED"]["order"]),
            "num_paras": len(init_values) + 1,
            "num_covars": num_covars,
            "labels": covar_label,
        }

    return init_dict
