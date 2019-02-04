"""This module provides auxiliary functions for the import process of the init file."""
import numpy as np


def init_dict_to_attr_dict(init_dict):
    """This function processes the imported initialization file so that it fulfills the requirements
     for the following simulation and estimation process.
    """

    init_dict["AUX"] = {"init_values"}

    init_values = []
    for key in ["TREATED", "UNTREATED", "CHOICE", "DIST"]:
        if "params" in init_dict[key].keys():
            init_dict[key]["params"] = np.array(init_dict[key]["params"])
            init_values += list(init_dict[key]["params"])
        else:
            init_values += [0.0] * len(init_dict[key]["order"])

    num_covars = len(
        set(
            init_dict["TREATED"]["order"]
            + init_dict["UNTREATED"]["order"]
            + init_dict["CHOICE"]["order"]
        )
    )

    if np.all(init_dict["DIST"]["params"] == 0):
        init_dict["DETERMINISTIC"] = True
    else:
        init_dict["DETERMINISTIC"] = False

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
