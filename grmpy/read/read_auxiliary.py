"""This module provides auxiliary functions for the import process of the init file."""
import numpy as np


def create_attr_dict(init_dict, semipar=False):
    """This function processes the imported initialization file so that it fulfills the
    requirements for the subsequent simulation and estimation process.
    """
    init_dict["AUX"] = {"init_values"}
    init_values = []

    if semipar is True:

        # Include constant if not provided by the user
        if "const" not in init_dict["CHOICE"]["order"]:
            init_dict["CHOICE"]["order"].insert(0, "const")
            init_dict["CHOICE"]["params"] = np.array([1.0])
        else:
            pass

        for key in ["TREATED", "UNTREATED", "CHOICE"]:
            if "params" in init_dict[key].keys():
                init_dict[key]["params"] = np.array(init_dict[key]["params"])
                init_values += list(init_dict[key]["params"])
            else:
                init_values += [0.0] * len(init_dict[key]["order"])

    # semipar is False
    else:

        # Include constant if not provided by the user
        for key in ["TREATED", "UNTREATED", "CHOICE"]:
            if "const" not in init_dict[key]["order"]:
                init_dict[key]["order"].insert(0, "const")
                init_dict[key]["params"] = np.array([1.0])
            else:
                pass

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

    #
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
