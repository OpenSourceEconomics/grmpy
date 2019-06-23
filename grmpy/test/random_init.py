"""The module provides a random dictionary generating process for test purposes."""
import collections
import uuid
from itertools import combinations

import numpy as np
import oyaml as yaml
from scipy.stats import wishart

from grmpy.check.check import UserError


def generate_random_dict(constr=None):
    """The module provides a random dictionary generating process for test purposes."""

    if constr is not None:
        if not isinstance(constr, dict):
            msg = "{} is not a dictionary.".format(constr)
            raise UserError(msg)
    else:
        constr = dict()

    if "DETERMINISTIC" in constr.keys():
        is_deterministic = constr["DETERMINISTIC"]
    else:
        is_deterministic = np.random.random_sample() < 0.1

    if "STATE_DIFF" in constr.keys():
        state_diff = constr["STATE_DIFF"]
    else:
        state_diff = np.random.random_sample() < 0.5

    if "OPTIMIZER" in constr.keys():
        optimizer = constr["OPTIMIZER"]
    else:
        optimizer = str(
            np.random.choice(a=["SCIPY-BFGS", "SCIPY-POWELL"], p=[0.5, 0.5])
        )

    if "SAME_SIZE" in constr.keys():
        same_size = constr["SAME_SIZE"]
    else:
        same_size = False

    if "IS_ZERO" in constr.keys() and not is_deterministic:
        is_zero = np.random.random_sample() < 0.1 / (1 - 0.1)
    else:
        is_zero = False

    if "OVERLAP" in constr.keys():
        overlap = constr["OVERLAP"]
    else:
        overlap = np.random.random_sample() < 0.5

    if "MAXITER" in constr.keys():
        maxiter = constr["MAXITER"]
    else:
        maxiter = np.random.randint(0, 10000)

    if "AGENTS" in constr.keys():
        agents = constr["AGENTS"]

    else:
        agents = np.random.randint(1, 1000)

    if "START" in constr.keys():
        start = constr["START"]
    else:
        start = str(np.random.choice(a=["init", "auto"]))
    if "SEED" in constr.keys():
        seed = constr["SEED"]
    else:
        seed = int(np.random.randint(1, 10000))

    source = str(uuid.uuid4()).upper().replace("-", "")[0:8]

    # Specify the number of variables/parameters for every section
    treated_num = np.random.randint(1, 10)
    choice_num = np.random.randint(1, 10)

    # Determine if there are different variables that affect the outcome states
    if state_diff:
        untreated_num = treated_num + np.random.randint(1, 10)
        choice_num = untreated_num + choice_num
        num = [
            [1, treated_num],
            [treated_num, untreated_num],
            [untreated_num, choice_num],
        ]

    else:
        untreated_num = treated_num
        choice_num = treated_num + choice_num
        num = [[1, treated_num], [1, treated_num], [treated_num, choice_num]]

    init_dict = {}

    # Coefficients
    for counter, section in enumerate(["TREATED", "UNTREATED", "CHOICE"]):
        init_dict[section] = {}
        init_dict[section]["params"], init_dict[section]["order"] = generate_coeff(
            num[counter], is_zero
        )

    # Specify if there are variables that affect a combination of sections
    init_dict = comb_overlap(init_dict, state_diff, overlap)

    # Specify if some variables are binary
    init_dict["VARTYPES"] = {}
    for variable in set(
        init_dict["TREATED"]["order"]
        + init_dict["UNTREATED"]["order"]
        + init_dict["CHOICE"]["order"]
    ):
        init_dict["VARTYPES"][variable] = "nonbinary"

    init_dict = types(init_dict)

    # Simulation parameters
    init_dict["SIMULATION"] = {}
    init_dict["SIMULATION"]["seed"] = seed
    init_dict["SIMULATION"]["agents"] = agents
    init_dict["SIMULATION"]["source"] = source

    # Estimation parameters
    init_dict["ESTIMATION"] = {}
    if same_size:
        init_dict["ESTIMATION"]["agents"] = agents
    else:
        init_dict["ESTIMATION"]["agents"] = np.random.randint(1, 1000)
    init_dict["ESTIMATION"]["file"] = source + ".grmpy.txt"
    init_dict["ESTIMATION"]["optimizer"] = optimizer
    init_dict["ESTIMATION"]["start"] = start
    init_dict["ESTIMATION"]["maxiter"] = maxiter
    init_dict["ESTIMATION"]["dependent"] = "Y"
    init_dict["ESTIMATION"]["indicator"] = "D"
    init_dict["ESTIMATION"]["output_file"] = "est.grmpy.info"
    init_dict["ESTIMATION"]["comparison"] = "0"
    init_dict["ESTIMATION"]["print_output"] = "0"

    init_dict["SCIPY-BFGS"], init_dict["SCIPY-POWELL"] = {}, {}
    init_dict["SCIPY-BFGS"]["gtol"] = np.random.uniform(1.5e-05, 0.8e-05)
    init_dict["SCIPY-BFGS"]["eps"] = np.random.uniform(
        1.4901161193847655e-08, 1.4901161193847657e-08
    )
    init_dict["SCIPY-POWELL"]["xtol"] = np.random.uniform(0.00009, 0.00011)
    init_dict["SCIPY-POWELL"]["ftol"] = np.random.uniform(0.00009, 0.00011)

    # Variance and covariance parameters
    init_dict["DIST"] = {}
    if not is_deterministic:
        scale_matrix = np.identity(3) * 0.1
        b = wishart.rvs(df=10, scale=scale_matrix, size=1)
        for i in [0, 1, 2]:
            b[i, i] = b[i, i] ** 0.5
    else:
        b = np.zeros((3, 3))
    init_dict["DIST"]["params"] = np.around(
        [float(i) for i in list(b[np.triu_indices(3)])], 4
    ).tolist()

    print_dict(init_dict)

    return init_dict


def generate_coeff(num, is_zero):
    """The function generates random coefficients for creating the random init
    dictionary.
    """

    # Generate a random paramterization and specify the variable order
    if not is_zero:
        params = np.around(
            np.random.normal(0.0, 2.0, [len(range(num[0] - 1, num[1]))]), 4
        ).tolist()
    else:
        params = np.array([0] * num).tolist()

    order = ["X1"] + ["X{}".format(i + 1) for i in range(num[0], num[1])]

    return params, order


def types(init_dict):
    """This function determines if there are any binary variables. If so the funtion
    specifies the rate for which the variable is equal to one.
    """

    variables = [i for i in init_dict["VARTYPES"].keys() if i != "X1"]
    for var in variables:
        if np.random.random_sample() < 0.1:
            frac = np.random.uniform(0, 0.8)
            init_dict["VARTYPES"][var] = ["binary", frac]
        else:
            pass

    return init_dict


def comb_overlap(init_dict, state_diff, overlap):
    """This function evaluates which variables affect more than one section."""

    # List all possible overlaps between the different sections and chose a random
    # combination
    if state_diff and overlap:
        cases = [list(i) for i in combinations(list(init_dict.keys()), 2)] + [
            list(init_dict.keys())
        ]
        case = np.random.choice(cases)
        case = [i for i in case if len(init_dict[i]["order"]) > 1]
    elif not state_diff and overlap:
        case = list(init_dict.keys())
        case = [i for i in case if len(init_dict[i]["order"]) > 1]
    else:
        case = []

    # Select a random number of variables that effect the chosen combination of sections
    if len(case) != 0:
        aux_dict = {j: len(init_dict[j]["order"]) for j in case}
        min_key = min(aux_dict, key=aux_dict.get)
        num_overlap = np.random.choice(range(1, aux_dict[min_key]))
        for section in case:
            init_dict[section]["order"][1 : 1 + num_overlap] = init_dict[min_key][
                "order"
            ][1 : num_overlap + 1]

    return init_dict


def print_dict(init_dict, file_name="test"):
    """This function prints the initialization dict as a yaml file."""

    # Transfer the init dict in an ordered one to ensure that the init file is aligned
    #  appropriately
    ordered_dict = collections.OrderedDict()
    order = [
        "SIMULATION",
        "ESTIMATION",
        "TREATED",
        "UNTREATED",
        "CHOICE",
        "DIST",
        "VARTYPES",
        "SCIPY-BFGS",
        "SCIPY-POWELL",
    ]

    for key_ in order:
        ordered_dict[key_] = init_dict[key_]
    for section in ["TREATED", "CHOICE", "UNTREATED", "DIST"]:
        if isinstance(ordered_dict[section]["params"], list):
            pass
        else:
            ordered_dict[section]["params"] = ordered_dict[section]["params"].tolist()

    # Print the initialization file
    with open("{}.grmpy.yml".format(file_name), "w") as outfile:
        yaml.dump(
            ordered_dict,
            outfile,
            explicit_start=True,
            indent=4,
            width=99,
            default_flow_style=False,
        )
