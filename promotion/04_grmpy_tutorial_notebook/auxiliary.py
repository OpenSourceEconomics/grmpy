"""This module contains auxiliary function which are used in the grmpy application notebook."""

import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm
import seaborn as sns
import pandas as pd
import numpy as np
import linecache
import shlex
import json

from linearmodels.iv import IV2SLS
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker


from grmpy.simulate.simulate_auxiliary import simulate_unobservables
from grmpy.estimate.estimate_output import calculate_mte
from grmpy.test.random_init import print_dict
from grmpy.estimate.estimate import fit
from grmpy.read.read import read


def process_data(df, output_file):
    """This function adds squared and interaction terms to the Cainero data set."""

    # Delete redundant columns\n",
    for key_ in ["newid", "caseid"]:
        del df[key_]

    # Add squared terms
    for key_ in ["mhgc", "cafqt", "avurate", "lurate_17", "numsibs", "lavlocwage17"]:
        str_ = key_ + "sq"
        df[str_] = df[key_] ** 2

    # Add interaction terms
    for j in ["pub4", "lwage5_17", "lurate_17", "tuit4c"]:
        for i in ["cafqt", "mhgc", "numsibs"]:
            df[j + i] = df[j] * df[i]

    df.to_pickle(output_file + ".pkl")


def effects(data):
    """This function plots the distribution of benefits and the related conventional effects."""
    from pylab import rcParams

    rcParams["figure.figsize"] = 15, 10

    benefit = data["Y1"] - data["Y0"]
    TT = np.mean(data[data.D == 1]["Y1"] - data[data.D == 1]["Y0"])
    TUT = np.mean(data[data.D == 0]["Y1"] - data[data.D == 0]["Y0"])
    ATE = np.mean(benefit)
    fmt = "ATE: {}\nTT:  {}\nTUT: {} \n"
    print(fmt.format(ATE, TT, TUT))
    ay = plt.figure().add_subplot(111)

    sns.distplot(benefit, kde=True, hist=False)

    ay.set_xlim(-1.5, 2.5)
    ay.set_ylim(0.0, None)
    ay.set_yticks([])

    # Rename axes
    ay.set_ylabel("$f_{Y_1 - Y_0}$", fontsize=16)
    ay.set_xlabel("$Y_1 - Y_0$", fontsize=16)

    for effect in [ATE, TT, TUT]:
        if effect == ATE:
            label = "$B^{ATE}$"
        elif effect == TT:
            label = "$B^{TT}$"
        else:
            label = "$B^{TUT}$"
        ay.plot([effect, effect], [0, 5], label=label)
    plt.legend(prop={"size": 15})


def update_tutorial(file, rho=None):
    """This function enables us to rewrite the grmpy tutorial file so that it correspond to a
    parameterization with essential heterogeneity"""

    if rho is None:
        rho = []
        rho += [np.random.uniform(0.3, 0.7, 1)]
        rho += [np.random.uniform(-0.1, -0.345, 1)]

    init_dict = read(file)

    init_dict["SIMULATION"]["source"] = "data_eh"

    sd1 = init_dict["DIST"]["params"][0]
    sd0 = init_dict["DIST"]["params"][3]
    sdv = init_dict["DIST"]["params"][-1]

    init_dict["DIST"]["params"][2] = sd1 * sdv * rho[0]

    init_dict["DIST"]["params"][-2] = sd0 * sdv * rho[1]

    print_dict(init_dict, "files/tutorial_eh")


def create_data(file):
    """This function creates the a data set based for the monte carlo simulation setup."""
    # Read in initialization file and the data set
    init_dict = read(file)
    df = pd.read_pickle(init_dict["SIMULATION"]["source"] + ".grmpy.pkl")

    # Distribute information
    indicator, dep = init_dict["ESTIMATION"]["indicator"], init_dict["ESTIMATION"]["dependent"]
    label_out = init_dict["TREATED"]["order"]
    label_choice = init_dict["CHOICE"]["order"]
    seed = init_dict["SIMULATION"]["seed"]

    # Set random seed to ensure recomputabiltiy
    np.random.seed(seed)

    # Simulate unobservables
    U = simulate_unobservables(init_dict)

    df["U1"], df["U0"], df["V"] = U["U1"], U["U0"], U["V"]

    # Simulate choice and output
    df[dep + "1"] = np.dot(df[label_out], init_dict["TREATED"]["params"]) + df["U1"]
    df[dep + "0"] = np.dot(df[label_out], init_dict["UNTREATED"]["params"]) + df["U0"]
    df[indicator] = np.array(
        np.dot(df[label_choice], init_dict["CHOICE"]["params"]) - df["V"] > 0
    ).astype(int)
    df[dep] = df[indicator] * df[dep + "1"] + (1 - df[indicator]) * df[dep + "0"]

    # Save the data
    df.to_pickle(init_dict["SIMULATION"]["source"] + ".grmpy.pkl")

    return df


def update_correlation_structure(model_dict, rho):
    """This function takes a valid model specification and updates the correlation structure
    among the unobservables."""

    # We first extract the baseline information from the model dictionary.
    sd_v = model_dict["DIST"]["params"][-1]
    sd_u1 = model_dict["DIST"]["params"][0]

    # Now we construct the implied covariance, which is relevant for the initialization file.
    cov1v = rho * sd_v * sd_u1

    model_dict["DIST"]["params"][2] = cov1v

    # We print out the specification to an initialization file with the name mc_init.grmpy.ini.
    print_dict(model_dict, "files/mc")


def get_effect_grmpy(file):
    """This function simply returns the ATE of the data set."""
    dict_ = read(file)
    df = pd.read_pickle(dict_["SIMULATION"]["source"] + ".grmpy.pkl")
    beta_diff = dict_["TREATED"]["params"] - dict_["UNTREATED"]["params"]
    covars = dict_["TREATED"]["order"]
    ATE = np.dot(np.mean(df[covars]), beta_diff)

    return ATE


def monte_carlo(file, grid_points):
    """This function estimates the ATE for a sample with different correlation structures between U1
     and V. Two different strategies for (OLS,LATE) are implemented.
     """

    ATE = 0.5

    # Define a dictionary with a key for each estimation strategy
    effects = {}
    for key_ in ["grmpy", "ols", "true", "random", "rho", "iv", "means"]:
        effects[key_] = []

    # Loop over different correlations between V and U_1
    for rho in np.linspace(0.00, 0.99, grid_points):
        effects["rho"] += [rho]
        # Readjust the initialization file values to add correlation
        model_spec = read(file)
        X = model_spec["TREATED"]["order"]
        update_correlation_structure(model_spec, rho)
        sim_spec = read(file)
        # Simulate a Data set and specify exogeneous and endogeneous variables
        df_mc = create_data(file)
        endog, exog, exog_ols = df_mc["wage"], df_mc[X], df_mc[["state"] + X]
        instr = sim_spec["CHOICE"]["order"]
        instr = [i for i in instr if i != "const"]
        # Calculate true average treatment effect
        ATE = np.mean(df_mc["wage1"] - df_mc["wage0"])
        effects["true"] += [ATE]

        # Estimate  via grmpy
        rslt = fit(file)
        beta_diff = rslt["TREATED"]["params"] - rslt["UNTREATED"]["params"]
        stat = np.dot(np.mean(exog), beta_diff)

        effects["grmpy"] += [stat]

        # Estimate via OLS
        ols = sm.OLS(endog, exog_ols).fit()
        stat = ols.params[0]
        effects["ols"] += [stat]

        # Estimate via 2SLS
        iv = IV2SLS(endog, exog, df_mc["state"], df_mc[instr]).fit()
        stat = iv.params["state"]
        effects["iv"] += [stat]

        # Estimate via random
        random = np.mean(df_mc[df_mc.state == 1]["wage"]) - np.mean(df_mc[df_mc.state == 0]["wage"])
        stat = random
        effects["random"] += [stat]

        # outcomes
        stat = [
            [np.mean(df_mc[df_mc.state == 1]["wage"]), df_mc[df_mc.state == 1].shape[0]],
            [np.mean(df_mc[df_mc.state == 0]["wage"]), df_mc[df_mc.state == 0].shape[0]],
        ]
        effects["means"] += stat

    create_plots(effects, effects["true"])


def create_plots(effects, true):
    """The function creates the figures that illustrates the behavior of each estimator of the ATE
    when the correlation structure changes from 0 to 1."""

    grid = np.linspace(0.00, 0.99, len(effects["ols"]))

    # Plot all graphs in one plot
    ax2 = plt.figure(figsize=(17.5, 10)).add_subplot(111)
    ax2.set_xlim([-0.005, 1.005])
    ax2.set_ylim(0.375, 0.625)
    ax2.tick_params(axis="both", which="major", labelsize=18)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax2.set_ylabel(r"$B^{ATE}$", fontsize=20)
    ax2.set_xlabel(r"$\rho_{U_1, V}$", fontsize=20)
    ax2.plot(grid, true, label="True", color="blue", linewidth=3.0)
    ax2.plot(grid, effects["grmpy"], label="grmpy", color="orange", linewidth=3.0)
    ax2.plot(grid, effects["random"], label="Naive comparison", color="green", linewidth=3.0)
    ax2.plot(grid, effects["iv"], label="Instrumental variables", color="red", linewidth=3.0)
    ax2.plot(grid, effects["ols"], label="Ordinary Leastsquares", color="purple", linewidth=3.0)
    plt.rc("xtick", labelsize=18)
    plt.rc("ytick", labelsize=18)

    blue_patch = mpatches.Patch(color="blue", label="True")
    orange_patch = mpatches.Patch(color="orange", label="grmpy")
    green_patch = mpatches.Patch(color="green", label="Naive comparison")
    red_patch = mpatches.Patch(color="red", label="Instrumental Variables")
    purple_patch = mpatches.Patch(color="purple", label="Ordinary Least Squares")

    plt.legend(
        handles=[blue_patch, orange_patch, green_patch, red_patch, purple_patch], prop={"size": 13}
    )

    plt.show()


def plot_est_mte(rslt, file):
    """This function calculates the marginal treatment effect for different quartiles of the
    unobservable V. ased on the calculation results."""

    init_dict = read(file)
    data_frame = pd.read_pickle(init_dict["ESTIMATION"]["file"])

    # Define the Quantiles and read in the original results
    quantiles = [0.0001] + np.arange(0.01, 1.0, 0.01).tolist() + [0.9999]
    mte_ = json.load(open("data/mte_original.json", "r"))
    mte_original = mte_[1]
    mte_original_d = mte_[0]
    mte_original_u = mte_[2]

    # Calculate the MTE and confidence intervals
    mte = calculate_mte(rslt, init_dict, data_frame, quantiles)
    mte = [i / 4 for i in mte]
    mte_up, mte_d = calculate_cof_int(rslt, init_dict, data_frame, mte, quantiles)

    # Plot both curves
    ax = plt.figure(figsize=(17.5, 10)).add_subplot(111)

    ax.set_ylabel(r"$B^{MTE}$", fontsize=24)
    ax.set_xlabel("$u_D$", fontsize=24)
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.plot(quantiles, mte, label="grmpy $B^{MTE}$", color="blue", linewidth=4)
    ax.plot(quantiles, mte_up, color="blue", linestyle=":", linewidth=3)
    ax.plot(quantiles, mte_d, color="blue", linestyle=":", linewidth=3)
    ax.plot(quantiles, mte_original, label="original$B^{MTE}$", color="orange", linewidth=4)
    ax.plot(quantiles, mte_original_d, color="orange", linestyle=":", linewidth=3)
    ax.plot(quantiles, mte_original_u, color="orange", linestyle=":", linewidth=3)
    ax.set_ylim([-0.41, 0.51])
    ax.set_xlim([-0.005, 1.005])

    blue_patch = mpatches.Patch(color="blue", label="original $B^{MTE}$")
    orange_patch = mpatches.Patch(color="orange", label="grmpy $B^{MTE}$")
    plt.legend(handles=[blue_patch, orange_patch], prop={"size": 16})
    plt.show()

    return mte


def calculate_cof_int(rslt, init_dict, data_frame, mte, quantiles):
    """This function calculates the confidence interval of the marginal treatment effect."""

    # Import parameters and inverse hessian matrix
    hess_inv = rslt["AUX"]["hess_inv"] / data_frame.shape[0]
    params = rslt["AUX"]["x_internal"]

    # Distribute parameters
    dist_cov = hess_inv[-4:, -4:]
    param_cov = hess_inv[:46, :46]
    dist_gradients = np.array([params[-4], params[-3], params[-2], params[-1]])

    # Process data
    covariates = init_dict["TREATED"]["order"]
    x = np.mean(data_frame[covariates]).tolist()
    x_neg = [-i for i in x]
    x += x_neg
    x = np.array(x)

    # Create auxiliary parameters
    part1 = np.dot(x, np.dot(param_cov, x))
    part2 = np.dot(dist_gradients, np.dot(dist_cov, dist_gradients))
    # Prepare two lists for storing the values
    mte_up = []
    mte_d = []

    # Combine all auxiliary parameters and calculate the confidence intervals
    for counter, i in enumerate(quantiles):
        value = part2 * (norm.ppf(i)) ** 2
        aux = np.sqrt(part1 + value) / 4
        mte_up += [mte[counter] + norm.ppf(0.95) * aux]
        mte_d += [mte[counter] - norm.ppf(0.95) * aux]

    return mte_up, mte_d


def plot_joint_distribution_unobservables(df, df_eh):
    """This function plots the joint distribution of the relevant unobservables."""
    g1 = sns.jointplot(df["V"], df["U1"], stat_func=None).set_axis_labels(
        "$V$", "$U_1$", fontsize=15
    )
    g1.fig.subplots_adjust(top=0.9)
    g1.fig.suptitle("Abscence of essential heterogeneity", fontsize=18)

    g2 = sns.jointplot(df_eh["V"], df_eh["U1"], stat_func=None).set_axis_labels(
        "$V$", "$U_1$", fontsize=15
    )
    g2.fig.subplots_adjust(top=0.9)
    g2.fig.suptitle("Presence of essential heterogeneity", fontsize=18)


def plot_marginal_effects(file1, file2):
    """This function plots the marginal effect of treatment given the output files of a grmpy
    simulation process.
    """
    ax = plt.figure().add_subplot(111)

    ax.set_xlim(0, 1)
    ax.set_ylabel(r"$B^{MTE}$", fontsize=18)
    ax.set_xlabel("$u_S$", fontsize=18)

    for fname in [file1, file2]:
        parameter = []
        linecache.clearcache()
        for num in range(40, 60):
            line = linecache.getline(fname, num)
            parameter += [float(shlex.split(line)[1])]

        if parameter.count(parameter[0]) == len(parameter):
            label = "Absence"
        else:
            label = "Presence"

        grid = np.linspace(0.01, 1, num=20, endpoint=True)
        ax.plot(grid, parameter, label=label)

    plt.legend(prop={"size": 18})

    plt.show()


def plot_joint_distribution_potential(df):
    """This function plots the joint distribution of potential outcomes."""
    sns.jointplot(df["Y1"], df["Y0"], stat_func=None).set_axis_labels(
        "$Y_1$", r"$Y_0$", fontsize=15
    )
