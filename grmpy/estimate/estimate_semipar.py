"""
This module contains the semiparametric estimation process.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from grmpy.KernReg.locpoly import locpoly

from skmisc.loess import loess

lowess = sm.nonparametric.lowess


def semipar_fit(dict_, data):
    """"This function runs the semiparametric estimation via
    local instrumental variables"""
    # Process the information specified in the initialization file
    nbins, logit, bandwidth, gridsize, a, b = process_user_input(dict_)
    trim, rbandwidth, reestimate_p = process_default_input(dict_)

    show_output = dict_["ESTIMATION"]["show_output"]

    # Prepare the sample for the estimation process
    # Compute propensity score, define common support and trim the sample
    data, ps = process_mte_data(
        dict_, data, logit, nbins, trim, reestimate_p, show_output
    )

    # Estimate the observed and unobserved component of the MTE
    X, b1_b0, b0, mte_u = mte_components(
        dict_, data, ps, rbandwidth, bandwidth, gridsize, a, b, show_output
    )

    # Generate the quantiles of the final MTE
    quantiles = np.linspace(a, b, gridsize)

    # Construct the MTE
    # Calculate the MTE component that depends on X
    mte_x = np.dot(X, b1_b0)

    # Put the MTE together
    mte = mte_x.mean(axis=0) + mte_u

    # Account for variation in X
    mte_min = np.min(mte_x) + mte_u
    mte_max = np.max(mte_x) + mte_u

    b1 = np.array(b1_b0) + np.array(b0)

    rslt = {
        "quantiles": quantiles,
        "mte": mte,
        "mte_x": mte_x,
        "mte_u": mte_u,
        "mte_min": mte_min,
        "mte_max": mte_max,
        "X": X,
        "b1": b1,
        "b0": b0,
    }

    return rslt


def process_user_input(dict_):
    """This functions processes the input parameters that need to be
    specified by the user"""
    nbins = dict_["ESTIMATION"]["nbins"]
    logit = dict_["ESTIMATION"]["logit"]
    bandwidth = dict_["ESTIMATION"]["bandwidth"]
    gridsize = dict_["ESTIMATION"]["gridsize"]
    a = dict_["ESTIMATION"]["ps_range"][0]
    b = dict_["ESTIMATION"]["ps_range"][1]

    return nbins, logit, bandwidth, gridsize, a, b


def process_default_input(dict_):
    """This functions processes the default input parameters"""
    try:
        dict_["ESTIMATION"]["trim_support"]
    # Set default to True
    except KeyError:
        trim = True
    else:
        trim = dict_["ESTIMATION"]["trim_support"]

    try:
        dict_["ESTIMATION"]["reestimate_p"]
    # Set default to False
    except KeyError:
        reestimate_p = False
    else:
        reestimate_p = dict_["ESTIMATION"]["reestimate_p"]

    try:
        dict_["ESTIMATION"]["rbandwidth"]
    # Set default to 0.05
    except KeyError:
        rbandwidth = 0.05
    else:
        rbandwidth = dict_["ESTIMATION"]["rbandwidth"]

    return trim, rbandwidth, reestimate_p


def process_choice_data(dict_, data):
    """This functions processes the inputs for the
    decision equation"""
    indicator = dict_["ESTIMATION"]["indicator"]
    D = data[indicator].values
    Z = data[dict_["CHOICE"]["order"]]

    return indicator, D, Z


def process_mte_data(dict_, data, logit, nbins, trim, reestimate_p, show_output):
    """This functions prepares the data for the semiparametric estimation stage"""
    indicator, D, Z = process_choice_data(dict_, data)

    # Estimate propensity score P(z)
    ps = estimate_treatment_propensity(D, Z, logit, show_output)

    # Define common support and trim the data, if trim=True
    data, ps = trim_support(
        dict_, data, logit, ps, indicator, nbins, trim, reestimate_p, show_output
    )

    return data, ps


def trim_support(
    dict_, data, logit, ps, indicator, nbins, trim, reestimate_p, show_output
):
    """This function defines common support and trims the data.
    Optionally p is re-estimated on the trimmed sample"""
    # Find common support
    common_support = define_common_support(ps, indicator, data, nbins, show_output)

    # Trim the data
    if trim is True:
        data, ps = trim_data(ps, common_support, data)

    # Optional. Not recommended
    # Re-estimate baseline propensity score on the trimmed sample
    if reestimate_p is True:
        # Re-estimate the parameters of the decision equation based
        # on the new trimmed data set
        indicator, D, Z = process_choice_data(dict_, data)

        # Re-estimate propensity score P(z)
        ps = estimate_treatment_propensity(D, Z, logit, show_output)

    else:
        pass

    data = data.sort_values(by="ps", ascending=True)
    ps = np.sort(ps)

    return data, ps


def define_common_support(
    ps,
    indicator,
    data,
    nbins,
    show_output,
    figsize=(10, 6),
    fontsize=15,
    plot_title=False,
    save_output=False,
):
    """
    This function defines the common support as the region under the histograms
    where propensities in the treated and untreated subsample overlap.

    Carneiro et al (2011) choose 25 bins for a total sample of 1747
    observations, so nbins=25 is set as a default.
    """
    hist, treated, untreated = plot_common_support(
        ps, indicator, data, nbins, show_output, figsize, fontsize, plot_title
    )

    lower_limit, upper_limit = find_limits_support(hist, treated, untreated)
    common_support = [lower_limit, upper_limit]

    if show_output is True:
        print(
            """
    Common support lies beteen:

        {0} and
        {1}""".format(
                lower_limit, upper_limit
            )
        )

    if save_output is not False:
        plt.savefig(save_output, dpi=300)

    return common_support


def mte_components(dict_, data, ps, rbandwidth, bandwidth, gridsize, a, b, show_output):
    """This functions produces the observed on unobserved components
    of the final MTE"""
    # Double Residual Regression
    X = data[dict_["TREATED"]["order"]]
    Xp = construct_Xp(X, ps)
    Y = data[[dict_["ESTIMATION"]["dependent"]]]

    b0, b1_b0 = double_residual_reg(ps, X, Xp, Y, rbandwidth, show_output)

    # Turn the X, Xp, and Y DataFrames into np.ndarrays
    X_arr = np.array(X)
    Xp_arr = np.array(Xp)
    Y_arr = np.array(Y).ravel()

    # Compute the unobserved part of Y
    Y_tilde = Y_arr - np.dot(X_arr, b0) - np.dot(Xp_arr, b1_b0)

    # Estimate mte_u, the unobserved component of the MTE,
    # through a locally quadratic regression
    mte_u = locpoly(ps, Y_tilde, 1, 2, bandwidth, gridsize, a, b)

    return X, b1_b0, b0, mte_u


def estimate_treatment_propensity(D, Z, logit, show_output):
    """
    This function estimates the propensity of selecting into treatment
    for both treated and untreated individuals based on instruments Z.
    Z subsumes all the observable components that influence the treatment
    decision, e.g. the decision to enroll into college (D = 1) or not (D = 0).

    Estimate propensity scores via Logit (default) or Probit.
    """
    if logit is True:
        logitRslt = sm.Logit(D, Z).fit(disp=0)
        ps = logitRslt.predict(Z)

        if show_output is True:
            print(logitRslt.summary())

    else:
        probitRslt = sm.Probit(D, Z).fit(disp=0)
        ps = probitRslt.predict(Z)

        if show_output is True:
            print(probitRslt.summary())

    return ps.values


def plot_common_support(
    ps, indicator, data, nbins, show_output, figsize, fontsize, plot_title
):
    data.loc[:, "ps"] = ps

    treated = data[[indicator, "ps"]][data[indicator] == 1].values
    untreated = data[[indicator, "ps"]][data[indicator] == 0].values

    treated = treated[:, 1].tolist()
    untreated = untreated[:, 1].tolist()

    ltreat = len(treated)
    luntreat = len(untreated)

    # Make the histogram using a list of lists
    fig = plt.figure(figsize=figsize)
    hist = plt.hist(
        [treated, untreated],
        bins=nbins,
        weights=[np.ones(ltreat) / ltreat, np.ones(luntreat) / luntreat],
        density=0,
        alpha=0.55,
        label=["Treated", "Unreated"],
    )

    if show_output is True:
        # Plot formatting
        plt.tick_params(axis="both", labelsize=14)
        plt.legend(loc="upper right", prop={"size": 14})
        plt.xticks(np.arange(0, 1.1, step=0.1))
        plt.grid(axis="y", alpha=0.25)
        plt.xlabel("$P$", fontsize=fontsize)
        plt.ylabel("$f(P)$", fontsize=fontsize)

        if plot_title is True:
            plt.title("Support of $P(\hat{Z})$ for $D=1$ and $D=0$")

    else:
        plt.close(fig)

    return hist, treated, untreated


def find_limits_support(hist, treated, untreated):
    """Find the upper and lower limit of the common support"""
    # Treated Sample
    # Set the lowest frequency observed in the treated subsample
    # as the default for the lower limit of the common support
    lower_limit = np.min(treated)

    # The following algorithm checks for any empty histogram bins
    # (starting from 0 going up to 0.5).
    # If an empty histogram bin is found, the lower_limit is set to
    # the corresponding P(Z) value of the next bin above.
    for low in range(len(hist[0][0])):

        # Only consider values in the interval [0, 0.5)
        if hist[1][low] > 0.5:
            break

        # If the algorithm starts below the sample minimum,
        # move on to the next bin
        elif hist[1][low] < np.min(treated):
            continue

        else:
            # If the current bin is non-empty, we have still continuous
            # support and the sample minimum remains our lower limit
            if hist[0][0][low] > 0:
                pass

            # If an empty bin is found, set the lower limit to the next bin above
            # and move on to the next bin until P(Z) = 0.5 is reached
            else:
                lower_limit = hist[1][low + 1]

    # Untreated Sample
    # Set the highest frequency observed in the untreated subsample
    # as the default for the upper limit of the common support
    upper_limit = np.max(untreated)

    # The following algorithm checks for any empty histogram bins
    # (starting from 1 going down to 0.5).
    # If an empty histogram bin is found, the upper_limit is set to the
    # current next bin.
    for up in reversed(range(len(hist[0][1]))):

        # Only consider values in the interval (0.5, 1]
        if hist[1][up] < 0.5:
            break

        # If the algorithm starts above the sample maximum, move on to the next bin
        elif hist[1][up] > np.max(untreated):
            continue

        else:
            # If the current bin is non-empty, we have still continuous support and
            # the sample maximum remains our upper limit
            if hist[0][1][up] > 0:
                pass

            # If an empty bin is found, set the upper limit to the next bin below
            # and move on to the next bin until P(Z) = 0.5 is reached
            else:
                upper_limit = hist[1][up]

    return lower_limit, upper_limit


def trim_data(ps, common_support, data):
    """This function trims the data below and above the common support."""
    data_trim = data[(data.ps >= common_support[0]) & (data.ps <= common_support[1])]
    ps_trim = ps[(ps >= common_support[0]) & (ps <= common_support[1])]

    return data_trim, ps_trim


def construct_Xp(X, ps):
    """
    This function generates the X * ps regressors.
    """
    # To multiply each elememt in X (shape N x k) with the corresponding ps,
    # set up a ps matrix of same size.
    ps = pd.Series(ps)
    P_z = pd.concat([ps] * len(X.columns), axis=1, ignore_index=True)

    # Construct Xp
    Xp = pd.DataFrame(
        X.values * P_z.values, columns=[key_ + "_ps" for key_ in list(X)], index=X.index
    )

    return Xp


def generate_residuals(x, y, bandwidth=0.05):
    """
    This function runs a series of loess regressions for different
    response variables (y) on a single explanatory variable (x)
    and computes the corresponding residuals.
    """
    # Turn input data into np.ndarrays.
    y = np.array(y)
    x = np.array(x)

    # Determine number of observations and number of columns for the
    # outcome variable.
    n = len(y)
    col_len = len(y[0])

    res = np.zeros([n, col_len])

    for i in range(col_len):
        yfit = loess(x, y[:, i], span=bandwidth, degree=1)
        yfit.fit()
        res[:, i] = yfit.outputs.fitted_residuals

    return res


def double_residual_reg(ps, X, Xp, Y, rbandwidth, show_output):
    """
    This function performs a Double Residual Regression of X, Xp, and Y on ps.

    A local linear kernel regression (polynomial of degree 1)
    is implemented to generate the residuals.
    """
    # 1) Fit a separate local linear regression of X, Xp, and Y on ps,
    # which yields residuals e_X, e_Xp, and e_Y.
    res_X = generate_residuals(ps, X, rbandwidth)
    res_Xp = generate_residuals(ps, Xp, rbandwidth)
    res_Y = generate_residuals(ps, Y, rbandwidth)

    # Append res_X and res_Xp.
    col_names = list(X) + list(Xp)
    res_X_Xp = pd.DataFrame(np.append(res_X, res_Xp, axis=1), columns=col_names)

    # 2) Run a single OLS regression of e_Y on e_X and e_Xp without intercept:
    # e_Y = e_X * beta_0 + e_Xp * (beta_1 - beta_0),
    # to estimate the values of beta_0 and (beta_1 - beta_0).
    model = sm.OLS(res_Y, res_X_Xp)
    results = model.fit()
    b0 = results.params[: len(list(X))]
    b1_b0 = results.params[len((list(X))) :]

    if show_output is True:
        print(results.summary())

    return b0, b1_b0
