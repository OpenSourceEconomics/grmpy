"""
This module contains the semiparametric estimation process.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from skmisc.loess import loess

from grmpy.check.auxiliary import read_data
from grmpy.KernReg.locpoly import locpoly


def semipar_fit(dict_):
    """This functions estimates the MTE via Local Instrumental Variables"""
    # Read the data
    data = read_data(dict_["ESTIMATION"]["file"])

    # Process data for the semiparametric estimation.
    indicator = dict_["ESTIMATION"]["indicator"]
    D = data[indicator].values
    Z = data[dict_["CHOICE"]["order"]]

    nbins = dict_["ESTIMATION"]["nbins"]
    trim = dict_["ESTIMATION"]["trim_support"]
    reestimate = dict_["ESTIMATION"]["reestimate_p"]
    rbandwidth = dict_["ESTIMATION"]["rbandwidth"]
    bandwidth = dict_["ESTIMATION"]["bandwidth"]
    gridsize = dict_["ESTIMATION"]["gridsize"]
    a = dict_["ESTIMATION"]["ps_range"][0]
    b = dict_["ESTIMATION"]["ps_range"][1]

    logit = dict_["ESTIMATION"]["logit"]
    show_output = dict_["ESTIMATION"]["show_output"]

    # The Local Instrumental Variables (LIV) estimator
    # 1. Estimate propensity score P(z)
    ps = estimate_treatment_propensity(D, Z, logit, show_output)

    # 2a. Find common support
    treated, untreated, common_support = define_common_support(
        ps, indicator, data, nbins, show_output
    )

    # 2b. Trim the data
    if trim is True:
        data, ps = trim_data(ps, common_support, data)

    # 2c. Re-estimate baseline propensity score on the trimmed sample
    if reestimate is True:
        D = data[indicator].values
        Z = data[dict_["CHOICE"]["order"]]

        # Re-estimate propensity score P(z)
        ps = estimate_treatment_propensity(D, Z, logit, show_output)

    # 3. Double Residual Regression
    # Sort data by ps
    data = data.sort_values(by="ps", ascending=True)
    ps = np.sort(ps)

    X = data[dict_["TREATED"]["order"]]
    Xp = construct_Xp(X, ps)
    Y = data[[dict_["ESTIMATION"]["dependent"]]]

    b0, b1_b0 = double_residual_reg(ps, X, Xp, Y, rbandwidth, show_output)

    # Turn the X, Xp, and Y DataFrames into np.ndarrays
    X_arr = np.array(X)
    Xp_arr = np.array(Xp)
    Y_arr = np.array(Y).ravel()

    # 4. Compute the unobserved part of Y
    Y_tilde = Y_arr - np.dot(X_arr, b0) - np.dot(Xp_arr, b1_b0)

    # 5. Estimate mte_u, the unobserved component of the MTE,
    # through a locally quadratic regression
    quantiles, mte_u = locpoly(ps, Y_tilde, 1, 2, bandwidth, gridsize, a, b)

    return quantiles, mte_u, X, b1_b0


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


def define_common_support(ps, indicator, data, nbins=25, show_output=True):
    """
    This function defines the common support as the region under the histograms
    where propensities in the treated and untreated subsample overlap.

    Carneiro et al (2011) choose 25 bins for a total sample of 1747
    observations, so nbins=25 is set as a default.
    """
    data["ps"] = ps

    treated = data[[indicator, "ps"]][data[indicator] == 1].values
    untreated = data[[indicator, "ps"]][data[indicator] == 0].values

    treated = treated[:, 1].tolist()
    untreated = untreated[:, 1].tolist()

    # Make the histogram using a list of lists
    fig = plt.figure(figsize=(10, 6))
    hist = plt.hist(
        [treated, untreated],
        bins=nbins,
        weights=[
            np.ones(len(treated)) / len(treated),
            np.ones(len(untreated)) / len(untreated),
        ],
        density=0,
        alpha=0.55,
        label=["Treated", "Unreated"],
    )

    if show_output is True:
        # Plot formatting
        plt.legend(loc="upper right")
        plt.xticks(np.arange(0, 1.1, step=0.1))
        plt.grid(axis="y", alpha=0.25)
        plt.xlabel("$P$")
        plt.ylabel("$f(P)$")
        plt.title("Support of $P(\hat{Z})$ for $D=1$ and $D=0$")
        # fig

    else:
        plt.close(fig)

    if nbins is None:
        lower_limit = np.min(treated[:, 1])
        upper_limit = np.max(untreated[:, 1])

    # Find the true common support
    else:
        # Treated [0][0]
        # Set the lowest frequency observed in the treated subsample
        # as the default for the lower limit of the common support
        lower_limit = np.min(treated)

        # The following algorithm checks for any empty histogram bins (starting from 0 going up to 0.5).
        # If an empty histogram bin is found, the lower_limit is set to the corresponding P(Z)
        # value of the next bin above.
        # This may go up to the extreme case where empty bins are found very close 0.5
        # and the common support is very small.
        # Below, the algorithm for the untreated sample will start from above (0.5, 1]
        # to find the upper_limit.
        # If no empty bin is found in the interval [0, 0.5),
        # np.min(treated) remains the true lower limit
        for low in range(len(hist[0][0])):

            # Only consider values in the interval [0, 0.5)
            if hist[1][low] > 0.5:
                break

            # If the algorithm starts below the sample minimum, move on to the next bin
            elif hist[1][low] < np.min(treated):
                continue

            else:
                # If the current bin is non-empty, we have still continuous support and
                # the sample minimum remains our lower limit
                if hist[0][0][low] > 0:
                    pass

                # If an empty bin is found, set the lower limit to the next bin above
                # and move on to the next bin until P(Z) = 0.5 is reached
                else:
                    lower_limit = hist[1][low + 1]

        # Untreated [0][1]
        # Set the highest frequency observed in the untreated subsample
        # as the default for the upper limit of the common support
        upper_limit = np.max(untreated)

        # The following algorithm checks for any empty histogram bins (starting from 1 going down to 0.5).
        # If an empty histogram bin is found, the upper_limit is set to the corresponding P(Z)
        # value of the next bin below.
        # We may reach extreme case where empty bins are found very close 0.5
        # and the common support is very small.
        # The algorithm above proceeds analogously for the treated sample
        # to find the lower limit in the interval [0, 0.5).
        # If no empty bin is found in the interval (0.5, 1],
        # np.max(untreated) remains the true upper limit
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

    return treated, untreated, common_support


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
    N = len(X)
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
    N = len(y)
    col_len = len(y[0])

    res = np.zeros([N, col_len])

    for i in range(col_len):
        yfit = loess(x, y[:, i], span=bandwidth, degree=1)
        yfit.fit()
        res[:, i] = yfit.outputs.fitted_residuals

    return res


def double_residual_reg(ps, X, Xp, Y, rbandwidth, show_output):
    """
    This function performs a Double Residual Regression of X, Xp, and Y on ps.

    The LOESS (Locally Estimated Scatterplot Smoothing) method is implemented
    to perform the local linear fit and generate the residuals.
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
