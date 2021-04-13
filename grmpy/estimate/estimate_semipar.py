"""
This module contains the semiparametric estimation process.
"""
import kernreg as kr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from skmisc.loess import loess

lowess = sm.nonparametric.lowess


def semipar_fit(dict_, data):
    """ "
    This function runs the semiparametric estimation of the
    marginal treatment effect via local instrumental variables.

    Parameters
    ----------
    dict_: dict
        Estimation dictionary. Returned by grmpy.read(init_file)).
    data: pandas.DataFrame
        Data set to perform the estimation on. Specified
        under dict_["ESTIMATION"]["file"].

    Returns
    ------
    rslt: dict
        Result dictionary containing
        - quantiles
        - mte
        - mte_x
        - mte_u
        - mte_min
        - mte_max
        - X
        - b1
        - b0
    """
    # Process information specified in the initialization file and
    # assign default values if missing
    bins, logit, bandwidth, gridsize, startgrid, endgrid = process_primary_inputs(dict_)
    trim, rbandwidth, reestimate_p, show_output = process_secondary_inputs(dict_)

    # Prepare the sample for the estimation process
    # Compute propensity score, define common support and trim the sample
    # data, prop_score = process_mte_data(
    #     dict_, data, logit, bins, trim, reestimate_p, show_output
    # )
    data = estimate_treatment_propensity(dict_, data, logit, show_output)

    X, Y, prop_score = trim_support(
        dict_, data, logit, bins, trim, reestimate_p, show_output
    )

    b0, b1_b0 = double_residual_reg(X, Y, prop_score)

    # # Construct the MTE
    # Generate the quantiles of the final MTE
    quantiles = np.linspace(startgrid, endgrid, gridsize)

    mte_x = mte_observed(X, b1_b0)
    mte_u = mte_unobserved_semipar(
        X, Y, b0, b1_b0, prop_score, bandwidth, gridsize, startgrid, endgrid
    )

    # Put the MTE together
    mte = mte_x.mean(axis=0) + mte_u

    # Account for variation in X
    mte_min = np.min(mte_x) + mte_u
    mte_max = np.max(mte_x) + mte_u

    rslt = {
        "quantiles": quantiles,
        "mte": mte,
        "mte_x": mte_x,
        "mte_u": mte_u,
        "mte_min": mte_min,
        "mte_max": mte_max,
        "X": X,
        "b0": b0,
        "b1": b1_b0 + b0,
    }

    return rslt


def process_primary_inputs(dict_):
    """
    This functions processes the parameters specified
    by the user in the initialization dictionary.

    Parameters
    ----------
    dict_: dict
        Estimation dictionary. Returned by grmpy.read(init_file).

    Returns
    -------
    bins: int
        Number of histogram bins used to determine common support.
    logit: bool
        Probability model for the choice equation.
        If True: logit, else: probit.
    bandwidth: float
        Kernel bandwidth for the local polynomial regression.
    gridsize: int
        Number of equally-spaced grid points of u_D over which the
        MTE shall be estimated.
    startgrid: int
        Start point of the grid of unobservable resistance (u_D),
        over which the MTE is evaluated.
    end_grind: int
        End point of the grid of unobservable resistance (u_D),
        over which the MTE is evaluated.
    """
    try:
        dict_["ESTIMATION"]["bins"]
    except KeyError:
        bins = 25
    else:
        bins = dict_["ESTIMATION"]["bins"]

    try:
        dict_["ESTIMATION"]["logit"]
    except KeyError:
        logit = True
    else:
        logit = dict_["ESTIMATION"]["logit"]

    try:
        dict_["ESTIMATION"]["bandwidth"]
    except KeyError:
        bandwidth = 0.32
    else:
        bandwidth = dict_["ESTIMATION"]["bandwidth"]

    try:
        dict_["ESTIMATION"]["gridsize"]
    except KeyError:
        gridsize = 500
    else:
        gridsize = dict_["ESTIMATION"]["gridsize"]

    try:
        dict_["ESTIMATION"]["ps_range"]
    except KeyError:
        prop_score_range = [0.005, 0.995]
    else:
        prop_score_range = dict_["ESTIMATION"]["ps_range"]

    start_grid = prop_score_range[0]
    endgrid = prop_score_range[1]

    return bins, logit, bandwidth, gridsize, start_grid, endgrid


def process_secondary_inputs(dict_):
    """
    This functions processes the secondary input parameters.

    Parameters
    ----------
    dict_: dict
        Estimation dictionary. Returned by grmpy.read(init_file).

    Returns
    -------
    trim: bool, default True
        Trim the data outside the common support, recommended.
    rbandwidth: float, default 0.05
        Bandwidth for the Double Residual Regression.
    reestimate_p: bool, default False
        Re-estimate P(Z) after trimming, not recommended.
    show_output: bool, default False
        Show intermediate outputs of the estimation process.
    """
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

    try:
        dict_["ESTIMATION"]["show_output"]
    # Set default to True
    except KeyError:
        show_output = False
    else:
        show_output = dict_["ESTIMATION"]["show_output"]

    return trim, rbandwidth, reestimate_p, show_output


def estimate_treatment_propensity(dict_, data, logit, show_output=False):
    """
    This function estimates the propensity of selecting into treatment
    for both treated and untreated individuals based on instruments Z.
    Z subsumes all the observable components that influence the treatment
    decision, e.g. the decision to enroll into college (D = 1) or not (D = 0).

    Estimate propensity scores via Logit (default) or Probit.

    Parameters
    ----------
    dict_: dict
        Estimation dictionary. Returned by grmpy.read(init_file)).
    data: pandas.DataFrame
        Data set to perform the estimation on. Specified
        under dict_["ESTIMATION"]["file"].
    logit: bool
        Probability model for the choice equation.
        If True: logit, else: probit.
    show_output: bool
        If True, intermediate outputs of the estimation process are displayed.

    Returns
    -------
    data: pandas.DataFrame
        Propensity score (range between [0, 1]). Values closer to 1
        denote a higher inclination to treatment.
    """
    D = data[dict_["ESTIMATION"]["indicator"]].values
    Z = data[dict_["CHOICE"]["order"]]

    if logit is True:
        logitRslt = sm.Logit(D, Z).fit(disp=0)
        prop_score = logitRslt.predict(Z)

        if show_output is True:
            print(logitRslt.summary())

    else:
        probitRslt = sm.Probit(D, Z).fit(disp=0)
        prop_score = probitRslt.predict(Z)

        if show_output is True:
            print(probitRslt.summary())

    data.loc[:, "prop_score"] = prop_score

    return data


def trim_support(
    dict_, data, logit, bins=25, trim=True, reestimate_p=False, show_output=False
):
    """
    This function defines common support and trims the data.
    Optionally p is re-estimated on the trimmed sample.

    Parameters
    ----------
    dict_: dict
        Estimation dictionary. Returned by grmpy.read(init_file).
    data: pandas.DataFrame
        Data set to perform the estimation on. Specified
        under dict_["ESTIMATION"]["file"].
    logit: bool
        Probability model for the choice equation.
        If True: logit, else: probit.
    bins: int
        Number of histogram bins used to determine common support.
    trim: bool
        Trim the data outside the common support.
    reestimate_p: bool
        Re-estimate P(Z) after trimming.
    show_output: bool
        If True, intermediate outputs of the estimation process are displayed.

    Returns
    -------
    X: pandas.DataFrame
        Trimmed X data (observables), i.e. observations outside the common support
        of the propensity score (prop_score) have been dropped.
        Sorted in ascending order by *prop_score*.
    Y: pandas.DataFrame
        Trimmed Y data (wage), i.e. observations outside the common support
        of the propensity score (prop_score) have been dropped.
        Sorted in ascending order by *prop_score*.
    prop_score: pandas.Series
        Propensity score (range between [0, 1]). Values closer to 1
        denote a higher inclination to treatment.
        Sorted in ascending order.
    """
    # Find common support
    prop_score = data["prop_score"]
    common_support = _define_common_support(dict_, data, bins, show_output)

    # Trim the data. Recommended.
    if trim is True:
        # data, prop_score = trim_data(prop_score, common_support, data)
        data = data[
            (data.prop_score >= common_support[0])
            & (data.prop_score <= common_support[1])
        ]
        prop_score = prop_score[
            (prop_score >= common_support[0]) & (prop_score <= common_support[1])
        ]

        # Optional. Not recommended
        # Re-estimate baseline propensity score on the trimmed sample
        if reestimate_p is True:
            # Re-estimate the parameters of the decision equation based
            # on the new trimmed data set
            data = estimate_treatment_propensity(dict_, data, logit, show_output)

        else:
            pass
    else:
        pass

    data = data.sort_values(by="prop_score", ascending=True)
    prop_score = prop_score.sort_values(axis=0, ascending=True)
    X = data[dict_["TREATED"]["order"]]
    Y = data[[dict_["ESTIMATION"]["dependent"]]]

    return X, Y, prop_score


def double_residual_reg(X, Y, prop_score, rbandwidth=0.05, show_output=False):
    """
    This function performs a Double Residual Regression (DDR)
    of X, Xp, and Y on *prop_score*.

    A local linear kernel regression (polynomial of degree 1)
    is implemented to generate the residuals.

    Parameters
    ----------
    X: pandas.DataFrame
        DataFrame of observables (i.e. covariates).
    Y: pandas.DataFrame
        Individuals' wage data.
    prop_score: pandas.Series
        Propensity score (range between [0, 1]). Values closer to 1
        denote a higher inclination to treatment.
        Sorted in ascending order.

    Returns
    -------
    b0: np.ndarray
        Beta0 coefficient of the DDR (no-intercept OLS regression
        of the residuals of X, Xp, and Y on *prop_score*).
    b1: np.ndarray
        Beta1 coefficient of the DDR (no-intercept OLS regression
        of the residuals of X, Xp, and Y on *prop_score*).
    """
    # 0) Construct Xp := X * P(z)
    Xp = _construct_Xp(X, prop_score)

    # 1) Fit a separate local linear regression of X, Xp, and Y on prop_score,
    # which yields residuals e_X, e_Xp, and e_Y.
    res_X = _generate_residuals(prop_score, X, rbandwidth)
    res_Xp = _generate_residuals(prop_score, Xp, rbandwidth)
    res_Y = _generate_residuals(prop_score, Y, rbandwidth)

    # Append res_X and res_Xp.
    col_names = list(X) + list(Xp)
    res_X_Xp = pd.DataFrame(np.append(res_X, res_Xp, axis=1), columns=col_names)

    # 2) Run a single OLS regression of e_Y on e_X and e_Xp without intercept:
    # e_Y = e_X * beta_0 + e_Xp * (beta_1 - beta_0),
    # to estimate the values of beta_0 and (beta_1 - beta_0).
    model = sm.OLS(res_Y, res_X_Xp)
    results = model.fit()
    b0 = results.params[: len(list(X))]
    b1_b0 = results.params[len(list(X)) :]

    if show_output is True:
        print(results.summary())

    return np.array(b0), np.array(b1_b0)


def mte_observed(X, b1_b0):
    """
    This function computes the observed component of the MTE (*mte_x*)
    that depends on observables X:

    mte = *mte_x* + mte_u

    Parameters
    ----------
    X: pandas.DataFrame
        Data of observables (covariates).
    b1_b0: np.ndarray
        Difference of the coefficients in the Double Residual Regression,
        i.e. the no-intercept OLS regression of the residuals of
        X, Xp, and Y on *prop_score*.

    Returns
    -------
    mte_x: np.ndarray
        Part of the MTE that depends on observables X.
    """
    mte_x = np.dot(X, b1_b0)

    return mte_x


def mte_unobserved_semipar(
    X, Y, b0, b1_b0, prop_score, bandwidth, gridsize, startgrid, endgrid
):
    """
    This function computes the unobserved component of the MTE
    in MTE = mte_x + *mte_u*, where *mte_u* depends on the unobserved
    esistance to treatment u_D.

    Parameters
    ----------
    X: pandas.DataFrame
        DataFrame of observables (i.e. covariates).
    Xp: pandas.DataFrame
        X data multiplied by *prop_score* X * P(z).
    Y: pandas.DataFrame
        Individuals' wage data.
    b0: np.ndarray
        Beta0 coefficient in the Double Residual Regression,
        i.e. the no-intercept OLS regression of the residuals of
        X, Xp, and Y on *prop_score*.
    b1_b0: np.ndarray
        Difference of the coefficients in the Double Residual Regression,
        i.e. the no-intercept OLS regression of the residuals of
        X, Xp, and Y on *prop_score*.
    prop_score: pandas.Series
        Propensity score (range between [0, 1]). Values closer to 1
        denote a higher inclination to treatment.
        Sorted in ascending order.
    bandwidth: float
        Kernel bandwidth for the local polynomial regression.
    gridsize: int
        Number of equally-spaced grid points of u_D over which the
        MTE shall be estimated.
    startgrid: int
        Start point of the grid of unobservable resistance (u_D),
        over which the MTE is evaluated.
    endgrid: int
        End point of the grid of unobservable resistance (u_D),
        over which the MTE is evaluated.

    Returns
    -------
    mte_u: np.ndarray
        Part of the MTE that depends on the unobserved resistance
        to treatment (u_D).
    """
    # 0) Construct Xp := X * P(z)
    Xp = _construct_Xp(X, prop_score)

    # Turn the X, Xp, and Y DataFrames as well as the
    # propensity score Series into np.ndarrays
    X_arr = np.array(X)
    Xp_arr = np.array(Xp)
    Y_arr = np.array(Y).ravel()
    prop_score = np.array(prop_score)

    # Compute the unobserved part of Y
    Y_tilde = Y_arr - np.dot(X_arr, b0) - np.dot(Xp_arr, b1_b0)

    # Estimate mte_u, the unobserved component of the MTE,
    # through a locally quadratic regression
    rslt_locpoly = kr.locpoly(
        x=prop_score,
        y=Y_tilde,
        derivative=1,
        degree=2,
        bandwidth=bandwidth,
        gridsize=gridsize,
        a=startgrid,
        b=endgrid,
    )

    mte_u = rslt_locpoly["curvest"]

    return mte_u


def _define_common_support(
    dict_,
    data,
    bins=25,
    show_output=False,
    figsize=(10, 6),
    fontsize=15,
    plot_title=False,
    save_output=False,
):
    """
    This function defines the common support as the region under the histograms
    where propensities in the treated and untreated subsample overlap.

    Carneiro et al (2011) choose 25 bins for a total sample of 1747
    observations, so *nbins*=25 is set as a default.

    Parameters
    ----------
    dict_: dict
        Estimation dictionary. Returned by grmpy.read(init_file)).
    data: pandas.DataFrame
        Data set to perform the estimation on. Specified
        under dict_["ESTIMATION"]["file"].
    bins: int
        Number of histogram bins used to determine common support.
    show_output: bool
        If True, intermediate outputs of the estimation process are displayed.
    figsize: tuple, default is (10, 6)
        Tuple denoting the size of the resulting figure.
    fontsize: int, default is 15
        Parameter specifying the font size used in the figure.
    plot_title: bool, default is False
        If True, a title for the figure is displayed.
    save_output: bool or str or PathLike or file-like object, default is False
        If False, the resulting plot is shown but not saved.
        If True, the MTE plot is saved as 'common_support.png'.
        Else, if a str or Pathlike or file-like object is specified,
        the plot is saved according to *save_output*.
        The output format is inferred from the extension ('png', 'pdf', 'svg'... etc.)
        By default, '.png' is assumed.

    Returns
    -------
    common_support: list
        List containing lower and upper bound of the propensity score.
    """
    hist, treated, untreated = _make_histogram(
        dict_, data, bins, show_output, figsize, fontsize, plot_title
    )
    lower_limit, upper_limit = _find_limits(hist, treated, untreated)
    common_support = [lower_limit, upper_limit]

    if show_output is True:
        print(
            """
    Common support lies beteen:

        {} and
        {}""".format(
                lower_limit, upper_limit
            )
        )

    if save_output is False:
        pass
    elif save_output is True:
        plt.savefig("common_support.png", dpi=300)
    else:
        plt.savefig(save_output, dpi=300)

    return common_support


def _make_histogram(
    dict_,
    data,
    bins=25,
    show_output=False,
    figsize=(10, 6),
    fontsize=15,
    plot_title=False,
):
    """
    This function plots the common supports, i.e. the overlapping regions under the
    histograms of both the treated and untreated individuals.
    The plot is only displayed if *show_output* is True.

    Parameters
    ----------
    dict_: dict
        Estimation dictionary. Returned by grmpy.read(init_file)).
    data: pandas.DataFrame
        Data set to perform the estimation on. Specified
        under dict_["ESTIMATION"]["file"].
    bins: int
        Number of histogram bins used to determine common support.
    show_output: bool
        If True, intermediate outputs of the estimation process are displayed.
    figsize: tuple, default is (10, 6)
        Tuple denoting the size of the resulting figure.
    fontsize: int, default is 15
        Parameter specifying the font size used in the figure.
    plot_title: bool, default is False
        If True, a title for the figure is displayed.

    Returns
    -------
    hist: np.ndarray
        Array containing positional parameters of the histogram bins.
    treated: list
        List containing the propensity scores of the treated
        individuals.
    untreated: list
        List containing the propensity scores of the treated
        individuals.
    """
    indicator = dict_["ESTIMATION"]["indicator"]

    treated = data[[indicator, "prop_score"]][data[indicator] == 1].values
    untreated = data[[indicator, "prop_score"]][data[indicator] == 0].values

    treated = treated[:, 1].tolist()
    untreated = untreated[:, 1].tolist()

    # Make the histogram using a list of lists
    fig = plt.figure(figsize=figsize)
    hist = plt.hist(
        [treated, untreated],
        bins=bins,
        weights=[
            np.ones(len(treated)) / len(treated),
            np.ones(len(untreated)) / len(untreated),
        ],
        density=0,
        alpha=0.55,
        label=["Treated", "Unreated"],
    )

    if show_output is True:
        plt.tick_params(axis="both", labelsize=14)
        plt.legend(loc="upper right", prop={"size": 14})
        plt.xticks(np.arange(0, 1.1, step=0.1))
        plt.grid(axis="y", alpha=0.25)
        plt.xlabel("$P$", fontsize=fontsize)
        plt.ylabel("$f(P)$", fontsize=fontsize)

        if plot_title is True:
            plt.title(r"Support of $P(\hat{Z})$ for $D=1$ and $D=0$")

    else:
        plt.close(fig)

    return hist, treated, untreated


def _find_limits(hist, treated, untreated):
    """
    Find the upper and lower limit of the common support.

    Parameters
    ----------
    hist: np.ndarray
        Array containing positional parameters of the histogram bins.
    treated: list
        List containing the propensity scores of the treated
        individuals.
    untreated: list
        List containing the propensity scores of the treated
        individuals.

    Returns
    -------
    lower_limit: float
        Lower limit of the common support.
    upper_limit: float
        Upper limit of the common support.
    """
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


def _construct_Xp(X, prop_score):
    """
    This function generates the X * *prop_score* regressors:

    Xp = X * P(z) for each individual.

    Parameters
    ----------
    X: pandas.DataFrame
        Data of observables (covariates).
    prop_score: pandas.Series
        Propensity score (range between [0, 1]). Values closer to 1
        denote a higher inclination to treatment.
        Sorted in ascending order.

    Returns
    -------
    X: pandas.DataFrame
        Data of observables (covariates) multiplied with *prop_score*.
    """
    # To multiply each elememt in X (shape N x k) with the corresponding prop_score,
    # set up a prop_score matrix of same size.
    P_z = pd.concat([prop_score] * len(X.columns), axis=1, ignore_index=True)

    # Construct Xp
    Xp = pd.DataFrame(
        X.values * P_z.values, columns=[key_ + "_ps" for key_ in list(X)], index=X.index
    )

    return Xp


def _generate_residuals(exog, endog, bandwidth=0.05):
    """
    This function runs a series of loess regressions (degree=1)
    for a set of response variables (*endog*) on a single explanatory
    variable (*exog*) and computes the corresponding residuals.

    For clarity, *exog* denotes Y and *endog* denotes X data.

    To avoid any confusion with the naming convention in
    the rest of this module, we use the terms *exog* and *endog* here.

    The actual loess regressions (in the terminology of the generalized
    Roy model) look like this in pseudo-code:

    For each k in NumberOfCovariates:
        X[k] = prop_score + e_X[k]
        Xp[k] = prop_score + e_Xp[k]

    And similarly:
        Y = prop_score + e_Y,

    where the goal is to retrieve e_X, e_Xp, e_Y.
    These residuals are then use by **double_residual_reg()**


    Parameters
    ----------
    exog: pandas.DataFrame or pandas.Series
        Data of exogenous variable(s). May have one- or multiple columns.
    endog: pandas.Series or pandas.DataFrame
        Single column DataFrame or Series of endogenous data.
    bandwidth: float, default is 0.05
        Bandwidth (i.e. span) used in the the loess regression.

    Returns
    -------
    res: numpy.ndarray
        Loess residuals of shape [ len(*y*), len(*y*[0]) ]
    """
    # Turn input data into np.ndarrays.
    exog = np.array(exog)
    endog = np.array(endog)

    # Determine number of observations and number of columns of the
    # outcome variable.
    n = endog.shape[0]

    # *y* is a column vector
    if endog.ndim == 1:
        y_fit = loess(exog, endog, span=bandwidth, degree=1)
        y_fit.fit()
        res = y_fit.outputs.fitted_residuals

    else:
        columns = endog.shape[1]
        res = np.zeros([n, columns])

        for col in range(columns):
            y_fit = loess(exog, endog[:, col], span=bandwidth, degree=1)
            y_fit.fit()
            res[:, col] = y_fit.outputs.fitted_residuals

    return res
