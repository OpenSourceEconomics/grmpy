"""
This module contains the plot function, which plot the parametric or
semiparametric MTE along with 90 percent confidence bands.
"""

from grmpy.check.auxiliary import read_data
from grmpy.plot.plot_auxiliary import (
    mte_and_cof_int_par,
    mte_and_cof_int_semipar,
    plot_curve,
)
from grmpy.read.read import check_append_constant, read


def plot_mte(
    rslt,
    init_file,
    college_years=4,
    font_size=22,
    label_size=16,
    color="blue",
    semipar=False,
    nboot=250,
    save_plot=False,
):
    """
    This function calculates the marginal treatment effect for
    different quantiles u_D of the unobservables.
    Depending on the model specification, either the parametric or
    semiparametric MTE is plotted along with the corresponding
    90 percent confidence bands.

    Parameters
    ----------
    rslt: dict
        Result dictionary returned by grmpy.fit().
    init_file: yaml
        Initialization file containing parameters for the estimation
        process.
    college_years: int, default is 4
        Average duration of college degree. The MTE plotted will thus
        refer to the returns per one year of college education.
    font_size: int, default is 22
        Font size of the MTE graph.
    label_size: int, default is 16
        Label size of the MTE graph
    color: str, default is "blue"
        Color of the MTE curve.
    semipar: bool, default is False
        Option to indicate the semiparametric estimation.
        If semipar is False, the parametric normal model is assumed and
        confidence intervals are computed analytically.
        Else (semipar is True), confidence bands are bootstrapped.
    nboot: int, default is 250
        Only relevant for semiparametric estimation (semipar=True).
        Number of of bootstrap iterations used to compute
        confidence intervals.
    save_plot: bool or str or PathLike or file-like object, default is False
        If False, the resulting plot is shown but not saved.
        If True, the MTE plot is saved as 'MTE_plot.png'.
        Else, if a str or Pathlike or file-like object is specified,
        the plot is saved according to *save_plot*.
        The output format is inferred from the extension ('png', 'pdf', 'svg'... etc.)
        By default, '.png' is assumed.
    """
    # Read init dict and data
    dict_ = read(init_file, semipar)
    data = read_data(dict_["ESTIMATION"]["file"])

    dict_, data = check_append_constant(init_file, dict_, data, semipar)

    if semipar is True:
        quantiles, mte, con_u, con_d = mte_and_cof_int_semipar(
            rslt, init_file, college_years, nboot
        )

    else:
        quantiles, mte, con_u, con_d = mte_and_cof_int_par(
            rslt, dict_, data, college_years
        )

    # Add confidence intervals to rslt dictionary
    rslt.update({"con_u": con_u, "con_d": con_d})

    plot_curve(mte, quantiles, con_u, con_d, font_size, label_size, color, save_plot)
