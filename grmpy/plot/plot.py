"""This module contains the plot function, which plot the parametric or
semiparametric MTE along with 90 percent confidence bands."""

from grmpy.plot.plot_auxiliary import mte_and_cof_int_semipar
from grmpy.plot.plot_auxiliary import mte_and_cof_int_par
from grmpy.plot.plot_auxiliary import plot_curve

from grmpy.read.read import read, check_append_constant
from grmpy.check.auxiliary import read_data


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
    """This function calculates the marginal treatment effect for
    different quantiles u_D of the unobservables.
    Depending on the model specification, either the parametric or
    semiparametric MTE is plotted along with the corresponding
    90 percent confidence bands.
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
