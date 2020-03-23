import matplotlib.pyplot as plt

from grmpy.plot.plot_auxiliary import mte_and_cof_int_semipar
from grmpy.plot.plot_auxiliary import mte_and_cof_int_par

from grmpy.check.check import check_append_constant
from grmpy.check.auxiliary import read_data
from grmpy.read.read import read


def plot_mte(
    rslt,
    init_file,
    college_years=4,
    font_size=20,
    label_size=14,
    color="blue",
    semipar=False,
    nboot=250,
):
    """This function calculates the marginal treatment effect for
    different quantiles u_D of the unobservables.

    Depending on the model specification, either the parametric or
    semiparametric MTE is plotted along with the corresponding
    90 percent confindence bands.
    """
    # Read init dict and data
    init_dict = read(init_file)
    data = read_data(init_dict["ESTIMATION"]["file"])

    dict_, data = check_append_constant(init_file, init_dict, data, semipar)

    if semipar is True:
        quantiles, mte, mte_up, mte_d = mte_and_cof_int_semipar(
            rslt, init_file, college_years, nboot
        )

    else:
        quantiles, mte, mte_up, mte_d = mte_and_cof_int_par(rslt, init_dict, data)

    # Plot curve
    ax = plt.figure(figsize=(17.5, 10)).add_subplot(111)

    ax.set_ylabel(r"$MTE$", fontsize=font_size)
    ax.set_xlabel("$u_D$", fontsize=font_size)
    ax.tick_params(
        axis="both",
        direction="in",
        length=5,
        width=1,
        grid_alpha=0.25,
        labelsize=label_size,
    )
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")

    ax.plot(quantiles, mte, color=color, linewidth=4)
    ax.plot(quantiles, mte_up, color=color, linestyle=":", linewidth=3)
    ax.plot(quantiles, mte_d, color=color, linestyle=":", linewidth=3)

    plt.show()

    return mte, quantiles
