""""""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from fig_config import OUTPUT_DIR

plt.style.use("resources/grmpy.mplstyle")


def plot_treatment_effect():

    x_axis = np.arange(-2, 4, 0.001)
    ax = plt.figure(figsize=(14, 6))

    for fig in [121, 122]:
        if fig == 121:
            TT, TUT, ATE = [1.3, "TT"], [0.7, "TUT"], [1.0, "ATE"]
        elif fig == 122:
            TT, TUT, ATE = [1.0, "TT"], [1.0, "TUT"], [1.0, "ATE"]

        ay = ax.add_subplot(fig)
        ay.plot(x_axis, norm.pdf(x_axis, 1, 1))
        ay.set_xlim(-2, 4)
        ay.set_ylim(0.0, None)
        ay.set_yticks([])

        # Rename axes
        ay.set_ylabel("$f_{Y_1 - Y_0}$")
        ay.set_xlabel("$Y_1 - Y_0$")

        for effect in [ATE, TT, TUT]:
            ay.plot([effect[0], effect[0]], [0, 5], label=effect[1])
    plt.legend(prop={"size": 15})

    plt.subplots_adjust(
        top=0.95, bottom=0.15, left=0.05, right=0.95, hspace=0.15, wspace=0.1
    )

    plt.savefig(OUTPUT_DIR + "/fig-treatment-effects-with-and-without-eh.png", dpi=300)


if __name__ == "__main__":
    plot_treatment_effect()
