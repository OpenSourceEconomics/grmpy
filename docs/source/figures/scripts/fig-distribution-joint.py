"""This module creates the contour plots for the bivariate normal distribution."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from fig_config import OUTPUT_DIR

plt.style.use("resources/grmpy.mplstyle")

y_min, y_max = -4, 4

x = np.linspace(y_min, y_max, 100)
y = np.linspace(y_min, y_max, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

mean = np.tile(0.2, 2)
cov = np.identity(2) * 2

rv = multivariate_normal(mean, cov)
rv.pdf(pos)

ax = plt.figure().add_subplot(111)

levels = np.linspace(0.1, 1.0, 10, endpoint=True)
cns = plt.contourf(X, Y, rv.pdf(pos) / np.max(rv.pdf(pos)), levels=levels)

ax.set_ylabel("$Y_1$")
ax.set_xlabel("$Y_0$")
ax.set_ylim(y_min, y_max)
ax.set_xlim(y_min, y_max)

ax.set_yticklabels([])
ax.set_xticklabels([])

plt.plot(
    np.arange(y_min, y_max), np.arange(y_min, y_max), color="black", linestyle="--"
)

plt.colorbar(cns)

ax.text(3.3, 3.3, r"$45^o$")

plt.savefig(OUTPUT_DIR + "/fig-distribution-joint-potential.png", dpi=300)

# This plot shows the joint distribution of surplus and benefits.
x = np.linspace(y_min, y_max, 100)
y = np.linspace(y_min, y_max, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

mean = np.tile(0.2, 2)
cov = np.identity(2) * 2

rv = multivariate_normal(mean, cov)
rv.pdf(pos)

ax = plt.figure().add_subplot(111)

levels = np.linspace(0.1, 1.0, 10, endpoint=True)
cns = plt.contourf(X, Y, rv.pdf(pos) / np.max(rv.pdf(pos)), levels=levels)

ax.set_ylabel("$B$")
ax.set_xlabel("$S$")
ax.set_ylim(y_min, y_max)
ax.set_xlim(y_min, y_max)

ax.set_yticklabels(["", "", "", "", 0])
ax.set_xticklabels(["", "", "", "", 0])


plt.colorbar(cns)

ax.axvline(x=0, color="black", linestyle="--")
ax.axhline(y=0, color="black", linestyle="--")

ax.axvline(x=-0.5, color="lightgray", linestyle="--")


ax.text(x=-3.5, y=3.5, s="I")

ax.text(x=3.5, y=3.5, s="II")

ax.text(x=3.5, y=-3.5, s="III")

ax.text(x=-3.5, y=-3.5, s="IV")

plt.savefig(OUTPUT_DIR + "/fig-distribution-joint-surplus.png", dpi=300)
