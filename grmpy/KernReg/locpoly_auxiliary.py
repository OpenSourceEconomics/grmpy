"""This module provides auxiliary functions for locpoly."""
import numpy as np
import math

from numba import jit


@jit(nopython=True)
def get_kernelweights(tau, bandwidth, delta):
    """This function computes approximated weights for the Gaussian kernel."""
    L = math.floor(tau * bandwidth / delta)
    lenkernel = 2 * L + 1
    kernelweights = np.zeros(lenkernel)

    # Determine midpoint of kernelweights
    mid = L + 1

    # Compute the kernel weights
    for j in range(L + 1):

        # Note that the mid point (kernelweights[mid - 1]) receives a weight of 1.
        kernelweights[mid - 1 + j] = math.exp(-(delta * j / bandwidth) ** 2 / 2)

        # Because of the kernel's symmetry, weights in equidistance
        # below and above the midpoint are identical.
        kernelweights[mid - 1 - j] = kernelweights[mid - 1 + j]

    return L, lenkernel, kernelweights, mid


@jit(nopython=True)
def combine_bincounts_kernelweights(
    xcounts, ycounts, gridsize, colx, coly, L, lenkernel, kernelweights, mid, binwidth
):
    """
    This function combines the bin counts (xcounts) and bin averages (ycounts) with
    kernel weights via a series of direct convolutions. As a result, binned
    approximations to X'W X and X'W y, denoted by weigthedx and weigthedy, are computed.

    Recall that the local polynomial curve estimator beta_ and its derivatives are
    minimizers to a locally weighted least-squares problem. At each grid
    point g = 1,..., M in the grid, beta_ is computed as the solution to the
    linear matrix equation:

    X'W X * beta_ = X'W y,

    where W are kernel weights approximated by the Gaussian density function.
    X'W X and X'W y are approximated by weigthedx and weigthedy,
    which are the result of a direct convolution of bin counts (xcounts) and kernel
    weights, and bin averages (ycounts) and kernel weights, respectively.

    The terms "kernel" and "kernel function" are used interchangeably
    throughout.

    For more information see the documentation of the main function locpoly
    under KernReg.locpoly.

    Parameters
    ----------
    xcounts: np.ndarry
        1-D array of binned x-values ("bin counts") of length gridsize.
    ycounts: np.ndarry
        1-D array of binned y-values ("bin averages") of length gridsize.
    gridsize: int
        Number of equally-spaced grid points.
    colx: int
        Number of columns of output array weigthedx, i.e. the binned approximation to X'W X.
    coly: int
        Number of columns of output array weigthedy, i.e the binned approximation to X'W y.
    lenkernel: int
        Length of 1-D array kernelweights.
    kernelweights: np.ndarry
        1-D array of length lenfkap containing
        approximated weights for the Gaussian kernel
        (W in the notation above).
    L: int
        Parameter defining the number of times the kernel function
        has to be evaluated.
        Note that L < N, where N is the total number of observations.
    mid: int
        Midpoint of kernelweights.
    binwidth: float
        Bin width.

    Returns
    -------
    weigthedx: np.ndarry
        Dimensions (M, colx). Binned approximation to X'W X.
    weigthedy: np.ndarry
        Dimensions (M, coly). Binned approximation to X'W y.
    """
    weigthedx = np.zeros((gridsize, colx))
    weigthedy = np.zeros((gridsize, coly))

    for g in range(gridsize):
        if xcounts[g] != 0:
            for i in range(max(0, g - L - 1), min(gridsize, g + L)):

                if 0 <= i <= gridsize - 1 and 0 <= g - i + mid - 1 <= lenkernel - 1:
                    fac_ = 1

                    weigthedx[i, 0] += xcounts[g] * kernelweights[g - i + mid - 1]
                    weigthedy[i, 0] += ycounts[g] * kernelweights[g - i + mid - 1]

                    for j in range(1, colx):
                        fac_ = fac_ * binwidth * (g - i)

                        weigthedx[i, j] += (
                            xcounts[g] * kernelweights[g - i + mid - 1] * fac_
                        )

                        if j < coly:
                            weigthedy[i, j] += (
                                ycounts[g] * kernelweights[g - i + mid - 1] * fac_
                            )
    return weigthedx, weigthedy


@jit(nopython=True)
def get_curve_estimator(weigthedx, weigthedy, coly, derivative, gridsize):
    """
    This functions solves the locally weighted least-squares regression
    problem and returns an estimator for the v-th derivative of beta_,
    the local polynomial estimator, at all points in the grid.

    Before doing so, the function first turns each row in weightedx into
    an array of size (coly, coly) and each row in weigthedy into
    an array of size (coly,), called xmat and yvec, respectively.

    The local polynomial curve estimator beta_ and its derivatives are
    minimizers to a locally weighted least-squares problem. At each grid
    point, beta_ is computed as the solution to the linear matrix equation:

    X'W X * beta_ = X'W y,

    where W are kernel weights approximated by the Gaussian density function.
    X'W X and X'W y are approximated by weigthedx and weigthedy,
    which are the result of a direct convolution of bin counts (xcounts) and kernel
    weights, and bin averages (ycounts) and kernel weights, respectively.

    Note that for a v-th derivative the order of the polynomial
    should be p = v + 1.

    Parameters
    ----------
    weigthedx: np.ndarry
        Dimensions (M, colx). Binned approximation to X'W X.
    weigthedy: np.ndarry
        Dimensions (M, coly). Binned approximation to X'W y.
    coly: int
        Number of columns of output array weigthedy, i.e the binned approximation to X'W y.
    derivative: int
        Order of the derivative to be estimated.
    gridsize: int
        Number of equally-spaced grid points.

    Returns
    -------
    curvest: np.ndarray
        1-D array of length gridsize. Estimator for the specified
        derivative at each grid point.
    """
    xmat = np.zeros((coly, coly))
    yvec = np.zeros(coly)
    curvest = np.zeros(gridsize)

    for g in range(gridsize):
        for row in range(0, coly):
            for column in range(0, coly):
                colindex = row + column
                xmat[row, column] = weigthedx[g, colindex]

                yvec[row] = weigthedy[g, row]

        # Calculate beta_ as the solution to the linear matrix equation
        # X'W X * beta_ = X'W y.
        # Note that xmat and yvec are binned approximations to X'W X and
        # X'W y, respectively, evaluated at the given grid point.
        beta_ = np.linalg.solve(xmat, yvec)

        # Obtain curve estimator for the desired derivative of beta_.
        curvest[g] = beta_[derivative]

    curvest = math.gamma(derivative + 1) * curvest

    return curvest


@jit(nopython=True)
def is_sorted(a):
    """This function checks if the input array is sorted ascendingly."""
    for i in range(a.size - 1):
        if a[i + 1] < a[i]:
            return False
        return True
