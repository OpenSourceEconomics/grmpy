"""This module contains a function that implements the linear binning procedure."""
import numpy as np
from numba import jit


@jit(nopython=True)
def linear_binning(x, y, gridsize, startgrid, binwidth, truncate=True):
    """
    This function generates bin counts and bin averages over an equally spaced
    grid via the linear binning strategy.
    In essence, bin counts are obtained by assigning the raw data to
    neighboring grid points. A bin count can be thought of as representing the
    amount of data in the neighborhood of its corresponding grid point.
    Counts on the y-axis display the respective bin averages.

    The linear binning strategy is based on the transformation
    xgrid = ((x - a) / delta) + 1, which maps each x_i onto its corresponding
    gridpoint. The integer part of xgrid_i indicates the two
    nearest bin centers to x_i. This calculation already does the trick
    for simple binning. For linear binning, however, we additionally compute the
    "fractional part" or binweights = xgrid - bincenters, which gives the weights
    attached to the two nearest bin centers, namely (1 - binweights) for the bin
    considered and binweights for the next bin.

    If truncate is True, end observations are truncated.
    Otherwise, weight from end observations is given to corresponding
    end grid points.

    Parameters
    ----------
    x: np.ndarray
        Array of the predictor variable. Shape (N,).
        Missing values are not accepted. Must be sorted.
    y: np.ndarray
        Array of the response variable. Shape (N,).
        Missing values are not accepted. Must come presorted by x.
    gridsize: int
        Number of equally-spaced grid points
        over which x and y are to be evaluated.
    startgrid: int
        Start point of the grid.
    binwidth: float
        Bin width.
    truncate: bool
        If True, then endpoints are truncated.

    Returns
    -------
    xcounts: np.ndarry
        Array of binned x-values ("bin counts") of length M.
    ycounts: np.ndarry
        Array of binned y-values ("bin averages") of length M.
    """
    N = len(x)

    xcounts = np.zeros(gridsize)
    ycounts = np.zeros(gridsize)
    xgrid = np.zeros(N)
    binweights = np.zeros(N)
    bincenters = [0] * N

    # Map x into set of corresponding grid points
    for i in range(N):
        xgrid[i] = ((x[i] - startgrid) / binwidth) + 1

        # The integer part of xgrid indicates the two nearest bin centers to x[i]
        bincenters[i] = int(xgrid[i])
        binweights[i] = xgrid[i] - bincenters[i]

    for gridpoint in range(gridsize):
        for index, element in enumerate(bincenters):
            if element == gridpoint:
                xcounts[gridpoint - 1] += 1 - binweights[index]
                xcounts[gridpoint] += binweights[index]

                ycounts[gridpoint - 1] += (1 - binweights[index]) * y[index]
                ycounts[gridpoint] += binweights[index] * y[index]

    # By default, end observations are truncated.
    if truncate is True:
        pass

    # Truncation is implicit if there are no points in bincenters
    # beyond the grid's boundary points.
    # Note that bincenters is sorted. So it is sufficient to check if
    # the conditions below hold for the bottom and top
    # observation, respectively
    elif 1 <= bincenters[0] and bincenters[N - 1] < gridsize:
        pass

    # If truncate=False, weight from end observations is given to
    # corresponding end grid points.
    elif truncate is False:
        for index, element in enumerate(xgrid):
            if element < 1:
                xcounts[0] += 1
                ycounts[0] += y[index]
            elif element >= gridsize:
                xcounts[gridsize - 1] += 1
                ycounts[gridsize - 1] += y[index]

    return xcounts, ycounts
