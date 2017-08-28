"""This module provides some auxilary functions for the testing process."""

from pandas.util.testing import assert_series_equal


def check_series(s1, s2):
    """The function checks the equality of given pandas series objects."""
    try:
        assert_series_equal(s1, s2, check_less_precise=10, check_names=False)
        return True
    except AssertionError:
        return False
