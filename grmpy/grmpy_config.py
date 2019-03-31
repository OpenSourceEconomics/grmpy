"""This module provides some configuration for the package."""
import warnings
import sys
import os

from statsmodels.tools.sm_exceptions import ConvergenceWarning
import numpy as np

# Set Debug mode dependend on the environment in which the test battery is running
IS_PRODUCTION = False

if os.getenv("GRMPY_DEV") == "TRUE":
    IS_PRODUCTION = True

# We want to turn off selected warnings.
if IS_PRODUCTION is False:
    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.simplefilter(action="ignore", category=ConvergenceWarning)
    warnings.simplefilter(action="ignore", category=RuntimeWarning)

# We only support modern Python.
np.testing.assert_equal(sys.version_info[0], 3)
np.testing.assert_equal(sys.version_info[1] >= 5, True)

# We rely on relative paths throughout the package.
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_RESOURCES_DIR = PACKAGE_DIR + "/test/resources"
