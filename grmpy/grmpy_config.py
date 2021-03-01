"""This module provides some configuration for the package."""
import os
import sys
import warnings

import numpy as np
from pathlib import Path
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Obtain the root directory of the package.
# Do not import grmpy which creates a circular import.
ROOT_DIR = Path(__file__).parent


# Directory with additional resources for the testing harness
TEST_DIR = ROOT_DIR / "tests"
TEST_RESOURCES_DIR = ROOT_DIR / "tests" / "resources"

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
