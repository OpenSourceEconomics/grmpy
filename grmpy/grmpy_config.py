"""This module provides some configuration for the package."""
import warnings
import sys
import os

import numpy as np

# We want to turn off selected warnings.
warnings.simplefilter(action='ignore', category=FutureWarning)

# We only support modern Python.
np.testing.assert_equal(sys.version_info[0], 3)
np.testing.assert_equal(sys.version_info[1] >= 5, True)

# We rely on relative paths throughout the package.
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_RESOURCES_DIR = PACKAGE_DIR + '/test/resources'
