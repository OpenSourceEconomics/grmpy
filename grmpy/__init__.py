"""The module allows to run tests from inside the interpreter."""
import os

import pytest

from grmpy.simulate.simulate import simulate
from grmpy.grmpy_config import PACKAGE_DIR
from grmpy.estimate.estimate import fit
import grmpy.grmpy_config


def test():
    """The function allows to run the tests from inside the interpreter."""
    current_directory = os.getcwd()
    os.chdir(PACKAGE_DIR)
    pytest.main()
    os.chdir(current_directory)
