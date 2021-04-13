"""The module allows to run tests from inside the interpreter."""
import os

import pytest

from grmpy.estimate.estimate import fit
from grmpy.simulate.simulate import simulate


def test():
    """The function allows to run the tests from inside the interpreter."""

    package_directory = os.path.dirname(os.path.realpath(__file__))
    current_directory = os.getcwd()

    os.chdir(package_directory)

    pytest.main()

    os.chdir(current_directory)
