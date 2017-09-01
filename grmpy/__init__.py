"""The module allows to run tests from inside the interpreter."""
import os

import pytest


def test():
    """The function allows to run the tests from inside the interpreter."""

    package_directory = os.path.dirname(os.path.realpath(__file__))
    current_directory = os.getcwd()

    os.chdir(package_directory)

    pytest.main()

    os.chdir(current_directory)
