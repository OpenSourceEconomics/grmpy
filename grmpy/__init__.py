"""The module allows to run tests from inside the interpreter
"""
import os

try:
    import pytest
except ImportError:
    pass

__version__ = '1.0.0'


def test():
    """ Run PYTEST for the package.
    """

    package_directory = os.path.dirname(os.path.realpath(__file__))
    current_directory = os.getcwd()

    os.chdir(package_directory)

    pytest.main()

    os.chdir(current_directory)
