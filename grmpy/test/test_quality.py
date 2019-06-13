"""This module contains test that check the code quality of the package."""
import os
import subprocess
from subprocess import CalledProcessError

from grmpy.grmpy_config import PACKAGE_DIR


def test1():
    """This test runs flake8 to ensure the code quality. However, this is only relevant
    during development."""
    cwd = os.getcwd()
    os.chdir(PACKAGE_DIR)
    try:
        subprocess.check_call(["flake8"])
        os.chdir(cwd)
    except CalledProcessError:
        os.chdir(cwd)
        raise CalledProcessError
