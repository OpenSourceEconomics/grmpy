"""This module contains test that check the code quality of the package."""
import os
import subprocess
from subprocess import CalledProcessError

# import pytest
from grmpy.grmpy_config import PACKAGE_DIR


# @pytest.mark.skip(reason="Obsolete. Would be used during development.")
def test_flake8():
    """Run flake8 to ensure the code quality. However, this is only relevant
    during development."""
    cwd = os.getcwd()
    os.chdir(PACKAGE_DIR)
    try:
        subprocess.check_call(["flake8"])
        os.chdir(cwd)
    except CalledProcessError:
        os.chdir(cwd)
        raise CalledProcessError(returncode=1, cmd="Flake8 standards not met.")
