#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine

import os
import sys
from pathlib import Path
from shutil import rmtree

from setuptools import Command, find_packages, setup

# Package meta-data.
NAME = "grmpy"
DESCRIPTION = (
    "grmpy is a Python package for the simulation and estimation of the "
    "generalized Roy model."
)
README = Path("README.md").read_text()
URL = "http://grmpy.readthedocs.io"
EMAIL = "eisenhauer@policy-lab.org"
AUTHOR = "The grmpy Development Team"

# What packages are required for this module to be executed?
REQUIRED = [
    "numpy",
    "scipy",
    "pytest",
    "pandas",
    "statsmodels",
    "linearmodels",
    "oyaml",
    "matplotlib",
    "seaborn",
    "scikit-misc",
    "numba",
]

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.rst' is present in your MANIFEST.in file!
# with io.open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
#     long_description = '\n' + f.read()

# Load the package's __version__.py module as a dictionary.
about = {}
with open(os.path.join(here, NAME, "__version__.py")) as f:
    exec(f.read(), about)


class PublishCommand(Command):
    """Support setup.py publish."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
        except FileNotFoundError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPi via Twine…")
        os.system("twine upload dist/*")

        sys.exit()


setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=DESCRIPTION + "\n\n" + README,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages(exclude=("tests",)),
    install_requires=REQUIRED,
    license="MIT",
    include_package_data=True,
    cmdclass={"publish": PublishCommand},
)
