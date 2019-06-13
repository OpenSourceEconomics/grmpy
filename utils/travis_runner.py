#!/usr/bin/env python
"""This script manages all tasks for the TRAVIS build server."""
import os
import subprocess as sp

if __name__ == "__main__":
    os.chdir("promotion/grmpy_tutorial_notebook")
    notebook = "grmpy_tutorial_notebook.ipynb"
    cmd = " jupyter nbconvert --execute {}  --ExecutePreprocessor.timeout=-1".format(
        notebook
    )
    sp.check_call(cmd, shell=True)
    os.chdir("../..")
