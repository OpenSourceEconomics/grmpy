#!/usr/bin/env python
"""This script manages all tasks for the TRAVIS build server."""
import os
import subprocess

if __name__ == "__main__":
    os.chdir("promotion/grmpy_tutorial_notebook")
    cmd = [
        "jupyter",
        "nbconvert",
        "--execute",
        "grmpy_tutorial_notebook.ipynb",
        "--ExecutePreprocessor.timeout=-1",
    ]
    subprocess.check_call(cmd)
    os.chdir("../..")


# if __name__ == "__main__":
#     os.chdir("promotion/grmpy_tutorial_notebook")
#     cmd = [
#         "jupyter",
#         "nbconvert",
#         "--execute",
#         "tutorial_semipar_notebook.ipynb",
#         "--ExecutePreprocessor.timeout=-1",
#     ]
#     subprocess.check_call(cmd)
