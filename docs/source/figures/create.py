"""This module creates all available figures and then provides them in a compiled
document for review.
"""
import os
import subprocess

from scripts.fig_config import SCRIPTS_DIR


def create_figures():
    """This function creates all the figures that are available in its
    subdirectories.
    """
    for script in os.listdir(SCRIPTS_DIR):
        os.chdir(SCRIPTS_DIR)
        if script.startswith("fig-"):
            subprocess.check_call(["python", script])
        else:
            continue
        os.chdir("../")


if __name__ == "__main__":

    create_figures()
