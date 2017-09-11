"""The module provides basic axiliary functions for the test modules."""
import glob
import os


def cleanup():
    for f in glob.glob("*.grmpy.*"):
        os.remove(f)
