"""The module provides basic axiliary functions for the test modules."""
import glob
import os


def cleanup(options=None):
    """The function deletes package related output files."""
    if options is None:
        for f in glob.glob("*.grmpy.*"):
            os.remove(f)
    elif options == 'regression':
        for f in glob.glob("*.grmpy.*"):
            if f.startswith('regression'):
                pass
            else:
                os.remove(f)

