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


def save_output(file, option):
    """The function renames a given file and moves it in an output directory."""
    assert os.path.isfile(file)
    dir = os.path.join(os.getcwd(),'estimation_output')
    os.rename(file, option )
    if not os.path.isdir(dir):
        os.makedirs(dir)
    os.rename(os.path.join(os.getcwd(), option),
              os.path.join(dir, option))

