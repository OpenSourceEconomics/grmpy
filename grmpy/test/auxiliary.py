"""The module provides basic axiliary functions for the test modules."""
import shlex
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
    directory = os.path.join(os.getcwd(), 'estimation_output')
    os.rename(file, option)
    if not os.path.isdir(directory):
        os.makedirs(directory)
    os.rename(os.path.join(os.getcwd(), option),
              os.path.join(directory, option))


def read_desc(fname):
    dict_ = {}
    with open(fname, 'r') as handle:
        for i, line in enumerate(handle):
            list_ = shlex.split(line)
            if i >= 6 and i < 9:
                if list_[0] in ['All', 'Treated', 'Untreated']:
                    dict_[list_[0]] = {}
                    dict_[list_[0]]['Number'] = list_[1:]
            elif i >= 19 and i < 22:
                if list_[0] == 'Observed':
                    dict_['All'][list_[0] + ' ' + list_[1]] = list_[2:]
                else:
                    dict_['All'][list_[0] + ' ' + list_[1] + ' ' + list_[2]] = list_[3:]
            elif i >= 28 and i < 31:
                if list_[0] == 'Observed':
                    dict_['Treated'][list_[0] + ' ' + list_[1]] = list_[2:]
                else:
                    dict_['Treated'][list_[0] + ' ' + list_[1] + ' ' + list_[2]] = list_[3:]
            elif i >= 37 and i < 40:
                if list_[0] == 'Observed':
                    dict_['Untreated'][list_[0] + ' ' + list_[1]] = list_[2:]
                else:
                    dict_['Untreated'][list_[0] + ' ' + list_[1] + ' ' + list_[2]] = list_[3:]

    return dict_
