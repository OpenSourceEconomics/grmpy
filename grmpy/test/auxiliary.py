"""The module provides basic axiliary functions for the test modules."""
import shlex
import glob
import os

from grmpy.estimate.estimate_auxiliary import transform_rslt_DIST
from grmpy.test.random_init import print_dict
from grmpy.read.read import read


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
    elif options == 'init_file':
        for f in glob.glob("*.grmpy.*"):
            if f.startswith('test.grmpy'):
                pass
            else:
                os.remove(f)

def save_output(file, option):
    """The function renames a given file and moves it in an output directory."""
    if not os.path.isfile(file):
        raise AssertionError()
    directory = os.path.join(os.getcwd(), 'estimation_output')
    os.rename(file, option)
    if not os.path.isdir(directory):
        os.makedirs(directory)
    os.rename(os.path.join(os.getcwd(), option),
              os.path.join(directory, option))


def read_desc(fname):
    """The function reads the descriptives output file and returns a dictionary that contains the
    relevant parameters for test6 in test_integration.py.
    """
    dict_ = {}
    with open(fname, 'r') as handle:
        for i, line in enumerate(handle):
            list_ = shlex.split(line)
            if 7 <= i < 10:
                if list_[0] in ['All', 'Treated', 'Untreated']:
                    dict_[list_[0]] = {}
                    dict_[list_[0]]['Number'] = list_[1:]
            elif 20 <= i < 23:
                print(list_)
                if list_[0] == 'Observed':
                    dict_['All'][list_[0] + ' ' + list_[1]] = list_[2:]
                else:
                    dict_['All'][list_[0] + ' ' + list_[1] + ' ' + list_[2]] = list_[3:]
            elif 29 <= i < 32:
                if list_[0] == 'Observed':
                    dict_['Treated'][list_[0] + ' ' + list_[1]] = list_[2:]
                else:
                    dict_['Treated'][list_[0] + ' ' + list_[1] + ' ' + list_[2]] = list_[3:]
            elif 38 <= i < 41:
                if list_[0] == 'Observed':
                    dict_['Untreated'][list_[0] + ' ' + list_[1]] = list_[2:]
                else:
                    dict_['Untreated'][list_[0] + ' ' + list_[1] + ' ' + list_[2]] = list_[3:]

    return dict_


def adjust_output_cholesky(output):
    """The function transfers the output of the cholesky decomposition process so that it is similar
    in regards of to the distributional information of the init file.
    """
    output[1] = output[1] * (output[0] * output[3])
    output[2] = output[2] * (output[0] * output[5])
    output[4] = output[4] * (output[3] * output[5])
    return output


def refactor_results(dict_, file, newfile):
    """The function generates a new init file based on a dictionary with parameter values from a previous
    estimation process.
    """
    
    pseudo = read(file)

    for key in ['TREATED', 'UNTREATED', 'COST', 'DIST']:
        if key == 'DIST':
            pseudo['DIST']['all'] = dict_['AUX']['x_internal'][-6:]
        else:
            pseudo[key]['all'] = dict_[key]['all'].tolist()

    pseudo = transform_rslt_DIST(dict_['AUX']['x_internal'], pseudo)
    print_dict(pseudo, newfile)
