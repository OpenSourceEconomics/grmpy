

import shlex

from grmpy.read.read_auxiliary import auxiliary
from grmpy.read.read_auxiliary import process


def read(file_):
    """reads the initialization file and provides an dictionary with parameters for the simulation"""
    dict_ = {}
    for line in open(file_).readlines():

        list_ = shlex.split(line)

        is_empty = (list_ == [])

        if not is_empty:
            is_keyword = list_[0].isupper()
        else:
            is_keyword = False

        if is_empty:
            continue

        if is_keyword:
            keyword = list_[0]
            dict_[keyword] = {}
            continue

        process(list_, dict_, keyword)

    dict_ = auxiliary(dict_)

    return dict_

