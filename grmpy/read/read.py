"""The module contains the main function of the init file import process."""
import shlex
import os

from grmpy.read.read_auxiliary import auxiliary
from grmpy.read.read_auxiliary import process
from grmpy.check.check import UserError


def read(file_):
    """The function reads the initialization file and returns a dictionary with parameters for the
    simulation.
    """
    if not os.path.isfile(file_):
        msg = '{}: There is no such file or directory.'.format(file_)
        raise UserError(msg)

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

