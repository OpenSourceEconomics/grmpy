#!/usr/bin/env python
"""This module compiles the lecture notes."""
import subprocess
import argparse
import shutil
import glob
import os


def compile_single(is_update):
    """Compile a single lecture."""
    for task in ['pdflatex', 'bibtex', 'pdflatex', 'pdflatex']:
        subprocess.check_call(task + ' main', shell=True)

    if is_update:
        shutil.copy('main.pdf', '../../distribution/' + os.getcwd().split('/')[-1] + '.pdf')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(' Create slides for lecture')

    parser.add_argument('--update', action='store_true', dest='update',
                        help='update public slides')

    is_complete = 'lectures' == os.getcwd().split('/')[-1]
    is_update = parser.parse_args().update

    if is_complete:
        for dirname in glob.glob("0*"):
            os.chdir(dirname)
            compile_single(is_update)
            os.chdir('../')

        # I also want to have a complete deck of slides available. This is not intended for
        # public distribution.
        fnames = []
        for fname in sorted(glob.glob("0*")):
            fnames += [fname + '/main.pdf']
        cmd = 'pdftk ' + ' '.join(fnames) + ' cat output course_deck.pdf'
        subprocess.check_call(cmd, shell=True)
        if is_update:
            shutil.copy('course_deck.pdf', '../distribution/course_deck.pdf')

    else:

        compile_single(is_update)
