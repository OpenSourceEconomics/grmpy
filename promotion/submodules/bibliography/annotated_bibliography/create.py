#!/usr/bin/env python
"""This module compiles the notes with the figures."""
import subprocess
import shutil
import os

if __name__ == '__main__':

    os.chdir('sources')

    for task in ['pdflatex', 'bibtex', 'pdflatex', 'pdflatex']:
        subprocess.check_call(task + ' main', shell=True)

    shutil.move('main.pdf', '../annotated-bibliography.pdf')

    os.chdir('../')
    subprocess.check_call('git clean -d -f', shell=True)
