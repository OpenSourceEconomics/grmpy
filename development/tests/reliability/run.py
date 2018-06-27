#!/usr/bin/env python
""" This module contains a first stab at a reliability test."""
import grmpy
import sys

from grmpy.simulate.simulate import simulate
from grmpy.estimate.estimate import estimate



option = 'true_values'
estimate('test.grmpy.ini')

for type in ['grmpy', 'ols', '2sls', 'random']:
    filename = 'fig_{}_average_effect_estimation.png'.format(type)
    move(join(directory, filename), join(target, filename))
