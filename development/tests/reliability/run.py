#!/usr/bin/env python
""" This module contains a first stab at a reliability test."""
import grmpy
import sys

from grmpy.simulate.simulate import simulate
from grmpy.estimate.estimate import estimate



option = 'true_values'
estimate('test.grmpy.ini')

