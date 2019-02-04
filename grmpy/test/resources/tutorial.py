"""This module contains a tutorial illustrating the basic capabilities of the grmpy package."""
import os

from grmpy.simulate.simulate import simulate
from grmpy.estimate.estimate import estimate

f = os.path.dirname(__file__) + "/tutorial.grmpy.ini"
simulate(f)
rslt = estimate(f)
