"""This module contains a tutorial illustrating the basic capabilities of the grmpy
package.
"""
import os

from grmpy.estimate.estimate import fit
from grmpy.simulate.simulate import simulate

f = os.path.dirname(__file__) + "/tutorial.grmpy.yml"
simulate(f)
rslt = fit(f)
