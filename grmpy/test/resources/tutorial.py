"""This module contains a tutorial illustrating the basic capabilities of the grmpy package."""
import os

import grmpy

if __name__ == '__main__':
    f = os.path.dirname(grmpy.__file__) + '/test/resources/tutorial.grmpy.ini'
    grmpy.simulate(f)
    rslt = grmpy.estimate(f)
