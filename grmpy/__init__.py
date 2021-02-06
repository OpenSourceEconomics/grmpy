"""This is the entry-point to the grmpy package.

Include only imports which should be available using

.. code-block::

    import grmpy as gp

    gp.<func>
"""
import pytest

from grmpy.estimate.estimate import fit  # noqa: F401
from grmpy.grmpy_config import ROOT_DIR
from grmpy.plot.plot import plot_mte  # noqa: F401
from grmpy.simulate.simulate import simulate  # noqa: F401


def test(*args, **kwargs):
    """Run basic tests of the package."""
    pytest.main([str(ROOT_DIR), *args], **kwargs)
