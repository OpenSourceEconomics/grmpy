Software Engineering
====================

We now briefly discuss our software engineering practices that help us to ensure the transparency, reliability, scalability, and extensibility of the ``grmpy`` package. Please visit us at the `Software Engineering for Economists Initiative <http://softecon.github.io/>`_ for an accessible introduction on how to integrate these practices in your own research.

Test Battery
------------
.. image:: https://codecov.io/gh/OpenSourceEconomics/grmpy/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/OpenSourceEconomics/grmpy

We use `pytest <http://docs.pytest.org>`_ as our test runner. We broadly group our tests in three categories:

* **property-based testing**

    We create random model parameterizations and estimation requests and test for a valid return of the program.

* **reliability testing**

    We conduct numerous Monte Carlo exercises to ensure that we can recover the true underlying parameterization with an estimation. Also by varying the tuning parameters of the estimation (e.g. random draws for integration) and the optimizers, we learn about their effect on estimation performance.

* **regression testing**

    We provide a regression test. For this purpose we generated random model parameterizations, simulated the coresponding outputs, summed them up and saved both, the parameters and the sums in a json file.
    The json file is part of the package. Through this the provided test is able to draw parameterizations randomly from the json file. In the next step the test simulates the output variables and compares the sum of the simulated output with the associated json file information.
    This ensures that the package works accurate even after an update to a new version.

Documentation
-------------
.. image:: https://readthedocs.org/projects/grmpy/badge/?version=latest
   :target: http://grmpy.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

The documentation is created using `Sphinx <http://www.sphinx-doc.org/>`_ and hosted on `Read the Docs <https://readthedocs.org/>`_.

Code Review
-----------

.. image:: https://api.codacy.com/project/badge/Grade/e27b1ed4789f4d5596e84177a58dd2d8
   :target: https://www.codacy.com/app/eisenhauer/grmpy?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=grmToolbox/grmpy&amp;utm_campaign=Badge_Grade

We use several automatic code review tools to help us improve the readability and maintainability of our code base. For example, we work with `Codacy <https://www.codacy.com/app/eisenhauer/grmpy/dashboard>`_. However, we also conduct regular peer code-reviews using `Reviewable <https://reviewable.io/>`_.


Continuous Integration Workflow
-------------------------------
.. image:: https://travis-ci.org/OpenSourceEconomics/grmpy.svg?branch=master
   :target: https://travis-ci.org/OpenSourceEconomics/grmpy

We set up a continuous integration workflow around our `GitHub Organization <https://github.com/OpenSourceEconomics>`_. We use the continuous integration services provided by `Travis CI <https://travis-ci.org/OpenSourceEconomics/grmpy>`_. `tox <https://tox.readthedocs.io/en/latest/>`_ ensures that the package installs correctly with different Python versions.
