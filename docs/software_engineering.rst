Software Engineering
====================

We now briefly discuss our software engineering practices that help us to ensure the transparency, reliability, scalability, and extensibility of the ``grmpy`` package.

Test Battery
------------

.. image:: https://codecov.io/gh/grmToolbox/grmpy/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/grmToolbox/grmpy

We use `pytest <http://docs.pytest.org>`_ as our test runner. We broadly group our tests in four categories:

* **property-based testing**

    We create random model parameterizations and estimation requests and test for a valid return of the program.

* **reliability testing**

    We conduct numerous Monte Carlo exercises to ensure that we can recover the true underlying parameterization with an estimation. Also by varying the tuning parameters of the estimation (e.g. random draws for integration) and the optimizers, we learn about their effect on estimation performance.

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

We use several automatic code review tools to help us improve the readability and maintainability of our code base. For example, we work with `Codacy <https://www.codacy.com/app/eisenhauer/grmpy/dashboard>`.

Continuous Integration Workflow
-------------------------------

.. image:: https://travis-ci.org/grmToolbox/grmpy.svg?branch=master
   :target: https://travis-ci.org/grmToolbox/grmpy

We set up a continuous integration workflow around our `GitHub Organization <https://github.com/grmToolbox>`_. We use the continuous integration services provided by `Travis CI <https://travis-ci.org/grmToolbox/grmpy>`.
