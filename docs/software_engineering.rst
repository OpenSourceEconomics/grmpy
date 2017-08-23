Software Engineering
====================

We now briefly discuss our software engineering practices that help us to ensure the transparency, reliability, scalability, and extensibility of the ``grmpy`` package.

Test Battery
------------

.. image:: https://codecov.io/gh/grmpy/package/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/grmpy/package

We use `pytest <http://docs.pytest.org>`_ as our test runner. We broadly group our tests in four categories:

* **property-based testing**

    We create random model parameterizations and estimation requests and test for a valid return of the program.

* **reliability testing**

    We conduct numerous Monte Carlo exercises to ensure that we can recover the true underlying parameterization with an estimation. Also by varying the tuning parameters of the estimation (e.g. random draws for integration) and the optimizers, we learn about their effect on estimation performance.


Documentation
-------------

.. image:: https://readthedocs.org/projects/grmpy/badge/?version=latest
   :target: http://respy.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

The documentation is created using `Sphinx <http://www.sphinx-doc.org/>`_ and hosted on `Read the Docs <https://readthedocs.org/>`_.

Code Review
-----------

.. image:: https://api.codacy.com/project/badge/Grade/3dd368fb739c49d78d910676c9264a81
   :target: https://www.codacy.com/app/eisenhauer/respy?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=restudToolbox/package&amp;utm_campaign=Badge_Grade

.. image:: https://landscape.io/github/restudToolbox/package/master/landscape.svg?style=flat
    :target: https://landscape.io/github/restudToolbox/package/master
    :alt: Code Health

We use several automatic code review tools to help us improve the readability and maintainability of our code base. For example, we work with `Codacy <https://www.codacy.com/app/eisenhauer/respy/dashboard>`_ and `Landscape <https://landscape.io/github/restudToolbox/package>`_

Continuous Integration Workflow
-------------------------------

.. image:: https://travis-ci.org/restudToolbox/package.svg?branch=master
   :target: https://travis-ci.org/restudToolbox/package

.. image:: https://requires.io/github/restudToolbox/package/requirements.svg?branch=master
    :target: https://requires.io/github/restudToolbox/package/requirements/?branch=master
    :alt: Requirements Status

.. image:: https://badge.fury.io/py/respy.svg
    :target: https://badge.fury.io/py/respy

We set up a continuous integration workflow around our `GitHub Organization <https://github.com/restudToolbox>`_. We use the continuous integration services provided by `Travis CI <https://travis-ci.org/restudToolbox/package>`_. `tox <https://tox.readthedocs.io>`_ helps us to ensure the proper workings of the package for alternative Python implementations. Our build process is managed by `Waf <https://waf.io/>`_. We rely on `Git <https://git-scm.com/>`_ as our version control system and follow the `Gitflow Workflow <https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow>`_. We use `GitLab <https://gitlab.com/restudToolbox/package/issues>`_ for our issue tracking. The package is distributed through `PyPI <https://pypi.python.org/pypi/respy>`_ which automatically updated from our development server.
