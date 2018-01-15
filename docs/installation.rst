Installation
============

The ``grmpy`` package can be conveniently installed from the `Python Package Index <https://pypi.python.org/pypi>`_ (PyPI) or directly from its source files. We currently support Python 2.7 and Python 3.6 on Linux systems.

Python Package Index
--------------------

You can install the stable version of the package the usual way.

.. code-block:: bash

   $ pip install grmpy

Source Files
------------

You can download the sources directly from our `GitHub repository <https://github.com/grmToolbox/grmpy>`_.

.. code-block:: bash

   $ git clone https://github.com/grmpy/package.git

Once you obtained a copy of the source files, installing the package in editable model is straightforward.

.. code-block:: bash

   $ pip install -e .

Test Suite
----------

Please make sure that the package is working properly by running our test suite using ``pytest``.

.. code-block:: bash

  $ python -c "import grmpy; grmpy.test()"
