.. grmpy documentation master file, created by
   sphinx-quickstart on Fri Aug 18 13:05:32 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to grmpy's documentation!
=================================

`PyPI <https://pypi.python.org/pypi/grmpy>`_ | `GitHub <https://github.com/OpenSourceEconomics/grmpy>`_  | `Issues <https://github.com/OpenSourceEconomics/grmpy/issues>`_

``grmpy``  is an open-source Python package for the simulation and estimation of the generalized Roy model. It serves as a teaching tool to promote the conceptual framework of the generalized Roy model, illustrate a variety of issues in the econometrics of policy evaluation, and showcase basic software engineering practices.

We build on the following main references:

    James J. Heckman and Edward J. Vytlacil. `Econometric evaluation of social programs, part I: Causal models, structural models and econometric policy evaluation. <http://ac.els-cdn.com/S1573441207060709/1-s2.0-S1573441207060709-main.pdf?_tid=b933f5c8-6bbe-11e7-8ae8-00000aacb35d&acdnat=1500385435_c69182d36b79b66bbce5f5a7c593617c>`_ In *Handbook of Econometrics*, volume 6B, chapter 70, pages 4779–4874. Elsevier Science, 2007.

    James J. Heckman and Edward J. Vytlacil. `Econometric evaluation of social programs, part II: Using the marginal treatment effect to organize alternative econometric estimators to evaluate social programs, and to forecast their effects in new environments. <http://ac.els-cdn.com/S1573441207060710/1-s2.0-S1573441207060710-main.pdf?_tid=5ccb4ace-6bbf-11e7-807b-00000aab0f26&acdnat=1500385710_c3706f18138fabe356b0f3ebddd75670>`_ In *Handbook of Econometrics*, volume 6B, chapter 71, pages 4875–5143. Elsevier Science, 2007.

The remainder of this documentation is structured as follows. We first present the basic economic model and provide installation instructions. We then illustrate the basic use case of the package in a tutorial and showcase some evidence regarding its reliability. The documentation concludes with some housekeeping issues.

Please see `here <https://github.com/HumanCapitalEconomics/econometrics/blob/master/README.md>`_ for a host of lectures material on the econometrics of policy evaluation.

.. htmlonly::
.. image:: https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000   :target:

.. toctree::
    :hidden:
    :maxdepth: 1

    economics
    installation
    tutorial
    reliability
    software_engineering
    contributing
    credits
    changes
    bibliography


.. todo::


    This `todoList` is just a random collection of future features to be implemented. It is not printed on Read the Docs.

    * If the interaction with pypi gets too cumbersome, we can consider using bumpversion, zest.releaser, or hatch ...

    * We need all figures formatted exactly identical, referenced in the main text and figure headings. There is no apparent naming convention for the python scripts and files in the figures subdirectory. Do we need the __init__ file, remove carneiro paper screenshot.

    * We want the table of content in the pdf two go to depth two, but now show up in the html version at all.

    * The main reference formatting does not look identical to the citations on the website.

    * We want to get rid of the malformatting warnings during build.
