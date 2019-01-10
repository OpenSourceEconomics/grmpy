.. grmpy documentation master file, created by
   sphinx-quickstart on Fri Aug 18 13:05:32 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to grmpy's documentation!
=================================

`PyPI <https://pypi.python.org/pypi/grmpy>`_ | `GitHub <https://github.com/OpenSourceEconomics/grmpy>`__  | `Issues <https://github.com/OpenSourceEconomics/grmpy/issues>`_

``grmpy``  is an open-source package for the simulation and estimation of the generalized Roy model. It serves as a teaching tool to promote the conceptual framework of the generalized Roy model, illustrate a variety of issues in the econometrics of policy evaluation, and showcase basic software engineering practices.

We build on the following main references:

    James J. Heckman and Edward J. Vytlacil. `Econometric evaluation of social programs, part I: Causal models, structural models and econometric policy evaluation. <https://www.sciencedirect.com/science/article/pii/S1573441207060709>`_ In *Handbook of Econometrics*, volume 6B, chapter 70, pages 4779–4874. Elsevier Science, 2007.

    James J. Heckman and Edward J. Vytlacil. `Econometric evaluation of social programs, part II: Using the marginal treatment effect to organize alternative econometric estimators to evaluate social programs, and to forecast their effects in new environments. <https://www.sciencedirect.com/science/article/pii/S1573441207060710>`_ In *Handbook of Econometrics*, volume 6B, chapter 71, pages 4875–5143. Elsevier Science, 2007.

    Jaap H. Abbring and James J. Heckman. `Econometric evaluation of social programs, part III: Distributional treatment effects, dynamic treatment effects, dynamic discrete choice, and general equilibrium policy evaluation. <https://www.sciencedirect.com/science/article/pii/S1573441207060722>`_ *Handbook of Econometrics*, volume 6B, chapter 72, pages 5145-5303. Elsevier Science, 2007.


The remainder of this documentation is structured as follows. We first present the basic economic model and provide installation instructions. We then illustrate the basic use case of the package in a tutorial and showcase some evidence regarding its reliability. In addition we provide some information on the software engineering tools that are used for transparency and dependability purposes. The documentation concludes with further information on contributing, contact details as well as a listing of the latest releases.

The package is used as a teaching tool for a course on the analysis human capital at the University of Bonn. The affiliated lecture material is available on `GitHub <https://github.com/HumanCapitalEconomics/econometrics/blob/master/README.md>`__ .

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
