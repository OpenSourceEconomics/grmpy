# grmpy

``grmpy``  is an open-source Python package for the simulation and estimation of the generalized Roy model. It serves as a teaching tool to promote the conceptual framework of the generalized Roy model, illustrate a variety of issues in the econometrics of policy evaluation, and showcases basic software engineering practices. <br>
Marginal Treatment Effects (MTE) can be estimated based on a parametric normal model or,
alternatively, via the semiparametric method of Local Instrumental Variables (LIV).

You can install ```grmpy``` either via pip

```
$ pip install grmpy
```
Or download it directly from our GitHub repository and install the package in editable mode

```
$ git clone https://github.com/OpenSourceEconomics/grmpy.git
$ pip install -e .
```

---
## Quick Start
> Initialization File

```grmpy``` relies on an ```"initialization.yml"``` file (referred to as ``ìnit_file`` below)
to perform both simulation and estimation.
For example, check out these two ``init_files`` for
[simulation and parametric estimation](https://github.com/OpenSourceEconomics/grmpy/blob/master/promotion/grmpy_tutorial_notebook/files/tutorial.grmpy.yml) as well as 
a [semiparametric estimation](https://github.com/OpenSourceEconomics/grmpy/blob/master/promotion/grmpy_tutorial_notebook/files/tutorial_semipar.yml) setup.

Below you'll find some example code you can copy to jump-start your project.  

> Simulation
```
import grmpy

# Specify the initilaization file you want to use, e.g.:
init_file = "ProjectFiles/simulation.yml"

data = grmpy.simulate(init_file)
```
> Estimation

```
import grmpy

# Specify the initilaization file you want to use, e.g.:
init_file = "ProjectFiles/estimation.yml"

# Parametric Normal Model
rslt = grmpy.fit(init_file, semipar=False)
grmpy.plot_mte(rslt, init_file, color="blue", semipar=False, save_plot="MTE_par.png")

# Local Instrumental Variables (Semiparametric Model)
rslt = grmpy.fit(init_file, semipar=True)
grmpy.plot_mte(rslt, init_file, color="orange", semipar=True, nboot= 250, save_plot="MTE_semipar.png")
```

Please visit our [online documentation](http://grmpy.readthedocs.io/) for tutorials and more.

-----
[![docs passing](https://travis-ci.org/OpenSourceEconomics/grmpy.svg?branch=master)]()
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)]()
[![code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
