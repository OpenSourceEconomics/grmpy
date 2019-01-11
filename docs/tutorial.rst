Tutorial
========

We now illustrate the basic capabilities of the ``grmpy`` package. We start with the assumptions about functional form and the distribution of unobservables and then turn to some simple use cases.

Assumptions
------------

The ``grmpy`` package implements the normal linear-in-parameters version of the generalized Roy model. Both potential outcomes and the choice :math:`(Y_1, Y_0, D)` are a linear function of the individual's observables :math:`(X, Z)` and random components :math:`(U_1, U_0, V)`.


.. math::
    Y_1  &= X \beta_1 + U_1 \\
    Y_0  &= X \beta_0 + U_0 \\
    D &= I[D^{*} > 0] \\
    D^{*}    &= Z \gamma -V

We collect all unobservables determining treatment choice in :math:`V = U_C - (U_1 - U_0)`. The unobservables follow a normal distribution :math:`(U_1, U_0, V) \sim \mathcal{N}(0, \Sigma)` with mean zero and covariance matrix :math:`\Sigma`.  Individuals decide to select into latent indicator variable :math:`D^{*}` is positive. Depending on their decision, we either observe :math:`Y_1` or :math:`Y_0`.

Model Specification
-------------------

You can specify the details of the model in an initialization file (`example <https://github.com/OpenSourceEconomics/grmpy/blob/master/docs/tutorial/tutorial.grmpy.yml>`_). This file contains several blocks:

**SIMULATION**

The *SIMULATION* block contains some basic information about the simulation request.

=======     ======      ==============================================
Key         Value       Interpretation
=======     ======      ==============================================
agents      int         number of individuals
seed        int         seed for the specific simulation
source      str         specified name for the simulation output files
=======     ======      ==============================================

**ESTIMATION**

The *ESTIMATION* block determines the basic information for the estimation process.

===========     ======      ===============================================
Key             Value       Interpretation
===========     ======      ===============================================
agents          int         number of individuals (for the comparison file)
file            str         name of the estimation specific init file
optimizer       str         optimizer used for the estimation process
start           str         flag for the start values
maxiter	        int         maximum numbers of iterations
dependent       str         indicates the dependent variable
indicator       str         label of the treatment indicator variable
output_file     str         name for the estimation output file
comparison	int         flag for enabling the comparison file creation
===========     ======      ===============================================



**TREATED**

The *TREATED* block specifies the number and order of the covariates determining the potential outcome in the treated state and the values for the coefficients :math:`\beta_1`. Note that the length of the list which determines the paramters has to be equal to the number of variables that are included in the order list.

=======   =========  ======     ===================================
Key       Container  Values     Interpretation
=======   =========  ======     ===================================
params    list       float      Paramters
order     list       str        Variable labels
=======   =========  ======     ===================================


**UNTREATED**

The *UNTREATED* block specifies the covariates that a the potential outcome in the untreated state and the values for the coefficients :math:`\beta_0`.

=======   =========  ======     ===================================
Key       Container  Values     Interpretation
=======   =========  ======     ===================================
params    list       float      Paramters
order     list       str        Variable labels
=======   =========  ======     ===================================

**CHOICE**

The *CHOICE* block specifies the number and order of the covariates determining the selection process and the values for the coefficients :math:`\gamma`.

=======   =========  ======     ===================================
Key       Container  Values     Interpretation
=======   =========  ======     ===================================
params    list       float      Paramters
order     list       str        Variable labels
=======   =========  ======     ===================================

**DIST**

The *DIST* block specifies the distribution of the unobservables.

=======   =========  ======     =========================================
Key       Container  Values     Interpretation
=======   =========  ======     =========================================
params    list       float      Upper triangular of the covariance matrix
=======   =========  ======     =========================================

**VARTYPES**

The *VARTYPES* section enables users to specify optional characteristics to specific variables in their simulated data. Currently there is only the option to determine binary variables. For this purpose the user have to specify a key which reflects the corresponding variable label and assign a list to this label which contains the type (*binary*) as a string as well as a float (<0.9) that determines the probability for which the variable is one.

================   =========  ================     =========================================
Key                Container  Values               Interpretation
================   =========  ================     =========================================
*Variable label*   list       string and float     Type of variable + additional information
================   =========  ================     =========================================




**SCIPY-BFGS**

The *SCIPY-BFGS* block contains the specifications for the *BFGS* minimization algorithm. For more information see: `SciPy documentation <https://docs.scipy.org/doc/scipy-0.19.0/reference/optimize.minimize-bfgs.html#optimize-minimize-bfgs>`__.

========  ======      ==================================================================================
Key       Value       Interpretation
========  ======      ==================================================================================
gtol      float       the value that has to be larger as the gradient norm before successful termination
eps       float       value of step size (if *jac* is approximated)
========  ======      ==================================================================================

**SCIPY-POWELL**

The *SCIPY-POWELL* block contains the specifications for the *POWELL* minimization algorithm. For more information see: `SciPy documentation <https://docs.scipy.org/doc/scipy-0.19.0/reference/optimize.minimize-powell.html#optimize-minimize-powell>`__.

========  ======      ===========================================================================
Key       Value       Interpretation
========  ======      ===========================================================================
xtol       float      relative error in solution values *xopt* that is acceptable for convergence
ftol       float      relative error in fun(*xopt*) that is acceptable for convergence
========  ======      ===========================================================================


Examples
--------

In the following chapter we explore the basic features of the ``grmpy`` package. The resources for the tutorial are also available `online <https://github.com/OpenSourceEconomics/grmpy/tree/master/docs/tutorial>`_.
So far the package provides the features to simulate a sample from the generalized Roy model and to estimate some parameters of interest for a provided sample as specified in your initialization file.

**Simulation**

First we will take a look on the simulation feature. For simulating a sample from the generalized Roy model you use the ``simulate()`` function provided by the package. For simulating a sample of your choice you have to provide the path of your initialization file as an input to the function.
::

    import grmpy

    grmpy.simulate('tutorial.grmpy.yml')


This creates a number of output files that contain information about the resulting simulated sample.

* **data.grmpy.info**, basic information about the simulated sample
* **data.grmpy.txt**, simulated sample in a simple text file
* **data.grmpy.pkl**, simulated sample as a pandas data frame


**Estimation**

The other feature of the package is the estimation of the parameters of interest. The specification regarding start values and and the optimizer options are determined in the *ESTIMATION* section of the initialization file.

::

    grmpy.fit('tutorial.grmpy.yml')

As in the simulation process this creates a number of output file that contains information about the estimation results.

* **est.grmpy.info**, basic information of the estimation process
* **comparison.grmpy.txt**, distributional characteristics of the input sample and the samples simulated from the start and result values of the estimation process
