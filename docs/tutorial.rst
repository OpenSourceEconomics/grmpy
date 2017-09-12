Tutorial
========

We now illustrate the basic capabilities of the ``grmpy`` package. We start with the assumptions about functional form and the distribution of unobservables and then turn to some simple use cases.

Assumptions
------------

The ``grmpy`` package implements the normal linear-in-parameters version of the generalized Roy model. Both potential outcomes and the cost associated with treatment participations :math:`(Y_1, Y_0, C)` are a linear function of the individual's observables :math:`(X, Z)` and random components :math:`(U_1, U_0, U_C)`.

.. math::
    Y_1 & = X \beta_1 + U_1 \\
    Y_0 & = X \beta_0 + U_0 \\
    C   & = Z \gamma + U_C \\

The unobservables follow a normal distribution :math:`(U_1, U_0, U_C) \sim \mathcal{N}(0, \Sigma)` with mean zero and covariance matrix :math:`\Sigma`. We collect all unobservables determining treatment choice in :math:`V = U_C - (U_1 - U_0)`. Individuals decide to select into treatment if their surplus from doing so is positive :math:`S = Y_1 - Y_0 - C`. Depending on their decision, we either observe :math:`Y_1` or :math:`Y_0`.

Model Specification
-------------------

You can specify the details of the model in an initialization file (`example <https://github.com/restudToolbox/package/blob/master/respy/tests/resources/kw_data_one.ini>`_). This file contains several blocks:

**SIMULATION**

The *SIMULATION* block contains some basic information about the simulation request.

=======     ======      ==================
Key         Value       Interpretation
=======     ======      ==================
agents      int         number of individuals
seed        int         seed for the specific simulation
source      str         specified name for the simulation output files
=======     ======      ==================

**TREATED**

The *TREATED* block specifies the number of covariates determining the potential outcome in the treated state and the values for the coefficients :math:`\beta_1`.

=======     ======  ==================
Key         Value   Interpretation
=======     ======  ==================
coeff       float   intercept coefficient
coeff       float   coefficient of the first covariate
coeff       float   coefficient of the second covariate
 ...
=======     ======  ==================

**UNTREATED**

The *UNTREATED* block specifies the number of covariates determining the potential outcome in the untreated state and the values for the coefficients :math:`\beta_0`. Note that the covariates need to be identical to the *TREATED* block.

=======     ======  ==================
Key         Value   Interpretation
=======     ======  ==================
coeff       float   intercept coefficient
coeff       float   coefficient of the first covariate
coeff       float   coefficient of the second covariate
 ...
=======     ======  ==================

**COST**

The *COST* block specifies the number of covariates determining the cost of treatment and the values for the coefficients :math:`\gamma`.

=======     ======  ==================
Key         Value   Interpretation
=======     ======  ==================
coeff       float   intercept coefficient
coeff       float   coefficient of the first covariate
coeff       float   coefficient of the second covariate
 ...
=======     ======  ==================

**DIST**

The **DIST** block specifies the distribution of the unobservables.

.. warning:: This block specifies the correlation structure between :math:`(U_1, U_0, V)` and not :math:`(U_1, U_0, U_C)`.

======= ======      ==========================
Key     Value       Interpretation
======= ======      ==========================
coeff    float      :math:`\sigma_{U_0}`
coeff    float      :math:`\sigma_{U_1,U_0}`
coeff    float      :math:`\sigma_{U_0,V}`
coeff    float      :math:`\sigma_{U_1}`
coeff    float      :math:`\sigma_{U_1,V}`
coeff    float      :math:`\sigma_{V}`
======= ======      ==========================

Examples
--------

In the following chapter we explore the basic features of the ``grmpy`` package. The module with this tutorial is available (`online <https://github.com/restudToolbox/package/blob/master/respy/tests/resources/kw_data_one.ini>`_).

Initially we simply load the package.

::

    import grmpy


    from grmpy.test.random_init import generate_random_dict()

**Specifiying Simulation Characteristics**

In the first step we determine the parametrization of our model. For this purpose you could create a initialization file by your own preferences. For information relating the structure of the initialization file see the **Model Specification** chapter above.
In our specific example we will generate a random initialization file by using the included ``generate_random_dict()`` function.
::


    generate_random_dict()


The function creates a random initialization file like the one below.

.. todo::
    insert example image of an initialization file

**Simulation**

Next we simulate a sample according to our pre specified characteristics.
::

    simulate('test.grmpy.ini)

During this process the functions returns the following output files:

    - ######.grmpy.info:
        An information file that provides information about
            * The number of all, treated and untreated individuals
            * The outcome distribution
            * The distribution of effects of interest
            * MTE by quantile
            * The parametrization.

    - ######.grmpy.txt: The simulated data frame as a txt file.

    - ######.grmpy.pkl: The simulated data frame as a pickle file.


.. Warning::

    - The prefix of the output files is determined by the given **source** entry in the **SIMULATION** section of your initialization file.

    - Note that you have to provide the name of your initialization file as an input in the simulate function. If you generate a random initialization file, the name is fixed to *test.grmpy.ini*.

    - The function is able to return a dataframe directly by setting ``data_frame = simulate('test.grmpy.ini')``
