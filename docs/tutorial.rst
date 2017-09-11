Tutorial
========

We now illustrate the basic capabilities of the ``grmpy`` package. We start with the model parametrization and specification before turning to some example use cases.


Parametric Assumptions
----------------------

The package focuses on the linear form of the generalized roy model for reasons of simplicity. The model is characterized by the following equations:

**Outcome**

:math:`Y_1` and :math:`Y_0` represent the ex post outcome for each individual depending on treatment status.

.. math::
    Y_1 & = X \beta_1 + U_1 \\
    Y_0 & = X \beta_0 + U_0 \\

**Costs**

The cost function illustrates the costs of an individual for selecting in the treatment group.

.. math::
        C & = Z \gamma + U_C \\

**Choice Parameters**

Individuals have an incentive to select themselves in the treatment group if their associated surplus is positive. The surplus :math:`S` is defined as the  difference between their outputs :math:`Y_1` and :math:`Y_0` minus the subjective costs :math:`C` for selecting into treatment.
Their specific choice is defined as a dummy variable :math:`D`.

.. math::
        S & = Y_1 - Y_0 - C\\
        D & = I\{S>0\}

**Unobservables**

The parameter :math:`V` denotes the collected unobservable variables :math:`U_1`, :math:`U_0` and :math:`U_C`.

.. math::
        V & = U_C -(U_1 - U_0)\\

The surplus can be rewritten as:

.. math::
        S & = X (\beta_1 - \beta_0) - Z \gamma - V\\



**Selecting Process**

.. todo::
    implement the self selection process

**Realized Outcome**

.. math::
        Y = D Y_1 + (1-D) Y_0\\

Model Specification
-------------------

The model is specified in an initialization file. The file includes the following sections:


**SIMULATION**

The *SIMULATION* part contains basic configurations that determine the simulation and output process.

=======     ======      ==================
Key         Value       Interpretation
=======     ======      ==================
agents      int         number of individuals
seed        int         seed for the specific simulation
source      str         specified name for the simulation output files
=======     ======      ==================

**TREATED & UNTREATED**

The *TREATED* and *UNTREATED* paragraph are similar regarding their structure. Both contain the parameters that determine the expected wage dependent on the treatment status. There can be added as many covariates as wished, but the number has to be the same for both cases.

=======     ======  =======  =========   ==================
Key         Value    Type    Fraction    Interpretation
=======     ======  =======  =========   ==================
coeff       float    ---      ---         intercept coefficient
coeff       float   string    float       coefficient of the first covariate
coeff       float   string    float       coefficient of the second covariate
 ...
=======     ======  =======  =========   ==================

**COST**

The *COST* section includes parameters related to the cost function.

=======     ======  =======  =========   ==================
Key         Value    Type    Fraction    Interpretation
=======     ======  =======  =========   ==================
coeff       float    ---      ---         intercept coefficient
coeff       float   string    float       coefficient of the first covariate
coeff       float   string    float       coefficient of the second covariate
 ...
=======     ======  =======  =========   ==================

.. Warning::

    - The first coefficient in the **TREATED**, **UNTREATED** and **COST** section is interpreted as an intercept.

    - You can add a desired number of different coefficients to all three sections. However it should be noted that the number of coefficients in the **TREATED** and **UNTREATED** sections has to be the same.

    - The **Type** column allows to set covariates to binary variables. For this purpose you just have to insert **binary** behind the coefficient value. The default value is **nonbinary**.

    - Note that if you want to create a binary in the **TREATED** and the **UNTREATED** section it is sufficent to implement the option in one of the sections. Further note that setting an intercept coefficient to **binary** will be ignored in the simulation process.

    - The **Fraction** column allows to set a specific rate for which the binary variable is one by adding a float value between 0 and 1. If no argument is inserted, the simulation process will define a random rate.

**DIST**

This Section determines the distributional characteristics of the unobservable variables.
The indices *0* and *1* denote the distributional information for the error terms of the untreated and treated outcomes :math:`(Y_0, Y_1)`, whereas *V* denotes the distributional characteristics of the collected unobservable variables.


======= ======      ==========================
Key     Value       Interpretation
======= ======      ==========================
coeff    float      :math:`\sigma_{0}`
coeff    float      :math:`\sigma_{01}`
coeff    float      :math:`\sigma_{0V}`
coeff    float      :math:`\sigma_{1}`
coeff    float      :math:`\sigma_{1V}`
coeff    float      :math:`\sigma_{V}`
======= ======      ==========================

Examples
--------
.. todo::
    - Ask Phillip why we can't use functions from the package by importing it via ``import grmpy``.

In the following chapter we explore the basic features of the ``grmpy`` package. Firstly you have to import the package and the related functions.
::


    from grmpy.simulate.simulate import simulate

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

    - Note that you have to insert the name of your initialization file as an input in the simulate function, if you generate a random initialization file the name is fixed to *test.grmpy.ini*.

    - Besides the ``.grmpy.txt`` the function is able to return a dataframe directly by setting ``data_frame = simulate('test.grmpy.ini')``




