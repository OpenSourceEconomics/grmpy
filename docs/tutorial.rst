Tutorial
========

We now illustrate the basic capabilities of the ``grmpy`` package. We start with the model parametrization and specification before turning to some example use cases.


Parametric Assumptions
----------------------

The package focuses on the linear form of the generalized roy model for reasons of simplicity. The model is characterized by the following equations:

**Outcome**

`m:math:`Y_1` and :math:`Y_0` represent the ex post outcome for each individual depending on treatment status.

.. math::
    Y_1 & = X \beta_1 + U_1 \\
    Y_2 & = X \beta_0 + U_0 \\
    :label: output

**Costs**

The cost function illustrates the costs of an individual for selecting in the treatment group.

.. math::
    C & = Z \gamma + U_C
    :label: cost

**Choice Parameters**

Individuals have an incentive to select themselves in the treatment group if their associated surplus is positive. The surplus :math:`S` is defined as the  difference between their outputs :math:`Y_1` and :math:`Y_0` minus the subjective costs :math:`C` for selecting into treatment.
Their specific choice is defined as a dummy variable :math:`D`.

.. math::
    S & = Y_1 - Y_0 - C\\
    D & = I{S>0}
    :label: surplus

**Unobservables**

The parameter :math:`V` denotes the collected unobservable variables :math:`U_1`, :math:`U_0` and :math:`U_C`.

.. math::
    V & = U_C -(U_1 - U_0)

The surplus can be rewritten as:

.. math::
    S & = X (\beta_1 - \beta_0) - Z \gamma - V



**Selecting Process**

.. to do:: implement the self selection process

**Realized Outcome**

.. math::
        Y = D Y_1 + (1-D) Y_0

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

=======     ======      ==================
Key         Value       Interpretation
=======     ======      ==================
coeff       float       intercept coefficient
coeff       float       coefficient of the first covariate
coeff       float       coefficient of the second covariate
 ...
=======     ======      ==================

**COST**

The *COST* section includes parameters related to the cost function.

=======     ======      ==================
Key         Value       Interpretation
=======     ======      ==================
coeff       float       intercept coefficient
coeff       float       coefficient of the first covariate
coeff       float       coefficient of the second covariate
 ...
=======     ======      ==================

.. Warning::

    - The first coefficient in the *TREATED*, *UNTREATED* and *COST* section is interpreted as an intercept.

    - You can add a desired number of different coefficients to all three sections. It should be noted however that the number of coefficients in the *TREATED* and *UNTREATED* sections has to be the same.


**DIST**

This Section determines the distributional characteristics of the unobservable variables.
The indices *0* and *1* denote the distributional information for the error terms of the untreated and treated outcomes :math:`(Y_0, Y_1)`, whereas *C* denotes the distributional characteristics related to the cost function error terms.


======= ======      ==========================
Key     Value       Interpretation
======= ======      ==========================
coeff    float      :math:`\sigma_{0}`
coeff    float      :math:`\sigma_{01}`
coeff    float      :math:`\sigma_{0C}`
coeff    float      :math:`\sigma_{1}`
coeff    float      :math:`\sigma_{1C}`
coeff    float      :math:`\sigma_{C}`
======= ======      ==========================
