Tutorial
========

We now illustrate the basic capabilities of the ``grmpy`` package. We start with the model parametrization and specification before turning to some example use cases.


Parametric Assumptions
----------------------


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
