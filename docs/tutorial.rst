Tutorial
========

We now illustrate the basic capabilities of the ``grmpy`` package. We start with the model parametrization and specification before turning to some example use cases.


Parametric Assumptions
----------------------


Model Specification
-------------------

The model is specified in an initialization file. The specific file includes the following sections:


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

*Note:* The first coefficient is interpreted as an intercept.

=======     ======      ==================
Key         Value       Interpretation
=======     ======      ==================
coeff       float       intercept coefficient
coeff       float       coefficient of the first covariate
coeff       float       coefficient of the second covariate
 ...
=======     ======      ==================

**COST**

*COST* includes parameters related to the cost function.

*Note:* The first coefficient is interpreted as an intercept.

=======     ======      ==================
Key         Value       Interpretation
=======     ======      ==================
coeff       float       intercept coefficient
coeff       float       coefficient of the first covariate
coeff       float       coefficient of the second covariate
 ...
=======     ======      ==================

**DIST**

This Section determines the distributional characteristics of the unobservable variables.
The indices *0* and *1* denote the error for the untreated and treated observations whereas *C* denotes the error term for the costs.


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




