Reliability
===========

The following section illustrates the power of the estimation strategy behind the ``grmpy`` package when facing agent heterogeneity.
For this purpose we present the results from two straightforward Monte Carlo exercises.

Both exercises run a certain amount of steps.  Each step increases the rate of correlation between the unobservables. Translated in the Roy model framework this is equivalent to an increase in the correlation between the unobservable variable `U_1`and the `V`, the variable in which all unobservables that determine the selection into treatment are collected.
The data is constructed and simulated by the package itself so that the dataset follows the structure of the generalized roy model. Additionally the specifications of the inititalization file ensure that the true average effect of treatment (ATE) is fixed to 0.5 independent of the correlation structure.
In the first exercise we estimate the ATE via a simple OLS approach whereas the second one makes use of the ``grmpy`` estimation process.

.. figure:: ../docs/figures/fig_ols_average_effect_estimation.png
   :align: center

.. figure:: ../docs/figures/fig_grmpy_average_effect_estimation.png
    :align: center

Nevertheless both estimation strategies lead to very different results.
The ``grmpy`` estimator returns an rate of effect that is really close to the true value even if the unobservables are almost perfectly correlated.
However the OLS estimator underestimates the effect. This is really obvious from the second figure. The stronger the correlation between the unobservables the smaller the estimated ATE.
This stresses the capabilities of the Local Average
