Reliability
===========

The following section illustrates the power of the estimation strategy behind the ``grmpy`` package when facing agent heterogeneity.
For this purpose we present the results of two straightforward Monte Carlo exercises.

Both exercises run a certain amount of steps. Each step increases the rate of correlation between the unobservables. Translated in the Roy model framework this is equivalent to an increase in the correlation between the unobservable variable :math:`U_1` and the :math:`V`, the variable in which all unobservables that determine the selection into treatment are collected.
The data is constructed and simulated by the package itself so that the dataset follows the structure of the generalized Roy model. Additionally the specifications of the inititalization file ensure that the true average effect of treatment (ATE) is fixed to 0.5 independent of the correlation structure.
In the first exercise we estimate the ATE via a simple OLS approach.


.. figure:: ../docs/figures/fig_ols_average_effect_estimation.png
    :align: center


As can be seen from the figure, the OLS estimator underestimates the effect significantly as soon as the correlation differs from zero. The stronger the correlation between the unobservable variables the stronger the downwards bias.

.. figure:: ../docs/figures/fig_grmpy_average_effect_estimation.png
    :align: center


The second figure shows the estimated ATE from the ``grmpy`` estimation process that is implemented in the second exercise.
Conversely to the OLS results the estimate of the average effect is really close to the true value even if the unobservables are almost perfectly correlated.
