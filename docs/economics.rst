Economics
=========

This section provides a general discussion of the generalized Roy model and selected issues in the econometrics of policy evaluation. The ``grmpy`` package implements a parametric normal version of the model, please see the next section for details.

Generalized Roy Model
*********************

The generalized Roy model (Roy, 1951; Heckman & Vytlacil, 2005) provides a coherent framework to explore the econometrics of policy evaluation. Its parametric version is characterized by the following set of equations.


.. math::
    \text{Potential Outcomes} &  \\
    Y_1 & = \mu_1(X) + U_1 \\
    Y_0 & = \mu_0(X) + U_0 \\
        & \\
    \text{Choice} &  \\
    D & = I[S  > 0 ] \\
    S & = E[Y_1 - Y_0 - C \mid \mathcal{I}] \\
    C & = \mu_C(Z) + U_C \\
    & \\
    \text{Observed Outcome} &  \\
    Y & = D Y_1 + (1 - D) Y_0

:math:`(Y_1, Y_0)` are objective outcomes associated with each potential treatment state :math:`D` and realized after the treatment decision. :math:`Y_1` refers to the outcome in the treated state and :math:`Y_0` in the untreated state. :math:`C` denotes the subjective cost of treatment participation. Any subjective benefits, e.g. job amenities, are included (as a negative contribution) in the subjective cost of treatment. Agents take up treatment D if they expect the objective benefit to outweigh the subjective cost. In that case, their subjective evaluation, i.e. the expected surplus from participation S, is positive. I denotes the agent’s information set at the time of the participation decision. The observed outcome Y is determined in a switching-regime fashion (Quandt, 1958, 1972). If agents take up treatment, then the observed outcome Y corresponds to the outcome in the presence of treatment Y1. Otherwise, Y0 is observed. The unobserved potential outcome is referred to as the counterfactual outcome. If costs are identically zero for all agents, there are no observed regressors, and (U1, U0) ∼ N (0, Σ), then the generalized Roy model corresponds to the original
Roy model (Roy, 1951).

From the perspective of the econometrician, (X, Z) are observable while (U1, U0, UC) are not. X are the observed determinants of potential outcomes (Y1, Y0), and Z are the observed determinants of the cost of treatment C. Potential outcomes and cost are decomposed into their means (µ1(X), µ0(X), µC(Z)) and their deviations from the mean (U1, U0, UC). (X, Z) might have common elements. Observables and unobservables jointly determine program participation D.

If their ex ante surplus S from participation is positive, then agents select into treatment. Yet, this does not require their expected objective returns to be positive as well. Subjective cost C might be negative such that agents which expect negative returns still participate. Moreover, in the case of imperfect information, an agent’s ex ante evaluation of treatment is potentially different from their ex post assessment.

The evaluation problem arises because either Y1 or Y0 is observed. Thus, the effect of treatment cannot be determined on an individual level. If the treatment choice D depends on the potential outcomes, then there is also a selection problem. If that is the case, then the treated and untreated differ not only in their treatment status but in other characteristics as well. A naive comparison of the treated and untreated leads to misleading conclusions. Jointly, the evaluation and selection problem are the two fundamental problems of causal inference (Holland, 1986).

Selected Issues
***************

We now highlight some selected issues in the econometrics of policy evaluation that can be fruitfully discussed within the framework of the model.

Agent Heterogeneity
-------------------

What gives rise to variation in choices and outcomes among, from the econometrician’s perspective, otherwise observationally identical agents? This is the central question in all econometric policy analyses (Browning et al., 1999; Heckman, 2001).

The individual benefit of treatment is defined as B = Y1 − Y0 = (µ1(X) − µ0(X)) + (U1 − U0). From the perspective of the econometrician, differences in benefits are the result of variation in observable X and unobservable characteristics (U1 − U0). However, (U1 − U0) might be (at least partly) included in the agent’s information set I and thus known to the agent at the time of the treatment decision.

As a result, unobservable treatment effect heterogeneity can be distinguished into private information and uncertainty. Private information is only known to the agent but not the econometrician; uncertainty refers to variability that is unpredictable by both.

The information available to the econometrician and the agent determines the set of valid estimation approaches for the evaluation of a policy. The concept of essential heterogeneity emphasizes this point (Heckman et al., 2006b).

Essential Heterogeneity
-----------------------

If agents select their treatment status based on benefits unobserved by the econometrician (selection on unobservables), then there is no unique effect of a treatment or a policy even after conditioning on observable characteristics. Average benefits are different from marginal benefits, and different policies select individuals at different margins. Conventional econometric methods that only account for selection on observables, like matching (Cochran and Rubin, 1973; Rosenbaum and Rubin, 1983; Heckman et al., 1998), are not able to identify any parameter of interest (Heckman and Vytlacil, 2005; Heckman et al., 2006b).

Objects of Interest
*******************

Treatment effect heterogeneity requires to be precise about the effect being discussed. There is no single effect of neither a policy nor a treatment. For each specific policy question, the object of interest must be carefully defined (Heckman and Vytlacil, 2005, 2007a,b). We present several potential objects of interest and discuss what question they are suited to answer. We start with the average effect parameters. However, these neglect possible effect heterogeneity. Therefore, we explore their distributional counterparts as well.

Treatment effect heterogeneity requires to be precise about the effect being discussed. There is no single effect of neither a policy nor a treatment. For each specific policy question, the object of interest must be carefully defined (Heckman and Vytlacil, 2005, 2007a,b). We present several potential objects of interest and discuss what question they are suited to answer. We start with the average effect parameters. However, these neglect possible effect heterogeneity. Therefore, we explore their distributional counterparts as well.

Conventional Average Treatment Effects
--------------------------------------

It is common to summarize the average benefits of treatment for different subsets of the population. In general, the focus is on the average effect in the whole population, the average treatment effect (AT E), or the average effect on the
treated (T T) or untreated (T UT).

AT E = E [Y1 − Y0]
T T = E [Y1 − Y0 | D = 1]
T UT = E [Y1 − Y0 | D = 0]

The relationship between these parameters depends on the assignment mechanism that matches agents to treatment. If agents select their treatment status based on their own benefits, then agents that take up treatment benefit more than those that do not and thus T T > T UT. If agents select their treatment status at random, then all parameters are equal. The policy relevance of the conventional treatment effect parameters is limited. They are only informative about extreme policy alternatives. The AT E is of interest to policy makers if they weigh the possibility of moving a full economy from a baseline to an alternative state or are able to assign agents to treatment at random. The T T is informative if the complete elimination of a program already in place is considered. Conversely, if the same program is examined for
compulsory participation, then the T UT is the policy relevant parameter. To ensure a tight link between the posed policy question and the parameter of interest, Heckman
and Vytlacil (2001b) propose the policy-relevant treatment effect (P RT E). They consider policies that do not change potential outcomes, but only affect individual choices. Thus, they account for voluntary program participation. Policy-Relevant Average Treatment Effects The P RT E captures the average change in outcomes per net person shifted by a change from a baseline state B to an alternative policy A. Let DB and DA denote the choice taken under the baseline and the alternative policy regime
respectively. Then, observed outcomes are determined as

YB = DBY1 + (1 − DB)Y0
YA = DAY1 + (1 − DA)Y0.

A policy change induces some agents to change their treatment status (DB 6= DA), while others are unaffected. More formally, the P RT E is then defined as

P RT E =
E [DA] − E [DB]
(E [YA] − E [YB]).

In our empirical illustration, in which we consider education policies, the lack of policy relevance of the conventional effect parameters is particularly evident. Rather than directly assigning individuals a certain level of education, policy makers can only indirectly affect schooling choices, e.g. by altering tuition cost through subsidies. The individuals drawn into treatment by such a policy will neither be a random sample of the whole population, nor the whole population of
the previously (un-)treated. That is why we estimate the policy-relevant effects of alternative education policies and contrast them with the conventional treatment effect parameters. We also show how the P RT E varies for alternative policy proposals as different agents are induced to change their treatment status.

Local Average Treatment Effect
------------------------------

The Local Average Treatment Effect (LATE) was introduced by \citet{Imbens.1994}. They show that instrumental variable estimator identify LATE, which measures the mean gross return to treatment for individuals induced into treatment by a change in an instrument.\\\newline
%
Unfortunately, the people induced to go into state 1 $(D=1)$ by a change in any particular instrument need not to be the same as the people induced to to go to state 1 by policy changes other than those corresponding exactly to the variation in the instrument. A desired policy effect may bot be directly correspond to the variation in the IV. Moreover, if there is a vector of instruments that generates choice and the components of the vector are intercorrelated, IV estimates using the components of $Z$ as the instruments, one at a time, do not, in general, identify the policy effect corresponding to varying that instruments, keeping all other instruments fixed, the ceteris paribus effect of the change in the instrument. \citet{Heckman.2010d} develops this argument in detail.

The average effect of a policy and the average effect of a treatment are linked by the marginal treatment effect (MT E). The MT E was introduced into the literature by Bj¨orklund and Moffitt (1987) and extended in Heckman and Vytlacil (2001a, 2005, 2007b).

Marginal Treatment Effect
--------------------------

The MT E is the treatment effect parameter that conditions on the unobserved desire to select into treatment. Let V = E[UC − (U1 − U0) | I ] summarize the expectations about all unobservables determining treatment choice and let US = FV (V ). Then, the MT E is defined as

MT E(x, uS) = E [ Y1 − Y0 | X = x, US = uS] .

The MT E is the average benefit for persons with observable characteristics X = x and unobservables US = uS. By construction, US denotes the different quantiles of V . So, when varying US but keeping X fixed, then the MT E shows how the average benefit varies along the distribution of V . For uS evaluation points close to zero, the MT E is the average effect of treatment for individuals with a value of V that makes them most likely to participate. The opposite is true for high values of uS.
The MT E provides the underlying structure for all average effect parameters previously discussed. These can be derived as weighted averages of the MT E (Heckman and Vytlacil, 2005).

Parameter j, ∆j (x), can be written as
∆j (x) = Z 1

MT E(x, uS)hj (x, uS) duS,

where the weights hj (x, uS) are specific to parameter j, integrate to one, and can be constructed from data.4 All parameters are identical only in the absence of essential heterogeneity. Then, the MT E(x, uS) is constant across the whole distribution of V as agents do not select their treatment status based on their unobservable benefits.

So far, we have only discussed average effect parameters. However, these conceal possible treatment effect heterogeneity, which provides important information about a treatment. Hence, we now present their distributional counterparts (Aakvik et al., 2005).

Distribution of Potential Outcomes
----------------------------------

Several interesting aspects of policies cannot be evaluated without knowing the joint distribution of potential outcomes (see Abbring and Heckman (2007) and Heckman et al. (1997)). The joint distribution of (Y1, Y0) allows to calculate the whole distribution of benefits. Based on it, the average treatment and policy effects can be
constructed just as the median and all other quantiles. In addition, the portion of people that benefit from treatment can be calculated for the overall population Pr(Y1 − Y0 > 0) or among any subgroup of particular interest to policy makers Pr(Y1 −Y0 > 0 | X).5 This is important as a treatment which is beneficial for agents on average can still be harmful for some. The absence of an average effect might be the result of part of the population having a positive effect, which is just offset by a negative effect on the rest of the population. This kind of treatment effect heterogeneity is informative as it provides the starting point for an adaptive research strategy that tries to understand the driving force behind these differences (Horwitz et al., 1996, 1997).
