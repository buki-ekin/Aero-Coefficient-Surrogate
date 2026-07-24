# Scientific Validation

## Claim Being Tested

Aero Coefficient Surrogate is intended to reproduce flow5 aerodynamic coefficients for NACA
4-digit geometries inside the sampled design and operating-condition domain.
It is not intended to establish aerodynamic truth.

## Verification Versus Validation

Software verification asks whether the importer, split logic, metrics, model
serialization, browser serialization, workflow, and interfaces behave as
implemented. This is addressed with automated tests and continuous
integration.

Model validation asks whether the surrogate reproduces reference values for
its intended use. This is addressed by withholding complete NACA geometries
and comparing their predictions with flow5.

No comparison with wind-tunnel measurements or high-fidelity CFD is made.

## Why Rows Are Not Split Randomly

Every airfoil appears at many angles of attack and three Reynolds numbers. A
row-random split would place the same geometry in both training and test data,
making the task easier than the intended use on a new geometry. Grouped
splitting keeps all rows of one NACA airfoil together.

## Two Complementary Assessments

The seed-42 holdout provides one concrete and reproducible example with five
airfoils and 302 rows. Repeated grouped validation provides 15 evaluations per
model using three repetitions of five folds.

Repeated folds compare:

- Random Forest with 300 trees and `min_samples_leaf=2`
- ordinary linear regression
- a mean-target baseline

The Random Forest achieves the lowest mean RMSE for CL, CD, and CM. Drag has
the highest relative fold sensitivity.

## Interpreting The Intervals

The reported 95% intervals are normal-approximation confidence intervals for
the mean across grouped folds. They quantify dependence on held-out geometry
selection. They do not include:

- uncertainty in flow5 itself
- measurement uncertainty
- model-form uncertainty outside the candidate models
- uncertainty caused by unobserved operating regimes

## Known Error Concentrations

For the seed-42 holdout, NACA 2424 produces the largest CL and CM RMSE. Errors
are generally larger at Reynolds number 500,000. Residual plots show that
aggregate metrics hide structure across angle of attack.

## Deployment Separation

The validation model is trained without the held-out airfoils and is used only
for evaluation. After evaluation, a separate deployment model is trained on
all 1,618 converged rows. Deployment performance must not be reported using
training error.

## Computational Cost And Sustainability

The recorded seed-42 run completed the fixed holdout, 15 grouped fold
evaluations for three model classes, final deployment training, reporting, and
figure generation in 5.70 seconds on the machine described in `meta.json`.
This makes rerunning the Python evidence inexpensive compared with regenerating
the external flow5 source data.

No direct energy measurement was taken, so the project does not translate
runtime into carbon or energy claims. A numerical convergence study is also
outside the surrogate stage: convergence belongs to the flow5 reference
calculations and is represented here by retaining only converged polar points
and preserving the source solver files and logs.
