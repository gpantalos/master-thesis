# Models

The models are implemented in this section. Included are the
weight-space ([VI](vi.py), [SVGD](svgd.py)) and function-space ([FVI](fvi.py)
, [FSVGD](fsvgd.py)) models.

## Abstract Methods

The models must implement a `fit(x_train, y_train)` and a `predict(x_test)`
method.

## Score Estimation

Score estimation for the functional models is done using the Spectral Stein
Gradient Estimator (SSGE)
