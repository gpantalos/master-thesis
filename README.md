# Functional Priors for Bayesian Neural Networks

This is a repository for training Bayesian Neural Networks (BNNs) using the
data-generating process in function space.

## Dependencies

To install the minimal dependencies needed to use the algorithms, run in the
main directory of this repository

```commandline
pip install .
```

Alternatively, you can install the required packages using the following
command:

```commandline
pip install -r requirements.txt
```

## Quick start

The following code snippet presents the core functionality of this repo

```python
from fpbnn.envs.sinusoids import Sinusoids
from fpbnn.models import FVI

# define a data-generating process
data_generator = Sinusoids()

# sample dataset
train_data, test_data = data_generator.sample_train_test(10, 100)

# initialize model
nn = FVI(train_data=train_data,
         functional_prior=data_generator,
         experiment=data_generator.name)

# fit to training data
nn.fit(plot=True, show=True, test_data=test_data)
```







