from ray import tune

hyper_params = dict(
    activation=tune.choice(['elu', 'relu', 'tanh']),
    learning_rate=tune.loguniform(1e-5, 1e-1),
    coeff_prior=tune.loguniform(1e-4, 1e0),
    weight_decay=tune.loguniform(1e-8, 1e-1),
    n_particles=tune.lograndint(10, 100),
    nn_prior_std=tune.loguniform(1e-3, 1e1),
    likelihood_prior_mean=tune.uniform(1e-3, 1e1),
    likelihood_prior_std=tune.loguniform(1e-3, 1e1),
    noise_std=1e-2,
)

from hyperparameter_tuning.main import tune_and_run

tune_and_run(__file__, hyper_params)
