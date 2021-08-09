from ray import tune

hyper_params = dict(
    activation=tune.choice(["elu", "tanh", "relu", "selu", "gelu"]),
    learning_rate=tune.loguniform(1e-5, 1e-1),
    n_particles=tune.lograndint(2, 100, base=2),
    bandwidth=tune.loguniform(1e-4, 1e1),
    likelihood_prior_mean=tune.loguniform(1e-3, 1e1),
    weight_decay=tune.loguniform(1e-8, 1e-5),
)

from hyperparameter_tuning.main import tune_and_run

tune_and_run(__file__, hyper_params)
