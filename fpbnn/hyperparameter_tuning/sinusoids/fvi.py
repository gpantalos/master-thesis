from ray import tune

hyper_params = dict(
    activation=tune.choice(["elu", "tanh"]),
    learning_rate=tune.loguniform(1e-3, 1e-1),
    n_particles=tune.lograndint(8, 64, base=2),
    likelihood_prior_mean=tune.loguniform(1e-3, 1e0),
    coeff_prior=tune.loguniform(1e-2, 1e0),
    coeff_entropy=tune.loguniform(1e-2, 1e0),
    coeff_cross_entropy=tune.loguniform(1e-2, 1.0),
    weight_decay=tune.loguniform(1e-8, 1e-4),
    prior_noise_std=tune.loguniform(1e-5, 1e-1),
    ssge_eta=1.0,
)

from hyperparameter_tuning.main import tune_and_run

tune_and_run(__file__, hyper_params)
