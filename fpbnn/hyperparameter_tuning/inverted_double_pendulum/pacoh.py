from ray import tune

hyper_params = dict(
    lr=tune.loguniform(1e-3, 1e-2),
    activation=tune.choice(['relu', 'elu', 'tanh']),
    likelihood_std=tune.loguniform(0.01, 1.0),
    meta_batch_size=4,
    num_iter_meta_train=30000,
    num_iter_meta_test=10000,
    n_samples_per_prior=10,
    num_hyper_posterior_particles=3,
    num_posterior_particles=5,
    prior_weight=tune.loguniform(0.01, 1.0),
    hyper_prior_weight=tune.loguniform(1e-5, 1e-3),
    hyper_prior_nn_std=tune.uniform(0.1, 1.0),
    hyper_prior_log_var_mean=tune.uniform(-4, -2),
    hyper_prior_likelihood_log_var_mean_mean=tune.uniform(-9, -7),
    hyper_prior_likelihood_log_var_log_var_mean=tune.uniform(-5, -3),
    bandwidth=tune.loguniform(0.001, 100),
)

from hyperparameter_tuning.main import tune_and_run

tune_and_run(__file__, hyper_params)
