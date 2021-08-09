import tensorflow as tf

from fpbnn.modules.abstract_regression import BayesianRegressionModel
from fpbnn.modules.prior_posterior import GaussianPosterior


class VI(BayesianRegressionModel):
    """Bayesian Neural Network trained using Variational Inference: https://arxiv.org/abs/1505.05424"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'vi'
        self.posterior = GaussianPosterior(self.nn_params, self.likelihood_param_size)

    def predict(self, x):
        x = self._broadcast_and_cast_dtype(x)
        x = self.normalize(x)
        nn_params, likelihood_std = self.split_params(self.posterior.sample([self.n_particles]))
        output_pred = self.nn.call_parametrized(x, nn_params)
        output_dist = self._predictive_mixture(output_pred, likelihood_std)
        output_pred = self._unnormalize_preds(output_pred)
        output_dist = self._unnormalize_predictive_dist(output_dist)
        return output_pred, output_dist

    @tf.function
    def __call__(self, x_batch, y_batch):
        with tf.GradientTape() as tape:
            tape.watch(self.posterior.trainable_variables)
            samples_posterior = self.posterior.sample([self.n_particles])
            nn_params, likelihood_std = self.split_params(samples_posterior)
            y_pred = self.nn.call_parametrized(x_batch, nn_params)
            likelihood = self.ll(y_batch, y_pred, likelihood_std)
            elbo = -likelihood

            # compute kl
            log_q = self.posterior.log_prob(samples_posterior)
            log_p = self.prior.log_prob(samples_posterior)
            kl_divergence = tf.reduce_mean(log_q - log_p)
            elbo += self.coeff_prior * kl_divergence

        # compute gradient of elbo wrt posterior parameters
        gradients = tape.gradient(elbo, self.posterior.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.posterior.trainable_variables))


if __name__ == '__main__':
    tf.random.set_seed(1234)
    from tensorflow_probability import distributions as tfd

    u = tfd.Uniform(-1, 1)

    input_dim = 1

    experiment = 'mujoco'
    if experiment == 'densities':
        from data import Densities

        functional_prior = Densities(input_dim, 1, (-0.5, 0.5), (0.25, 0.25))

        # hparams
        hparams = {
            'activation': 'elu',
            'n_particles': 10,
            'learning_rate': 0.002,
            'coeff_prior': 0.01,
            'likelihood_prior_mean': 0.1,
        }
    elif experiment == 'sinusoids':
        from data import Sinusoids

        functional_prior = Sinusoids(
            # uniform variables, format: (low, high)
            amplitude_range=(0, 2),
            period_range=(1, 1),
            # normal variables, format: (mean, std)
            slope=(2, 1e-6),
            x_shift=(0.0, 1e-6),
            y_shift=(0.0, 1e-6),
        )
        # hparams
        hparams = {
            'activation': 'elu',
            'n_particles': 20,
            'learning_rate': 0.01,
            'coeff_prior': 0.01,
            'likelihood_prior_mean': 0.01,
        }
    elif experiment == 'mujoco':
        from data import InvertedDoublePendulum

        functional_prior = InvertedDoublePendulum()
        hparams = {
            'activation': 'elu',
            'n_particles': 40,
            'learning_rate': 0.01,
            'bandwidth': 0.01,
            'likelihood_prior_mean': 0.03,
        }
        input_dim = functional_prior.input_dim
    else:
        raise NotImplemented

    x_train = u.sample([1e3, input_dim])
    x_test = u.sample([5e3, input_dim])
    x_val = u.sample([4e3, input_dim])

    # sample one function from prior
    f = functional_prior.sample_function()

    # evaluate it
    from time import time

    t0 = time()
    y_train = f(x_train)
    t1 = time()
    print(f'sampled train data in {t1 - t0:.3e}s.')
    y_test = f(x_test)
    t2 = time()
    print(f'sampled test data in {t2 - t1:.3e}s.')
    y_val = f(x_val)
    t3 = time()
    print(f'sampled val data in {t3 - t2:.3e}s.')

    params = {
        'train_data': (x_train, y_train),
        'experiment': experiment,
        'home_dir': '../',
        'verbose': 1,
        'weight_decay': 0.0,
        'batch_size': 8,
    }

    # init
    nn = VI(**params, **hparams)

    # train
    nn.fit(
        val_data=(x_val, y_val),
        test_data=(x_test, y_test),
        n_plots=10,
        show=True,
        plot=True,
    )
