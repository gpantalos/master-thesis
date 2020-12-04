import tensorflow as tf
from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp

from modules.neural_network import BatchedFullyConnectedNN
from modules.prior_posterior import GaussianPosteriorDist, GaussianPriorDist
from bnn.abstract import RegressionModel

import time
import math

class BayesianNeuralNetworkVI(RegressionModel):

    def __init__(self, x_train, y_train, hidden_layer_sizes=(32, 32), activation='elu',
                 likelihood_std=0.1, learn_likelihood=True, prior_std=1.0, prior_weight=0.1,
                 likelihood_prior_mean=math.log(0.1), likelihood_prior_std=1.0,
                 batch_size_vi=10, batch_size=10,
                 num_iter_fit=10000, lr=1e-3, normalize_data=True):

        self.prior_weight = prior_weight
        self.likelihood_std = likelihood_std
        self.learn_likelihood = learn_likelihood
        self.batch_size = batch_size
        self.batch_size_vi = batch_size_vi
        self.num_iter_fit = num_iter_fit

        # data handling
        self._process_train_data(x_train, y_train, normalize_data=normalize_data)

        # setup nn
        self.nn = BatchedFullyConnectedNN(self.batch_size_vi, self.output_size, hidden_layer_sizes, activation)
        self.nn.build((None, self.input_size))

        # setup prior
        self.nn_param_size = self.nn.get_variables_stacked_per_model().shape[-1]
        if self.learn_likelihood:
            self.likelihood_param_size = self.output_size
        else:
            self.likelihood_param_size = 0
        self.prior = GaussianPriorDist(self.nn_param_size, nn_prior_std=prior_std,
                                       likelihood_param_size=self.likelihood_param_size,
                                       likelihood_prior_mean=likelihood_prior_mean,
                                       likelihood_prior_std=likelihood_prior_std)

        # setup posterior
        self.posterior = GaussianPosteriorDist(stacked_nn_init_params=self.nn.get_variables_stacked_per_model(),
                                               likelihood_param_size=self.likelihood_param_size)

        # setup optimizern
        self.optim = tf.keras.optimizers.Adam(lr)

    def predict(self, x, num_posterior_samples=20):
        # data handling
        x = self._handle_input_data(x, convert_to_tensor=True)
        x = self._normalize_data(x)

        # nn prediction
        y_pred_batches = []
        likelihood_std_batches = []
        for _ in range(num_posterior_samples // self.batch_size_vi):
            sampled_params = self.posterior.sample((self.batch_size_vi,))
            nn_params, likelihood_std = self._split_into_nn_params_and_likelihood_std(sampled_params)
            likelihood_std_batches.append(likelihood_std)
            y_pred_batches.append(self.nn.call_parametrized(x, nn_params))
        y_pred = tf.concat(y_pred_batches, axis=0)
        likelihood_std = tf.concat(likelihood_std_batches, axis=0)

        pred_dist = self._predictive_mixture(y_pred, likelihood_std)

        # unnormalize preds
        y_pred = self._unnormalize_preds(y_pred)
        pred_dist = self._unnormalize_predictive_dist(pred_dist)
        return y_pred, pred_dist

    @tf.function
    def step(self, x_batch, y_batch):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.posterior.trainable_variables)
            sampled_params = self.posterior.sample((self.batch_size_vi,))
            avg_log_likelihood = tf.reduce_mean(self._log_likelihood(x_batch, y_batch, sampled_params))
            kl_term = tf.reduce_mean(self.posterior.log_prob(sampled_params) - self.prior.log_prob(sampled_params))
            elbo = - avg_log_likelihood + self.prior_weight / self.num_train_samples * kl_term
        grads = tape.gradient(elbo, self.posterior.trainable_variables)
        self.optim.apply_gradients(zip(grads, self.posterior.trainable_variables))
        return elbo

    def _split_into_nn_params_and_likelihood_std(self, params):
        tf.assert_equal(len(params.shape), 2)
        tf.assert_equal(params.shape[-1], self.nn_param_size + self.likelihood_param_size)
        n_particles = params.shape[0]

        # nn params
        nn_params = params[:, :self.nn_param_size]

        # likelihood params
        if self.likelihood_param_size > 0:
            likelihood_std = tf.exp(params[:, -self.likelihood_param_size:])
        else:
            likelihood_std = tf.ones((n_particles, self.output_size)) * self.likelihood_std

        tf.assert_equal(likelihood_std.shape, (n_particles, self.output_size))
        return nn_params, likelihood_std

    def _log_likelihood(self, x, y, params):
        nn_params, likelihood_std = self._split_into_nn_params_and_likelihood_std(params)
        y_pred = self.nn.call_parametrized(x, nn_params)
        return tfd.Independent(tfd.Normal(y_pred, likelihood_std), reinterpreted_batch_ndims=1).log_prob(y)



if __name__ == '__main__':
    import numpy as np
    from matplotlib import pyplot as plt

    # generate data
    n_train = 50
    x_train = np.random.uniform(-4, 4, size=(n_train, 1))
    y_train = np.sin(x_train) + np.random.normal(scale=0.1, size=x_train.shape)

    n_val = 200
    x_val = np.random.uniform(-4, 4, size=(n_val, 1))
    y_val = np.sin(x_val) + np.random.normal(scale=0.1, size=x_val.shape)

    nn = BayesianNeuralNetworkVI(x_train, y_train, hidden_layer_sizes=(32, 32), activation='elu', learn_likelihood=True,
                                 num_iter_fit=100000, prior_weight=0.1, likelihood_std=0.1,)

    x_plot = np.arange(-4, 4, 0.01)

    nn.plot_predictions(x_plot_range=(-8, 8))
    for i in range(10):
        n_iter_fit = 10000
        nn.fit(x_val=x_val, y_val=y_val, log_period=10000, num_iter_fit=n_iter_fit)
        fig, ax = nn.plot_predictions(show=False, x_plot_range=(-8, 8))
        ax.set_title('iter: %i'%int((i+1)*n_iter_fit))
        fig.show()




