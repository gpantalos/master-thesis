import tensorflow as tf
from tensorflow_probability import distributions as tfd

from modules.neural_network import BatchedFullyConnectedNN
from modules.prior_posterior import GaussianPriorDist
from modules.kernel import RBF_Kernel_TF
from bnn.abstract import RegressionModel

import time
import math

class BayesianNeuralNetworkSVGD(RegressionModel):

    def __init__(self, x_train, y_train, hidden_layer_sizes=(32, 32), activation='elu',
                 likelihood_std=0.1, learn_likelihood=True, prior_std=1.0, prior_weight=0.1,
                 likelihood_prior_mean=math.log(0.1), likelihood_prior_std=1.0,
                 num_particles=10, batch_size=10, bandwidth=0.01,
                 num_iter_fit=10000, lr=1e-3, normalize_data=True):

        self.prior_weight = prior_weight
        self.likelihood_std = likelihood_std
        self.learn_likelihood = learn_likelihood
        self.batch_size = batch_size
        self.num_iter_fit = num_iter_fit
        self.num_particles = num_particles

        # data handling
        self._process_train_data(x_train, y_train, normalize_data=normalize_data)

        # setup nn
        self.nn = BatchedFullyConnectedNN(num_particles, self.output_size, hidden_layer_sizes, activation)
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

        # setup particles & kernel
        nn_params = self.nn.get_variables_stacked_per_model()
        likelihood_params = tf.ones((self.num_particles, self.likelihood_param_size)) * likelihood_prior_mean
        self.particles = tf.Variable(tf.concat([nn_params, likelihood_params], axis=-1))
        self.kernel = RBF_Kernel_TF(bandwidth)

        # setup optimizer
        self.optim = tf.keras.optimizers.Adam(lr)

    def predict(self, x):
        # data handling
        x = self._handle_input_data(x, convert_to_tensor=True)
        x = self._normalize_data(x)

        # nn prediction
        nn_params, likelihood_std = self._split_into_nn_params_and_likelihood_std(self.particles)
        y_pred = self.nn.call_parametrized(x, nn_params)

        # form mixture of predictive distributions
        pred_dist = self._predictive_mixture(y_pred, likelihood_std)

        # unnormalize preds
        y_pred = self._unnormalize_preds(y_pred)
        pred_dist = self._unnormalize_predictive_dist(pred_dist)
        return y_pred, pred_dist

    @tf.function
    def step(self, x_batch, y_batch):
        # compute posterior score (gradient of log prob)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.particles)
            avg_log_likelihood = tf.reduce_mean(self._log_likelihood(x_batch, y_batch, self.particles))
            post_log_prob = avg_log_likelihood + self.prior_weight / self.num_train_samples * self.prior.log_prob(self.particles)
        score = tape.gradient(post_log_prob, self.particles)

        # compute kernel matrix and grads
        K_XX, K_grad = self._get_kernel_matrix_and_grad(self.particles)
        svgd_grads_stacked = tf.matmul(K_XX, score) - K_grad / self.num_particles

        # apply SVGD gradients
        self.optim.apply_gradients([(- svgd_grads_stacked, self.particles)])
        return - post_log_prob

    def _get_kernel_matrix_and_grad(self, stacked_particles):
        p2 = tf.identity(stacked_particles)
        with tf.GradientTape() as tape:
            tape.watch(stacked_particles)
            K_XX = self.kernel(stacked_particles, p2)
        K_grad = tape.gradient(K_XX, stacked_particles)
        return K_XX, K_grad

    def _log_likelihood(self, x, y, params):
        nn_params, likelihood_std = self._split_into_nn_params_and_likelihood_std(params)
        y_pred = self.nn.call_parametrized(x, nn_params)
        return tfd.Independent(tfd.Normal(y_pred, likelihood_std), reinterpreted_batch_ndims=1).log_prob(y)

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

if __name__ == '__main__':
    import numpy as np
    from matplotlib import pyplot as plt

    # generate data
    n_train = 20
    x_train = np.random.uniform(-4, 4, size=(n_train, 1))
    y_train = np.sin(x_train) + np.random.normal(scale=0.1, size=x_train.shape)

    n_val = 200
    x_val = np.random.uniform(-4, 4, size=(n_val, 1))
    y_val = np.sin(x_val) + np.random.normal(scale=0.1, size=x_val.shape)

    nn = BayesianNeuralNetworkSVGD(x_train, y_train, hidden_layer_sizes=(32, 32, 32, 32), activation='elu', learn_likelihood=True,
                                 num_iter_fit=100000, prior_weight=0.1, likelihood_std=0.1, bandwidth=100.0)

    x_plot = np.arange(-4, 4, 0.01)

    nn.plot_predictions(x_plot_range=(-8, 8))
    for i in range(10):
        n_iter_fit = 2000
        nn.fit(x_val=x_val, y_val=y_val, log_period=500, num_iter_fit=n_iter_fit)
        fig, ax = nn.plot_predictions(show=False, x_plot_range=(-8, 8))
        ax.scatter(x_val, y_val, alpha=0.2)
        ax.set_title('iter: %i'%int((i+1)*n_iter_fit))
        fig.show()




