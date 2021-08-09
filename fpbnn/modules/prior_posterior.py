import math

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class GaussianPosterior(tf.Module):
    def __init__(self, stacked_nn_init_params, likelihood_param_size):
        super().__init__()

        # mean & std for nn params
        nn_param_size = stacked_nn_init_params.shape[-1]
        nn_params_mean = tf.zeros(nn_param_size)
        nn_params_std = tfp.stats.stddev(stacked_nn_init_params, sample_axis=0) + 1.0 / nn_param_size

        # mean & std for likelihood params
        lk_mean = tf.zeros(likelihood_param_size)
        lk_std = tf.ones(likelihood_param_size) / likelihood_param_size

        self.mean = tf.Variable(tf.concat([nn_params_mean, lk_mean], axis=0))
        self.log_std = tf.Variable(tf.math.log(tf.concat([nn_params_std, lk_std], axis=0)))

    @property
    def stddev(self):
        return tf.exp(self.log_std)  # do not use softplus

    @property
    def dist(self):
        return tfp.distributions.Independent(tfp.distributions.Normal(self.mean, self.stddev), 1)

    def sample(self, size):
        return self.dist.sample(size)

    def log_prob(self, param_values):
        return self.dist.log_prob(param_values)


class GaussianPrior(tf.Module):
    def __init__(self, nn_param_size, likelihood_param_size=0, nn_prior_std=1.0,
                 likelihood_prior_mean=math.log(0.05), likelihood_prior_std=0.01):
        super().__init__()

        nn_prior_mean = tf.zeros(nn_param_size)
        nn_prior_std = tf.ones(nn_param_size) * nn_prior_std

        # mean and std of the Normal distribution over the log_std of the likelihood
        likelihood_prior_mean = tf.ones(likelihood_param_size) * likelihood_prior_mean
        likelihood_prior_std = tf.ones(likelihood_param_size) * likelihood_prior_std

        prior_mean = tf.concat([nn_prior_mean, likelihood_prior_mean], axis=0)
        prior_std = tf.concat([nn_prior_std, likelihood_prior_std], axis=0)

        self.dist = tfd.Independent(tfd.Normal(prior_mean, prior_std), reinterpreted_batch_ndims=1)

    def sample(self, size):
        return self.dist.sample(size)

    def log_prob(self, param_values):
        return self.dist.log_prob(param_values)
