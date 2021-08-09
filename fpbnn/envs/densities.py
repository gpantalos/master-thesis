"""Data generator for a function from R^m to R^n made from
the concatenation of joint gaussian densities."""

import tensorflow as tf
from tensorflow_probability import distributions as tfd

from .regression import RegressionDataset


class Densities(RegressionDataset):
    def __init__(
            self,
            input_dim=1,
            output_dim=1,
            mean_range=(-0.5, 0.5),
            variance_range=(0.25, 0.5),
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.variance_range = variance_range
        self.mean_range = mean_range
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dtype = tf.float32
        self.random_state = tf.random.Generator.from_seed(self.seed)

    def handle(self, x):
        if tf.rank(x) == 1:
            x = x[..., None]
        return tf.cast(x, self.dtype)

    def sample_function(self, mixtures=1):
        mean = self.random_state.uniform([self.output_dim, mixtures, self.input_dim], *self.mean_range)
        variance = self.random_state.uniform([self.output_dim, mixtures, self.input_dim], *self.variance_range)
        probs = [1 / mixtures for _ in range(mixtures)]
        mixture_distribution = tfd.Categorical(probs=probs)

        def g(x):
            """
            x: (b, m),
            y: (b, n)
            """
            x = self.handle(x)
            y = []
            for i in range(self.output_dim):
                dist = tfd.Independent(tfd.Normal(mean[i], variance[i]), 1)
                mixture = tfd.MixtureSameFamily(mixture_distribution, dist)
                y.append(mixture.log_prob(x))
            return tf.exp(tf.stack(y, -1))

        return g
