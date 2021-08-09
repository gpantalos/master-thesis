import tensorflow as tf
from tensorflow_probability import distributions as tfd


class DensitiesEnv:

    def __init__(self, mean_range=(-2, 2), variance_range=(0.25, 0.25),
                 x_range=(-4, 4), noise_std=0.01):
        self.noise_std = noise_std
        self.u = tfd.Uniform(*mean_range)
        self.v = tfd.Uniform(*variance_range)
        self.p = tfd.Uniform(*x_range)
        self.input_dim = 1
        self.output_dim = 1
        self.dtype = tf.float32

    def handle(self, x):
        if tf.rank(x) == 1:
            x = x[..., None]
        return tf.cast(x, self.dtype)

    def sample_function(self, mixtures=1):
        mean = self.u.sample([self.output_dim, mixtures, self.input_dim])
        variance = self.v.sample([self.output_dim, mixtures, self.input_dim])
        probs = [1 / mixtures for _ in range(mixtures)]
        mixture_distribution = tfd.Categorical(probs=probs)

        def g(x):
            """
            x: (b, m),
            y: (b, n)
            """
            x = self.handle(x)
            y = []
            for j in range(self.output_dim):
                dist = tfd.Independent(tfd.Normal(mean[j], variance[j]), 1)
                mixture = tfd.MixtureSameFamily(mixture_distribution, dist)
                y.append(mixture.log_prob(x))
            return tf.exp(tf.stack(y, -1))

        return g

    def generate_meta_train_data(self, n_tasks, n_samples):
        meta_train_tuples = []
        for i in range(n_tasks):
            f = self.sample_function()
            x = self.p.sample([n_samples, 1])
            y = f(x) + self.noise_std * tf.random.normal(f(x).shape)
            meta_train_tuples.append((x, y))
        return meta_train_tuples

    def generate_meta_test_data(self, n_tasks, n_samples_context, n_samples_test):
        assert n_samples_test > 0
        meta_test_tuples = []
        for i in range(n_tasks):
            f = self.sample_function()
            x = self.p.sample([n_samples_context + n_samples_test, 1])
            y = f(x) + self.noise_std * tf.random.normal(f(x).shape)
            meta_test_tuples.append(
                (x[:n_samples_context], y[:n_samples_context], x[n_samples_context:], y[n_samples_context:]))

        return meta_test_tuples
