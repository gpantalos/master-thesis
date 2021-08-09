import tensorflow as tf
import tensorflow_probability as tfp

from fpbnn.modules.abstract_regression import BayesianRegressionModel
from fpbnn.modules.ssge import SSGE

tfd = tfp.distributions
tfk = tfp.math.psd_kernels


class FunctionalRegressionModel(BayesianRegressionModel):
    """Functional Bayesian Neural Network base class"""

    def __init__(
            self,
            functional_prior,
            ssge_bandwidth=3.0,
            ssge_n_eigen=6,
            ssge_eta=0.01,
            coeff_entropy=0.1,
            coeff_cross_entropy=0.1,
            prior_noise_std=1e-3,
            buffer_size=10,
            **kwargs):
        super().__init__(**kwargs)
        self.buffer_size = buffer_size
        self.coeff_cross_entropy = coeff_cross_entropy
        self.coeff_entropy = coeff_entropy

        # setup tensor kernel
        self.kernel = tfk.ExponentiatedQuadratic(length_scale=self.bandwidth)

        # setup gradient estimator
        self.ssge = SSGE(eta=ssge_eta, bandwidth=ssge_bandwidth, n_eigen=ssge_n_eigen)

        # setup samples
        x_prior, y_prior = functional_prior.sample(self.buffer_size, self.batch_size, self.n_particles)
        x_prior, y_prior = self.normalize(x_prior, y_prior)

        # repeat buffer
        x_prior = tf.repeat(x_prior, self.n_iter // self.buffer_size + 1, axis=0)
        y_prior = tf.repeat(y_prior, self.n_iter // self.buffer_size + 1, axis=0)

        # add zero-mean noise
        y_prior += tf.random.normal(tf.shape(y_prior), 0.0, prior_noise_std)

        self.x_prior = iter(x_prior)
        self.y_prior = iter(y_prior)

    def __call__(self, x_batch, y_batch):
        pass

    def predict(self, x):
        pass
