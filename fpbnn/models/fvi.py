import tensorflow as tf
import tensorflow_probability as tfp

from fpbnn.modules.functional_regression import FunctionalRegressionModel
from fpbnn.modules.prior_posterior import GaussianPrior

tfd = tfp.distributions


class FVI(FunctionalRegressionModel):
    """Functional Variational Inference: https://arxiv.org/abs/1903.05779"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'fvi'
        self.prior = GaussianPrior(
            nn_param_size=0,
            nn_prior_std=0,
            likelihood_param_size=self.likelihood_param_size,
            likelihood_prior_mean=self.likelihood_prior_mean,
            likelihood_prior_std=self.likelihood_prior_std
        )

    def predict(self, x):
        # data handling
        x = self._broadcast_and_cast_dtype(x)
        x = self.normalize(x)

        # nn prediction
        sampled_params = self.posterior.sample([self.n_particles])
        nn_params, likelihood_std = self.split_params(sampled_params)
        y_pred = self.nn.call_parametrized(x, nn_params)
        pred_dist = self._predictive_mixture(y_pred, likelihood_std)

        # unnormalize preds
        y_pred = self._unnormalize_preds(y_pred)
        pred_dist = self._unnormalize_predictive_dist(pred_dist)
        return y_pred, pred_dist

    @tf.function
    def __call__(self, x_batch, y_batch):
        x_prior = next(self.x_prior)
        y_prior = next(self.y_prior)

        with tf.GradientTape() as tape:
            tape.watch(self.posterior.trainable_variables)
            samples_posterior = self.posterior.sample([self.n_particles])
            nn_params, likelihood_std = self.split_params(samples_posterior)
            y_pred = self.nn.call_parametrized(x_batch, nn_params)
            y_prior_pred = self.nn.call_parametrized(x_prior, nn_params)
            likelihood = self.ll(y_batch, y_pred, likelihood_std)
            felbo = -likelihood

            # minimize distance of likelihood std from prior value
            prior = tf.reduce_logsumexp(self.prior.log_prob(likelihood_std))
            felbo -= prior * self.coeff_prior

            # maximize prior entropy
            entropy_gradient = tf.stop_gradient(self.ssge.estimate_gradients(s=y_pred))
            entropy_surrogate = entropy_gradient * y_pred
            felbo += entropy_surrogate * self.coeff_entropy

            # maximize likelihood entropy
            entropy_gradient = tf.stop_gradient(self.ssge.estimate_gradients(s=y_prior_pred))
            entropy_surrogate = entropy_gradient * y_prior_pred
            felbo += entropy_surrogate * self.coeff_entropy

            # minimize prior cross-entropy
            cross_entropy_gradient = tf.stop_gradient(self.ssge.estimate_gradients(s=y_prior, x=y_prior_pred))
            cross_entropy_surrogate = cross_entropy_gradient * y_prior_pred
            felbo -= cross_entropy_surrogate * self.coeff_cross_entropy

            # for smoothness take mean instead of sum (which is the default in tape.gradient())
            felbo = tf.reduce_mean(felbo)

        # compute gradient of felbo wrt posterior parameters
        gradients = tape.gradient(felbo, self.posterior.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.posterior.trainable_variables))


if __name__ == '__main__':
    tf.random.set_seed(1234)
    u = tfd.Uniform(-1, 1)

    input_dim = 1
    experiment = 'sinusoids'
    if experiment == 'densities':
        from data import Densities

        data_generator = Densities(input_dim, 1, (-0.5, 0.5), (0.25, 0.25))
        # hparams
        hparams = dict(
            activation='elu',
            n_particles=20,
            learning_rate=0.01,
            weight_decay=1e-4,
            coeff_prior=1.0,
            coeff_entropy=10.0,
            coeff_cross_entropy=1.0,
            likelihood_prior_mean=1e-1,
            prior_noise_std=1e-2,
            noise_std=0.05,
        )
        n_train = 15
    elif experiment == 'sinusoids':
        from data import Sinusoids

        data_generator = Sinusoids(
            # uniform variables, format: (low, high)
            amplitude=(1, 2),
            period=(1, 1),
            # normal variables, format: (mean, std)
            slope=(2, 1e-6),
            x_shift=(0.0, 1e-6),
            y_shift=(0.0, 1e-6),
        )
        # hparams
        hparams = {
            'activation': 'elu',
            'n_particles': 30,
            # 'batch_size': 200,
            'learning_rate': 0.01,
            'weight_decay': 1e-6,
            'coeff_prior': 1.0,
            'coeff_entropy': 1.0,
            'coeff_cross_entropy': 1.0,
            'likelihood_prior_mean': 0.1,
            'prior_noise_std': 1e-3,
        }
        n_train = 100
    else:
        raise NotImplemented

    x_train = u.sample([n_train, input_dim])
    x_test = u.sample([5e3, input_dim])

    # sample one function from prior
    f = data_generator.sample_function()

    # evaluate it
    y_train = f(x_train)
    y_test = f(x_test)

    # init
    nn = FVI(
        train_data=(x_train, y_train),
        functional_prior=data_generator,
        experiment=experiment,
        verbose=1,
        **hparams
    )

    # train
    nn.fit(plot=True, show=True, test_data=(x_test, y_test))
    print(nn.eval(x_test, y_test, 'test'))
