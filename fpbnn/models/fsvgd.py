import tensorflow as tf

from fpbnn.modules.functional_regression import FunctionalRegressionModel


class FSVGD(FunctionalRegressionModel):
    """Functional Stein Variational Gradient Descent: https://arxiv.org/abs/1902.09754"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'fsvgd'
        likelihood_params = tf.ones([self.n_particles, self.likelihood_param_size]) * self.likelihood_prior_mean
        self.particles = tf.Variable(tf.concat([self.nn_params, likelihood_params], -1))

    def predict(self, x):
        x = self._broadcast_and_cast_dtype(x)
        x = self.normalize(x)
        nn_params, likelihood_std = self.split_params(self.particles)
        output_pred = self.nn.call_parametrized(x, nn_params)
        output_dist = self._predictive_mixture(output_pred, likelihood_std)
        output_pred = self._unnormalize_preds(output_pred)
        output_dist = self._unnormalize_predictive_dist(output_dist)
        return output_pred, output_dist

    @tf.function
    def __call__(self, x_batch, y_batch):
        x_prior = next(self.x_prior)
        y_prior = next(self.y_prior)
        with tf.GradientTape() as weight_space_tape:
            weight_space_tape.watch(self.particles)
            nn_params, likelihood_std = self.split_params(self.particles)
            y_pred = self.nn.call_parametrized(x_batch, nn_params)

        # function space gradient
        with tf.GradientTape() as function_space_tape:
            function_space_tape.watch(y_pred)
            likelihood = self.ll(y_batch, y_pred, likelihood_std)
        felbo = function_space_tape.gradient(likelihood, y_pred)

        # cross-entropy
        y_prior_pred = self.nn.call_parametrized(x_prior, nn_params)
        cross_entropy_gradient = self.ssge.estimate_gradients(s=y_prior, x=y_prior_pred)
        felbo += cross_entropy_gradient * self.coeff_cross_entropy

        # kernel
        copy = tf.identity(y_pred)
        with tf.GradientTape() as kernel_tape:
            kernel_tape.watch(copy)
            k = self.kernel.matrix(copy, y_pred)
        grad_kernel = kernel_tape.gradient(k, copy)
        v = -(k @ felbo + grad_kernel) / self.n_particles
        grads = weight_space_tape.gradient(y_pred, self.particles, output_gradients=v)
        self.optimizer.apply_gradients([(grads, self.particles)])


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
        hparams = dict(activation='tanh', n_particles=20, learning_rate=0.0001, bandwidth=0.001,
                       likelihood_prior_mean=0.1)
        n_train = 10
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
        hparams = dict(activation='elu', n_particles=10, learning_rate=0.0001, bandwidth=0.0001,
                       likelihood_prior_mean=0.1, coeff_cross_entropy=1.0)
        n_train = 15
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
        n_train = 1e3
    else:
        raise NotImplemented

    x_train = u.sample([n_train, input_dim])
    x_test = u.sample([1000, input_dim])
    x_val = u.sample([500, input_dim])

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

    params = dict(
        functional_prior=functional_prior,
        train_data=(x_train, y_train),
        experiment=experiment,
        verbose=1,
    )

    # init
    nn = FSVGD(**params, **hparams)

    # train
    nn.fit(
        val_data=(x_val, y_val),
        test_data=(x_test, y_test),
        n_plots=10,
        show=True,
        plot=True,
    )
