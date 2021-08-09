import tensorflow as tf

from fpbnn.modules.abstract_regression import BayesianRegressionModel


class BenchmarkModel(BayesianRegressionModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "mlp"

    def predict(self, x):
        x = self._broadcast_and_cast_dtype(x)
        x = self.normalize(x)
        y_pred = self.nn(x)
        likelihood_std = tf.ones(y_pred.shape[1:]) * 0.1
        pred_dist = self._predictive_mixture(y_pred, likelihood_std)
        y_pred = self._unnormalize_preds(y_pred)
        pred_dist = self._unnormalize_predictive_dist(pred_dist)
        return y_pred, pred_dist

    @tf.function
    def __call__(self, x_batch, y_batch):
        with tf.GradientTape() as tape:
            tape.watch(self.nn.trainable_variables)
            y_pred = self.nn(x_batch)
            loss = tf.reduce_mean((y_batch - y_pred) ** 2)
        gradients = tape.gradient(loss, self.nn.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.nn.trainable_variables))


if __name__ == '__main__':
    tf.random.set_seed(1234)
    from tensorflow_probability import distributions as tfd

    input_dim = 1

    u = tfd.Uniform(-1, 1)

    x_train = u.sample([5, input_dim])
    x_test = u.sample([500, input_dim])
    x_val = u.sample([200, input_dim])

    experiment = 'densities'
    if experiment == 'densities':
        from data import Densities

        functional_prior = Densities(input_dim, 1, (-0.5, 0.5), (0.25, 0.25))
        # hparams
        hparams = {
            'activation': 'elu',
            'n_particles': 20,
            'learning_rate': 0.001,
            'coeff_prior': 1.0,
            'likelihood_prior_mean': 0.001,
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
            'n_particles': 30,
            # 'batch_size': 200,
            'learning_rate': 0.002,
            'weight_decay': 1e-6,
            'coeff_prior': 1e0,
            'coeff_entropy': 1e0,
            'coeff_cross_entropy': 1e0,
            'likelihood_prior_mean': 0.01,
            'prior_noise_std': 1e-3,
        }

    else:
        raise NotImplemented

    # sample one function from prior
    f = functional_prior.sample_function()

    # evaluate it
    y_train = f(x_train)
    y_test = f(x_test)
    y_val = f(x_val)

    params = {
        'train_data': (x_train, y_train),
        'experiment': experiment,
        'home_dir': '../',
        'verbose': 1,
    }

    # init
    nn = BenchmarkModel(**params, **hparams)

    # train
    nn.fit(
        val_data=(x_val, y_val),
        test_data=(x_test, y_test),
        n_plots=10,
        show=True,
        plot=True,
    )
