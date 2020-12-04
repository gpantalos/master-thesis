import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

import numpy as np
import time
from modules.prob_dist import AffineTransform


class RegressionModel:

    def predict(self, x):
        raise NotImplementedError

    def step(self, x_batch, y_batch):
        raise NotImplementedError

    def fit(self, x_val=None, y_val=None, log_period=500, num_iter_fit=None):
        train_batch_sampler = self._get_batch_sampler(self.x_train, self.y_train)
        num_iter_fit = self.num_iter_fit if num_iter_fit is None else num_iter_fit

        t = time.time()
        loss_list = []
        for iter in range(num_iter_fit):
            x_batch, y_batch = next(train_batch_sampler)
            loss = self.step(x_batch, y_batch)
            loss_list.append(loss)

            if iter % log_period == 0:
                loss = np.mean(loss_list)
                loss_list = []
                message = '\n\tIter %d/%d - Train-Time %.2f sec' % (iter, self.num_iter_fit, time.time() - t)
                message += ' - Loss: %.4f' % loss

                if x_val is not None and y_val is not None:
                    metric_dict = self.eval(x_val, y_val)
                    message += ' - ' + ' - '.join(
                        ['%s: %.4f' % (metric, value) for metric, value in metric_dict.items()])
                print(message)
                t = time.time()

    def eval(self, x, y):
        x, y = self._handle_input_data(x, y, convert_to_tensor=True)
        _, pred_dist = self.predict(x)
        # compute metrics
        ll = tf.reduce_mean(tf.reduce_sum(pred_dist.log_prob(y), axis=-1)).numpy()
        rmse = tf.sqrt(tf.reduce_mean(tf.reduce_sum((pred_dist.mean()-y)**2, axis=-1), axis=-1)).numpy()
        return {'avg_ll': ll, 'rmse': rmse}

    def plot_predictions(self, x_plot_range=(-4, 4), show=True):
        from matplotlib import pyplot as plt
        assert self.input_size == 1 and self.output_size == 1
        x_plot = np.arange(*x_plot_range, 0.01)
        y_pred, pred_dist = self.predict(x_plot)
        fig, ax = plt.subplots(1,1)
        # plot predictive mean and confidence interval
        ax.plot(x_plot, pred_dist.mean())
        lcb, ucb = pred_dist.mean() - 2 * pred_dist.stddev(), pred_dist.mean() + 2 * pred_dist.stddev()
        ax.fill_between(x_plot, lcb.numpy().flatten(), ucb.numpy().flatten(), alpha=0.2)
        # plot individual particle predictions
        for y_hat in y_pred.numpy():
            ax.plot(x_plot, y_hat, color='green', alpha=0.3, linewidth=1.0)
        # unnormalize training data & plot it
        x_train = self.x_train * self.x_std + self.x_mean
        y_train = self.y_train * self.y_std + self.y_mean
        ax.scatter(x_train, y_train)
        if show:
            fig.show()
        return fig, ax

    def _process_train_data(self, x_train, y_train, normalize_data=True):
        self.x_train, self.y_train = self._handle_input_data(x_train, y_train, convert_to_tensor=True)
        self.input_size, self.output_size = self.x_train.shape[-1], self.y_train.shape[-1]
        self.num_train_samples = self.x_train.shape[0]
        self._compute_normalization_stats(self.x_train, self.y_train, normalize_data=normalize_data)
        self.x_train, self.y_train  = self._normalize_data(self.x_train, self.y_train)

    def _get_batch_sampler(self, x, y):
        x, y = self._handle_input_data(x, y, convert_to_tensor=True)
        num_train_points = x.shape[0]

        if self.batch_size == -1:
            batch_size = num_train_points
        elif self.batch_size > 0:
            batch_size = self.batch_size
        else:
            raise AssertionError('batch size must be either positive or -1')

        train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
        train_dataset = train_dataset.shuffle(buffer_size=num_train_points, reshuffle_each_iteration=True)\
            .repeat().batch(batch_size)
        train_batch_sampler = train_dataset.__iter__()
        return train_batch_sampler

    def _predictive_mixture(self, y_pred, likelihood_std):
        # check shapes
        tf.assert_equal(len(y_pred.shape), 3)
        tf.assert_equal(y_pred.shape[0], likelihood_std.shape[0])
        tf.assert_equal(y_pred.shape[-1], likelihood_std.shape[-1])

        num_mixture_components = y_pred.shape[0]
        comoponents = [tfd.Independent(tfd.Normal(y_pred[i], likelihood_std[i]), reinterpreted_batch_ndims=1)
         for i in range(num_mixture_components)]
        categorical = tfd.Categorical(logits=tf.transpose(tf.zeros(y_pred.shape[:2])))
        return tfp.distributions.Mixture(categorical, comoponents, name='predictive_mixture')

    def _compute_normalization_stats(self, x_train, y_train, normalize_data=True):
        self.x_mean = tf.reduce_mean(x_train, axis=0)
        self.x_std = tfp.stats.stddev(x_train, sample_axis=0)
        self.y_mean = tf.reduce_mean(y_train, axis=0)
        self.y_std = tfp.stats.stddev(y_train, sample_axis=0)

        self.affine_pred_dist_transform = AffineTransform(normalization_mean=self.y_mean,
                                                        normalization_std=self.y_std)

    def _normalize_data(self, x, y=None):
        x = (x - self.x_mean) / self.x_std
        if y is None:
            return x
        else:
            y = (y - self.y_mean) / self.y_std
            return x, y

    def _unnormalize_preds(self, y):
        return y * self.y_std + self.y_mean

    def _unnormalize_predictive_dist(self, pred_dist):
        return self.affine_pred_dist_transform.apply(pred_dist)

    @staticmethod
    def _handle_input_data(x, y=None, convert_to_tensor=True, dtype=tf.float32):
        if x.ndim == 1:
            x = np.expand_dims(x, -1)

        assert x.ndim == 2

        if y is not None:
            if y.ndim == 1:
                y = np.expand_dims(y, -1)
            assert x.shape[0] == y.shape[0]
            assert y.ndim == 2

            if convert_to_tensor:
                x, y = tf.cast(x, dtype=dtype), tf.cast(y, dtype=dtype)
            return x, y
        else:
            if convert_to_tensor:
                x = tf.cast(x, dtype=dtype)
            return x