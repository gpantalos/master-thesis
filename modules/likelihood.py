import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

from .prob_dist import EqualWeightedMixtureDist


class GaussianLikelihood(tf.Module):
    def __init__(self, config, output_dim, n_batched_models, name='gaussian_likelihood'):
        super().__init__(name=name)
        self.config = config
        self.output_dim = output_dim
        self.n_batched_models = n_batched_models

        log_var = self.config['log_var']

        if tf.rank(log_var) == 0:
            log_var = tf.ones((n_batched_models, 1, self.output_dim), tf.float32) * log_var
        elif tf.rank(log_var) == 1:
            assert (tf.size(log_var) == self.output_dim)
            log_var = tf.reshape(log_var, (n_batched_models, 1, self.output_dim))
        else:
            raise Exception('Unexpected dimensionality of likelihood log variance')

        if self.config['learn_likelihood_variables']:
            log_var = tf.Variable(log_var, dtype=tf.float32, name=f'{name}_log_var', trainable=True)
        else:
            log_var = tf.constant(log_var)

        self.log_var = log_var
        self._parameters_shape = None

    @tf.function
    def log_prob(self, y_pred, y_true, param=None):
        # y_pred has dimensions (num_batched_models, batch_size, output_dim)
        # y_true should have dimensions: (batch_size x output_dim), (1, batch_size x output_dim) or (num_batched_models, batch_size, output_dim)
        # returns a tensor with dimensions (n_models, 1)
        tf.assert_equal(tf.rank(y_true), 3)

        if param is None:
            std = tf.math.exp(0.5 * self.log_var)
        else:
            param = tf.reshape(param, self.log_var.shape)
            std = tf.math.exp(0.5 * param)

        dist = tfd.Independent(
            tfp.distributions.Normal(y_pred, std),
            reinterpreted_batch_ndims=1
        )

        # TODO remove keepdims
        return tf.reduce_mean(dist.log_prob(y_true), axis=-1, keepdims=True)

    def calculate_eval_metrics(self, y_pred, pred_dist, y_true):
        # log_prob returns a tensor of dims (n_batched,)
        avg_ll = tf.reduce_mean(pred_dist.log_prob(y_true))
        avg_rmse = self.rmse(pred_dist.mean(), y_true)
        avg_calibr_error = self.calib_error(pred_dist, y_true)

        o = {'avg_ll': avg_ll,
             'avg_rmse': avg_rmse,
             'cal_err': avg_calibr_error}

        return o

    @staticmethod
    def calib_error(pred_dist, y_true, use_circular_region=False):
        if y_true.ndim == 3:
            y_true = y_true[0]

        if use_circular_region or y_true.shape[-1] > 1:
            cdf_vals = pred_dist.cdf(y_true, circular=True)
        else:
            cdf_vals = pred_dist.cdf(y_true)
        cdf_vals = tf.reshape(cdf_vals, (-1, 1))

        num_points = tf.cast(tf.size(cdf_vals), tf.float32)
        conf_levels = tf.linspace(0.05, 0.95, 20)
        emp_freq_per_conf_level = tf.reduce_sum(tf.cast(cdf_vals <= conf_levels, tf.float32), axis=0) / num_points

        #TODO change back
        #calib_err = tf.reduce_mean(tf.abs((emp_freq_per_conf_level - conf_levels)))
        calib_err = tf.sqrt(tf.reduce_mean((emp_freq_per_conf_level - conf_levels)**2))
        return calib_err

    def get_pred_dist(self, y_pred, param=None):
        if param is None:
            std = tf.math.exp(0.5 * self.log_var)
        else:
            param = tf.reshape(param, self.log_var.shape)
            std = tf.math.exp(0.5 * param)

        pred_dists = [tfp.distributions.Independent(
            NormalDistr(y_pred[i], std[i]),
            reinterpreted_batch_ndims=1) for i in range(self.n_batched_models)]

        pred_dist = EqualWeightedMixtureDist(pred_dists)

        return pred_dist

    def set_variables_vectorized(self, parameters):
        if self._parameters_shape is None:
            self._parameters_shape = parameters.shape

        self.log_var.assign(tf.reshape(parameters, self.log_var.shape))

    def reshape_gradients(self, gradients):
        vectorized_gradients = tf.concat([tf.reshape(g, (-1, 1)) for g in gradients], axis=0)
        return tf.reshape(vectorized_gradients, self._parameters_shape)

    def rmse(self, y_pred_mean, y_true):
        """
        Args:
            y_pred_mean (tf.Tensor): mean prediction
            y_true (tf.Tensor): true target variable

        Returns: (tf.Tensor) Root mean squared error (RMSE)

        """
        tf.assert_equal(y_pred_mean.shape, y_true.shape)
        return tf.sqrt(tf.reduce_mean(tf.square(y_pred_mean - y_true)))

class ClassificationLikelihood(tf.Module):
    def __init__(self, output_dim):
        self.output_dim = output_dim

    def log_prob(self, y_pred, y_true):
        if self.output_dim == 1:
            return -1 * self._binary_cross_entropy(y_pred, y_true)
        else:
            return -1 * self._categorical_cross_entropy(y_pred, y_true)

    @tf.function
    def _binary_cross_entropy(self, y_pred, y_true):
        log_prob_per_sample = tf.squeeze(tf.keras.backend.binary_crossentropy(y_true, y_pred))
        log_prob_per_model = tf.reduce_mean(log_prob_per_sample, axis=-1)
        return log_prob_per_model

    @tf.function
    def _categorical_cross_entropy(self, y_pred, y_true):
        log_prob_per_sample = tf.keras.backend.categorical_crossentropy(
            y_true, y_pred, from_logits=False, axis=-1
        )
        log_prob_per_model = tf.reduce_mean(log_prob_per_sample, axis=-1)
        return log_prob_per_model

    def get_pred_dist(self, y_pred):
        probs = tf.reduce_mean(y_pred, axis=0)  # marginalize posterior samples
        pred_dist = tfp.distributions.Categorical(probs=probs)
        return pred_dist

    def calculate_eval_metrics(self, y_pred, pred_dist, y_true):
        prob_per_class = tf.reduce_mean(y_pred, axis=0)
        avg_acc = self.accuracy(prob_per_class, y_true)
        avg_ll = self.avg_log_likelihood(prob_per_class, y_true)
        avg_calibr_error = self.calib_error(prob_per_class, y_true)

        eval_metrics = {'avg_ll': avg_ll,
                        'avg_acc': avg_acc,
                        'cal_err': avg_calibr_error}
        return eval_metrics

    def avg_log_likelihood(self, prob_per_class, y_true):
        return - self._categorical_cross_entropy(prob_per_class, y_true)

    def calib_error(self, prob_per_class, y_true, n_bins=5):
        """
        Computes the calibration error
        Args:
            prob_per_class (tf.Tensor):  predicted class probabilities, Tensor of shape (batch_size, n_classes)
            y_true (tf.Tensor): Tensor of one-hot encodings (batch_size, n_classes)
            n_bins (int): number of bins used for estimating the calibration error
        Returns: classification accuracy
        """
        # check whether there is suffifient data for binning
        # there should be at lest 10 samples per bin
        n_samples = y_true.shape[0]
        if n_samples < 10 * n_bins:
            import warnings
            warnings.warn("Insufficient data for computing a reliable estimate of the calibration "
                          "error with %i bins." % n_bins)

        # split data into bins of similar prediction confidence
        p_max = tf.reduce_max(prob_per_class, axis=-1)
        bins_idx = _split(tf.argsort(p_max), n_splits=n_bins)

        # for each bin compute the avg. confidence and actual accuracy
        residuals = []
        for i, bin_idx in enumerate(bins_idx):
            prob_per_class_bin = tf.gather(prob_per_class, bin_idx)
            p_max_bin = tf.gather(p_max, bin_idx)
            y_true_bin = tf.gather(y_true, bin_idx)
            accuracy_bin = self.accuracy(prob_per_class_bin, y_true_bin)
            avg_confidence_bin = tf.reduce_mean(p_max_bin)
            residuals.append(avg_confidence_bin - accuracy_bin)
        residuals = tf.stack(residuals)

        # Across the bins, compute the average absolute difference between confidence and accuracy
        calib_error = tf.reduce_mean(tf.abs(residuals))
        return calib_error

    def accuracy(self, prob_per_class, y_true):
        """
        Computes the classification accuracy
        Args:
            prob_per_class (tf.Tensor):  predicted class probabilities, Tensor of shape (batch_size, n_classes)
            y_true (tf.Tensor): Tensor of one-hot encodings (batch_size, n_classes)

        Returns: classification accuracy
        """
        tf.assert_equal(tf.rank(prob_per_class), 2)

        if prob_per_class.shape[-1] > 1:
            metric = tf.keras.metrics.CategoricalAccuracy(name='cat_acc', dtype=None)
        else:
            metric = tf.keras.metrics.BinaryAccuracy(name='bin_acc', dtype=None)

        metric.update_state(y_true, prob_per_class)
        return metric.result()

""" helper function """

def _split(array, n_splits):
    """
        splits array into n_splits of potentially unequal sizes
    """
    assert array.ndim == 1
    n_elements = array.shape[0]

    remainder = n_elements % n_splits
    split_sizes = []
    for i in range(n_splits):
        if i < remainder:
            split_sizes.append(n_elements //  n_splits + 1)
        else:
            split_sizes.append(n_elements // n_splits)
    return tf.split(array, split_sizes)

class NormalDistr(tfp.distributions.Normal):

    def mahalanobis_distance(self, x):
        mean, std = self.loc, self.scale
        return tf.reduce_sum(((x - mean) / std) ** 2, axis=-1)

    def circular_cdf(self, x, **kwargs):
        assert x.ndim >= 2
        event_dim = x.shape[-1] # interpret last dim as event dim
        mahalanobis_dist = self.mahalanobis_distance(x)
        cdf_vals = tfp.distributions.Chi2(df=event_dim).cdf(mahalanobis_dist)
        assert cdf_vals.shape == x.shape[:-1]
        return cdf_vals

    def cdf(self, x, circular=False, **kwargs):
        if circular:
            return self.circular_cdf(x, **kwargs)
        else:
            return super().cdf(x, **kwargs)

    def _log_cdf(self, x, circular=False, **kwargs):
        if circular:
            return tf.expand_dims(tf.math.log(self.circular_cdf(x, **kwargs)), axis=-1)
        else:
            return super()._log_cdf(x, **kwargs)



