import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


class EqualWeightedMixtureDist(tfd.Distribution):
    def __init__(self, dists):
        self.dists = dists
        self.n_dists = len(dists)
        super().__init__(tf.float32, tfd.NOT_REPARAMETERIZED, False, False, name='EqualWeightedMixtureDist')

    def _mean(self):
        return tf.reduce_mean(tf.stack([dist.mean() for dist in self.dists], axis=0), axis=0)

    def _stddev(self):
        return tf.math.sqrt(self.variance())

    def _variance(self):
        means = tf.stack([dist.mean() for dist in self.dists], axis=0)

        var1 = tf.reduce_mean((means - tf.reduce_mean(means, axis=0)) ** 2, axis=0)
        var2 = tf.reduce_mean(tf.stack([dist.variance() for dist in self.dists], axis=0), axis=0)

        # check shape
        original_shape = self.dists[0].mean().shape
        # assert var1.shape == var2.shape == original_shape
        tf.assert_equal(var1.shape, var2.shape)
        tf.assert_equal(var2.shape, original_shape)

        return var1 + var2

    def _log_prob(self, value):
        log_probs_dists = tf.stack([dist.log_prob(value) for dist in self.dists], axis=0)
        return tf.reduce_logsumexp(log_probs_dists, axis=0) - tf.math.log(
            tf.convert_to_tensor(self.n_dists, dtype=tf.float32))

    def cdf(self, value, **kwargs):
        cum_p = tf.stack([dist.cdf(value, **kwargs) for dist in self.dists])
        return tf.reduce_mean(cum_p, axis=0)

    def _batch_shape_tensor(self):
        return self.dists[0].batch_shape_tensor()

    def _batch_shape(self):
        return self.dists[0].batch_shape

    def _event_shape_tensor(self):
        return self.dists[0].event_shape_tensor()

    def _event_shape(self):
        return self.dists[0].event_shape


class AffineTransform:
    def __init__(self, normalization_mean, normalization_std):

        self.loc_tensor = tf.cast(normalization_mean, dtype=tf.float32)
        self.scale_tensor = tf.cast(normalization_std, dtype=tf.float32)

        shift = tfb.Shift(self.loc_tensor)

        if tf.size(self.scale_tensor) == 1:
            scale = tfb.Scale(self.scale_tensor)
        else:
            scale = tfb.ScaleMatvecDiag(self.scale_tensor)

        self.transform = shift(scale)

    def apply(self, base_dist):
        if isinstance(base_dist, tfp.distributions.Categorical):
            # Categorical distribution --> make sure that normalization stats are mean=0 and std=1
            tf.assert_equal(tf.math.count_nonzero(self.loc_tensor), tf.constant(0, dtype=tf.int64))
            tf.assert_equal(tf.math.count_nonzero(self.scale_tensor - 1.0), tf.constant(0, dtype=tf.int64))
            return base_dist
        else:
            base_dist = base_dist

            d = self.transform(base_dist)
            d.transform = self.transform
            d.base_dist = base_dist

            def cdf(y, **kwargs):
                x = self.transform.inverse(y)
                return base_dist.cdf(x, **kwargs)

            def mean():
                return self.transform(base_dist.mean())

            def stddev():
                return tf.math.exp(tf.math.log(base_dist.stddev()) + tf.math.log(self.scale_tensor))

            def variance():
                return tf.math.exp(tf.math.log(base_dist.variance()) + 2 * tf.math.log(self.scale_tensor))

            d.cdf = cdf
            d.mean = mean
            d.stddev = stddev
            d.variance = variance

            return d