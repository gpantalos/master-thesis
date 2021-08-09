import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

exp = tf.math.exp
log = tf.math.log

dtype = tf.float32


class Affine:
    """Y = transform(X) = normalization_std @ X + normalization_mean"""

    def __init__(self, normalization_mean, normalization_std):
        self.loc_tensor = tf.cast(normalization_mean, dtype=dtype)
        self.scale_tensor = tf.cast(normalization_std, dtype=dtype)

        shift = tfb.Shift(self.loc_tensor)
        if tf.size(self.scale_tensor) == 1:
            scale = tfb.Scale(self.scale_tensor)
        else:
            scale = tfb.ScaleMatvecDiag(self.scale_tensor)
        self.transform = tfb.Chain([shift, scale])

    def __call__(self, base_dist):
        # Transform distribution to access `log_prob` and `sample` methods
        transformed_dist = self.transform(base_dist)

        # Fill in missing methods
        mean, stddev, var = base_dist.mean(), base_dist.stddev(), base_dist.variance()
        transformed_dist.mean = self.transform(mean)
        transformed_dist.stddev = exp(log(stddev) + log(self.scale_tensor))
        transformed_dist.variance = exp(log(var) + 2 * log(self.scale_tensor))
        return transformed_dist


if __name__ == '__main__':
    p = tfd.Normal(0, 1)
    transform = Affine(1, 2)
    q = transform(p)
    print(q.mean)
    print(q.stddev)
    print(q.variance)
    print(q.cdf(.5))
