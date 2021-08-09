import tensorflow as tf


class RBF(tf.Module):
    def __init__(self, bandwidth=1.0, jitter=1e-8):
        super().__init__()
        self.bandwidth = bandwidth
        self.jitter = jitter

    def matrix(self, x1, x2, norm_axis=(-2, -1)):
        norm = tensor_squared_norm(x1, x2, axis=norm_axis)
        gamma = 1.0 / (2 * self.bandwidth ** 2 + self.jitter)
        return tf.exp(-gamma * norm)


def tensor_squared_norm(x1, x2, axis=(-2, -1)):
    """Computes squared norm of n-dimensional tensor"""
    x1 = tf.expand_dims(x1, 0)
    x2 = tf.expand_dims(x2, 1)
    norm = tf.norm(x1 - x2, axis=axis)
    return tf.square(norm)

# def matrix_squared_norm(x1, x2):
#     x1x2 = x1 @ tf.transpose(x2)
#
#     x1x1 = x1 @ tf.transpose(x1)
#     x1x1 = tf.expand_dims(tf.linalg.diag_part(x1x1), -1)
#
#     x2x2 = x2 @ tf.transpose(x2)
#     x2x2 = tf.expand_dims(tf.linalg.diag_part(x2x2), -2)
#
#     return x1x1 - 2 * x1x2 + x2x2
