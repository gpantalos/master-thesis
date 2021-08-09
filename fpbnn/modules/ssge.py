import tensorflow as tf


class AbstractScoreEstimator:
    @staticmethod
    def rbf_kernel(x1, x2, bandwidth):
        return tf.exp(-tf.reduce_sum(tf.square((x1 - x2) / bandwidth), axis=-1) / 2)

    def gram(self, x1, x2, bandwidth):
        """
        x1: [..., n1, D]
        x2: [..., n2, D]
        bandwidth: [..., 1, 1, D]
        returns: [..., n1, n2]
        """
        x_row = tf.expand_dims(x1, -2)
        x_col = tf.expand_dims(x2, -3)
        return self.rbf_kernel(x_row, x_col, bandwidth)

    def grad_gram(self, x1, x2, bandwidth):
        """
        x1: [..., n1, D]
        x2: [..., n2, D]
        bandwidth: [..., 1, 1, D]
        returns: [..., n1, n2], [..., n1, n2, D], [..., n1, n2, D]
        """
        x_row = tf.expand_dims(x1, -2)
        x_col = tf.expand_dims(x2, -3)
        # g: [..., n1, n2]
        g = self.rbf_kernel(x_row, x_col, bandwidth)
        # diff: [..., n1, n2, D]
        diff = (x_row - x_col) / (bandwidth ** 2)
        # g_expand: [..., n1, n2, 1]
        g_expand = tf.expand_dims(g, axis=-1)
        # grad_x1: [..., n1, n2, D]
        grad_x2 = g_expand * diff
        # grad_x2: [..., n1, n2, D]
        grad_x1 = g_expand * (-diff)
        return g, grad_x1, grad_x2

    @staticmethod
    def median_heuristic(x_samples, x_basis):
        """
        x_samples: [..., n_samples, d]
        x_basis: [..., n_basis, d]
        returns: [..., 1, 1, d]
        """
        d = tf.shape(x_samples)[-1]
        n_samples = tf.shape(x_samples)[-2]
        n_basis = tf.shape(x_basis)[-2]
        x_samples_expand = tf.expand_dims(x_samples, -2)
        x_basis_expand = tf.expand_dims(x_basis, -3)
        pairwise_dist = tf.abs(x_samples_expand - x_basis_expand)

        length = len(pairwise_dist.get_shape())
        reshape_dims = list(range(length - 3)) + [length - 1, length - 3, length - 2]
        pairwise_dist = tf.transpose(pairwise_dist, reshape_dims)

        k = n_samples * n_basis // 2
        k = k if k > 0 else 1
        top_k_values = tf.nn.top_k(tf.reshape(pairwise_dist, [-1, d, n_samples * n_basis]), k=k).values
        bandwidth = tf.reshape(top_k_values[:, :, -1], tf.concat([tf.shape(x_samples)[:-2], [1, 1, d]], axis=0))
        bandwidth *= (tf.cast(d, bandwidth.dtype) ** 0.5)
        bandwidth += tf.cast((bandwidth < 1e-6), bandwidth.dtype)
        return bandwidth


class SSGE(AbstractScoreEstimator):
    def __init__(self, n_eigen=6, eta=1e-3, n_eigen_threshold=None, bandwidth=2.0):
        self.n_eigen_threshold = n_eigen_threshold
        self.bandwidth = bandwidth
        self.n_eigen = n_eigen
        self.eta = eta

    def nystrom_ext(self, s, x, eigen_vectors, eigen_values, bandwidth):
        """
        s: [..., m, d]
        index_points: [..., n, d]
        eigen_vectors: [..., m, n_eigen]
        eigen_values: [..., n_eigen]
        returns: [..., n, n_eigen], by default n_eigen=m.
        """
        m = tf.shape(s)[-2]
        # kxq: [..., N, m]
        kxq = self.gram(x, s, bandwidth)
        # ret: [..., N, n_eigen]
        ret = tf.matmul(kxq, eigen_vectors)
        ret *= tf.cast(m, ret.dtype) ** 0.5 / tf.expand_dims(eigen_values, axis=-2)
        return ret

    def estimate_gradients(self, s, x=None):
        perm = (1, 0, 2)
        if x is not None:
            return tf.transpose(self.__call__(tf.transpose(s, perm), tf.transpose(x, perm)), perm)
        else:
            return tf.transpose(self.__call__(tf.transpose(s, perm)), perm)

    def __call__(self, s, x=None):
        """
        s: [..., m, d], samples
        x: [..., n, d], index points
        """
        if x is None:
            x = stacked_samples = s
        else:
            stacked_samples = tf.concat([s, x], axis=-2)

        if self.bandwidth is None:
            length_scale = self.median_heuristic(stacked_samples, stacked_samples)
        else:
            length_scale = self.bandwidth

        m = tf.shape(s)[-2]
        # kq: [..., m, m]
        # grad_k1: [..., m, m, d]
        # grad_k2: [..., m, m, d]
        kq, grad_k1, grad_k2 = self.grad_gram(s, s, length_scale)
        kq += self.eta * tf.eye(m)
        # eigen_vectors: [..., m, m]
        # eigen_values: [..., m]
        eigen_values, eigen_vectors = tf.linalg.eigh(kq)
        if (self.n_eigen is None) and (self.n_eigen_threshold is not None):
            eigen_arr = tf.reduce_mean(tf.reshape(eigen_values, [-1, m]), axis=0)
            eigen_arr = tf.reverse(eigen_arr, axis=[-1])
            eigen_arr /= tf.reduce_sum(eigen_arr)
            eigen_cum = tf.cumsum(eigen_arr, axis=-1)
            eigen_les = tf.cast(tf.less(eigen_cum, self.n_eigen_threshold), tf.int32)
            self.n_eigen = tf.reduce_sum(eigen_les)
        if self.n_eigen is not None:
            eigen_values = eigen_values[..., -self.n_eigen:]
            eigen_vectors = eigen_vectors[..., -self.n_eigen:]
        # eigen_ext: [..., n, n_eigen]
        eigen_ext = self.nystrom_ext(s, x, eigen_vectors, eigen_values, length_scale)
        # grad_k1_avg = [..., m, d]
        grad_k1_avg = tf.reduce_mean(grad_k1, axis=-3)
        # beta: [..., n_eigen, d]
        beta = - tf.matmul(eigen_vectors, grad_k1_avg, transpose_a=True)
        beta *= tf.cast(m, beta.dtype) ** 0.5 / tf.expand_dims(eigen_values, -1)
        # grads: [..., n, d]
        grads = tf.matmul(eigen_ext, beta)
        return grads
