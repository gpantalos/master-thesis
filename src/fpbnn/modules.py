import math
from collections.abc import Callable
from typing import Union

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras

tfd = tfp.distributions
tfb = tfp.bijectors
exp = tf.math.exp
log = tf.math.log
dtype = tf.float32


class AbstractScoreEstimator:
    @staticmethod
    def rbf_kernel(x1: tf.Tensor, x2: tf.Tensor, bandwidth: tf.Tensor) -> tf.Tensor:
        diff = tf.subtract(x1, x2)
        scaled_diff = tf.divide(diff, bandwidth)
        return tf.exp(-tf.reduce_sum(tf.square(scaled_diff), axis=-1) / 2)

    def gram(self, x1: tf.Tensor, x2: tf.Tensor, bandwidth: tf.Tensor) -> tf.Tensor:
        """
        x1: [..., n1, D]
        x2: [..., n2, D]
        bandwidth: [..., 1, 1, D]
        returns: [..., n1, n2]
        """
        x_row = tf.expand_dims(x1, -2)
        x_col = tf.expand_dims(x2, -3)
        return self.rbf_kernel(x_row, x_col, bandwidth)

    def grad_gram(self, x1: tf.Tensor, x2: tf.Tensor, bandwidth: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        x1: [..., n1, D]
        x2: [..., n2, D]
        bandwidth: [..., 1, 1, D]
        returns: [..., n1, n2], [..., n1, n2, D], [..., n1, n2, D]
        """
        x_row = tf.expand_dims(x1, -2)
        x_col = tf.expand_dims(x2, -3)
        g = self.rbf_kernel(x_row, x_col, bandwidth)
        diff = tf.divide(tf.subtract(x_row, x_col), tf.square(bandwidth))
        g_expand = tf.expand_dims(g, axis=-1)
        grad_x2 = g_expand * diff
        grad_x1 = g_expand * (-diff)
        return g, grad_x1, grad_x2

    @staticmethod
    def median_heuristic(x_samples: tf.Tensor, x_basis: tf.Tensor) -> tf.Tensor:
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
        bandwidth = tf.reshape(
            top_k_values[:, :, -1],
            tf.concat([tf.shape(x_samples)[:-2], [1, 1, d]], axis=0),
        )
        bandwidth *= tf.cast(d, bandwidth.dtype) ** 0.5
        bandwidth += tf.cast((bandwidth < 1e-6), bandwidth.dtype)
        return bandwidth


class SSGE(AbstractScoreEstimator):
    def __init__(
        self,
        n_eigen: int | None = 6,
        eta: float = 1e-3,
        n_eigen_threshold: float | None = None,
        bandwidth: float | None = 2.0,
    ) -> None:
        self.n_eigen_threshold = n_eigen_threshold
        self.bandwidth = bandwidth
        self.n_eigen = n_eigen
        self.eta = eta

    def nystrom_ext(
        self,
        s: tf.Tensor,
        x: tf.Tensor,
        eigen_vectors: tf.Tensor,
        eigen_values: tf.Tensor,
        bandwidth: tf.Tensor,
    ) -> tf.Tensor:
        """
        s: [..., m, d]
        index_points: [..., n, d]
        eigen_vectors: [..., m, n_eigen]
        eigen_values: [..., n_eigen]
        returns: [..., n, n_eigen], by default n_eigen=m.
        """
        m = tf.shape(s)[-2]
        kxq = self.gram(x, s, bandwidth)
        ret = tf.matmul(kxq, eigen_vectors)
        ret *= tf.cast(m, ret.dtype) ** 0.5 / tf.expand_dims(eigen_values, axis=-2)
        return ret

    def estimate_gradients(self, s: tf.Tensor, x: tf.Tensor | None = None) -> tf.Tensor:
        perm = (1, 0, 2)
        if x is not None:
            return tf.transpose(self.__call__(tf.transpose(s, perm), tf.transpose(x, perm)), perm)
        else:
            return tf.transpose(self.__call__(tf.transpose(s, perm)), perm)

    def __call__(self, s: tf.Tensor, x: tf.Tensor | None = None) -> tf.Tensor:
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
        kq, grad_k1, grad_k2 = self.grad_gram(s, s, length_scale)
        kq += self.eta * tf.eye(m)
        eigen_values, eigen_vectors = tf.linalg.eigh(kq)
        if (self.n_eigen is None) and (self.n_eigen_threshold is not None):
            eigen_arr = tf.reduce_mean(tf.reshape(eigen_values, [-1, m]), axis=0)
            eigen_arr = tf.reverse(eigen_arr, axis=[-1])
            eigen_arr /= tf.reduce_sum(eigen_arr)
            eigen_cum = tf.cumsum(eigen_arr, axis=-1)
            eigen_les = tf.cast(tf.less(eigen_cum, self.n_eigen_threshold), tf.int32)
            self.n_eigen = tf.reduce_sum(eigen_les)
        if self.n_eigen is not None:
            eigen_values = eigen_values[..., -self.n_eigen :]
            eigen_vectors = eigen_vectors[..., -self.n_eigen :]
        eigen_ext = self.nystrom_ext(s, x, eigen_vectors, eigen_values, length_scale)
        grad_k1_avg = tf.reduce_mean(grad_k1, axis=-3)
        beta = -tf.matmul(eigen_vectors, grad_k1_avg, transpose_a=True)
        beta *= tf.cast(m, beta.dtype) ** 0.5 / tf.expand_dims(eigen_values, -1)
        grads = tf.matmul(eigen_ext, beta)
        return grads


class GaussianPosterior(tf.Module):
    def __init__(self, stacked_nn_init_params: tf.Tensor, likelihood_param_size: int) -> None:
        super().__init__()

        nn_params_mean = tf.reduce_mean(stacked_nn_init_params, axis=0)

        nn_params_std = tf.math.reduce_std(stacked_nn_init_params, axis=0)
        nn_params_std = tf.maximum(nn_params_std, 0.1)  # Increased from 0.05 to 0.1

        lk_mean = tf.ones(likelihood_param_size) * tf.math.log(0.02)  # Even smaller noise for precise patterns
        lk_std = tf.ones(likelihood_param_size) * 0.05

        self.mean = tf.Variable(tf.concat([nn_params_mean, lk_mean], axis=0))
        self.log_std = tf.Variable(tf.math.log(tf.concat([nn_params_std, lk_std], axis=0)))

    @property
    def stddev(self) -> tf.Tensor:
        return tf.exp(tf.clip_by_value(self.log_std, -7.0, 7.0))

    @property
    def dist(self) -> tfd.Independent:
        return tfd.Independent(tfd.Normal(self.mean, self.stddev), 1)

    def sample(self, size: list[int], seed: int | None = None) -> tf.Tensor:
        if seed is None:
            return self.dist.sample(size)
        return self.dist.sample(size, seed=seed)

    def log_prob(self, param_values: tf.Tensor) -> tf.Tensor:
        return self.dist.log_prob(param_values)


class GaussianPrior(tf.Module):
    def __init__(
        self,
        nn_param_size: int,
        likelihood_param_size: int = 0,
        nn_prior_std: float = 1.0,
        likelihood_prior_mean: float = math.log(0.05),
        likelihood_prior_std: float = 0.01,
    ) -> None:
        super().__init__()

        nn_prior_mean = tf.zeros(nn_param_size)
        nn_prior_std = tf.ones(nn_param_size) * nn_prior_std

        likelihood_prior_mean = tf.ones(likelihood_param_size) * likelihood_prior_mean
        likelihood_prior_std = tf.ones(likelihood_param_size) * likelihood_prior_std

        prior_mean = tf.concat([nn_prior_mean, likelihood_prior_mean], axis=0)
        prior_std = tf.concat([nn_prior_std, likelihood_prior_std], axis=0)

        self.dist = tfd.Independent(tfd.Normal(prior_mean, prior_std), reinterpreted_batch_ndims=1)

    def sample(self, size: list[int]) -> tf.Tensor:
        return self.dist.sample(size)

    def log_prob(self, param_values: tf.Tensor) -> tf.Tensor:
        return self.dist.log_prob(param_values)


class Dense(tf.Module):
    def __init__(
        self, input_size: int, output_size: int, activation: Callable | None = None, bias: bool = True
    ) -> None:
        super().__init__()
        self.bias = bias
        weight_initializer = keras.initializers.HeUniform(seed=42)
        weight = weight_initializer(shape=[input_size, output_size])
        if bias:
            bias_initializer = keras.initializers.HeUniform(seed=43)
            bias_value = bias_initializer(shape=[output_size])
            self.b = tf.Variable(bias_value)
        self.w = tf.Variable(weight)
        self.activation = activation

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        output = tf.matmul(x, self.w)
        if self.bias:
            output += self.b
        if self.activation is not None:
            output = self.activation(output)
        return output


class BatchedModule(tf.Module):
    """Base class for batched fully connected NNs."""

    def __init__(self, n_batched_models: int | None = None) -> None:
        super().__init__()
        self._variable_sizes = None
        self._parameters_shape = None
        self._n_batched_models = n_batched_models

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    def get_variables_stacked_per_model(self) -> tf.Tensor:
        vectorized_vars = [tf.reshape(v, (1, -1)) for v in self.variables]
        vectorized_vars = tf.concat(vectorized_vars, axis=1)
        return tf.reshape(vectorized_vars, (self._n_batched_models, -1))

    @tf.function
    def _variable_sizes_method(self) -> list[tf.Tensor]:
        if self._variable_sizes is None:
            self._variable_sizes = [tf.size(v) for v in self.variables]
        return self._variable_sizes

    @tf.function
    def _set_variables_vectorized(self, parameters: tf.Tensor) -> None:
        if self._parameters_shape is None:
            self._parameters_shape = parameters.shape

        parameters = tf.reshape(parameters, (-1, 1))
        split = tf.split(parameters, self._variable_sizes_method())

        for v, n_v in zip(self.variables, split, strict=False):
            v.assign(tf.reshape(n_v, v.shape))

    @tf.function
    def concat_and_vectorize_grads(self, gradients: list[tf.Tensor]) -> tf.Tensor:
        vectorized_gradients = tf.concat([tf.reshape(g, (-1, 1)) for g in gradients], axis=0)
        if self._parameters_shape is None:
            return tf.reshape(vectorized_gradients, (self._n_batched_models, -1))
        return tf.reshape(vectorized_gradients, self._parameters_shape)

    @tf.custom_gradient
    def call_parametrized(self, x: tf.Tensor, variables_vectorized: tf.Tensor) -> tuple[tf.Tensor, Callable]:
        self._set_variables_vectorized(variables_vectorized)

        tape = tf.GradientTape(persistent=True)
        with tape:
            tape.watch([x] + list(self.trainable_variables))
            y = self(x)

        def grad_fn(dy, variables):
            with tape:
                tampered_y = y * dy
            grads_x_w = tape.gradient(tampered_y, [x] + list(self.trainable_variables))
            grads_to_input = [
                grads_x_w[0],
                self.concat_and_vectorize_grads(grads_x_w[1:]),
            ]
            return grads_to_input, [None] * len(variables)

        return y, grad_fn


class MLP(BatchedModule):
    def __init__(
        self, input_size: int, output_size: int, hidden_sizes: list[int], activation: Callable | None = None
    ) -> None:
        super().__init__()
        self.n_hidden_layers = len(hidden_sizes)
        self.hidden_layers = []
        for hidden_size in hidden_sizes:
            hidden_layer = Dense(input_size, hidden_size, activation)
            self.hidden_layers.append(hidden_layer)
            input_size = hidden_size
        self.output_layer = Dense(input_size, output_size, bias=False)

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        return self.output_layer(x)


class BatchedMLP(BatchedModule):
    def __init__(
        self,
        n_batched_models: int,
        input_size: int,
        output_size: int,
        hidden_sizes: list[int],
        activation: Callable | None = None,
    ) -> None:
        super().__init__(n_batched_models)
        self.n_batched_models = n_batched_models
        self.models = []
        for i in range(n_batched_models):
            self.models.append(MLP(input_size, output_size, hidden_sizes, activation))

    def __call__(self, inputs: tf.Tensor, batched_input: bool = False) -> tf.Tensor:
        if batched_input:
            tf.assert_equal(tf.rank(inputs), 3)
            tf.assert_equal(inputs.shape[0], self.n_batched_models)
            outputs = tf.stack([self.models[i](tf.gather(inputs, i)) for i in range(self.n_batched_models)])
        else:
            tf.assert_equal(tf.rank(inputs), 2)
            outputs = tf.stack([self.models[i](inputs) for i in range(self.n_batched_models)])
        return outputs


def tensor_squared_norm(x1: tf.Tensor, x2: tf.Tensor, axis: tuple[int, int] = (-2, -1)) -> tf.Tensor:
    """Computes squared norm of n-dimensional tensor"""
    x1_expanded = tf.expand_dims(x1, 1)
    x2_expanded = tf.expand_dims(x2, 0)

    diff = x1_expanded - x2_expanded
    squared_norm = tf.reduce_sum(tf.square(diff), axis=-1)
    return squared_norm


class Affine:
    """Y = transform(X) = normalization_std @ X + normalization_mean"""

    def __init__(
        self, normalization_mean: Union[tf.Tensor, tf.Operation], normalization_std: Union[tf.Tensor, tf.Operation]
    ) -> None:
        self.loc_tensor = tf.cast(normalization_mean, dtype=dtype)
        self.scale_tensor = tf.cast(normalization_std, dtype=dtype)

        shift = tfb.Shift(self.loc_tensor)
        if tf.size(self.scale_tensor) == 1:
            scale = tfb.Scale(self.scale_tensor)
        else:
            scale = tfb.ScaleMatvecDiag(self.scale_tensor)
        self.transform = tfb.Chain([shift, scale])

    def __call__(self, base_dist: tfd.Distribution) -> tfd.Distribution:
        transformed_dist = self.transform(base_dist)

        mean, stddev, var = base_dist.mean(), base_dist.stddev(), base_dist.variance()
        transformed_dist.mean = self.transform(mean)
        transformed_dist.stddev = exp(log(stddev) + log(self.scale_tensor))
        transformed_dist.variance = exp(log(var) + 2 * log(self.scale_tensor))
        return transformed_dist
