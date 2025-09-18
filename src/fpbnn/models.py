import sys

import tensorflow as tf
import tensorflow_probability as tfp

from fpbnn.configs import Config
from fpbnn.modules import GaussianPrior
from fpbnn.regression import AbstractBayesianRegression, FunctionalPrior, FunctionalRegressionModel
from fpbnn.utils import keys_for_module


class MLP(AbstractBayesianRegression):
    """Multi-Layer Perceptron, used as a baseline model."""

    def __init__(self, train_data: tuple[tf.Tensor, tf.Tensor], config: Config) -> None:
        super().__init__(
            train_data=train_data,
            experiment=config.experiment,
            n_iter=config.train.n_iter,
            batch_size=config.train.batch_size,
            n_particles=config.n_particles,
            activation=config.arch.activation,
            width=config.arch.width,
            depth=config.arch.depth,
            bandwidth=config.arch.bandwidth,
            nn_prior_std=config.arch.nn_prior_std,
            likelihood_prior_mean=config.like.prior_mean,
            likelihood_prior_std=config.like.prior_std,
            learning_rate=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
            coeff_prior=config.train.coeff_prior,
            noise_std=config.like.noise_std,
            image_format=config.runtime.image_format,
            verbose=config.runtime.verbose,
            normalize_train_data=config.train.normalize_train_data,
            use_wandb=config.runtime.use_wandb,
            enable_early_stopping=config.train.enable_early_stopping,
            early_stopping_patience=config.train.early_stopping_patience,
            early_stopping_metric=config.train.early_stopping_metric,
            early_stopping_mode=config.train.early_stopping_mode,
            early_stopping_min_delta=config.train.early_stopping_min_delta,
        )

    @property
    def name(self) -> str:
        return "mlp"

    def predict(self, x: tf.Tensor, seed: int | None = None) -> tuple[tf.Tensor, tfp.distributions.Distribution]:
        x = self._broadcast_and_cast_dtype(x)
        x = self.normalize(x)
        y_pred = self.nn(x)
        if tf.rank(y_pred) == 2:
            y_pred = y_pred[None, ...]
        likelihood_std = tf.ones([tf.shape(y_pred)[0], 1], y_pred.dtype) * 0.1
        pred_dist = self._predictive_mixture(y_pred, likelihood_std)
        y_pred = self._unnormalize_preds(y_pred)
        pred_dist = self._unnormalize_predictive_dist(pred_dist)
        return y_pred, pred_dist

    @tf.function
    def __call__(self, x_batch: tf.Tensor, y_batch: tf.Tensor) -> None:
        with tf.GradientTape() as tape:
            tape.watch(self.nn.trainable_variables)
            y_pred = self.nn(x_batch)
            if tf.rank(y_pred) == 3:
                y_pred = tf.reduce_mean(y_pred, 0)
            loss = tf.reduce_mean(tf.square(y_batch - y_pred))
        grads = tape.gradient(loss, self.nn.trainable_variables)
        grads = [(g, v) for g, v in zip(grads, self.nn.trainable_variables, strict=False) if g is not None]
        if grads:
            self.optimizer.apply_gradients(grads)

    @classmethod
    def from_config(cls, train_data: tuple[tf.Tensor, tf.Tensor], config: Config) -> "MLP":
        return cls(train_data=train_data, config=config)


class FVI(FunctionalRegressionModel):
    """Functional Variational Inference: https://arxiv.org/abs/1903.05779"""

    def __init__(
        self,
        functional_prior: FunctionalPrior,
        train_data: tuple[tf.Tensor, tf.Tensor],
        config: Config,
    ) -> None:
        super().__init__(
            functional_prior=functional_prior,
            train_data=train_data,
            experiment=config.experiment,
            n_iter=config.train.n_iter,
            batch_size=config.train.batch_size,
            n_particles=config.n_particles,
            activation=config.arch.activation,
            width=config.arch.width,
            depth=config.arch.depth,
            bandwidth=config.arch.bandwidth,
            nn_prior_std=config.arch.nn_prior_std,
            likelihood_prior_mean=config.like.prior_mean,
            likelihood_prior_std=config.like.prior_std,
            learning_rate=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
            coeff_prior=config.train.coeff_prior,
            noise_std=config.like.noise_std,
            image_format=config.runtime.image_format,
            verbose=config.runtime.verbose,
            normalize_train_data=config.train.normalize_train_data,
            use_wandb=config.runtime.use_wandb,
            enable_early_stopping=config.train.enable_early_stopping,
            early_stopping_patience=config.train.early_stopping_patience,
            early_stopping_metric=config.train.early_stopping_metric,
            early_stopping_mode=config.train.early_stopping_mode,
            early_stopping_min_delta=config.train.early_stopping_min_delta,
            ssge_bandwidth=config.ssge.bandwidth,
            ssge_n_eigen=config.ssge.n_eigen,
            ssge_eta=config.ssge.eta,
            coeff_entropy=config.ssge.coeff_entropy,
            coeff_cross_entropy=config.ssge.coeff_cross_entropy,
            prior_noise_std=config.ssge.prior_noise_std,
            buffer_size=config.train.buffer_size,
        )
        self._likelihood_prior = GaussianPrior(
            nn_param_size=0,
            nn_prior_std=0,
            likelihood_param_size=self.likelihood_param_size,
            likelihood_prior_mean=self.likelihood_prior_mean,
            likelihood_prior_std=self.likelihood_prior_std,
        )

        self._train_step = tf.Variable(0, trainable=False, dtype=tf.int32)
        self._kl_anneal_steps = tf.constant(max(1, self.n_iter // 10), dtype=tf.int32)

    @property
    def name(self) -> str:
        return "fvi"

    @classmethod
    def from_config(
        cls,
        functional_prior: FunctionalPrior,
        train_data: tuple[tf.Tensor, tf.Tensor],
        config: Config,
    ) -> "FVI":
        return cls(functional_prior=functional_prior, train_data=train_data, config=config)

    def predict(self, x: tf.Tensor, seed: int | None = None) -> tuple[tf.Tensor, tfp.distributions.Distribution]:
        x = self._broadcast_and_cast_dtype(x)
        x = self.normalize(x)

        sampled_params = self.posterior.sample([self.n_particles], seed=int(seed) if seed is not None else None)
        nn_params, likelihood_std = self.split_params(sampled_params)
        y_pred = self.nn.call_parametrized(x, nn_params)
        pred_dist = self._predictive_mixture(y_pred, likelihood_std)

        y_pred = self._unnormalize_preds(y_pred)
        pred_dist = self._unnormalize_predictive_dist(pred_dist)
        return y_pred, pred_dist

    @tf.function
    def __call__(self, x_batch: tf.Tensor, y_batch: tf.Tensor) -> None:
        x_prior = next(self.x_prior)
        y_prior = next(self.y_prior)
        x_concat = tf.concat([x_batch, x_prior], axis=0)
        batch_size = tf.shape(x_batch)[0]

        with tf.GradientTape() as tape:
            tape.watch(self.posterior.trainable_variables)

            samples_posterior = self.posterior.sample([self.n_particles])
            nn_params, likelihood_std = self.split_params(samples_posterior)
            f_all = self.nn.call_parametrized(x_concat, nn_params)
            f_batch = f_all[:, :batch_size]

            ll_sum = self.ll(y_batch, f_batch, likelihood_std, reduction="sum")
            num_train = tf.cast(self.num_train_samples, ll_sum.dtype)
            batch_float = tf.cast(batch_size, ll_sum.dtype)
            like_term = -(num_train / batch_float) * ll_sum

            g_q = tf.stop_gradient(self.ssge.estimate_gradients(s=f_all))
            y_prior_combined = tf.concat([f_batch, y_prior], axis=1)
            g_p = tf.stop_gradient(self.ssge.estimate_gradients(s=y_prior_combined, x=f_all))
            kl_sur = tf.reduce_sum((g_q - g_p) * f_all) / tf.cast(tf.shape(f_all)[1], f_all.dtype)

            frac = tf.cast(tf.minimum(self._train_step, self._kl_anneal_steps), tf.float32) / tf.cast(
                tf.maximum(self._kl_anneal_steps, 1), tf.float32
            )
            lambda_kl = frac * tf.cast(self.coeff_prior, tf.float32)

            loss = like_term + lambda_kl * kl_sur

            coeff_noise_prior = tf.constant(0.0, dtype=loss.dtype)
            if coeff_noise_prior > 0.0:
                loss += -coeff_noise_prior * tf.reduce_mean(self._likelihood_prior.log_prob(likelihood_std))

        gradients = tape.gradient(loss, self.posterior.trainable_variables)
        grads_and_vars = []
        for g, v in zip(gradients, self.posterior.trainable_variables, strict=False):
            if g is None:
                continue
            num_elems = v.shape.num_elements() if v.shape is not None else None
            if num_elems is not None and num_elems == 0:
                continue
            grads_and_vars.append((g, v))
        if grads_and_vars:
            self.optimizer.apply_gradients(grads_and_vars)

        self._train_step.assign_add(1)


class SVGD(AbstractBayesianRegression):
    """Stein Variational Gradient Descent: https://arxiv.org/abs/1608.04471"""

    def __init__(self, train_data: tuple[tf.Tensor, tf.Tensor], config: Config) -> None:
        super().__init__(
            train_data=train_data,
            experiment=config.experiment,
            n_iter=config.train.n_iter,
            batch_size=config.train.batch_size,
            n_particles=config.n_particles,
            activation=config.arch.activation,
            width=config.arch.width,
            depth=config.arch.depth,
            bandwidth=config.arch.bandwidth,
            nn_prior_std=config.arch.nn_prior_std,
            likelihood_prior_mean=config.like.prior_mean,
            likelihood_prior_std=config.like.prior_std,
            learning_rate=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
            coeff_prior=config.train.coeff_prior,
            noise_std=config.like.noise_std,
            image_format=config.runtime.image_format,
            verbose=config.runtime.verbose,
            normalize_train_data=config.train.normalize_train_data,
            use_wandb=config.runtime.use_wandb,
            enable_early_stopping=config.train.enable_early_stopping,
            early_stopping_patience=config.train.early_stopping_patience,
            early_stopping_metric=config.train.early_stopping_metric,
            early_stopping_mode=config.train.early_stopping_mode,
            early_stopping_min_delta=config.train.early_stopping_min_delta,
        )
        likelihood_params = tf.ones([self.n_particles, self.likelihood_param_size]) * self.likelihood_prior_mean
        self.particles = tf.Variable(tf.concat([self.nn_params, likelihood_params], -1))

    @classmethod
    def from_config(cls, train_data: tuple[tf.Tensor, tf.Tensor], config: Config) -> "SVGD":
        return cls(train_data=train_data, config=config)

    @property
    def name(self) -> str:
        return "svgd"

    def predict(self, x: tf.Tensor, seed: int | None = None) -> tuple[tf.Tensor, tfp.distributions.Distribution]:
        x = self._broadcast_and_cast_dtype(x)
        x = self.normalize(x)
        nn_params, likelihood_std = self.split_params(self.particles)
        output_pred = self.nn.call_parametrized(x, nn_params)
        output_dist = self._predictive_mixture(output_pred, likelihood_std)
        output_pred = self._unnormalize_preds(output_pred)
        output_dist = self._unnormalize_predictive_dist(output_dist)
        return output_pred, output_dist

    @tf.function
    def __call__(self, x_batch: tf.Tensor, y_batch: tf.Tensor) -> None:
        batch_size = tf.cast(tf.shape(x_batch)[0], tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(self.particles)
            nn_params, likelihood_std = self.split_params(self.particles)
            y_pred = self.nn.call_parametrized(x_batch, nn_params)
            ll_sum = self.ll(y_batch, y_pred, likelihood_std, reduction="sum")
            num_train = tf.cast(self.num_train_samples, ll_sum.dtype)
            batch_float = tf.cast(batch_size, ll_sum.dtype)
            log_like = (num_train / batch_float) * ll_sum
            log_prior = self.prior.log_prob(self.particles)
            log_post = log_like + log_prior

        score = tape.gradient(log_post, self.particles)
        if score is None:
            score = tf.zeros_like(self.particles)

        with tf.GradientTape() as kernel_tape:
            kernel_tape.watch(self.particles)
            kernel_matrix = self.kernel.matrix(self.particles, tf.stop_gradient(self.particles))
        grad_kernel = kernel_tape.gradient(tf.reduce_sum(kernel_matrix, axis=1), self.particles)
        if grad_kernel is None:
            grad_kernel = tf.zeros_like(self.particles)

        phi = (kernel_matrix @ score + grad_kernel) / tf.cast(self.n_particles, self.particles.dtype)
        self.optimizer.apply_gradients([(-phi, self.particles)])


class VI(AbstractBayesianRegression):
    """Bayesian Neural Network trained using Variational Inference: https://arxiv.org/abs/1505.05424"""

    def __init__(
        self,
        train_data: tuple[tf.Tensor, tf.Tensor],
        config: Config,
    ) -> None:
        super().__init__(
            train_data=train_data,
            experiment=config.experiment,
            n_iter=config.train.n_iter,
            batch_size=config.train.batch_size,
            n_particles=config.n_particles,
            activation=config.arch.activation,
            width=config.arch.width,
            depth=config.arch.depth,
            bandwidth=config.arch.bandwidth,
            nn_prior_std=config.arch.nn_prior_std,
            likelihood_prior_mean=config.like.prior_mean,
            likelihood_prior_std=config.like.prior_std,
            learning_rate=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
            coeff_prior=config.train.coeff_prior,
            noise_std=config.like.noise_std,
            image_format=config.runtime.image_format,
            verbose=config.runtime.verbose,
            normalize_train_data=config.train.normalize_train_data,
            use_wandb=config.runtime.use_wandb,
            enable_early_stopping=config.train.enable_early_stopping,
            early_stopping_patience=config.train.early_stopping_patience,
            early_stopping_metric=config.train.early_stopping_metric,
            early_stopping_mode=config.train.early_stopping_mode,
            early_stopping_min_delta=config.train.early_stopping_min_delta,
        )

    @classmethod
    def from_config(cls, train_data: tuple[tf.Tensor, tf.Tensor], config: Config) -> "VI":
        return cls(train_data=train_data, config=config)

    @property
    def name(self) -> str:
        return "vi"

    def predict(self, x: tf.Tensor, seed: int | None = None) -> tuple[tf.Tensor, tfp.distributions.Distribution]:
        x = self._broadcast_and_cast_dtype(x)
        x = self.normalize(x)
        nn_params, likelihood_std = self.split_params(
            self.posterior.sample([self.n_particles], seed=int(seed) if seed is not None else None)
        )
        output_pred = self.nn.call_parametrized(x, nn_params)
        output_dist = self._predictive_mixture(output_pred, likelihood_std)
        output_pred = self._unnormalize_preds(output_pred)
        output_dist = self._unnormalize_predictive_dist(output_dist)
        return output_pred, output_dist

    @tf.function
    def __call__(self, x_batch: tf.Tensor, y_batch: tf.Tensor) -> None:
        with tf.GradientTape() as tape:
            tape.watch(self.posterior.trainable_variables)
            samples = self.posterior.sample([self.n_particles])
            nn_params, likelihood_std = self.split_params(samples)
            y_pred = self.nn.call_parametrized(x_batch, nn_params)

            ll_sum = self.ll(y_batch, y_pred, likelihood_std, reduction="sum")
            num_train = tf.cast(self.num_train_samples, ll_sum.dtype)
            batch_float = tf.cast(tf.shape(x_batch)[0], ll_sum.dtype)
            neg_log_like = -(num_train / batch_float) * (ll_sum / tf.cast(self.n_particles, ll_sum.dtype))

            kl = tfp.distributions.kl_divergence(self.posterior.dist, self.prior.dist)

            if not hasattr(self, "_vi_step"):
                self._vi_step = tf.Variable(0, trainable=False, dtype=tf.int32)

            warmup_steps = tf.cast(self.n_iter * 0.2, tf.int32)
            step_ratio = tf.cast(tf.minimum(self._vi_step, warmup_steps), tf.float32) / tf.cast(
                warmup_steps, tf.float32
            )
            kl_weight = 0.01 + 0.99 * step_ratio

            loss = neg_log_like + kl_weight * kl

        grads = tape.gradient(loss, self.posterior.trainable_variables)

        clip_value = 5.0 * (1.0 - 0.8 * step_ratio)
        grads = [tf.clip_by_value(g, -clip_value, clip_value) for g in grads if g is not None]
        grads = [(g, v) for g, v in zip(grads, self.posterior.trainable_variables, strict=False) if g is not None]
        if grads:
            self.optimizer.apply_gradients(grads)

        self._vi_step.assign_add(1)


class FSVGD(FunctionalRegressionModel):
    """Functional Stein Variational Gradient Descent: https://arxiv.org/abs/1902.09754"""

    def __init__(
        self,
        functional_prior: FunctionalPrior,
        train_data: tuple[tf.Tensor, tf.Tensor],
        config: Config,
    ) -> None:
        super().__init__(
            functional_prior=functional_prior,
            train_data=train_data,
            experiment=config.experiment,
            n_iter=config.train.n_iter,
            batch_size=config.train.batch_size,
            n_particles=config.n_particles,
            activation=config.arch.activation,
            width=config.arch.width,
            depth=config.arch.depth,
            bandwidth=config.arch.bandwidth,
            nn_prior_std=config.arch.nn_prior_std,
            likelihood_prior_mean=config.like.prior_mean,
            likelihood_prior_std=config.like.prior_std,
            learning_rate=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
            coeff_prior=config.train.coeff_prior,
            noise_std=config.like.noise_std,
            image_format=config.runtime.image_format,
            verbose=config.runtime.verbose,
            normalize_train_data=config.train.normalize_train_data,
            use_wandb=config.runtime.use_wandb,
            enable_early_stopping=config.train.enable_early_stopping,
            early_stopping_patience=config.train.early_stopping_patience,
            early_stopping_metric=config.train.early_stopping_metric,
            early_stopping_mode=config.train.early_stopping_mode,
            early_stopping_min_delta=config.train.early_stopping_min_delta,
            ssge_bandwidth=config.ssge.bandwidth,
            ssge_n_eigen=config.ssge.n_eigen,
            ssge_eta=config.ssge.eta,
            coeff_entropy=config.ssge.coeff_entropy,
            coeff_cross_entropy=config.ssge.coeff_cross_entropy,
            prior_noise_std=config.ssge.prior_noise_std,
            buffer_size=config.train.buffer_size,
        )
        self._config = config

        perturbations = tf.random.normal(tf.shape(self.nn_params), mean=0.0, stddev=self.nn_prior_std * 0.1)
        nn_params_randomized = self.nn_params + perturbations

        likelihood_params = tf.random.normal(
            [self.n_particles, self.likelihood_param_size],
            mean=self.likelihood_prior_mean,
            stddev=self.likelihood_prior_std,
        )
        self.particles = tf.Variable(tf.concat([nn_params_randomized, likelihood_params], -1))

    @classmethod
    def from_config(
        cls,
        functional_prior: FunctionalPrior,
        train_data: tuple[tf.Tensor, tf.Tensor],
        config: Config,
    ) -> "FSVGD":
        return cls(functional_prior=functional_prior, train_data=train_data, config=config)

    @property
    def name(self) -> str:
        return "fsvgd"

    def predict(self, x: tf.Tensor, seed: int | None = None) -> tuple[tf.Tensor, tfp.distributions.Distribution]:
        x = self._broadcast_and_cast_dtype(x)
        x = self.normalize(x)
        nn_params, likelihood_std = self.split_params(self.particles)
        output_pred = self.nn.call_parametrized(x, nn_params)
        output_dist = self._predictive_mixture(output_pred, likelihood_std)
        output_pred = self._unnormalize_preds(output_pred)
        output_dist = self._unnormalize_predictive_dist(output_dist)
        return output_pred, output_dist

    def ll(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        likelihood_std: tf.Tensor,
        reduction: str = "sum",
    ) -> tf.Tensor:
        """
        Compute log likelihood for particles.
        y_true: (batch_size, output_dim)
        y_pred: (n_particles, batch_size, output_dim)
        likelihood_std: (n_particles, 1)
        """
        y_true_expanded = tf.expand_dims(y_true, 0)
        y_true_expanded = tf.tile(y_true_expanded, [self.n_particles, 1, 1])

        likelihood_std = tf.expand_dims(likelihood_std, axis=1)
        likelihood = tfp.distributions.Independent(
            tfp.distributions.Normal(y_pred, likelihood_std),
            reinterpreted_batch_ndims=1,
        )
        log_likelihood = likelihood.log_prob(y_true_expanded)

        if reduction == "sum":
            return tf.reduce_sum(log_likelihood)
        elif reduction == "mean":
            return tf.reduce_mean(log_likelihood)
        else:
            return log_likelihood

    def _median_heuristic(self, x: tf.Tensor) -> tf.Tensor:
        """
        Compute bandwidth using median heuristic for RBF kernel.
        x: [n_particles, feature_dim] - flattened function vectors
        Returns: scalar bandwidth
        """
        x_sq = tf.reduce_sum(x**2, axis=1, keepdims=True)
        pairwise_sq_dists = x_sq + tf.transpose(x_sq) - 2 * tf.matmul(x, x, transpose_b=True)

        pairwise_sq_dists = tf.maximum(pairwise_sq_dists, 0.0)

        n = tf.shape(x)[0]
        mask = tf.linalg.band_part(tf.ones((n, n), dtype=tf.bool), 0, -1) & ~tf.eye(n, dtype=tf.bool)
        sq_dists = tf.boolean_mask(pairwise_sq_dists, mask)

        median_sq_dist = tfp.stats.percentile(sq_dists, 50.0)
        median_sq_dist = tf.maximum(median_sq_dist, 1e-6)
        bandwidth = tf.sqrt(median_sq_dist)

        bandwidth = tf.clip_by_value(bandwidth, 1e-3, 1e3)
        return bandwidth

    def _rbf_kernel_matrix(self, x: tf.Tensor, y: tf.Tensor, bandwidth: tf.Tensor) -> tf.Tensor:
        """
        Compute RBF kernel matrix between X and Y.
        x: [n, d], y: [m, d]
        Returns: [n, m] kernel matrix
        """
        x_sq = tf.reduce_sum(x**2, axis=1, keepdims=True)
        y_sq = tf.reduce_sum(y**2, axis=1, keepdims=True)
        sq_dists = x_sq + tf.transpose(y_sq) - 2 * tf.matmul(x, y, transpose_b=True)

        sq_dists = tf.maximum(sq_dists, 0.0)

        exponent = -sq_dists / (2 * bandwidth**2)
        exponent = tf.clip_by_value(exponent, -50.0, 0.0)
        return tf.exp(exponent)

    @tf.function
    def __call__(self, x_batch: tf.Tensor, y_batch: tf.Tensor) -> None:
        x_prior = next(self.x_prior)
        y_prior = next(self.y_prior)

        with tf.GradientTape() as weight_space_tape:
            weight_space_tape.watch(self.particles)
            nn_params, likelihood_std = self.split_params(self.particles)
            x_concat = tf.concat([x_batch, x_prior], axis=0)
            y_concat = self.nn.call_parametrized(x_concat, nn_params)
            batch_length = tf.shape(x_batch)[0]
            y_batch_pred = y_concat[:, :batch_length]
            y_prior_pred = y_concat[:, batch_length:]

        y_true_expanded = tf.expand_dims(y_batch, 0)
        likelihood_scale = tf.reshape(likelihood_std, [self.n_particles, 1, -1])
        likelihood_scale = tf.broadcast_to(likelihood_scale, tf.shape(y_batch_pred))
        grad_likelihood = (y_true_expanded - y_batch_pred) / tf.square(likelihood_scale)

        batch_size = tf.cast(tf.shape(y_batch)[0], grad_likelihood.dtype)
        num_train = tf.cast(self.num_train_samples, grad_likelihood.dtype)
        grad_likelihood = grad_likelihood * (num_train / batch_size)

        grad_prior = self.ssge.estimate_gradients(s=y_prior, x=y_prior_pred)
        grad_prior = grad_prior * self.coeff_cross_entropy

        function_values = tf.concat([y_batch_pred, y_prior_pred], axis=1)
        gradients_combined = tf.concat([grad_likelihood, grad_prior], axis=1)

        function_values_flat = tf.reshape(function_values, [self.n_particles, -1])
        gradients_flat = tf.reshape(gradients_combined, [self.n_particles, -1])

        bandwidth = tf.stop_gradient(self._median_heuristic(function_values_flat))
        kernel_matrix = self._rbf_kernel_matrix(function_values_flat, tf.stop_gradient(function_values_flat), bandwidth)

        ones_vector = tf.ones([tf.shape(kernel_matrix)[1], 1], dtype=kernel_matrix.dtype)
        sum_kernel_rows = tf.matmul(kernel_matrix, ones_vector)
        kernel_function_product = tf.matmul(kernel_matrix, function_values_flat)
        grad_kernel = -((sum_kernel_rows * function_values_flat) - kernel_function_product) / tf.square(bandwidth)

        phi_vector = (tf.matmul(kernel_matrix, gradients_flat) + grad_kernel) / tf.cast(
            self.n_particles, function_values_flat.dtype
        )

        phi_vector = tf.where(tf.math.is_finite(phi_vector), phi_vector, tf.zeros_like(phi_vector))
        phi_vector = tf.clip_by_value(phi_vector, -1e6, 1e6)

        phi = tf.reshape(phi_vector, tf.shape(function_values))

        gradients = weight_space_tape.gradient(y_concat, self.particles, output_gradients=phi)

        if gradients is not None:
            gradients = tf.where(tf.math.is_finite(gradients), gradients, tf.zeros_like(gradients))
            gradients = tf.clip_by_value(gradients, -1e3, 1e3)
            self.optimizer.apply_gradients([(-gradients, self.particles)])


available_models = keys_for_module(sys.modules[__name__], "lower")
