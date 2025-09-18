import importlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import pyplot as plt
from tensorflow import keras

import wandb
from fpbnn.envs import AbstractMujocoEnv
from fpbnn.modules import SSGE, Affine, BatchedMLP, GaussianPosterior, GaussianPrior
from fpbnn.utils import paths

tfd = tfp.distributions
tfk = tfp.math.psd_kernels


class AbstractBayesianRegression(ABC):
    """Base class for bayesian regression."""

    def __init__(
        self,
        train_data: tuple[tf.Tensor, tf.Tensor],
        experiment: str,
        n_iter: float = 1e4,
        batch_size: int = 8,
        n_particles: int = 10,
        activation: str = "elu",
        width: int = 32,
        depth: int = 4,
        bandwidth: float = 1e-2,
        nn_prior_std: float = 1.0,
        likelihood_prior_mean: float = 1e-2,
        likelihood_prior_std: float = 1e-2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        coeff_prior: float = 1.0,
        noise_std: float = 1e-2,
        image_format: str = "png",
        figure_dpi: int = 200,
        verbose: int = 0,
        normalize_train_data: bool = True,
        use_wandb: bool = True,
        enable_early_stopping: bool = True,
        early_stopping_patience: int = 1000,
        early_stopping_metric: str = "val_nll",
        early_stopping_mode: str = "min",
        early_stopping_min_delta: float = 0.0,
    ) -> None:
        assert nn_prior_std > 0.0
        assert likelihood_prior_std > 0.0
        x_train, y_train = train_data
        self.image_format = image_format
        self.figure_dpi = int(figure_dpi)
        assert self.figure_dpi > 0
        self.num_train_samples = len(x_train)

        assert batch_size > 0, "Batch size must be positive"
        self.batch_size = min(int(batch_size), self.num_train_samples)

        self.n_iter = int(n_iter)
        self.likelihood_prior_std = likelihood_prior_std
        self.likelihood_prior_mean = np.log(likelihood_prior_mean)
        self.nn_prior_std = nn_prior_std
        self.bandwidth = bandwidth

        self.kernel = tfk.ExponentiatedQuadratic(length_scale=self.bandwidth)
        activation_map = {
            "elu": tf.nn.elu,
            "relu": tf.nn.relu,
            "tanh": tf.nn.tanh,
            "sigmoid": tf.nn.sigmoid,
            "swish": tf.nn.swish,
            "gelu": tf.nn.gelu,
            "leaky_relu": tf.nn.leaky_relu,
            "softplus": tf.nn.softplus,
        }
        if activation not in activation_map:
            raise ValueError(f"Unsupported activation '{activation}'. Available: {list(activation_map.keys())}")
        self.activation = activation_map[activation]
        self.coeff_prior = coeff_prior / self.num_train_samples
        self.n_particles = n_particles
        self.hidden_layer_sizes = tuple([width] * depth)
        self.noise_std = noise_std

        x_train, y_train = self._broadcast_and_cast_dtype(x_train, y_train)
        self.x_mean = tf.reduce_mean(x_train, 0)
        self.y_mean = tf.reduce_mean(y_train, 0)
        eps = tf.constant(1e-6, tf.float32)
        self.x_std = tf.maximum(tf.math.reduce_std(x_train, axis=0), eps)
        self.y_std = tf.maximum(tf.math.reduce_std(y_train, axis=0), eps)

        if normalize_train_data:
            self.x_train, self.y_train = self.normalize(x_train, y_train)
        else:
            self.x_train, self.y_train = x_train, y_train
        self.y_train += tf.random.normal(tf.shape(self.y_train), 0.0, self.noise_std)

        self.unnormalized_x_train = x_train
        self.unnormalized_y_train = self._unnormalize_preds(self.y_train)

        self.affine_transform = Affine(self.y_mean, self.y_std)

        self.input_size = self.x_train.shape[-1]
        self.output_size = self.y_train.shape[-1]

        self.likelihood_param_size = self.output_size

        self.optimizer = keras.optimizers.AdamW(
            weight_decay=weight_decay,
            learning_rate=learning_rate,
        )

        self.nn = BatchedMLP(
            self.n_particles,
            self.input_size,
            self.output_size,
            self.hidden_layer_sizes,
            self.activation,
        )
        self.nn_params = self.nn.get_variables_stacked_per_model()
        self.nn_param_size = self.nn_params.shape[-1]
        self.prior = GaussianPrior(
            nn_param_size=self.nn_param_size,
            nn_prior_std=self.nn_prior_std,
            likelihood_param_size=self.likelihood_param_size,
            likelihood_prior_mean=self.likelihood_prior_mean,  # use logged mean
            likelihood_prior_std=likelihood_prior_std,
        )
        self.posterior = GaussianPosterior(self.nn_params, self.likelihood_param_size)

        self.experiment = experiment

        self._viz_seed_base: int = 1337

        self.wandb_enabled = use_wandb
        if self.wandb_enabled and not wandb.run:
            wandb.init(
                project="fpbnn-experiments",
                name=f"{self.experiment}-{self.name}",
                config={
                    "experiment": self.experiment,
                    "model": self.name,
                    "n_particles": self.n_particles,
                    "n_iter": self.n_iter,
                    "batch_size": self.batch_size,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "bandwidth": self.bandwidth,
                    "nn_prior_std": self.nn_prior_std,
                },
            )

        self.verbose = verbose

        self.enable_early_stopping = bool(enable_early_stopping)
        self.early_stopping_patience = int(early_stopping_patience)
        self.early_stopping_metric = str(early_stopping_metric)
        self.early_stopping_mode = str(early_stopping_mode)
        self.early_stopping_min_delta = float(early_stopping_min_delta)

        self._deferred_plot_cache: list[dict[str, np.ndarray | int]] = []
        self._general_plot_cache: list[dict[str, np.ndarray | int]] = []

    def fit(
        self,
        val_data: tuple[tf.Tensor, tf.Tensor] | None = None,
        test_data: tuple[tf.Tensor, tf.Tensor] | None = None,
        n_plots: int = 50,
        plot: bool = False,
        n_seconds_gif: int = 5,
        show: bool = False,
        report_callback: Callable | None = None,
    ) -> None:
        """
        Fit train to test data.
        Args:
            val_data: tuple (x_val, y_val). If set to None, does not compute metrics for validation data.
            test_data: tuple (x_test, y_test). If set to None, does not compute metrics for test data.
            n_plots: number of plots to produce during training phase. Requires test data not to be None.
            plot: boolean for plotting results. Plots are rendered at the end of training for optimal performance.
            n_seconds_gif: duration of gif of training procedure in seconds.
        """
        from tqdm import trange

        plot_period = self.n_iter // n_plots
        if plot_period == 0:
            plot_period = 1
        eval_period = self.n_iter // 100
        if eval_period == 0:
            eval_period = 1
        plot1d = (self.output_size == 1) and (self.input_size == 1) and plot and (test_data is not None)
        plot_mujoco = self._is_mujoco_environment() and plot
        plot_general = plot and (test_data is not None) and not plot1d and not plot_mujoco

        train_batch_sampler = self._get_batch_sampler()
        if self.verbose < 1:
            pbar = trange(self.n_iter, disable=True)
        else:
            pbar = trange(self.n_iter, desc=f"training {self.name} on {self.experiment}")
        train_eval = (self.unnormalized_x_train, self.unnormalized_y_train)
        message = dict()

        if plot1d or plot_mujoco or plot_general:
            target_folder = paths.figures_for_experiment(self.experiment, self.name)
            if target_folder.is_dir():
                for file_path in target_folder.iterdir():
                    if file_path.suffix == f".{self.image_format}":
                        file_path.unlink()
            target_folder.mkdir(parents=True, exist_ok=True)

        best_metric = float("inf") if self.early_stopping_mode == "min" else -float("inf")
        epochs_without_improvement = 0
        best_state = None
        for i in pbar:
            x_batch, y_batch = next(train_batch_sampler)
            self(x_batch, y_batch)

            if i % eval_period == 0:
                eval_seed = self._predict_seed_for_iteration(i)
                message.update(self.eval_(*train_eval, "train", seed=eval_seed))
                if self.wandb_enabled and wandb.run:
                    wandb.log({"train_nll": message["train_nll"], "train_rmse": message["train_rmse"], "iteration": i})

                if val_data is not None:
                    message.update(self.eval_(*val_data, "val", seed=eval_seed))
                    if self.wandb_enabled and wandb.run:
                        wandb.log({"val_nll": message["val_nll"], "val_rmse": message["val_rmse"], "iteration": i})

                    if self.enable_early_stopping:
                        metric_value = message.get(self.early_stopping_metric)
                        if metric_value is not None:
                            improved = (
                                metric_value < (best_metric - self.early_stopping_min_delta)
                                if self.early_stopping_mode == "min"
                                else metric_value > (best_metric + self.early_stopping_min_delta)
                            )
                            if improved:
                                best_metric = metric_value
                                epochs_without_improvement = 0

                                best_state = self._capture_trainable_state()
                            else:
                                epochs_without_improvement += 1
                                evaluation_patience = max(1, self.early_stopping_patience // eval_period)
                                if epochs_without_improvement >= evaluation_patience:
                                    pbar.set_postfix(message)
                                    break

                if report_callback is not None:
                    payload: dict[str, float] = {"iteration": float(i)}
                    for k, v in message.items():
                        if isinstance(v, (int, float)):
                            payload[k] = float(v)
                    report_callback(payload)

                pbar.set_postfix(message)

            if i % plot_period == 0:
                if plot1d:
                    self._cache_plot_data_1d(test_data, i)
                elif plot_mujoco:
                    self.plot_mujoco_training_frame(i, show=show)
                elif plot_general:
                    self._cache_plot_data_general(test_data, i)

        if best_state is not None:
            self._restore_trainable_state(best_state)

        final_iteration = i  # i is the last iteration that was actually executed
        if final_iteration % plot_period != 0:
            if plot1d:
                self._cache_plot_data_1d(test_data, final_iteration)
            elif plot_mujoco:
                self.plot_mujoco_training_frame(final_iteration, show=show)
            elif plot_general:
                self._cache_plot_data_general(test_data, final_iteration)

        if plot1d and len(self._deferred_plot_cache) > 0:
            self._render_deferred_plots()
        elif plot_mujoco and hasattr(self, "_mujoco_training_cache"):
            if len(self._mujoco_training_cache) > 0:
                self._render_deferred_mujoco_frames()
        elif plot_general and hasattr(self, "_general_plot_cache"):
            if len(self._general_plot_cache) > 0:
                self._render_deferred_general_plots()

        if plot1d:
            self.gif_path = self.make_gif(duration_seconds=n_seconds_gif)
        elif plot_mujoco:
            self.gif_path = self.make_mujoco_training_gif(duration_seconds=n_seconds_gif)
        elif plot_general:
            self.gif_path = self.make_general_gif()

    def eval_(
        self,
        x: tf.Tensor,
        y: tf.Tensor,
        data_type: Literal["val", "test", "train"] = "val",
        *,
        seed: int | None = None,
    ) -> dict[str, float]:
        """
        Evaluate model according to NLL and RMSE.
        Args:
            x: index points
            y: true targets
            data_type: 'val', 'test', or 'train'
        """
        assert data_type in ["val", "test", "train"], f"{data_type=} must be one of 'val', 'test', or 'train'"
        x, y = self._broadcast_and_cast_dtype(x, y)
        y_pred, pred_dist = self.predict(x, seed=seed)
        output_dim = y.shape[-1]

        nll = -tf.reduce_mean(pred_dist.log_prob(y))
        nll_bits_per_dim = nll / float(output_dim) / tf.math.log(2.0)

        y_pred_mean = tf.reduce_mean(y_pred, axis=0)  # Average across particles
        squared_errors = tf.square(y_pred_mean - y)  # [batch_size, output_dim]
        mse_per_component = tf.reduce_mean(squared_errors, axis=0)  # [output_dim]
        rmse_per_component = tf.sqrt(tf.reduce_mean(mse_per_component))  # Average across components

        return {
            data_type + "_nll": float(nll_bits_per_dim.numpy()),
            data_type + "_rmse": float(rmse_per_component.numpy()),
        }

    def split_params(self, params: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Splits model parameters into neural network parameters and likelihood standard deviation.
        """
        total_params = params.shape[-1]
        likelihood_start_idx = total_params - self.likelihood_param_size

        likelihood_params = params[..., likelihood_start_idx:]
        likelihood_std = tf.exp(likelihood_params)
        nn_params = params[..., : self.nn_param_size]
        return nn_params, likelihood_std

    def _cache_plot_data_1d(self, plot_data: tuple[tf.Tensor, tf.Tensor], iteration: int) -> None:
        """
        Cache plot data for rendering at the end of training.
        """
        assert self.input_size == 1 and self.output_size == 1, "Data is not 1D"
        x_plot, y_plot = plot_data
        x_plot = tf.squeeze(x_plot)
        indices = tf.argsort(x_plot)
        x_plot = tf.gather(x_plot, indices)
        y_plot = tf.gather(y_plot, indices)

        y_preds, pred_dist = self.predict(x_plot, seed=self._predict_seed_for_iteration(iteration))

        mu_np = pred_dist.mean.numpy().flatten()
        sigma_np = pred_dist.stddev.numpy().flatten()
        y_preds_np = np.stack([yp.numpy().flatten() for yp in y_preds], axis=0)
        cache_entry: dict[str, np.ndarray | int] = {
            "iteration": int(iteration),
            "x_plot": x_plot.numpy().flatten(),
            "y_plot": y_plot.numpy().flatten(),
            "mu": mu_np,
            "sigma": sigma_np,
            "y_preds": y_preds_np,
            "x_train": self.unnormalized_x_train.numpy().flatten(),
            "y_train": self.unnormalized_y_train.numpy().flatten(),
        }
        self._deferred_plot_cache.append(cache_entry)

    def _render_deferred_plots(self) -> None:
        """Render all cached frames to disk in the standard figures folder.

        This function runs once at the end of training for optimal performance.
        """
        if len(self._deferred_plot_cache) == 0:
            return
        cache = sorted(self._deferred_plot_cache, key=lambda d: int(d["iteration"]))
        target_folder = paths.figures_for_experiment(self.experiment, self.name)
        target_folder.mkdir(parents=True, exist_ok=True)

        for entry in cache:
            iteration = int(entry["iteration"])
            x_plot = entry["x_plot"]
            y_plot = entry["y_plot"]
            mu = entry["mu"]
            sigma = entry["sigma"]
            y_preds = entry["y_preds"]
            x_train = entry["x_train"]
            y_train = entry["y_train"]

            plt.figure(figsize=(6, 4))
            for y_pred in y_preds:
                plt.plot(x_plot, y_pred, lw=0.1, c="r")
            plt.plot(x_plot, mu, lw=1, c="r")
            lcb, ucb = mu - 2.0 * sigma, mu + 2.0 * sigma
            plt.fill_between(x_plot, lcb, ucb, alpha=0.1, color="r")
            plt.scatter(x_train, y_train, label="train", marker="x", lw=1, color="b")
            plt.scatter(x_plot, y_plot, label="test", s=1, alpha=0.1, color="g")
            y_plot_tensor = tf.convert_to_tensor(y_plot, dtype=tf.float32)
            if y_plot_tensor.ndim == 1:
                y_plot_tensor = tf.expand_dims(y_plot_tensor, -1)
            mu_tensor = tf.convert_to_tensor(mu.reshape(-1, 1), dtype=tf.float32)
            sigma_tensor = tf.convert_to_tensor(sigma.reshape(-1, 1), dtype=tf.float32)
            pred_dist = tfp.distributions.Independent(tfp.distributions.Normal(loc=mu_tensor, scale=sigma_tensor), 1)
            nll_value = float(-tf.reduce_mean(pred_dist.log_prob(y_plot_tensor)).numpy())
            assert mu_tensor.shape[0] == y_plot_tensor.shape[0], "Mismatch between cached prediction and targets"
            rmse_value = float(tf.sqrt(tf.reduce_mean((y_plot_tensor - mu_tensor) ** 2)).numpy())
            plt.title(
                f"iteration $\\mathtt{{{iteration + 1:7d}/{self.n_iter}}}$  |  nll: $\\mathtt{{{nll_value:+.2e}}}$  |  rmse: $\\mathtt{{{rmse_value:.2e}}}$"
            )
            height = np.max(y_plot) - np.min(y_plot)
            plt.ylim([np.min(y_plot) - height, np.max(y_plot) + height])
            plt.legend(loc="upper right")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(
                target_folder / f"{iteration:010d}.{self.image_format}",
                dpi=self.figure_dpi,
            )
            plt.close()
        self._deferred_plot_cache.clear()

    def _cache_plot_data_general(self, plot_data: tuple[tf.Tensor, tf.Tensor], iteration: int) -> None:
        """
        Cache plot data for general (multi-dimensional) environments for rendering at the end of training.
        """
        x_plot, y_plot = plot_data

        y_preds, pred_dist = self.predict(x_plot, seed=self._predict_seed_for_iteration(iteration))

        mu_np = pred_dist.mean.numpy()
        sigma_np = pred_dist.stddev.numpy()
        y_preds_np = np.stack([yp.numpy() for yp in y_preds], axis=0)

        cache_entry: dict[str, np.ndarray | int] = {
            "iteration": int(iteration),
            "x_plot": x_plot.numpy(),
            "y_plot": y_plot.numpy(),
            "mu": mu_np,
            "sigma": sigma_np,
            "y_preds": y_preds_np,
            "x_train": self.unnormalized_x_train.numpy(),
            "y_train": self.unnormalized_y_train.numpy(),
        }
        self._general_plot_cache.append(cache_entry)

    def _render_deferred_general_plots(self) -> None:
        """Render all cached general plot frames to disk."""
        if len(self._general_plot_cache) == 0:
            return

        cache = sorted(self._general_plot_cache, key=lambda d: int(d["iteration"]))
        target_folder = paths.figures_for_experiment(self.experiment, self.name)
        target_folder.mkdir(parents=True, exist_ok=True)

        for entry in cache:
            iteration = int(entry["iteration"])
            x_plot = entry["x_plot"]
            y_plot = entry["y_plot"]
            mu = entry["mu"]
            sigma = entry["sigma"]
            y_preds = entry["y_preds"]
            x_train = entry["x_train"]
            y_train = entry["y_train"]

            self._render_general_plot_frame(
                x_plot, y_plot, mu, sigma, y_preds, x_train, y_train, iteration, target_folder
            )

        self._general_plot_cache.clear()

    def _render_general_plot_frame(
        self,
        x_plot: np.ndarray,
        y_plot: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        y_preds: np.ndarray,
        x_train: np.ndarray,
        y_train: np.ndarray,
        iteration: int,
        target_folder: Path,
    ) -> None:
        """Render a single frame for general multi-dimensional plotting."""
        import matplotlib.pyplot as plt

        output_dim = y_plot.shape[-1]
        _ = x_plot.shape[-1]

        if output_dim == 1:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            self._plot_prediction_vs_truth(ax, y_plot.flatten(), mu.flatten(), sigma.flatten(), "Output")
        else:
            n_cols = min(3, output_dim)  # Max 3 columns
            n_rows = (output_dim + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()

            for i in range(output_dim):
                self._plot_prediction_vs_truth(axes[i], y_plot[:, i], mu[:, i], sigma[:, i], f"Output Dim {i + 1}")

            for i in range(output_dim, len(axes)):
                axes[i].set_visible(False)

        y_plot_tensor = tf.convert_to_tensor(y_plot, dtype=tf.float32)
        mu_tensor = tf.convert_to_tensor(mu, dtype=tf.float32)
        sigma_tensor = tf.convert_to_tensor(sigma, dtype=tf.float32)

        pred_dist = tfp.distributions.Independent(tfp.distributions.Normal(loc=mu_tensor, scale=sigma_tensor), 1)
        nll_value = float(-tf.reduce_mean(pred_dist.log_prob(y_plot_tensor)).numpy())
        rmse_value = float(tf.sqrt(tf.reduce_mean((y_plot_tensor - mu_tensor) ** 2)).numpy())

        fig.suptitle(
            f"Iteration {iteration + 1:7d}/{self.n_iter} | NLL: {nll_value:+.2e} | RMSE: {rmse_value:.2e}", fontsize=14
        )

        plt.tight_layout()
        plt.savefig(
            target_folder / f"{iteration:010d}.{self.image_format}",
            dpi=self.figure_dpi,
        )
        plt.close()

    def _plot_prediction_vs_truth(
        self, ax, y_true: np.ndarray, y_pred: np.ndarray, sigma: np.ndarray, title: str
    ) -> None:
        """Create a prediction vs ground truth scatter plot with uncertainty."""

        ax.scatter(y_true, y_pred, alpha=0.6, s=20, c=sigma, cmap="viridis")

        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.8, label="Perfect prediction")

        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Prediction")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax.set_aspect("equal", adjustable="box")

    def make_general_gif(self) -> Path:
        """Create a GIF from general plot frames."""
        import imageio

        project_root = paths.root
        gifs_dir = paths.gifs
        gifs_dir.mkdir(parents=True, exist_ok=True)
        gif_path = gifs_dir / f"{self.experiment}-{self.name}.gif"
        figures_dir = paths.figures_for_experiment(self.experiment, self.name)
        image_suffix = f".{self.image_format}"
        image_files = sorted([f for f in figures_dir.iterdir() if f.suffix == image_suffix])
        frames: list = []
        for image_file in image_files:
            frames.append(imageio.imread(image_file))

        num_frames = len(frames)
        if num_frames == 0:
            fps = 10  # fallback
        else:
            max_fps = 30  # Reasonable maximum FPS for smooth playback
            target_animation_frames = 4 * max_fps  # 120 frames for 4 seconds at 30fps

            if num_frames > target_animation_frames:
                step = num_frames // target_animation_frames
                frames = frames[::step] + [frames[-1]]  # Keep every nth frame + final frame
                num_frames = len(frames) - 1  # Subtract the duplicate final frame

            pause_frames = max_fps  # 30 frames for 1 second pause
            frames = frames + [frames[-1]] * pause_frames

            fps = max_fps
        imageio.mimsave(gif_path, frames, fps=fps, loop=0, optimize=True)
        return gif_path.relative_to(project_root)

    def _predict_seed_for_iteration(self, iteration: int) -> int:
        base = 0
        key = f"{self.experiment}:{self.name}"
        for ch in key:
            base = (base * 131 + ord(ch)) % 2147483647
        seed = (base * 1000003 + int(iteration) + self._viz_seed_base) % 2147483647
        return int(seed)

    def _get_batch_sampler(self) -> tf.data.Iterator:
        train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        shuffle_seed = 1234
        if hasattr(self, "seed") and self.seed is not None:
            assert isinstance(self.seed, (int, np.integer)), "seed must be an integer"
            shuffle_seed = int(self.seed)
        train_dataset = train_dataset.shuffle(self.batch_size, seed=shuffle_seed, reshuffle_each_iteration=False)
        train_dataset = train_dataset.repeat()
        train_dataset = train_dataset.batch(self.batch_size)
        return iter(train_dataset)

    def _capture_trainable_state(self) -> dict[str, tf.Tensor]:
        """Capture model trainable state for early stopping restore.

        We capture variables that represent the learned state across models:
        - For VI/FVI-like models: posterior.trainable_variables
        - For SVGD/FSVGD: particles variable
        - For MLP baseline: nn.trainable_variables
        """
        state: dict[str, tf.Tensor] = {}
        if hasattr(self, "posterior") and isinstance(getattr(self, "posterior"), (GaussianPosterior,)):
            for idx, var in enumerate(self.posterior.trainable_variables):
                state[f"posterior_{idx}"] = tf.identity(var)
        if hasattr(self, "particles") and isinstance(getattr(self, "particles"), tf.Variable):
            state["particles"] = tf.identity(self.particles)
        if hasattr(self, "nn") and hasattr(self.nn, "trainable_variables"):
            for idx, var in enumerate(self.nn.trainable_variables):
                state[f"nn_{idx}"] = tf.identity(var)
        return state

    def _restore_trainable_state(self, state: dict[str, tf.Tensor]) -> None:
        """Restore trainable state captured earlier."""
        if hasattr(self, "posterior") and isinstance(getattr(self, "posterior"), (GaussianPosterior,)):
            for idx, var in enumerate(self.posterior.trainable_variables):
                key = f"posterior_{idx}"
                if key in state:
                    var.assign(state[key])
        if hasattr(self, "particles") and isinstance(getattr(self, "particles"), tf.Variable):
            if "particles" in state:
                self.particles.assign(state["particles"])
        if hasattr(self, "nn") and hasattr(self.nn, "trainable_variables"):
            for idx, var in enumerate(self.nn.trainable_variables):
                key = f"nn_{idx}"
                if key in state:
                    var.assign(state[key])

    def normalize(self, x: tf.Tensor, y: tf.Tensor | None = None) -> tf.Tensor | tuple[tf.Tensor, tf.Tensor]:
        x = (x - self.x_mean) / self.x_std
        if y is None:
            return x
        else:
            y = (y - self.y_mean) / self.y_std
            return x, y

    def _unnormalize_preds(self, y: tf.Tensor) -> tf.Tensor:
        return y * self.y_std + self.y_mean

    def _unnormalize_predictive_dist(self, pred_dist: tfp.distributions.Distribution) -> tfp.distributions.Distribution:
        return self.affine_transform(pred_dist)

    @staticmethod
    def _predictive_mixture(y_pred: tf.Tensor, likelihood_std: tf.Tensor) -> tfp.distributions.MixtureSameFamily:
        """
        Creates a mixture of isotropic Gaussian distributions using y_pred as mean and likelihood_std as
        standard deviation. Uses MixtureSameFamily for better performance.
        """
        num_particles = tf.shape(y_pred)[0]
        batch_size = tf.shape(y_pred)[1]

        likelihood_scale = tf.convert_to_tensor(
            likelihood_std, dtype=y_pred.dtype
        )  # [num_particles, 1] or [num_particles, output_dim]
        likelihood_scale = tf.reshape(likelihood_scale, [num_particles, 1, -1])  # [num_particles, 1, 1 or output_dim]
        likelihood_scale = tf.broadcast_to(
            likelihood_scale, tf.shape(y_pred)
        )  # [num_particles, batch_size, output_dim]

        location = tf.transpose(y_pred, [1, 0, 2])  # [batch_size, num_particles, output_dim]
        scale = tf.transpose(likelihood_scale, [1, 0, 2])  # [batch_size, num_particles, output_dim]

        components = tfp.distributions.Independent(
            tfp.distributions.Normal(loc=location, scale=scale),
            reinterpreted_batch_ndims=1,
        )
        categorical = tfp.distributions.Categorical(logits=tf.zeros([batch_size, num_particles], y_pred.dtype))
        return tfp.distributions.MixtureSameFamily(categorical, components)

    @staticmethod
    def _broadcast_and_cast_dtype(
        x: tf.Tensor, y: tf.Tensor | None = None, dtype: tf.DType = tf.float32
    ) -> tf.Tensor | tuple[tf.Tensor, tf.Tensor]:
        if x.ndim == 1:
            x = tf.expand_dims(x, -1)
        if y is not None:
            assert len(x) == len(y)
            if y.ndim == 1:
                y = tf.expand_dims(y, -1)
            return tf.cast(x, dtype=dtype), tf.cast(y, dtype=dtype)
        else:
            return tf.cast(x, dtype=dtype)

    def make_gif(self, duration_seconds: int = 5) -> Path:
        import imageio

        project_root = paths.root
        gifs_dir = paths.gifs
        gifs_dir.mkdir(parents=True, exist_ok=True)
        gif_path = gifs_dir / f"{self.experiment}-{self.name}.gif"
        figures_dir = paths.figures_for_experiment(self.experiment, self.name)
        image_suffix = f".{self.image_format}"
        image_files = sorted([f for f in figures_dir.iterdir() if f.suffix == image_suffix])
        frames: list = []
        for image_file in image_files:
            frames.append(imageio.imread(image_file))

        num_frames = len(frames)
        if num_frames == 0:
            fps = 10  # fallback
        else:
            target_fps = 30  # Reasonable FPS for smooth playback
            target_total_frames = duration_seconds * target_fps  # 150 frames for 5 seconds
            pause_frames = target_fps  # 30 frames for 1 second pause
            target_animation_frames = target_total_frames - pause_frames  # 120 frames for animation

            if num_frames > target_animation_frames:
                step = max(1, num_frames // target_animation_frames)
                subsampled_frames = frames[::step]

                if len(subsampled_frames) == 0 or subsampled_frames[-1] is not frames[-1]:
                    subsampled_frames = subsampled_frames + [frames[-1]]

                if len(subsampled_frames) > target_animation_frames:
                    subsampled_frames = subsampled_frames[: target_animation_frames - 1] + [subsampled_frames[-1]]

                frames = subsampled_frames

            frames = frames + [frames[-1]] * pause_frames

            fps = target_fps

        imageio.mimsave(gif_path, frames, fps=fps, loop=0, optimize=True)
        return gif_path.relative_to(project_root)

    def _preferred_project_root(self) -> Path:
        """Return the repository root deterministically."""
        return paths.root

    def _is_mujoco_environment(self) -> bool:
        """Dynamically check if the current experiment corresponds to a MuJoCo env.
        Resolves the class from `fpbnn.envs` using simple name resolution and checks subclassing.
        """
        from fpbnn.utils import resolve_class_name

        envs_module = importlib.import_module("fpbnn.envs")
        class_name = resolve_class_name(envs_module, self.experiment)
        env_class = getattr(envs_module, class_name)
        return issubclass(env_class, AbstractMujocoEnv)

    def _resolve_mujoco_env(self) -> AbstractMujocoEnv | None:
        """Return a MuJoCo environment instance with a `create_gif_from_model` method.
        Returns None if the env cannot be resolved.
        """
        if hasattr(self, "functional_prior") and hasattr(self.functional_prior, "create_gif_from_model"):
            return self.functional_prior

        from fpbnn.utils import resolve_class_name

        envs_module = importlib.import_module("fpbnn.envs")
        env_class_name = resolve_class_name(envs_module, self.experiment)
        assert hasattr(envs_module, env_class_name), "Environment class not found for MuJoCo resolution"
        env_class = getattr(envs_module, env_class_name)
        return env_class()

    def _make_mujoco_gif(self, fps: int = 10) -> Path:
        """Create a GIF for MuJoCo environments using the trained model."""
        env_instance = self._resolve_mujoco_env()
        assert env_instance is not None, "Could not resolve MuJoCo environment for GIF creation"

        def model_predict_func(x):
            """Predict function that takes state+action and returns next state."""
            x = tf.convert_to_tensor(x, dtype=tf.float32)
            y_pred, _ = self.predict(x)  # already unnormalized
            return tf.reduce_mean(y_pred, axis=0)

        base_root = paths.root
        output_dir = paths.gifs
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{self.experiment}-{self.name}.gif"

        gif_path = env_instance.create_gif_from_model(
            model_predict_func=model_predict_func,
            n_episodes=2,
            episode_length=50,
            fps=fps,
            output_path=str(output_path),
        )
        print(f"Created MuJoCo GIF at {Path(gif_path).relative_to(base_root)}!")
        from pathlib import Path as PathLib

        return PathLib(gif_path).relative_to(base_root)

    def plot_mujoco_training_frame(self, iteration: int, show: bool = False) -> None:
        """
        Create a training frame showing prediction convergence for MuJoCo environments.
        Similar to the 1D plotting functionality but for MuJoCo state prediction.
        """
        if not hasattr(self, "_mujoco_training_state"):
            env_instance = self._resolve_mujoco_env()
            assert env_instance is not None, "Could not resolve MuJoCo environment"

            numpy_state = np.random.get_state()

            import os

            user_seed = int(os.environ.get("FPBNN_VISUALIZATION_SEED", 42))
            env_base_name = self.experiment.split("_")[0] if "_" in self.experiment else self.experiment
            viz_seed = user_seed + hash(env_base_name) % 1000  # User seed + small env-specific offset
            np.random.seed(viz_seed)
            if hasattr(env_instance, "seed") and callable(getattr(env_instance, "seed", None)):
                env_instance.seed(viz_seed)

            (sample_inputs, _), _ = env_instance.sample_train_test(n_train=1, n_test=1)
            sample_input = sample_inputs[0]
            initial_state = sample_input[: env_instance.state_space_size]
            action = sample_input[env_instance.state_space_size :]

            ground_truth_state = env_instance.step(sample_input)

            env_instance.sample_new_env()

            np.random.set_state(numpy_state)

            fixed_model = env_instance.env.model
            fixed_data = env_instance.env.data

            self._mujoco_training_state = {
                "initial_state": initial_state,
                "action": action,
                "ground_truth_state": ground_truth_state,
                "model_input": sample_input,
                "env_instance": env_instance,
                "fixed_model": fixed_model,
                "fixed_data": fixed_data,
                "total_iterations": self.n_iter,  # Store total iterations for progress display
            }

            if not hasattr(self, "_mujoco_training_cache"):
                self._mujoco_training_cache = []

        model_input = self._mujoco_training_state["model_input"].reshape(1, -1)
        model_input_tensor = tf.convert_to_tensor(model_input, dtype=tf.float32)

        y_pred_particles, pred_dist = self.predict(model_input_tensor)
        predicted_state = tf.reduce_mean(y_pred_particles, axis=0).numpy().flatten()
        predicted_uncertainty = pred_dist.stddev.numpy().flatten()

        ground_truth_tensor = tf.convert_to_tensor(
            self._mujoco_training_state["ground_truth_state"].reshape(1, -1), dtype=tf.float32
        )
        nll = float(-tf.reduce_mean(pred_dist.log_prob(ground_truth_tensor)).numpy())

        rmse = float(
            tf.sqrt(
                tf.reduce_mean((ground_truth_tensor - tf.convert_to_tensor(predicted_state.reshape(1, -1))) ** 2)
            ).numpy()
        )

        cache_entry = {
            "iteration": int(iteration),
            "initial_state": self._mujoco_training_state["initial_state"].copy(),
            "predicted_state": predicted_state.copy(),
            "predicted_uncertainty": predicted_uncertainty.copy(),
            "ground_truth_state": self._mujoco_training_state["ground_truth_state"].copy(),
            "total_iterations": self._mujoco_training_state["total_iterations"],
            "nll": nll,
            "rmse": rmse,
        }
        self._mujoco_training_cache.append(cache_entry)

        if show:
            pass

    def _get_mujoco_frame_path(self, iteration: int) -> Path:
        """Get the path for a MuJoCo training frame."""
        folder = paths.figures_for_experiment(self.experiment, self.name)
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{iteration:010d}.{self.image_format}"

    def _render_deferred_mujoco_frames(self) -> None:
        """Render all cached MuJoCo training frames."""
        if not hasattr(self, "_mujoco_training_cache") or len(self._mujoco_training_cache) == 0:
            return

        cache = sorted(self._mujoco_training_cache, key=lambda d: int(d["iteration"]))

        target_folder = paths.figures_for_experiment(self.experiment, self.name)
        if target_folder.is_dir():
            for file_path in target_folder.iterdir():
                if file_path.suffix == f".{self.image_format}":
                    file_path.unlink()
        target_folder.mkdir(parents=True, exist_ok=True)

        env_instance = self._mujoco_training_state["env_instance"]
        for entry in cache:
            output_path = self._get_mujoco_frame_path(entry["iteration"])
            env_instance.create_state_comparison_image(
                predicted_state=entry["predicted_state"],
                predicted_uncertainty=entry["predicted_uncertainty"],
                ground_truth_state=entry["ground_truth_state"],
                output_path=str(output_path),
                use_fixed_env=True,
                fixed_model=self._mujoco_training_state["fixed_model"],
                fixed_data=self._mujoco_training_state["fixed_data"],
                iteration=entry["iteration"],
                total_iterations=entry["total_iterations"],
                nll=entry["nll"],
                rmse=entry["rmse"],
            )

        self._mujoco_training_cache.clear()

    def make_mujoco_training_gif(self, duration_seconds: int = 5) -> Path:
        """
        Create a GIF showing training convergence for MuJoCo environments.
        Similar to make_gif but for MuJoCo state prediction convergence.
        """
        if hasattr(self, "_mujoco_training_cache") and len(self._mujoco_training_cache) > 0:
            self._render_deferred_mujoco_frames()

        import imageio

        project_root = paths.root
        figures_root = paths.figures
        gifs_root = paths.gifs
        gifs_root.mkdir(parents=True, exist_ok=True)
        gif_output_path = gifs_root / f"{self.experiment}-{self.name}.gif"
        experiment_figures_dir = figures_root / self.experiment / self.name

        frame_suffix = f".{self.image_format}"
        frame_files = sorted([file for file in experiment_figures_dir.iterdir() if file.suffix == frame_suffix])

        if not frame_files:
            print(f"warning: no frames found for mujoco training gif in {experiment_figures_dir!s}")
            return gif_output_path

        frame_images = []
        for frame_file in frame_files:
            frame_images.append(imageio.imread(frame_file))

        num_frames = len(frame_images)
        if num_frames == 0:
            fps = 10  # fallback
        else:
            max_fps = 30  # Reasonable maximum FPS for smooth playback
            target_animation_frames = 4 * max_fps  # 120 frames for 4 seconds at 30fps

            if num_frames > target_animation_frames:
                step = num_frames // target_animation_frames
                frame_images = frame_images[::step] + [frame_images[-1]]  # Keep every nth frame + final frame
                num_frames = len(frame_images) - 1  # Subtract the duplicate final frame

            pause_frames = max_fps  # 30 frames for 1 second pause
            frame_images = frame_images + [frame_images[-1]] * pause_frames

            fps = max_fps

        imageio.mimsave(gif_output_path, frame_images, fps=fps, loop=0, optimize=True)

        return gif_output_path.relative_to(project_root)

    @staticmethod
    def ll(
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        likelihood_std: tf.Tensor,
        reduction: str = "sum",
    ) -> tf.Tensor:
        """
        Computes log likelihood of
        """
        likelihood_std = tf.expand_dims(likelihood_std, axis=1)
        likelihood = tfp.distributions.Independent(
            tfp.distributions.Normal(y_pred, likelihood_std),
            reinterpreted_batch_ndims=1,
        )
        log_likelihood = likelihood.log_prob(y_true)
        if reduction == "sum":
            return tf.reduce_sum(log_likelihood)
        elif reduction == "mean":
            return tf.reduce_mean(log_likelihood)
        elif reduction == "logsumexp":
            return tf.reduce_logsumexp(log_likelihood)
        else:
            raise NotImplementedError

    @abstractmethod
    def predict(self, x: tf.Tensor) -> tuple[tf.Tensor, tfp.distributions.Distribution]:
        pass

    @abstractmethod
    def __call__(self, x_batch: tf.Tensor, y_batch: tf.Tensor) -> None:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class FunctionalPrior(Protocol):
    def sample(self, _n_buffers: int, batch_size: int, n_particles: int) -> tuple[tf.Tensor, tf.Tensor]: ...


class FunctionalRegressionModel(AbstractBayesianRegression, ABC):
    def __init__(
        self,
        functional_prior: FunctionalPrior,
        ssge_bandwidth: float = 3.0,
        ssge_n_eigen: int = 6,
        ssge_eta: float = 0.01,
        coeff_entropy: float = 0.1,
        coeff_cross_entropy: float = 0.1,
        prior_noise_std: float = 1e-3,
        buffer_size: int = 10,
        *,
        train_data: tuple[tf.Tensor, tf.Tensor],
        experiment: str,
        n_iter: float = 1e4,
        batch_size: int = 8,
        n_particles: int = 10,
        activation: str = "elu",
        width: int = 32,
        depth: int = 4,
        bandwidth: float = 1e-2,
        nn_prior_std: float = 1.0,
        likelihood_prior_mean: float = 1e-2,
        likelihood_prior_std: float = 1e-2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-6,
        coeff_prior: float = 1.0,
        noise_std: float = 1e-2,
        image_format: str = "png",
        verbose: int = 0,
        normalize_train_data: bool = True,
        use_wandb: bool = True,
        enable_early_stopping: bool = True,
        early_stopping_patience: int = 1000,
        early_stopping_metric: str = "val_nll",
        early_stopping_mode: str = "min",
        early_stopping_min_delta: float = 0.0,
    ) -> None:
        super().__init__(
            train_data=train_data,
            experiment=experiment,
            n_iter=n_iter,
            batch_size=batch_size,
            n_particles=n_particles,
            activation=activation,
            width=width,
            depth=depth,
            bandwidth=bandwidth,
            nn_prior_std=nn_prior_std,
            likelihood_prior_mean=likelihood_prior_mean,
            likelihood_prior_std=likelihood_prior_std,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            coeff_prior=coeff_prior,
            noise_std=noise_std,
            image_format=image_format,
            verbose=verbose,
            normalize_train_data=normalize_train_data,
            use_wandb=use_wandb,
            enable_early_stopping=enable_early_stopping,
            early_stopping_patience=early_stopping_patience,
            early_stopping_metric=early_stopping_metric,
            early_stopping_mode=early_stopping_mode,
            early_stopping_min_delta=early_stopping_min_delta,
        )
        self.functional_prior = functional_prior  # Store the functional prior
        self.buffer_size = buffer_size
        self.coeff_cross_entropy = coeff_cross_entropy
        self.coeff_entropy = coeff_entropy

        self.ssge = SSGE(eta=ssge_eta, bandwidth=ssge_bandwidth, n_eigen=ssge_n_eigen)

        x_prior, y_prior = functional_prior.sample(self.buffer_size, self.batch_size, self.n_particles)
        x_prior, y_prior = self.normalize(x_prior, y_prior)

        x_prior = tf.repeat(x_prior, self.n_iter // self.buffer_size + 1, axis=0)
        y_prior = tf.repeat(y_prior, self.n_iter // self.buffer_size + 1, axis=0)

        y_prior += tf.random.normal(tf.shape(y_prior), 0.0, prior_noise_std)

        self.x_prior = iter(x_prior)
        self.y_prior = iter(y_prior)
