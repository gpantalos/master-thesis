import importlib
import os
import sys
from typing import Any

from fpbnn.configs import ExperimentConfig, create_model_config
from fpbnn.utils import get_dataset_sizes, seed_everything


def train_model_with_config(
    config: ExperimentConfig,
    model_name: str,
    env_class_name: str,
    seed: int = 1234,
    env_additional_params: dict[str, Any] | None = None,
    report_to_tune: bool = False,
    create_gif: bool = False,
) -> dict[str, float]:
    """Train a model with given configuration.

    This function serves as the unified training interface for both:
    1. Ray Tune hyperparameter optimization (when report_to_tune=True)
    2. Regular experiment execution (when report_to_tune=False)

    Args:
        config: Flat configuration dictionary containing all hyperparameters
        model_name: Name of the model class
        env_class_name: Name of the environment class
        seed: Random seed for reproducibility
        env_additional_params: Additional environment parameters
        report_to_tune: Whether to report metrics to Ray Tune
        create_gif: Whether to create GIF during training

    Returns:
        Dictionary with evaluation metrics
    """

    if env_additional_params is None:
        env_additional_params = {}

    seed_everything(int(seed))

    if report_to_tune:
        config = dict(config)
        config["use_wandb"] = False

    models_module = importlib.import_module("fpbnn.models")
    model_class = getattr(models_module, model_name.upper())
    envs_module = importlib.import_module("fpbnn.envs")
    env_class = getattr(envs_module, env_class_name)

    env = env_class(seed=seed, **env_additional_params)
    n_train_default, n_test_default = get_dataset_sizes(env.name)
    n_train = config.train_size or n_train_default
    n_test = config.test_size or n_test_default

    data = env.generate_meta_test_data(n_tasks=1, n_samples_context=n_train, n_samples_test=n_test)
    x_train, y_train, x_test, y_test = data[0]

    model_config = create_model_config(model_name.lower(), env.name, config)

    is_functional_model = model_name.lower() in {"fvi", "fsvgd"}

    if is_functional_model:
        model_instance = model_class.from_config(
            functional_prior=env,
            train_data=(x_train, y_train),
            config=model_config,
        )
    else:
        model_instance = model_class.from_config(
            train_data=(x_train, y_train),
            config=model_config,
        )

    assert hasattr(model_config, "train"), "Model config missing 'train' section"
    val_fraction = float(model_config.train.val_fraction)

    n_samples = int(x_train.shape[0])
    n_val = int(max(0, round(val_fraction * n_samples)))
    x_val = y_val = None
    if n_val > 0:
        import tensorflow as tf

        indices = tf.range(n_samples)
        indices = tf.random.shuffle(indices, seed=seed)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        x_val = tf.gather(x_train, val_idx)
        y_val = tf.gather(y_train, val_idx)
        x_train = tf.gather(x_train, train_idx)
        y_train = tf.gather(y_train, train_idx)

    is_macos = sys.platform == "darwin"
    has_display_env = os.environ.get("DISPLAY") is not None
    has_display = is_macos or has_display_env
    enable_plotting = create_gif and not report_to_tune and has_display

    if create_gif and not report_to_tune and not has_display:
        print("Warning: DISPLAY not available and not macOS, disabling GIF creation...")

    val_tuple = (x_val, y_val) if (x_val is not None and y_val is not None) else None
    report_callback = None
    tune_module = None
    if report_to_tune:
        from ray import tune

        tune_module = tune

        def _report(payload: dict[str, float]) -> None:
            tune.report(payload)

        report_callback = _report

    model_instance.fit(
        val_data=val_tuple,
        test_data=(x_test, y_test),
        plot=enable_plotting,
        n_plots=config.n_plots or 50,
        report_callback=report_callback,
    )

    results = model_instance.eval_(x_test, y_test, "test")
    if x_val is not None and y_val is not None:
        val_results = model_instance.eval_(x_val, y_val, "val")
        if isinstance(results, dict):
            results.update(val_results)

    assert isinstance(results, dict), "Model evaluation must return a dictionary of metrics"
    test_nll = float(results.get("test_nll", float("inf")))
    test_rmse = float(results.get("test_rmse", float("inf")))
    val_nll = float(results.get("val_nll", float("inf")))
    val_rmse = float(results.get("val_rmse", float("inf")))
    metrics = {"test_nll": test_nll, "test_rmse": test_rmse, "val_nll": val_nll, "val_rmse": val_rmse}

    if report_to_tune and tune_module is not None:
        tune_module.report(metrics)

    return metrics


def train_for_tune(
    config: ExperimentConfig,
    model_name: str,
    env_class_name: str,
    seed: int,
    env_additional_params: dict[str, Any],
) -> dict[str, float]:
    """Training function specifically for Ray Tune trials."""
    return train_model_with_config(
        config=config,
        model_name=model_name,
        env_class_name=env_class_name,
        seed=seed,
        env_additional_params=env_additional_params,
        report_to_tune=True,
    )
