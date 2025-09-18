import importlib
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from fpbnn.models import available_models as model_names
from fpbnn.training import train_for_tune
from fpbnn.utils import paths

Activation = Literal["elu", "relu", "tanh"]


@dataclass(frozen=True, slots=True)
class SpaceMixin:
    @classmethod
    def _get_space(cls) -> dict[str, Any]:
        """Override in subclasses to define search space."""
        return {}

    @classmethod
    def space(cls, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        return {**cls._get_space(), **(overrides or {})}


@dataclass(frozen=True, slots=True)
class ArchitectureHyperparams(SpaceMixin):
    width: int = 32
    depth: int = 4
    activation: Activation = "elu"
    nn_prior_std: float = 1.0
    bandwidth: float = 1e-2

    @classmethod
    def _get_space(cls) -> dict[str, Any]:
        from ray import tune

        return {
            "width": tune.choice([16, 32, 64, 128]),
            "depth": tune.choice([2, 3, 4, 5]),
            "activation": tune.choice(["relu", "elu", "tanh"]),
            "nn_prior_std": tune.loguniform(0.5, 5.0),
            "bandwidth": tune.loguniform(1e-3, 1e-1),
        }


@dataclass(frozen=True, slots=True)
class TrainingHyperparams(SpaceMixin):
    learning_rate: float = 1e-3
    batch_size: int = 8
    weight_decay: float = 1e-6
    n_particles: int = 10
    n_iter: int = 10_000

    val_fraction: float = 0.1
    enable_early_stopping: bool = True
    early_stopping_patience: int = 3000
    early_stopping_metric: str = "val_nll"
    early_stopping_mode: str = "min"
    early_stopping_min_delta: float = 1e-4

    @classmethod
    def _get_space(cls) -> dict[str, Any]:
        from ray import tune

        return {
            "learning_rate": tune.loguniform(1e-5, 1e-2),
            "batch_size": tune.choice([4, 8, 16, 32]),
            "weight_decay": tune.loguniform(1e-8, 1e-4),
            "n_particles": tune.choice([5, 10, 15, 20, 30, 50]),
        }


@dataclass(frozen=True, slots=True)
class LikelihoodHyperparams(SpaceMixin):
    likelihood_prior_mean: float = 1e-2
    likelihood_prior_std: float = 1e-2
    noise_std: float = 1e-2

    @classmethod
    def _get_space(cls) -> dict[str, Any]:
        from ray import tune

        return {
            "likelihood_prior_mean": tune.loguniform(1e-4, 1e-1),
            "likelihood_prior_std": tune.loguniform(1e-4, 1e-1),
            "noise_std": tune.loguniform(1e-4, 1e-1),
        }


@dataclass(frozen=True, slots=True)
class SSGEHyperparams(SpaceMixin):
    """Used for functional methods (FVI, FSVGD)."""

    ssge_bandwidth: float = 3.0
    ssge_n_eigen: int = 6
    ssge_eta: float = 0.01
    coeff_entropy: float = 0.1
    coeff_cross_entropy: float = 0.1
    prior_noise_std: float = 1e-3
    coeff_prior: float = 1.0

    @classmethod
    def _get_space(cls) -> dict[str, Any]:
        from ray import tune

        return {
            "ssge_bandwidth": tune.loguniform(0.1, 3.0),
            "ssge_n_eigen": tune.choice([4, 6, 8, 12, 16]),
            "ssge_eta": tune.loguniform(1e-3, 1e-1),
            "coeff_entropy": tune.loguniform(1e-3, 1e-1),
            "coeff_cross_entropy": tune.loguniform(1e-3, 1e-1),
            "prior_noise_std": tune.loguniform(1e-5, 1e-2),
            "coeff_prior": tune.loguniform(0.1, 5.0),
        }


@dataclass(frozen=True, slots=True)
class EnvironmentParams:
    seed: int = 1234
    additional_params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class HyperparameterConfig:
    architecture: ArchitectureHyperparams = field(default_factory=ArchitectureHyperparams)
    training: TrainingHyperparams = field(default_factory=TrainingHyperparams)
    likelihood: LikelihoodHyperparams = field(default_factory=LikelihoodHyperparams)
    ssge: SSGEHyperparams | None = None
    env_params: EnvironmentParams = field(default_factory=EnvironmentParams)
    n_samples: int = 100
    metric: str = "val_nll"
    mode: str = "min"

    def to_tune_config(self, model: str | None = None) -> dict[str, Any]:
        cfg: dict[str, Any] = {}

        cfg.update(asdict(self.architecture))

        training_params = asdict(self.training)
        if model and model.lower() == "mlp":
            training_params.pop("n_particles", None)
        cfg.update(training_params)

        cfg.update(asdict(self.likelihood))

        if self.ssge is not None and model and model.lower() in {"fvi", "fsvgd"}:
            cfg.update(asdict(self.ssge))

        return cfg

    @classmethod
    def with_search_spaces(cls, model: str, **overrides) -> "HyperparameterConfig":
        ao = overrides.get("arch_overrides")
        to = overrides.get("training_overrides")
        lo = overrides.get("likelihood_overrides")
        so = overrides.get("ssge_overrides")

        arch = ArchitectureHyperparams(**ArchitectureHyperparams.space(ao))

        training_space = TrainingHyperparams.space(to)
        if model.lower() == "mlp":
            training_space.pop("n_particles", None)
        training = TrainingHyperparams(**training_space)

        likelihood = LikelihoodHyperparams(**LikelihoodHyperparams.space(lo))
        ssge = SSGEHyperparams(**SSGEHyperparams.space(so)) if model.lower() in {"fvi", "fsvgd"} else None

        cfg_kwargs = {
            k: v
            for k, v in overrides.items()
            if k not in {"arch_overrides", "training_overrides", "likelihood_overrides", "ssge_overrides"}
        }

        return cls(architecture=arch, training=training, likelihood=likelihood, ssge=ssge, **cfg_kwargs)


def _setup_output(env_name: str, model_name: str) -> tuple[Path, Path]:
    cfg = paths.tuned_config_for_env_model(env_name, model_name)
    res = paths.tuning_results_for_env_model(env_name, model_name)
    cfg.parent.mkdir(parents=True, exist_ok=True)
    res.parent.mkdir(parents=True, exist_ok=True)
    return cfg, res


def _to_serializable(d: dict[str, Any]) -> dict[str, Any]:
    conv = {}
    for k, v in d.items():
        if hasattr(v, "item"):
            conv[k] = v.item()
        else:
            conv[k] = v
    return conv


def _save_results(results, config: HyperparameterConfig, cfg_file: Path, res_file: Path) -> None:
    df = results.get_dataframe()
    df.to_csv(res_file, index=False)
    best = results.get_best_result(metric=config.metric, mode=config.mode)
    if not best or not best.config:
        raise ValueError("No best configuration found.")

    unwanted_hp_keys = {"verbose"}
    filtered_config = {k: v for k, v in best.config.items() if k not in unwanted_hp_keys}

    rounded_config = _to_serializable(filtered_config)
    metric_keys = {"test_nll", "test_rmse", "val_nll", "val_rmse"}
    metrics = {k: v for k, v in best.metrics.items() if k in metric_keys}
    result_data = {"config": rounded_config, "metrics": metrics}

    yaml.add_representer(float, lambda d, x: d.represent_scalar("tag:yaml.org,2002:float", f"{x:.1e}"))
    cfg_file.write_text(yaml.dump(result_data, default_flow_style=False, allow_unicode=True))


def _load_best_config(env_name: str, model_name: str) -> dict[str, Any]:
    file = paths.tuned_config_for_env_model(env_name, model_name)
    if not file.exists():
        raise FileNotFoundError(f"No tuned configuration for {model_name} on {env_name}. Expected: {file}")
    data = yaml.safe_load(file.read_text())
    assert isinstance(data, dict), f"Configuration file must contain a YAML object: {file}"
    return data["config"]


class HyperparameterTuner:
    def __init__(
        self,
        enable_wandb: bool = True,
    ):
        self.enable_wandb = enable_wandb

    @staticmethod
    def _init_ray() -> None:
        import os

        import ray

        if not ray.is_initialized():
            original_tmpdir = os.environ.get("TMPDIR")
            if Path("/tmp").exists():
                os.environ["TMPDIR"] = "/tmp"

            ray.init(
                dashboard_host="0.0.0.0",
                dashboard_port=8265,
                include_dashboard=True,
                log_to_driver=True,
            )

            if original_tmpdir is not None:
                os.environ["TMPDIR"] = original_tmpdir
            elif "TMPDIR" in os.environ:
                del os.environ["TMPDIR"]

    def tune_model(
        self,
        model_name: str,
        env_name: str,
        config: HyperparameterConfig,
        verbose: int = 1,
        verbose_ray: int = 0,
    ):
        if model_name not in model_names:
            raise ValueError(f"Unknown model '{model_name}'. Available: {model_names}")

        envs_module = importlib.import_module("fpbnn.envs")
        if not hasattr(envs_module, env_name):
            avail = getattr(envs_module, "__all__", [])
            raise ValueError(f"Unknown environment '{env_name}'. Available: {list(avail)}")

        env_class = getattr(envs_module, env_name)
        env_tmp = env_class(seed=config.env_params.seed, **config.env_params.additional_params)

        cfg_file, res_file = _setup_output(env_tmp.name, model_name)
        tune_cfg = config.to_tune_config(model_name) | {"verbose": verbose}

        if self.enable_wandb:
            os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

        self._init_ray()

        from ray import tune

        trainable = tune.with_parameters(
            train_for_tune,
            model_name=model_name,
            env_class_name=env_name,
            seed=config.env_params.seed,
            env_additional_params=config.env_params.additional_params,
        )

        callbacks = []
        if self.enable_wandb:
            from ray.air.integrations.wandb import WandbLoggerCallback

            wandb_callback = WandbLoggerCallback(
                project="fpbnn-experiments",
                group=f"{env_tmp.name}-{model_name.lower()}",
                log_config=True,
                upload_checkpoints=False,  # Disable checkpoint uploads for faster runs
            )
            callbacks.append(wandb_callback)

        def _trial_name_creator(trial: object) -> str:
            suffix = getattr(trial, "trial_id", "00000")
            if isinstance(suffix, str) and "_" in suffix:
                suffix = suffix.split("_")[-1]
            return f"{env_tmp.name}-{model_name.lower()}-{suffix}"

        tuner = tune.Tuner(
            trainable,
            param_space=tune_cfg,
            tune_config=tune.TuneConfig(
                metric=config.metric,
                mode=config.mode,
                num_samples=config.n_samples,
                trial_name_creator=_trial_name_creator,
            ),
            run_config=tune.RunConfig(
                storage_path=str(paths.ray_results),
                name=f"{model_name.lower()}_{env_tmp.name}",
                verbose=verbose_ray,
                failure_config=tune.FailureConfig(fail_fast=True),
                progress_reporter=tune.CLIReporter(
                    metric_columns=["val_nll", "val_rmse", "test_nll", "test_rmse"],
                    parameter_columns=[],
                    max_progress_rows=10,
                    max_report_frequency=300,
                ),
                callbacks=callbacks,
            ),
        )

        results = tuner.fit()
        _save_results(results, config, cfg_file, res_file)
        return results

    def tune_from_file(self, file_str: str, config: HyperparameterConfig):
        p = Path(file_str)
        return self.tune_model(p.stem.upper(), p.parent.name.title().replace("_", ""), config)

    @staticmethod
    def create_config_with_search_spaces(model: str, n_samples: int = 100, **overrides) -> HyperparameterConfig:
        return HyperparameterConfig.with_search_spaces(model=model, n_samples=n_samples, **overrides)

    @staticmethod
    def load_best_config(env_name: str, model_name: str) -> dict[str, Any]:
        return _load_best_config(env_name, model_name)


def tune_model(
    model_name: str,
    env_name: str,
    config: HyperparameterConfig,
    verbose: int = 1,
    verbose_ray: int = 0,
    enable_wandb: bool = True,
):
    tuner = HyperparameterTuner(enable_wandb=enable_wandb)
    return tuner.tune_model(model_name, env_name, config, verbose, verbose_ray)


def tune_from_file(file_str: str, config: HyperparameterConfig):
    return HyperparameterTuner().tune_from_file(file_str, config)


def create_config_with_search_spaces(model: str, n_samples: int = 100, **overrides) -> HyperparameterConfig:
    return HyperparameterConfig.with_search_spaces(model=model, n_samples=n_samples, **overrides)


def load_best_config(env_name: str, model_name: str) -> dict[str, Any]:
    print(f"loading tuned config for ({model_name=},{env_name=})")
    return HyperparameterTuner.load_best_config(env_name, model_name)
