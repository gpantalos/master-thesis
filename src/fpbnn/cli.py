import argparse
import itertools
import random
import sys
from abc import ABC, abstractmethod
from types import ModuleType
from typing import get_args

import numpy as np
import yaml
from pydantic import BaseModel, Field, field_validator
from tqdm import tqdm

import fpbnn.envs as envs_module
import fpbnn.models as models_module
from fpbnn.configs import ExperimentConfig
from fpbnn.envs import available_envs
from fpbnn.models import available_models
from fpbnn.training import train_model_with_config
from fpbnn.tuning import load_best_config
from fpbnn.utils import camel_to_snake, paths, resolve_class_name


def _get_class_name(module: ModuleType, name_key: str) -> str:
    return resolve_class_name(module, name_key)


def _make_yaml_serializable(obj: object) -> object:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _make_yaml_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_yaml_serializable(i) for i in obj]
    return obj


class CLIArgs(BaseModel):
    model: str | list[str] | None = None
    env: str | list[str] | None = None
    list: bool = False
    tune: bool = False
    num_samples: int = Field(16, gt=0)
    n_particles: int | None = Field(None, gt=0)
    n_iter: int | None = Field(None, gt=0)
    batch_size: int | None = Field(None, gt=0)
    learning_rate: float | None = Field(None, gt=0)
    bandwidth: float | None = Field(None, gt=0)
    train_size: int | None = Field(None, gt=0)
    test_size: int | None = Field(None, gt=0)
    seed: int | None = None
    verbose: int | None = Field(None, ge=0, le=2)
    verbose_ray: int | None = Field(0, ge=0, le=2)
    no_plot: bool = False
    no_logging: bool = False
    no_gif: bool = False
    ssge_bandwidth: float | None = Field(None, gt=0)
    ssge_n_eigen: int | None = Field(None, gt=0)
    coeff_entropy: float | None = Field(None, gt=0)
    no_early_stopping: bool = False
    early_stopping_patience: int | None = Field(None, gt=0)
    early_stopping_metric: str | None = None
    early_stopping_mode: str | None = None
    early_stopping_min_delta: float | None = None
    use_defaults: bool = False
    n_plots: int | None = Field(None, gt=0)

    @field_validator("model", "env", mode="before")
    @classmethod
    def parse_comma_separated(cls, v):
        if isinstance(v, str) and "," in v:
            return [item.strip() for item in v.split(",")]
        return v

    @field_validator("model", "env")
    @classmethod
    def validate_choices(cls, v, info):
        choices = available_models if info.field_name == "model" else available_envs
        if v is not None:
            values = [v] if isinstance(v, str) else v
            for value in values:
                if value not in choices:
                    raise ValueError(f"Unknown {info.field_name} '{value}'.")
        return v

    def to_experiment_config(self) -> ExperimentConfig:
        config_data = self.model_dump(include=ExperimentConfig.model_fields.keys(), exclude_none=True)
        config_data.update({"plot": not self.no_plot, "wandb": not self.no_logging, "create_gif": not self.no_gif})
        if self.no_early_stopping:
            config_data["enable_early_stopping"] = False
        return ExperimentConfig(**config_data)


class Command(ABC):
    @abstractmethod
    def execute(self, args: CLIArgs) -> int:
        raise NotImplementedError


class TrainCommand(Command):
    def execute(self, args: CLIArgs) -> int:
        cli_config = args.to_experiment_config()
        if args.env:
            environments = [args.env] if isinstance(args.env, str) else args.env
        else:
            environments = available_envs
        if args.model:
            models = [args.model] if isinstance(args.model, str) else args.model
        else:
            models = available_models
        if args.env and args.model:
            environments = [args.env] if isinstance(args.env, str) else args.env
            models = [args.model] if isinstance(args.model, str) else args.model
        experiments = list(itertools.product(environments, models))
        completed_experiments = []
        for env_key, model_key in experiments:
            env_class_name = _get_class_name(envs_module, env_key)
            canonical_env_name = camel_to_snake(env_class_name)
            cfg_path = paths.tuned_config_for_env_model(canonical_env_name, model_key)
            if args.use_defaults:
                base_config = {}
            else:
                if cfg_path.exists():
                    base_config = load_best_config(canonical_env_name, model_key.upper())
                else:
                    base_config = {}
            final_config = base_config | cli_config.model_dump(exclude_none=True, exclude_defaults=True)
            output_dir = paths.logs_for_experiment(canonical_env_name)
            output_dir.mkdir(parents=True, exist_ok=True)
            experiment_config = ExperimentConfig(**final_config)
            results = train_model_with_config(
                config=experiment_config,
                model_name=model_key.upper(),
                env_class_name=env_class_name,
                seed=cli_config.seed or 1234,
                create_gif=cli_config.create_gif,
            )
            yaml.add_representer(float, lambda d, data: d.represent_scalar("tag:yaml.org,2002:float", f"{data:.1e}"))
            results_file = output_dir / "results.yaml"
            results_file.write_text(
                yaml.dump(
                    _make_yaml_serializable(results),
                    default_flow_style=False,
                    allow_unicode=True,
                )
            )
            config_file = output_dir / "config.yaml"
            config_file.write_text(
                yaml.dump(
                    final_config,
                    default_flow_style=False,
                    allow_unicode=True,
                )
            )
            completed_experiments.append((canonical_env_name, model_key.upper()))
        return 0


class TuneCommand(Command):
    def execute(self, args: CLIArgs) -> int:
        from fpbnn.tuning import HyperparameterConfig, tune_model

        if args.env:
            environments = [args.env] if isinstance(args.env, str) else args.env
        else:
            environments = available_envs

        if args.model:
            models = [args.model] if isinstance(args.model, str) else args.model
        else:
            models = available_models
        env_cls = [_get_class_name(envs_module, name) for name in environments]
        model_cls = [_get_class_name(models_module, name) for name in models]
        experiments = list(itertools.product(env_cls, model_cls))
        random.shuffle(experiments)
        print(f"Starting hyperparameter tuning for {len(experiments)} combination(s)...")
        for env_class, model_class in tqdm(experiments, desc="Tuning Progress"):
            all_overrides = args.to_experiment_config().model_dump(exclude_none=True)

            training_params = {
                "learning_rate",
                "batch_size",
                "weight_decay",
                "n_particles",
                "n_iter",
                "val_fraction",
                "enable_early_stopping",
                "early_stopping_patience",
                "early_stopping_metric",
                "early_stopping_mode",
                "early_stopping_min_delta",
            }
            arch_params = {"width", "depth", "activation", "nn_prior_std", "bandwidth"}
            likelihood_params = {"likelihood_prior_mean", "likelihood_prior_std", "noise_std"}
            ssge_params = {
                "ssge_bandwidth",
                "ssge_n_eigen",
                "ssge_eta",
                "coeff_entropy",
                "coeff_cross_entropy",
                "prior_noise_std",
                "coeff_prior",
            }

            training_overrides = {k: v for k, v in all_overrides.items() if k in training_params}
            arch_overrides = {k: v for k, v in all_overrides.items() if k in arch_params}
            likelihood_overrides = {k: v for k, v in all_overrides.items() if k in likelihood_params}
            ssge_overrides = {k: v for k, v in all_overrides.items() if k in ssge_params}

            search_config = HyperparameterConfig.with_search_spaces(
                model=model_class.lower(),
                n_samples=args.num_samples,
                training_overrides=training_overrides,
                arch_overrides=arch_overrides,
                likelihood_overrides=likelihood_overrides,
                ssge_overrides=ssge_overrides,
            )
            tune_model(
                model_name=model_class.lower(),
                env_name=env_class,
                config=search_config,
                verbose=args.verbose or 0,
                verbose_ray=args.verbose_ray,
                enable_wandb=not args.no_logging,
            )
        print(f"Tuning complete for all {len(experiments)} combination(s).")
        return 0


class CliRunner:
    def __init__(self):
        self.parser = self._create_parser()
        self.commands = {
            "tune": TuneCommand(),
            "default": TrainCommand(),
        }

    def _create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="CLI for FPBNN")
        modes = parser.add_mutually_exclusive_group()
        modes.add_argument("--tune", action="store_true", help="Run HP tuning (default: train)")

        def _annotation_to_arg_type(annotation):
            if annotation is bool:
                return None
            if annotation in (int, float, str):
                return annotation
            args = get_args(annotation)
            if args:
                non_none = [a for a in args if a is not type(None)]
                if len(non_none) == 1 and non_none[0] in (int, float, str):
                    return non_none[0]
            return None

        for field, props in CLIArgs.model_fields.items():
            if field in ["model", "env", "tune"]:
                continue
            arg_name = f"--{field.replace('_', '-')}"
            kwargs = {"help": props.description or ""}
            if props.annotation is bool:
                kwargs["action"] = "store_true"
            else:
                arg_type = _annotation_to_arg_type(props.annotation)
                if arg_type is not None:
                    kwargs["type"] = arg_type
            if props.default is not ...:
                kwargs["default"] = props.default
            parser.add_argument(arg_name, **kwargs)
        parser.add_argument("--model", "-m", help="Model(s) to use (comma-separated for multiple)")
        parser.add_argument("--env", "-e", help="Environment(s) to use (comma-separated for multiple)")
        return parser

    def run(self, argv: list[str] | None = None) -> int:
        raw_args = self.parser.parse_args(argv)
        args = CLIArgs(**vars(raw_args))

        cmd_key = next((key for key in self.commands if getattr(args, key, False)), "default")
        return self.commands[cmd_key].execute(args)


def main() -> int:
    import os
    import warnings

    import absl.logging

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
    absl.logging.set_verbosity(absl.logging.ERROR)
    return CliRunner().run()


if __name__ == "__main__":
    sys.exit(main())
