import inspect
import os
import random
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Literal


def _get_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        root = parent if parent.is_dir() else parent.parent
        if (root / "pyproject.toml").exists() or (root / ".git").exists():
            return root
    raise RuntimeError("Could not find project root (no pyproject.toml or .git found)")


@dataclass
class Paths:
    """Simple dataclass for project paths."""

    root: Path = _get_root()
    output: Path = root / "output"
    logs: Path = output / "logs"
    experiments: Path = output / "experiments"
    figures: Path = output / "figures"
    gifs: Path = output / "gifs"
    results: Path = output / "results"
    configs: Path = root / "configs"
    ray_results: Path = root / "ray_results"
    state_comparisons: Path = output / "state_comparisons"

    def logs_for_experiment(self, experiment: str, when: datetime | None = None) -> Path:
        """Get logs directory for a specific experiment with timestamp."""
        timestamp = (when or datetime.now()).strftime("%Y%m%d-%H%M%S")
        return self.logs / experiment / timestamp

    def figures_for_experiment(self, experiment: str, name: str) -> Path:
        """Get figures directory for a specific experiment and name."""
        return self.figures / experiment / name

    def experiments_for_env_model(self, env: str, model: str) -> Path:
        """Get experiments directory for a specific environment and model."""
        return self.experiments / f"{env}-{model.lower()}"

    def tuned_config_for_env_model(self, env: str, model: str) -> Path:
        """Get path to tuned config file for a specific environment and model."""
        return self.configs / env / f"{model.lower()}.yaml"

    def tuning_results_for_env_model(self, env: str, model: str) -> Path:
        """Get path to tuning results file for a specific environment and model."""
        return self.results / env / f"{model.lower()}.csv"


paths = Paths()


def generate_commands(
    model_names: Iterable[str],
    experiments: Iterable[str],
    n_cpus: int = 1,
    n_gpus: int = 1,
    mem: int = 1024,
    long: bool = False,
    interpreter: str = "/cluster/project/infk/krause/pgeorges/miniconda3/envs/thesis/bin/python",
    dry: bool = False,
) -> None:
    for experiment in experiments:
        log_dir = paths.output / "tuning_logs" / experiment
        err_dir = paths.output / "tuning_errs" / experiment

        log_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        for model_name in model_names:
            log_path = log_dir / (model_name + ".log")
            err_path = err_dir / (model_name + ".err")

            log_path.unlink(missing_ok=True)
            err_path.unlink(missing_ok=True)

            command = (
                f"bsub -N "
                f"-n {int(n_cpus)} "
                f"-W {23 if long else 3}:59 "
                f'-R "rusage[mem={int(mem)}, ngpus_excl_p={int(n_gpus)}]" '
                f"-o {str(log_path)} "
                f"-e {str(err_path)} "
                f"-J {model_name}.{experiment[:3]} "
                f"{interpreter} {experiment}/{model_name}.py "
            )
            if dry:
                subprocess.run(command, shell=True, check=True)
            else:
                print(command)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and TensorFlow RNGs and enable deterministic ops."""
    import numpy as np
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()


def get_dataset_sizes(env_name: str, train_multiplier: int = 100, test_multiplier: int = 1000) -> tuple[int, int]:
    """Get dataset sizes for training and testing based on environment dimensionality."""

    d = get_env_input_dim(env_name)

    n_train = train_multiplier * d
    n_test = test_multiplier * d

    return n_train, n_test


def get_env_input_dim(env_name: str) -> int:
    """Get the input dimensionality for a given environment."""
    import fpbnn.envs as envs
    from fpbnn.envs import AbstractRegressionEnv, available_envs

    assert env_name in available_envs, f"Unknown environment: {env_name}. Available environments: {available_envs}"

    env_class_name = resolve_class_name(envs, env_name)
    env_class = getattr(envs, env_class_name)
    env: AbstractRegressionEnv = env_class()
    return env.input_dim


def camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case.
    (e.g., "XMLHttp" -> "xml_http").
    """
    assert isinstance(name, str) and name != "", "name must be a non-empty string"
    return "".join(("_" + c.lower()) if c.isupper() else c for c in name).lstrip("_")


def module_canonical_names(module: ModuleType) -> list[str]:
    """Return class names defined in the module, without relying on __all__.

    Selection rules:
    - Only classes defined in this module (cls.__module__ == module.__name__)
    - Exclude abstract/base classes by name (prefix "Abstract")
    - Keep only ClassNames starting with an uppercase letter
    """
    classes = [
        name
        for name, obj in inspect.getmembers(module, inspect.isclass)
        if getattr(obj, "__module__", None) == module.__name__
        and name
        and name[0].isupper()
        and not name.startswith("Abstract")
    ]
    return classes


def keys_for_module(module: ModuleType, style: Literal["lower", "snake"] = "lower") -> list[str]:
    """Return available CLI-friendly keys for a module's exported names.

    - lower: lowercase of canonical names
    - snake: snake_case of canonical names
    """
    assert style in ("lower", "snake"), f"Invalid style: {style}"
    names = module_canonical_names(module)
    if style == "lower":
        return [n.lower() for n in names]
    return [camel_to_snake(n) for n in names]


def resolve_class_name(module: ModuleType, name_key: str) -> str:
    """Resolve a user-provided key (lower/snake/case-insensitive) to the canonical exported name.

    Examples:
    - "mlp" -> "MLP"
    - "inverted_double_pendulum" -> "InvertedDoublePendulum"
    - "Vi" -> "VI"
    """
    assert isinstance(name_key, str) and name_key != "", "name_key must be a non-empty string"
    canonical = module_canonical_names(module)

    if name_key in canonical:
        return name_key

    lower_key = name_key.lower()
    for n in canonical:
        if n.lower() == lower_key:
            return n

    lower_map = {n.lower(): n for n in canonical}
    if lower_key in lower_map:
        return lower_map[lower_key]

    snake_map = {camel_to_snake(n): n for n in canonical}
    if name_key in snake_map:
        return snake_map[name_key]

    available_lower = list(lower_map.keys())
    available_snake = list(snake_map.keys())
    raise ValueError(
        f"Unknown name '{name_key}' for module {module.__name__}. "
        f"Available: lower={available_lower}, snake={available_snake}"
    )


def get_class_by_key(module: ModuleType, name_key: str) -> type:
    """Return the class exported by a module matching the provided key."""
    name = resolve_class_name(module, name_key)
    return getattr(module, name)
