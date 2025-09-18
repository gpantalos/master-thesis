from typing import Literal

from pydantic import BaseModel, Field


class ExperimentConfig(BaseModel):
    seed: int | None = None
    create_gif: bool | None = True
    n_particles: int | None = None
    train_size: int | None = None
    test_size: int | None = None
    plot: bool | None = True
    wandb: bool | None = True
    verbose: int | None = Field(None, ge=0, le=2)
    n_plots: int | None = Field(None, gt=0)

    n_iter: int | None = None
    batch_size: int | None = None
    learning_rate: float | None = None
    bandwidth: float | None = None
    activation: str | None = None
    width: int | None = None
    depth: int | None = None
    nn_prior_std: float | None = None
    weight_decay: float | None = None
    buffer_size: int | None = None
    normalize_train_data: bool | None = None
    coeff_prior: float | None = None
    val_fraction: float | None = None
    enable_early_stopping: bool | None = None
    early_stopping_patience: int | None = Field(None, gt=0)
    early_stopping_metric: str | None = None
    early_stopping_mode: str | None = None
    early_stopping_min_delta: float | None = None
    ssge_bandwidth: float | None = None
    ssge_n_eigen: int | None = None
    coeff_entropy: float | None = None


class Architecture(BaseModel):
    activation: str = "elu"
    width: int = 32
    depth: int = 4
    bandwidth: float = 1e-2
    nn_prior_std: float = 1.0


class Training(BaseModel):
    n_iter: int = 10_000
    batch_size: int = 8
    learning_rate: float = 5e-3
    weight_decay: float = 1e-6
    buffer_size: int = 10
    normalize_train_data: bool = True
    coeff_prior: float = 1.0

    val_fraction: float = 0.2
    enable_early_stopping: bool = True
    early_stopping_patience: int = 1000
    early_stopping_metric: Literal["val_nll", "val_rmse"] = "val_nll"
    early_stopping_mode: Literal["min", "max"] = "min"
    early_stopping_min_delta: float = 0.0


class Likelihood(BaseModel):
    prior_mean: float = 0.3
    prior_std: float = 0.03
    noise_std: float = 0.3


class SSGE(BaseModel):
    bandwidth: float = 3.0
    n_eigen: int = 6
    eta: float = 0.01
    coeff_entropy: float = 0.1
    coeff_cross_entropy: float = 0.1
    prior_noise_std: float = 1e-3


class Runtime(BaseModel):
    use_wandb: bool = True
    verbose: int = 1
    image_format: str = "png"


class Config(BaseModel):
    experiment: str
    n_particles: int = 10
    arch: Architecture = Field(default_factory=Architecture)
    train: Training = Field(default_factory=Training)
    like: Likelihood = Field(default_factory=Likelihood)
    runtime: Runtime = Field(default_factory=Runtime)
    ssge: SSGE = Field(default_factory=SSGE)


def create_model_config(model: str, env_name: str, config: ExperimentConfig):
    """Create model-specific config from ExperimentConfig (loading tuned configs)."""
    assert isinstance(config, ExperimentConfig), f"config must be an ExperimentConfig not {type(config)}"

    _arch_def = Architecture()
    _train_def = Training()
    _like_def = Likelihood()
    _runtime_def = Runtime()
    _ssge_def = SSGE()

    arch = Architecture(
        activation=config.activation or _arch_def.activation,
        width=config.width or _arch_def.width,
        depth=config.depth or _arch_def.depth,
        bandwidth=config.bandwidth or _arch_def.bandwidth,
        nn_prior_std=config.nn_prior_std or _arch_def.nn_prior_std,
    )

    train = Training(
        n_iter=config.n_iter or _train_def.n_iter,
        batch_size=config.batch_size or _train_def.batch_size,
        learning_rate=config.learning_rate or _train_def.learning_rate,
        weight_decay=config.weight_decay or _train_def.weight_decay,
        buffer_size=config.buffer_size or _train_def.buffer_size,
        normalize_train_data=config.normalize_train_data
        if config.normalize_train_data is not None
        else _train_def.normalize_train_data,
        coeff_prior=config.coeff_prior or _train_def.coeff_prior,
        val_fraction=config.val_fraction or _train_def.val_fraction,
        enable_early_stopping=config.enable_early_stopping
        if config.enable_early_stopping is not None
        else _train_def.enable_early_stopping,
        early_stopping_patience=config.early_stopping_patience or _train_def.early_stopping_patience,
        early_stopping_metric=config.early_stopping_metric or _train_def.early_stopping_metric,
        early_stopping_mode=config.early_stopping_mode or _train_def.early_stopping_mode,
        early_stopping_min_delta=config.early_stopping_min_delta or _train_def.early_stopping_min_delta,
    )

    runtime = Runtime(
        use_wandb=config.wandb if config.wandb is not None else _runtime_def.use_wandb,
        verbose=config.verbose if config.verbose is not None else _runtime_def.verbose,
        image_format=_runtime_def.image_format,
    )

    like = Likelihood()

    ssge = SSGE(
        bandwidth=config.ssge_bandwidth or _ssge_def.bandwidth,
        n_eigen=config.ssge_n_eigen or _ssge_def.n_eigen,
        eta=_ssge_def.eta,
        coeff_entropy=config.coeff_entropy or _ssge_def.coeff_entropy,
        coeff_cross_entropy=_ssge_def.coeff_cross_entropy,
        prior_noise_std=_ssge_def.prior_noise_std,
    )

    common_kwargs = {
        "experiment": env_name,
        "n_particles": config.n_particles or 10,
        "arch": arch,
        "train": train,
        "like": like,
        "runtime": runtime,
    }

    m = model.lower()
    if m in ("fvi", "fsvgd"):
        common_kwargs["ssge"] = ssge

    return Config(**common_kwargs)
