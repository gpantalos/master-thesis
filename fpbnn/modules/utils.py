from pathlib import Path

PARENT_LOG_DIR = Path.home() / "tuning_logs"
PARENT_ERR_DIR = Path.home() / "tuning_errs"


def generate_commands(
        model_names: iter,
        experiments: iter,
        n_cpus=1,
        n_gpus=1,
        mem=1024,
        long=False,
        interpreter='/cluster/project/infk/krause/pgeorges/miniconda3/envs/thesis/bin/python',
        dry=False,
):
    for experiment in experiments:
        log_dir = PARENT_LOG_DIR / experiment
        err_dir = PARENT_ERR_DIR / experiment

        # make directories
        log_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        for model_name in model_names:
            log_path = log_dir / (model_name + '.log')
            err_path = err_dir / (model_name + '.err')

            # delete previous files
            log_path.unlink(missing_ok=True)
            err_path.unlink(missing_ok=True)

            command = (
                f'bsub -N '
                f'-n {int(n_cpus)} '
                f'-W {23 if long else 3}:59 '
                f'-R "rusage[mem={int(mem)}, ngpus_excl_p={int(n_gpus)}]" '
                f'-o {str(log_path)} '
                f'-e {str(err_path)} '
                f'-J {model_name}.{experiment[:3]} '
                f'{interpreter} {experiment}/{model_name}.py '
            )
            if dry:
                import os
                os.system(command)
            else:
                print(command)


def get_dataset_sizes(env_name):
    if env_name == "sinusoids":
        return 10, 100
    elif env_name == "densities":
        return 100, 1000
    elif env_name == "inverted_double_pendulum":
        return 1000, 10000
    elif env_name == "swimmer":
        return 1000, 10000
    elif env_name == "half_cheetah":
        return 5000, 50000
    else:
        raise NotImplemented
