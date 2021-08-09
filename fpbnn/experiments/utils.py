from pathlib import Path


def generate_commands(
        models_and_experiements: list,
        n_cpus=1,
        n_gpus=1,
        mem=1024,
        long=False,
        dry=False,
        interpreter='/cluster/project/infk/krause/pgeorges/miniconda3/envs/thesis/bin/python',
):
    for model, experiment in models_and_experiements:
        # specify paths
        log_dir = Path.home() / "tuning_logs" / experiment
        err_dir = Path.home() / "tuning_errs" / experiment
        log_path = log_dir / (model + '.log')
        err_path = err_dir / (model + '.err')

        # make directories
        log_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

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
            f'-J {model}.{experiment[:3]} '
            f'{interpreter} {experiment}/{model}.py '
        )
        if dry:
            import os
            os.system(command)
        else:
            print(command)


def get_models_and_experiments():
    config_path = Path(__file__).parents[1] / 'configs'
    mne = []
    for experiment_path in [x for x in config_path.iterdir() if x.is_dir()]:
        models = list(experiment_path.glob('**/*.json'))
        mne.extend([(model.stem, experiment_path.stem) for model in models])
    return mne
