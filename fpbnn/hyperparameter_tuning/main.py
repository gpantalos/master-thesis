from pathlib import Path

import pandas as pd
from ray import tune

# noinspection PyUnresolvedReferences
import fpbnn.envs
# noinspection PyUnresolvedReferences
import fpbnn.models
from fpbnn.modules.utils import get_dataset_sizes

ENV_SEED = 1234

RESULTS_PATH = Path.home() / 'output' / 'results'
CONFIGS_PATH = Path(__file__).parents[1] / 'configs'


def tune_bayesian_model(model_name: str, env_name: str, config, num_samples=100, env_params=None):
    """
    Hyperparameter tuning for all models except PACOH
    """
    # setup model and environment
    model = eval('fpbnn.models.' + model_name)
    if env_params is None:
        env_params = {}
    env = eval('fpbnn.envs.' + env_name)(seed=ENV_SEED, **env_params)

    # setup files
    path_configs = CONFIGS_PATH / env.name
    path_configs.mkdir(parents=True, exist_ok=True)
    file_configs = path_configs / (model_name.lower() + '.json')
    file_configs.unlink(missing_ok=True)

    path_results = RESULTS_PATH / env.name
    path_results.mkdir(parents=True, exist_ok=True)
    file_results = path_results / (model_name.lower() + '.csv')
    file_results.unlink(missing_ok=True)

    # generate train and test data once
    n_train, n_test = get_dataset_sizes(env.name)
    data = env.generate_meta_test_data(n_tasks=1, n_samples_context=n_train, n_samples_test=n_test)
    x_train, y_train, x_test, y_test = data[0]

    # define objective
    def objective(config, model=model):
        other_params = {'functional_prior': env} if model_name[0] == 'F' else {}
        model = model(train_data=(x_train, y_train), experiment=env.name, **config, **other_params)
        model.fit()
        return model.eval(x_test, y_test, 'test')

    def training_function(config):
        tune.report(**objective(config))

    # run analysis
    analysis = tune.run(training_function, config=config, num_samples=num_samples)
    best_config = analysis.get_best_config(metric='test_nll', mode='min')

    # create pandas dataframes
    df_results = analysis.results_df
    df_best_config = pd.Series(best_config)

    # write results to csv
    df_results.to_csv(str(file_results))

    # write best config to file for experiments
    df_best_config.to_json(str(file_configs), default_handler=str)


def tune_pacoh(env_name: str, config, num_samples=100, env_params=None):
    """
    Hyperparameter tuning for PACOH.
    """
    # setup environment
    from pacoh_nn import PACOH_NN_Regression
    if env_params is None:
        env_params = {}
    env = eval('fpbnn.envs.' + env_name)(seed=ENV_SEED, **env_params)

    # setup files for config
    path_config = CONFIGS_PATH / env.name
    path_config.mkdir(parents=True, exist_ok=True)
    file_config = path_config / 'pacoh.json'
    file_config.unlink(missing_ok=True)

    # setup files for results
    path_results = RESULTS_PATH / env.name
    path_results.mkdir(parents=True, exist_ok=True)
    file_results = path_results / 'pacoh.csv'
    file_results.unlink(missing_ok=True)

    # generate train and test data once
    n_train, n_test = get_dataset_sizes(env.name)
    meta_train_data = env.generate_meta_train_data(n_tasks=10, n_samples=100)
    meta_test_data = env.generate_meta_test_data(n_tasks=10, n_samples_context=n_train, n_samples_test=n_test)

    # define objective
    def objective(config):
        model = PACOH_NN_Regression(meta_train_data=meta_train_data, **config)
        model.meta_fit()
        mean, std = model.meta_eval_datasets(meta_test_data)
        return mean

    def training_function(config):
        tune.report(**objective(config))

    # run search
    analysis = tune.run(training_function, config=config, num_samples=num_samples)
    best = analysis.get_best_config(metric='test_nll', mode='min')

    # create dfs
    df_best = pd.Series(best)
    df_results = analysis.results_df

    # save results
    df_results.to_csv(str(file_results))

    # save best config
    df_best.to_json(str(file_config), default_handler=str)


def tune_model(file_str: str, hyper_params: dict):
    """
    Args:
        file_str: corresponds to the result of the `__file__` 
          keyword for each file in the `hyperparameter_tuning` directory
        hyper_params: dict of hyperparameters both from tune and fixed
    """
    model_name = file_str.split('/')[-1].split('.')[0].upper()
    env_name = file_str.split('/')[-2].title().replace('_', '')

    assert model_name in {'VI', 'SVGD', 'FVI', 'FSVGD', 'PACOH'}
    assert env_name in {'InvertedDoublePendulum', 'Swimmer', 'Sinusoids', 'Densities', 'HalfCheetah'}

    if model_name == 'PACOH':
        tune_pacoh(env_name, config=hyper_params)
    else:
        tune_bayesian_model(model_name, env_name, config=hyper_params)


def tune_and_run(file_str: str, hyper_params: dict):
    from experiments.main import run_model
    tune_model(file_str, hyper_params)
    run_model(file_str)
