from pathlib import Path

import pandas as pd

# noinspection PyUnresolvedReferences
import fpbnn.envs
# noinspection PyUnresolvedReferences
import fpbnn.models
from fpbnn.modules.utils import get_dataset_sizes

ENV_SEED = 1234
N_RUNS = 10
N_ITER = int(1e5)

RESULTS_PATH = Path.home() / 'output' / 'experiments'
CONFIGS_PATH = Path(__file__).parents[1] / 'configs'


def run_bayesian_model(model_name: str, env_name: str, env_params=None):
    """
    Runner for all models except pacoh
    :param model_name: class name in the models directory
    :param env_name: experiment name in snake case
    :param env_params: dictionnary with environment parameters
    :return: None
    """
    # setup model and environment
    model = eval('fpbnn.models.' + model_name)
    if env_params is None:
        env_params = {}
    env = eval('fpbnn.envs.' + env_name)(seed=ENV_SEED, **env_params)

    # generate train and test data
    n_train, n_test = get_dataset_sizes(env.name)
    data = env.generate_meta_test_data(n_tasks=N_RUNS, n_samples_context=n_train, n_samples_test=n_test)

    # load hyper-parameters from json
    file = CONFIGS_PATH / env.name / (model_name.lower() + '.json')
    config = pd.read_json(str(file), typ='series').to_dict()

    # run model
    f_param = {'functional_prior': env} if model_name[0] == 'F' else {}
    results = []
    for x_train, y_train, x_test, y_test in data:
        model_instance = model(train_data=(x_train, y_train), experiment=env.name, **config, **f_param, n_iter=N_ITER)
        model_instance.fit()
        results.append(model_instance.eval(x_test, y_test, 'test'))

    # setup files
    path = RESULTS_PATH / env.name
    path.mkdir(parents=True, exist_ok=True)
    file = path / (model_name.lower() + '.csv')
    file.unlink(missing_ok=True)

    # just keep mean and std of results
    df1 = pd.DataFrame(results)
    df2 = pd.DataFrame()
    df1.columns = [_[5:] for _ in df1.columns]
    df2['mean'] = df1.mean()
    df2['std'] = df1.std()

    # write results to csv
    df2.to_csv(str(file))


def run_pacoh(env_name: str, env_params=None):
    # setup environment
    from pacoh_nn.pacoh_nn_regression import PACOH_NN_Regression
    if env_params is None:
        env_params = {}
    env = eval('data.' + env_name)(seed=ENV_SEED, **env_params)

    # read hyper-parameters
    config = pd.read_json(str(CONFIGS_PATH / env.name / 'pacoh.json'))

    # generate meta-train and meta-test data
    meta_train_data = env.generate_meta_train_data(n_tasks=100, n_samples=100)
    n_train, n_test = get_dataset_sizes(env.name)
    meta_test_data = env.generate_meta_test_data(n_tasks=N_RUNS,
                                                 n_samples_context=n_train, n_samples_test=n_test)
    # run model
    model = PACOH_NN_Regression(meta_train_data=meta_train_data, **config, num_iter_meta_test=N_ITER)
    model.meta_fit()
    mean, std = model.meta_eval_datasets(meta_test_data)
    results = dict(mean=mean, std=std)

    # setup files
    path = RESULTS_PATH / env.name
    path.mkdir(parents=True, exist_ok=True)
    file = path / 'pacoh.csv'
    file.unlink(missing_ok=True)

    # write results to csv
    df = pd.DataFrame(results)
    df.to_csv(str(file))


def run_model(file_str):
    """
    Runner for all models.
    Args:
        file_str: __file__ attribute from the respective file in hyperparameter_tuning directory.
    """

    model_name = file_str.split('/')[-1].split('.')[0].upper()
    env_name = file_str.split('/')[-2].title().replace('_', '')

    assert model_name in {'VI', 'SVGD', 'FVI', 'FSVGD', 'PACOH'}
    assert env_name in {'InvertedDoublePendulum', 'Swimmer', 'Sinusoids', 'Densities', 'HalfCheetah'}
    if model_name == 'PACOH':
        run_pacoh(env_name)
    else:
        run_bayesian_model(model_name, env_name)
