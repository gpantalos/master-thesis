from experiments.utils import *

generate_commands(
    models_and_experiements=get_models_and_experiments(),
    n_cpus=10,
    mem=2 ** 15,
    # long=True,
    dry=True,
)
