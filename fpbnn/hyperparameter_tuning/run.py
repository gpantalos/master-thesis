from fpbnn.modules.utils import generate_commands

model_names = {
    'fvi',
    'fsvgd',
    'pacoh',
    'svgd',
    'vi',
}

experiments = {
    'swimmer',
    'sinusoids',
    'densities',
    'half_cheetah',
    'inverted_double_pendulum',
}

generate_commands(
    model_names=model_names,
    n_cpus=100,
    mem=2 ** 13,
    experiments=experiments,
    dry=True,
    long=True,
)
