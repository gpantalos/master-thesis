"""
Imports all environments necessary for data generation.
Can be used independently to provide numpy arrays, e.g.
>>> from fpbnn.envs.densities import Densities
>>> env = Densities()
>>> train_data, test_data = env.sample_train_test(100, 10)
"""

experiments = [
    'densities',
    'half_cheetah',
    'inverted_double_pendulum',
    'sinusoids',
    'swimmer',
]
