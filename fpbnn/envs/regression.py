import numpy as np
from tqdm import trange


def camel_to_snake(camel_string):
    return ''.join(['_' + i.lower() if i.isupper() else i for i in camel_string]).lstrip('_')


class RegressionDataset:
    def __init__(self, x_range=(-1, 1), noise_std=0.0, seed=None, verbose=0):
        self.verbose = verbose
        self.noise_std = noise_std
        self.x_range = x_range
        self.input_dim = 1
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        self.name = camel_to_snake(self.__class__.__name__)
        self.dtype = np.float32

    def handle(self, x):
        if np.ndim(x) == 1:
            x = x[..., None]
        return x.astype(self.dtype)

    def sample(self, n_samples, batch_size, n_particles):
        x_samples, y_samples = [], []
        for _ in trange(n_samples, desc='generating samples', disable=self.verbose < 1):
            x = self.random_state.uniform(*self.x_range, [batch_size, self.input_dim]).astype(np.float32)
            y = []
            for _ in range(n_particles):
                f = self.sample_function()
                y.append(f(x))
            y = np.stack(y)
            y += self.random_state.normal(0, self.noise_std, y.shape)
            x_samples.append(x)
            y_samples.append(y)
        return np.stack(x_samples), np.stack(y_samples)

    def generate_meta_train_data(self, n_tasks, n_samples):
        meta_train_tuples = []
        for _ in trange(n_tasks, desc='generating meta-train tasks', disable=self.verbose < 1):
            f = self.sample_function()
            x = self.random_state.uniform(*self.x_range, [n_samples, self.input_dim])
            y = f(x)
            y += self.random_state.normal(0, self.noise_std, y.shape)
            meta_train_tuples.append((x, y))
        return meta_train_tuples

    def generate_meta_test_data(self, n_tasks, n_samples_context, n_samples_test):
        """
        Args:
            n_tasks: number of runs
            n_samples_context: number of training points
            n_samples_test: number of test points
        """
        assert n_samples_test > 0
        meta_test_tuples = []
        for _ in trange(n_tasks, desc='generating meta-test tasks', disable=self.verbose < 1):
            f = self.sample_function()
            x = self.random_state.uniform(*self.x_range, [n_samples_context + n_samples_test, self.input_dim])
            y = f(x)
            y += self.random_state.normal(0, self.noise_std, y.shape)
            x_train, x_test = x[:n_samples_context], x[n_samples_context:]
            y_train, y_test = y[:n_samples_context], y[n_samples_context:]
            meta_test_tuples.append((x_train, y_train, x_test, y_test))
        return meta_test_tuples

    def sample_function(self):
        raise NotImplemented

    def sample_train_test(self, n_train, n_test):
        """
        :param n_train: number of training points
        :param n_test: number of test points
        :return: two tuples containing observations and labels for train and test
        """
        f = self.sample_function()

        # sample and evaluate train data
        x_train = self.random_state.uniform(*self.x_range, [n_train, self.input_dim]).astype(np.float32)
        y_train = f(x_train)

        # add noise to training data
        y_train += self.random_state.normal(0, self.noise_std, y_train.shape)

        # sample and evaluate test data
        x_test = self.random_state.uniform(*self.x_range, [n_test, self.input_dim]).astype(np.float32)
        y_test = f(x_test)
        return (x_train, y_train), (x_test, y_test)
