import os

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
from matplotlib import pyplot as plt

from fpbnn.modules.affine_transform import Affine
from fpbnn.modules.neural_network import BatchedMLP
from fpbnn.modules.prior_posterior import GaussianPrior, GaussianPosterior

tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tfp.math.psd_kernels


class BayesianRegressionModel:
    """Base class for bayesian regression."""

    def __init__(
            self,
            train_data,
            experiment,
            n_iter=1e4,
            batch_size=8,
            n_particles=10,
            activation="elu",
            width=32,
            depth=4,
            bandwidth=1e-2,
            nn_prior_std=1.0,
            likelihood_prior_mean=1e-2,
            likelihood_prior_std=1e-2,
            learning_rate=1e-3,
            weight_decay=1e-6,
            coeff_prior=1.0,
            noise_std=1e-2,
            image_format="png",
            verbose=0,
            normalize_train_data=True,
    ):
        assert nn_prior_std > 0.
        assert likelihood_prior_std > 0.
        x_train, y_train = train_data
        self.image_format = image_format
        self.num_train_samples = len(x_train)

        # batch size
        if self.num_train_samples < batch_size:
            self.batch_size = self.num_train_samples
            raise Warning('batch size bigger than number of training points,'
                          f' reducing batch size to {self.num_train_samples}')
        else:
            self.batch_size = int(batch_size)

        # hyperparameters
        self.n_iter = int(n_iter)
        self.likelihood_prior_std = likelihood_prior_std
        self.likelihood_prior_mean = np.log(likelihood_prior_mean)
        self.nn_prior_std = nn_prior_std
        self.bandwidth = bandwidth
        self.kernel = tfk.ExponentiatedQuadratic(length_scale=self.bandwidth)
        self.activation = eval("tf.nn." + activation)
        self.coeff_prior = coeff_prior / self.num_train_samples
        self.n_particles = n_particles
        self.hidden_layer_sizes = tuple([width] * depth)
        self.noise_std = noise_std

        # compute normalization stats
        x_train, y_train = self._broadcast_and_cast_dtype(x_train, y_train)
        self.x_mean = tf.reduce_mean(x_train, 0)
        self.y_mean = tf.reduce_mean(y_train, 0)
        self.x_std = tfp.stats.stddev(x_train, 0)
        self.y_std = tfp.stats.stddev(y_train, 0)

        # normalize training data
        if normalize_train_data:
            self.x_train, self.y_train = self.normalize(x_train, y_train)
        else:
            self.x_train, self.y_train = x_train, y_train
        self.y_train += tf.random.normal(tf.shape(self.y_train), 0.0, self.noise_std)

        # setup training data
        self.unnormalized_x_train = x_train
        self.unnormalized_y_train = self._unnormalize_preds(self.y_train)

        # define affine transform for predictive distribution
        self.affine_transform = Affine(self.y_mean, self.y_std)

        # record dimension stats
        self.input_size = self.x_train.shape[-1]
        self.output_size = self.y_train.shape[-1]

        # setup place for likelihood parameters
        self.likelihood_param_size = self.output_size

        # setup optimizer
        self.optimizer = tfa.optimizers.AdamW(
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            name='optimizerAdamW'
        )

        # setup nn
        self.nn = BatchedMLP(self.n_particles, self.input_size, self.output_size, self.hidden_layer_sizes,
                             self.activation)
        self.nn_params = self.nn.get_variables_stacked_per_model()
        self.nn_param_size = self.nn_params.shape[-1]
        self.prior = GaussianPrior(
            nn_param_size=self.nn_param_size,
            nn_prior_std=self.nn_prior_std,
            likelihood_param_size=self.likelihood_param_size,
            likelihood_prior_mean=likelihood_prior_mean,
            likelihood_prior_std=likelihood_prior_std
        )
        # prior for nn parameters (weights & biases)
        self.prior = GaussianPrior(
            nn_param_size=self.nn_param_size,
            nn_prior_std=self.nn_prior_std,
            likelihood_param_size=self.likelihood_param_size,
            likelihood_prior_mean=likelihood_prior_mean,
            likelihood_prior_std=likelihood_prior_std
        )
        self.posterior = GaussianPosterior(self.nn_params, self.likelihood_param_size)

        # Abstract params
        self.name = None
        self.experiment = experiment + '/'
        self.home_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

        # summary writers
        # current_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        # log_dir = self.home_dir + 'logs/' + current_time
        # self.summary_writer = tf.summary.create_file_writer(log_dir)

        # other params
        self.verbose = verbose

    def fit(self, val_data=None, test_data=None, n_plots=50, plot=False, n_seconds_gif=3, **kwargs):
        """
        Fit train to test data.
        Args:
            val_data: tuple (x_val, y_val). If set to None, does not compute metrics for validation data.
            test_data: tuple (x_test, y_test). If set to None, does not compute metrics for test data.
            n_plots: number of plots to produce during training phase. Requires test data not to be None.
            plot: boolean for plotting results.
            n_seconds_gif: duration of gif of training procedure in seconds.
        """
        from tqdm import trange

        log_period = self.n_iter // n_plots
        if log_period == 0:
            log_period = 1
        plot1d = (self.output_size == 1) and (self.input_size == 1) and plot and (test_data is not None)
        train_batch_sampler = self._get_batch_sampler()
        if self.verbose < 1:
            pbar = trange(self.n_iter + 1, disable=True, position=0)
        else:
            pbar = trange(self.n_iter + 1, desc=f"Training {self.name:12}", position=0)
        train_data = (self.x_train, self.y_train)
        message = dict()

        # setup trainer
        for i in pbar:

            # training
            x_batch, y_batch = next(train_batch_sampler)
            self(x_batch, y_batch)

            if i % log_period == 0:
                message.update(self.eval(*train_data, 'train'))
                # with self.summary_writer.as_default():
                #     tf.summary.scalar('train_nll', message['train_nll'], step=i)
                #     tf.summary.scalar('train_rmse', message['train_rmse'], step=i)

                # validation
                if val_data is not None:
                    message.update(self.eval(*val_data, 'val'))
                    # with self.summary_writer.as_default():
                    #     tf.summary.scalar('val_nll', message['val_nll'], step=i)
                    #     tf.summary.scalar('val_rmse', message['val_rmse'], step=i)
                if plot1d:
                    self.plot_predictions_1d(test_data, i, **kwargs)
                pbar.set_postfix(message)

        # make gif
        if plot1d:
            self.make_gif(fps=n_plots // n_seconds_gif)

    def eval(self, x, y, data_type='val'):
        """
        Evaluate model according to NLL and RMSE.
        Args:
            x: index points
            y: true targets
            data_type: 'val', 'test', or 'train'
        """
        x, y = self._broadcast_and_cast_dtype(x, y)
        y_pred, pred_dist = self.predict(x)
        nll = -tf.reduce_mean(pred_dist.log_prob(y))
        rmse = tf.sqrt(tf.reduce_mean(tf.square(y_pred - y)))
        return {
            data_type + '_nll': nll.numpy(),
            data_type + '_rmse': rmse.numpy(),
        }

    def split_params(self, params):
        """
        Splits model parameters into neural network parameters and likelihood standard deviation.
        """
        likelihood_params = params[:, -self.likelihood_param_size:]
        likelihood_std = tf.exp(likelihood_params)
        nn_params = params[:, :self.nn_param_size]
        return nn_params, likelihood_std

    def plot_predictions_1d(self, plot_data, iteration, show=False):
        """
        Plots predictions and saves images at (self.home_dir + "output/figures/")
        """
        assert self.input_size == 1 and self.output_size == 1, "Data is not 1D"
        x_plot, y_plot = plot_data
        x_plot = tf.squeeze(x_plot)
        indices = tf.argsort(x_plot)
        x_plot = tf.gather(x_plot, indices)
        y_plot = tf.gather(y_plot, indices)

        y_preds, pred_dist = self.predict(x_plot)
        plt.figure(figsize=(6, 4))

        # plot predictive mean and confidence interval
        mu, sigma = pred_dist.mean, pred_dist.stddev
        for y_pred in y_preds:
            plt.plot(x_plot, y_pred, lw=0.1, c='r')
        plt.plot(x_plot, mu, lw=1, c='r')
        lcb, ucb = mu - 2 * sigma, mu + 2 * sigma
        lcb, ucb = lcb.numpy().flatten(), ucb.numpy().flatten()
        plt.fill_between(x_plot, lcb, ucb, alpha=0.1, color='r')

        # unnormalize training data & plot it
        x_train, y_train = self.unnormalized_x_train, self.unnormalized_y_train
        plt.scatter(x_train, y_train, label="train", marker='x', lw=1, color='b')

        # plot test data
        plt.scatter(x_plot, y_plot, label="test", s=1, alpha=0.1, color='g')

        # plot properties
        plt.title(f"iteration {iteration}")
        height = np.max(y_plot) - np.min(y_plot)
        plt.ylim([np.min(y_plot) - height, np.max(y_plot) + height])
        plt.legend()
        plt.tight_layout()

        # save the images and delete previous ones
        folder = self.home_dir + "output/figures/" + self.experiment
        os.makedirs(folder, exist_ok=True)
        folder += self.name
        if iteration == 0:
            if os.path.isdir(folder):
                for filename in os.listdir(folder):
                    file_path = os.path.join(folder, filename)
                    os.remove(file_path)
            os.makedirs(folder, exist_ok=True)
        plt.savefig(folder + f"/{iteration:010d}.{self.image_format}")
        if show:
            plt.show()
        plt.close()

    def _get_batch_sampler(self):
        train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        train_dataset = train_dataset.shuffle(self.batch_size, reshuffle_each_iteration=True)
        train_dataset = train_dataset.repeat()
        train_dataset = train_dataset.batch(self.batch_size)
        return iter(train_dataset)

    def normalize(self, x, y=None):
        x = (x - self.x_mean) / self.x_std
        if y is None:
            return x
        else:
            y = (y - self.y_mean) / self.y_std
            return x, y

    def _unnormalize_preds(self, y):
        return y * self.y_std + self.y_mean

    def _unnormalize_predictive_dist(self, pred_dist):
        return self.affine_transform(pred_dist)

    @staticmethod
    def _predictive_mixture(y_pred, likelihood_std):
        """
        Creates a mixture of isotropic Gaussian distributions using y_pred as mean and likelihood_std as 
        standard deviation.
        """
        categorical = tfd.Categorical(tf.transpose(tf.zeros(y_pred.shape[:2])))

        def gaussian(params): return tfd.Independent(tfd.Normal(*params), 1)

        # create a list of length equal to the number of particles
        components = list(map(gaussian, zip(y_pred, likelihood_std)))
        mixture = tfp.distributions.Mixture(categorical, components)
        return mixture

    @staticmethod
    def _broadcast_and_cast_dtype(x, y=None, dtype=tf.float32):
        if x.ndim == 1:
            x = tf.expand_dims(x, -1)
        if y is not None:
            assert len(x) == len(y)
            if y.ndim == 1:
                y = tf.expand_dims(y, -1)
            return tf.cast(x, dtype=dtype), tf.cast(y, dtype=dtype)
        else:
            return tf.cast(x, dtype=dtype)

    def make_gif(self, fps=10):
        from tqdm import tqdm
        import imageio
        from pygifsicle import optimize

        folder1 = self.home_dir + "output/figures/"
        folder2 = self.home_dir + "output/gifs/"
        gif_folder = folder2 + self.experiment
        os.makedirs(folder2, exist_ok=True)
        os.makedirs(gif_folder, exist_ok=True)
        save_path = folder2 + self.experiment + self.name + ".gif"
        filenames = os.listdir(folder1 + self.experiment + self.name)
        filenames = sorted(filenames)
        fig_folder = folder1 + self.experiment + self.name + "/"
        filenames = [fig_folder + i for i in filenames if i.endswith(".png")]
        images = []
        for filename in tqdm(filenames, desc="Creating GIF"):
            images.append(imageio.imread(filename))
        images = images + [images[-1]] * fps
        imageio.mimsave(save_path, images, fps=fps)
        optimize(save_path)
        print(f'Created GIF at {os.path.abspath(save_path)}!')

    @staticmethod
    def ll(y_true, y_pred, likelihood_std, reduction='sum'):
        """
        Computes log likelihood of 
        """
        likelihood_std = tf.expand_dims(likelihood_std, axis=1)
        likelihood = tfd.Independent(tfd.Normal(y_pred, likelihood_std), reinterpreted_batch_ndims=1)
        log_likelihood = likelihood.log_prob(y_true)
        if reduction == 'sum':
            return tf.reduce_sum(log_likelihood)
        elif reduction == 'mean':
            return tf.reduce_mean(log_likelihood)
        elif reduction == 'logsumexp':
            return tf.reduce_logsumexp(log_likelihood)
        else:
            raise NotImplemented

    def predict(self, x):
        raise NotImplementedError

    def __call__(self, x_batch, y_batch):
        raise NotImplementedError

    def write_csv(self, results):
        import pandas as pd

        # save results to csv
        save_path = self.home_dir + "output/results/" + self.experiment
        os.makedirs(save_path, exist_ok=True)
        save_path += self.name + '.csv'
        df = pd.DataFrame(results)
        df.to_csv(save_path)
        print(f"Check results at {os.path.abspath(save_path)}!")
