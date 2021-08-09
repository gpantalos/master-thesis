import numpy as np

from fpbnn.envs.regression import RegressionDataset


class Sinusoids(RegressionDataset):
    def __init__(
            self,
            # uniform variables
            amplitude=(0.9, 1.1),
            period=(0.9, 1.1),
            # normal variables
            slope=(0.0, 0.01),
            x_shift=(0.0, 0.01),
            y_shift=(0.0, 0.01),
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.amplitude = amplitude
        self.period = period
        self.x_shift = x_shift
        self.y_shift = y_shift
        self.slope = slope

    def sample_function(self):
        a = self.random_state.uniform(*self.amplitude)
        t = self.random_state.uniform(*self.period)
        m = self.random_state.normal(*self.slope)
        x0 = self.random_state.normal(*self.x_shift)
        y0 = self.random_state.normal(*self.y_shift)

        def g(x):
            return m * x + a * np.sin(2 * np.pi / t * (x - x0)) + y0

        return g
