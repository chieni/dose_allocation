import numpy as np


class PosteriorPrediction:
    def __init__(self, num_subgroups, x_size):
        self.mean = np.empty((num_subgroups, x_size))
        self.lower = np.empty((num_subgroups, x_size))
        self.upper = np.empty((num_subgroups, x_size))
        self.variance = np.empty((num_subgroups, x_size))

    def set_variables(self, subgroup_idx, mean, lower=None, upper=None, variance=None):
        self.mean[subgroup_idx, :] = mean
        if lower is not None:
            self.lower[subgroup_idx, :] = lower
        if upper is not None:
            self.upper[subgroup_idx, :] = upper
        if variance is not None:
            self.variance[subgroup_idx, :] = variance