from abc import ABC, abstractmethod

import numpy as np
import torch as th


class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights


class Sampler(ScheduleSampler):
    def __init__(self, name, ddpm_num_steps):
        if name == 'normal_left':
            x = np.arange(-3 + 3 / ddpm_num_steps, 3 / ddpm_num_steps, 3 / ddpm_num_steps)
            self._weights = np.exp(- (x ** 2 / 2))
        elif name == 'uniform':
            self._weights = np.ones([ddpm_num_steps])

    def weights(self):
        return self._weights
