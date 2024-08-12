import time

import torch
import numpy as np

from timm import utils
from .epsilon_greedy import EpsilonGreedy


class UCB(EpsilonGreedy):
    def __init__(self,
            logger,
            args,
            args_text,
            has_apex,
            has_native_amp,
            has_compile,
            has_wandb):
        super(UCB, self).__init__(
            logger,
            args,
            args_text,
            has_apex,
            has_native_amp,
            has_compile,
            has_wandb)

        self.confidence = self.args.confidence

        self.uncertainties = np.zeros(shape=(self.num_train_samples,))
        self.variances = np.zeros(shape=(self.num_train_samples,))

    def update_batch_score(self, loss, indices):
        # update scores
        uncertainty = loss.detach().cpu().numpy()

        self.variances[indices] = (1 - self.alpha) * self.variances[indices] + self.alpha * (uncertainty - self.uncertainties[indices])**2
        self.uncertainties[indices] = (1 - self.alpha) * self.uncertainties[indices] + self.alpha * uncertainty

        self.scores[indices] = self.uncertainties[indices] + self.confidence * self.variances[indices]