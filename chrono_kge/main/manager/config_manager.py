"""
Global
"""

import torch
import numpy as np
import random

from torch.backends import cudnn


class ConfigManager:

    def __init__(self):
        """"""
        return

    def init(self):
        """"""
        self.seed()
        return

    @staticmethod
    def seed(seed=85):
        """
        :param seed: Default is 85 (01010101). Reduce bias by setting bits (0/1) with same frequency.
        :return:
        """
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        '''CUDA'''
        if torch.backends.cudnn.is_available():
            cudnn.deterministic = True
            cudnn.benchmark = True

        if torch.cuda.is_available:
            torch.cuda.manual_seed_all(seed)

        return
