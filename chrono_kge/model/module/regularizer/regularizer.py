"""
Regularizer
"""

import torch.nn as nn

from abc import ABC, abstractmethod


class Regularizer(nn.Module, ABC):

    def __init__(self, weight: float = 0.0, power: int = 1):
        """"""
        super().__init__()
        self.weight = weight
        self.power = power
        return

    @abstractmethod
    def __call__(self, **kwargs):
        """"""
        pass
