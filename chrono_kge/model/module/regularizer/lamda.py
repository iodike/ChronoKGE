"""
Lambda regularizer
"""

import torch

from chrono_kge.model.module.regularizer.regularizer import Regularizer


class Lambda(Regularizer):
    """Lambda regularizer
    Smoothness of time representations, c.f., TNTComplEx page 4, eq. 5.

    Params:
    TuckerTNT: p=4, (q=2)
    TNTComplEx: p=4
    """

    def __init__(self, weight: float, power: int):
        super().__init__(weight, power)
        return

    def __call__(self, factor):
        ddiff = factor[1:] - factor[:-1]
        rank = int(ddiff.shape[1] / 2)
        diff = torch.sqrt(ddiff[:, :rank] ** 2 + ddiff[:, rank:] ** 2) ** self.power
        return self.weight * torch.sum(diff) / (factor.shape[0] - 1)
