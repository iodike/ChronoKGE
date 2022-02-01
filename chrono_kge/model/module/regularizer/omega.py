"""
Omega regularizer
"""

import torch

from chrono_kge.model.module.regularizer.regularizer import Regularizer


class Omega(Regularizer):
    """Omega regularizer
    Constraints embeddings using lp-norm, c.f., TNTComplEx page 4, eq. 4.

    Params:
    TuckerTNT: p=4, (q=2)
    TNTComplEx: p=3
    """

    def __init__(self, weight: float, power: int):
        super().__init__(weight, power)
        return

    def __call__(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(torch.abs(f) ** self.power)
        return norm / factors[0].shape[0]
