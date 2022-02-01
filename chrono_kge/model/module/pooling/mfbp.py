"""
Multi-modal factorized bilinear pooling (MFB)
"""

import torch.nn as nn

from chrono_kge.model.module.pooling.mlbp import LowRankBilinearPooling
from chrono_kge.utils.vars.defaults import Default


class FactorizedBilinearPooling(LowRankBilinearPooling):

    def __init__(self,
                 param_dim1,
                 param_dim2,
                 out_dim,
                 rank,
                 device=None,
                 **kwargs
                 ) -> None:
        """"""
        super().__init__(param_dim1, param_dim2, out_dim, rank, device, **kwargs)

        self.lin1 = nn.Linear(self.param_dim1, self.rank * self.out_dim, bias=False, device=self.device,
                              dtype=Default.DTYPE)
        self.lin2 = nn.Linear(self.param_dim2, self.rank * self.out_dim, bias=False, device=self.device,
                              dtype=Default.DTYPE)

        return

    def __call__(self, emb1, emb2, **kwargs):
        """"""
        x = self.fusion2(emb1, emb2, self.lin1, self.lin2)
        x = self.dh0(x)
        x = self.sum_pool(x)
        x = self.normalize(x)
        x = self.dh1(self.norm_out(x))
        return x

    def sum_pool(self, x):
        """Sum Pooling. Unfold and sum.
        n x (k*o) -> n x k x o -> n x o (d_e)
        """
        x = x.view(-1, self.out_dim, self.rank)
        x = x.sum(-1)
        return x

    @staticmethod
    def fusion2(emb1, emb2, proj1, proj2, act: nn.Module = nn.Identity()):
        """
        (BS x (k*o)) * (BS x (k*o)) -> BS x (k*o)
        """
        return act(proj1(emb1)) * act(proj2(emb2))
