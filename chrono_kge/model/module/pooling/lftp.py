"""
Factorized trilinear pooling (FTP)
"""

import torch.nn as nn

from chrono_kge.model.module.pooling.lfcp import FactorizedChainedPooling


class FactorizedTrilinearPooling(FactorizedChainedPooling):

    def __init__(self,
                 param_dim1,
                 param_dim2,
                 param_dim3,
                 out_dim,
                 rank,
                 device=None,
                 **kwargs
                 ) -> None:
        """"""
        super().__init__(param_dim1, param_dim2, param_dim3, out_dim, rank, device, **kwargs)

        return

    def __call__(self, emb1, emb2, **kwargs):
        """"""
        emb3 = kwargs.get('emb3')

        x = self.fusion3(emb1, emb2, emb3, self.lin1, self.lin2, self.lin3)
        x = self.dh0(x)
        x = self.sum_pool(x)
        x = self.normalize(x)
        x = self.dh1(self.norm_out(x))
        return x

    def sum_pool(self, x):
        """Sum Pooling. Unfold and sum.
        BS x (k*o) -> BS x ED
        """
        x = x.view(-1, self.out_dim, self.rank)
        x = x.sum(-1)
        return x

    @staticmethod
    def fusion3(emb1, emb2, emb3, proj1, proj2, proj3, act: nn.Module = nn.Identity()):
        """
        (BS x (k*o)) * (BS x (k*o)) -> BS x (k*o)
        """
        return act(proj1(emb1)) * act(proj2(emb2)) * act(proj3(emb3))
