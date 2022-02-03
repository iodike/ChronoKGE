"""
Multi-modal chained factorized bilinear pooling (MFB-C)
"""

import torch.nn as nn

from chrono_kge.model.module.pooling.lfbp import FactorizedBilinearPooling
from chrono_kge.utils.vars.defaults import Default


class FactorizedChainedPooling(FactorizedBilinearPooling):

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
        super().__init__(param_dim1, param_dim2, out_dim, rank, device, **kwargs)

        self.param_dim3 = param_dim3

        self.lin1 = nn.Linear(self.param_dim1, self.rank * self.out_dim, bias=False, device=self.device,
                              dtype=Default.DTYPE)
        self.lin2 = nn.Linear(self.param_dim2, self.rank * self.out_dim, bias=False, device=self.device,
                              dtype=Default.DTYPE)
        self.lin3 = nn.Linear(self.param_dim3, self.rank * self.out_dim, bias=False, device=self.device,
                              dtype=Default.DTYPE)
        self.lin4 = nn.Linear(self.out_dim * self.rank, self.rank * self.out_dim, bias=False, device=self.device,
                              dtype=Default.DTYPE)

        self.dh2 = nn.Dropout(self.dh[2])

        return

    def __call__(self, emb1, emb2, **kwargs):
        """"""
        emb3 = kwargs.get('emb3')

        z = self.fusion2(emb1, emb2, self.lin1, self.lin2)
        z = self.dh0(z)
        x = self.fusion2(emb3, z, self.lin3, self.lin4)
        x = self.dh1(x)
        x = self.sum_pool(x)
        x = self.normalize(x)
        x = self.dh2(self.norm_out(x))
        return x
