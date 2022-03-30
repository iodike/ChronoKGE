"""
Low-rank Bilinear Pooling (LRBP)
"""

import torch
import torch.nn as nn
import torch.nn.functional as func

from chrono_kge.utils.vars.defaults import Default


class LowRankBilinearPooling(nn.Module):

    def __init__(self,
                 param_dim1: int,
                 param_dim2: int,
                 out_dim: int,
                 rank: int,
                 device=None,
                 **kwargs
                 ) -> None:
        """"""
        super().__init__()

        self.device = device
        self.kwargs = kwargs

        self.param_dim1 = param_dim1
        self.param_dim2 = param_dim2
        self.out_dim = out_dim

        self.rank = rank

        # dropout
        self.dh = self.kwargs.get('dh')
        self.dh0 = nn.Dropout(self.dh[0])
        self.dh1 = nn.Dropout(self.dh[1])

        # normalization
        self.norm_out = nn.BatchNorm1d(self.out_dim)

        return

    @staticmethod
    def normalize(x, l2: bool = True, lp: bool = True, dim: int = -1, eps: float = Default.ALPHA):
        """"""
        if lp:
            x = torch.mul(torch.sign(x), torch.sqrt(torch.add(torch.abs(x), eps)))
        if l2:
            x = nn.functional.normalize(x, p=2, dim=dim)
        return x
