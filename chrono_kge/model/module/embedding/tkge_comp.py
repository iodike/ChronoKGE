"""
TKG Embedding - Time Components
"""

import torch
import torch.nn as nn
import torch.nn.functional as nf

from functools import reduce

from chrono_kge.model.module.embedding.tkge import TKGE
from chrono_kge.utils.vars.defaults import Default
from chrono_kge.model.module.embedding.submodule.tce import TCE
from chrono_kge.main.handler.model_handler import ModelHandler
from chrono_kge.main.handler.data_handler import DataHandler
from chrono_kge.main.handler.env_handler import EnvironmentHandler


class TKGE_Components(TKGE):

    def __init__(self,
                 model_handler: ModelHandler,
                 data_handler: DataHandler,
                 env_handler: EnvironmentHandler,
                 **kwargs
                 ) -> None:
        """"""
        super().__init__(model_handler, data_handler, env_handler, **kwargs)

        self.c_dim = self.MODEL.time_dim

        self.tce = TCE(self.DATA.kg, self.c_dim, self.ENV.device)

        self.lin_layers = [nn.Linear(self.c_dim, self.MODEL.time_dim, bias=False, device=self.ENV.device,
                           dtype=Default.DTYPE) for _ in range(len(self.tce.bounds))]

        return

    def init(self) -> None:
        """"""
        super().init()
        self.tce.init()
        return

    def __call__(self, x):
        """"""
        es = self.emb_E(x[:, 0])
        er_t = self.emb_R1(x[:, 1])
        er_nt = self.emb_R2(x[:, 1])
        eo = self.emb_E(x[:, 3])

        '''TCE'''
        et_cs = self.tce(x[:, 2])

        '''projection + aggregation'''
        et = reduce(torch.Tensor.add_, [self.lin_layers[i](et_c) for i, et_c in enumerate(et_cs)],
                    torch.zeros((x.shape[0], self.MODEL.time_dim), dtype=Default.DTYPE, device=self.ENV.device))

        '''normalization'''
        et = nf.normalize(et)

        '''input normalization + dropout'''
        es = self.drop_in(self.norm_in(es))

        '''modulate'''
        er = self.modulate(er_t, er_nt, et)

        return es, er, et, eo
