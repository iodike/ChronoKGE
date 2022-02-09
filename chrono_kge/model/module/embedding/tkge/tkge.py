"""
TKGE
"""

import torch
from torch import nn

from chrono_kge.model.module.embedding.kge.kge import KGE
from chrono_kge.utils.vars.modes import MOD
from chrono_kge.main.handler.model_handler import ModelHandler
from chrono_kge.main.handler.data_handler import DataHandler
from chrono_kge.main.handler.env_handler import EnvironmentHandler


class TKGE(KGE):

    def __init__(self,
                 model_handler: ModelHandler,
                 data_handler: DataHandler,
                 env_handler: EnvironmentHandler,
                 **kwargs
                 ) -> None:
        """"""
        super().__init__(model_handler, data_handler, env_handler, **kwargs)

        self.emb_T = nn.Embedding(self.DATA.kg.n_time, self.MODEL.time_dim,
                                  padding_idx=0, device=self.ENV.device)
        self.emb_R2 = nn.Embedding(self.DATA.kg.n_relation, self.MODEL.relation_dim,
                                   padding_idx=0, device=self.ENV.device)

        return

    def init(self) -> None:
        """Initialise embeddings using normal distribution.
        """
        super().init()
        nn.init.xavier_normal_(self.emb_T.weight.data)
        nn.init.xavier_normal_(self.emb_R2.weight.data)
        return

    def __call__(self, x):
        """"""
        es = self.emb_E(x[:, 0])
        er_t = self.emb_R1(x[:, 1])
        er_nt = self.emb_R2(x[:, 1])
        et = self.emb_T(x[:, 2])
        eo = self.emb_E(x[:, 3])

        '''normalization + dropout'''
        es = self.drop_in(self.norm_in(es))

        '''modulate'''
        er = self.modulate(er_t, er_nt, et)

        return es, er, et, eo

    def modulate(self, x1, x2, y) -> torch.Tensor:
        """"""
        if self.MODEL.mod_mode == MOD.TNT:
            z = x1 * y + x2
        elif self.MODEL.mod_mode == MOD.T:
            z = x1 * y
        else:
            z = x1

        return z
