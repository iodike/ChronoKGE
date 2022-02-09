"""
KGE - SimplE
"""

import math

from torch import nn

from chrono_kge.model.module.embedding.kge.kge import KGE
from chrono_kge.main.handler.model_handler import ModelHandler
from chrono_kge.main.handler.data_handler import DataHandler
from chrono_kge.main.handler.env_handler import EnvironmentHandler


class KGE_SimplE(KGE):

    def __init__(self,
                 model_handler: ModelHandler,
                 data_handler: DataHandler,
                 env_handler: EnvironmentHandler,
                 **kwargs
                 ) -> None:
        """"""
        super().__init__(model_handler, data_handler, env_handler, **kwargs)

        self.ent_h_embs = nn.Embedding(self.DATA.kg.n_entity, self.MODEL.entity_dim,
                                       padding_idx=0, device=self.ENV.device)
        self.ent_t_embs = nn.Embedding(self.DATA.kg.n_entity, self.MODEL.entity_dim,
                                       padding_idx=0, device=self.ENV.device)
        self.rel_embs = nn.Embedding(self.DATA.kg.n_relation, self.MODEL.entity_dim,
                                     padding_idx=0, device=self.ENV.device)
        self.rel_inv_embs = nn.Embedding(self.DATA.kg.n_relation, self.MODEL.entity_dim,
                                         padding_idx=0, device=self.ENV.device)

        return

    def init(self) -> None:
        """Initialise embeddings using normal distribution.
        """
        super().init()
        sqrt_size = 6.0 / math.sqrt(self.MODEL.entity_dim)
        nn.init.uniform_(self.ent_h_embs.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.ent_t_embs.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.rel_embs.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.rel_inv_embs.weight.data, -sqrt_size, sqrt_size)
        return

    def __call__(self, x):
        """"""
        e_hh = self.ent_h_embs(x[:, 0])
        e_ht = self.ent_h_embs(x[:, 2])
        e_th = self.ent_t_embs(x[:, 0])
        e_tt = self.ent_t_embs(x[:, 2])
        e_r = self.rel_embs(x[:, 1])
        e_r_inv = self.rel_inv_embs(x[:, 1])

        '''normalization + dropout'''
        e_hh = self.drop_in(self.norm_in(e_hh))
        e_th = self.drop_in(self.norm_in(e_th))

        return e_hh, e_ht, e_th, e_tt, e_r, e_r_inv
