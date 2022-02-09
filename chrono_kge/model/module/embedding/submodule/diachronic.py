"""
KGE - Diachronic
"""

import torch
from torch import nn

from chrono_kge.main.handler.model_handler import ModelHandler
from chrono_kge.main.handler.data_handler import DataHandler
from chrono_kge.main.handler.env_handler import EnvironmentHandler
from chrono_kge.model.module.embedding.submodule.submodule import SubModule


class DIACHRONIC(SubModule):

    def __init__(self,
                 model_handler: ModelHandler,
                 data_handler: DataHandler,
                 env_handler: EnvironmentHandler,
                 **kwargs
                 ) -> None:
        """"""
        super().__init__(model_handler, data_handler, env_handler, **kwargs)

        self.yd = int(self.MODEL.de_gamma * self.MODEL.entity_dim)

        self.emb_As = nn.Embedding(self.DATA.kg.n_entity, self.MODEL.entity_dim - self.yd, padding_idx=0,
                                   device=self.ENV.device)
        self.emb_At = nn.Embedding(self.DATA.kg.n_entity, self.yd, padding_idx=0, device=self.ENV.device)
        self.emb_Wt = nn.Embedding(self.DATA.kg.n_entity, self.yd, padding_idx=0, device=self.ENV.device)
        self.emb_Bt = nn.Embedding(self.DATA.kg.n_entity, self.yd, padding_idx=0, device=self.ENV.device)

        return

    def init(self) -> None:
        """"""
        nn.init.xavier_normal_(self.emb_As.weight.data)
        nn.init.xavier_normal_(self.emb_At.weight.data)
        nn.init.xavier_normal_(self.emb_Wt.weight.data)
        nn.init.xavier_normal_(self.emb_Bt.weight.data)
        return

    def __call__(self, e, t):
        """"""
        e_t = e[:, 0:self.yd-1]
        e_s = e[:, self.yd:]

        x_t = self.emb_At(e_t) * torch.sin(self.emb_Wt(e) * t + self.emb_Bt(e))
        x_s = self.emb_As(e_s)

        x = torch.cat((x_t, x_s), 1)

        return x
