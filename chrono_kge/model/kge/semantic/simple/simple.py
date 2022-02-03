"""
SimplE
"""

import torch

from chrono_kge.model.kge.kge_model import KGE_Model
from chrono_kge.model.module.embedding.kge_simple import KGE_SimplE
from chrono_kge.main.handler.exp_handler import ExperimentHandler
from chrono_kge.main.handler.model_handler import ModelHandler
from chrono_kge.main.handler.data_handler import DataHandler
from chrono_kge.main.handler.env_handler import EnvironmentHandler


class SimplE(KGE_Model):

    def __init__(self,
                 exp_handler: ExperimentHandler,
                 model_handler: ModelHandler,
                 data_handler: DataHandler,
                 env_handler: EnvironmentHandler,
                 **kwargs
                 ):
        """"""
        super().__init__(exp_handler, model_handler, data_handler, env_handler, **kwargs)

        self.kge = KGE_SimplE(model_handler, data_handler, env_handler, **kwargs)

        return

    def init(self):
        """"""
        super().init()
        self.kge.init()
        return

    def l2_loss(self):
        """"""
        return ((torch.norm(self.ent_h_embs.weight, p=2) ** 2) + (torch.norm(self.ent_t_embs.weight, p=2) ** 2) + (
                    torch.norm(self.rel_embs.weight, p=2) ** 2) + (torch.norm(self.rel_inv_embs.weight, p=2) ** 2)) / 2

    def forward(self, x):
        """"""
        e_hh, e_ht, e_th, e_tt, e_r, e_r_inv = self.kge(x)

        s1 = torch.sum(e_hh * e_r * e_tt, dim=1)
        s2 = torch.sum(e_ht * e_r_inv * e_th, dim=1)

        return torch.clamp((s1 + s2) / 2, -20, 20)
