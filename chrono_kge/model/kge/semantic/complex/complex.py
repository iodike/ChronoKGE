"""
ComplEx
"""

import torch

from chrono_kge.model.kge.kge_model import KGE_Model
from chrono_kge.main.handler.exp_handler import ExperimentHandler
from chrono_kge.main.handler.model_handler import ModelHandler
from chrono_kge.main.handler.data_handler import DataHandler
from chrono_kge.main.handler.env_handler import EnvironmentHandler


class ComplEx(KGE_Model):

    def __init__(self,
                 exp_handler: ExperimentHandler,
                 model_handler: ModelHandler,
                 data_handler: DataHandler,
                 env_handler: EnvironmentHandler,
                 **kwargs
                 ):
        """"""
        super().__init__(exp_handler, model_handler, data_handler, env_handler, **kwargs)
        return

    def forward(self, x):
        """"""
        es, er, eo = self.kge(x)

        es_re, es_im = torch.chunk(es, 2, dim=2)
        er_re, er_im = torch.chunk(er, 2, dim=2)
        eo_re, eo_im = torch.chunk(eo, 2, dim=2)

        m_re = es_re * er_re - es_im * er_im
        m_im = es_re * er_im + es_im * er_re
        m = m_re * eo_re + m_im * eo_im

        m = m.sum(dim=2)

        m = torch.sigmoid(m)

        return m
