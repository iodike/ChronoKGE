"""
TComplEx
"""

import torch

from chrono_kge.model.kge.tkge_model import TKGE_Model
from chrono_kge.main.handler.exp_handler import ExperimentHandler
from chrono_kge.main.handler.model_handler import ModelHandler
from chrono_kge.main.handler.data_handler import DataHandler
from chrono_kge.main.handler.env_handler import EnvironmentHandler
from chrono_kge.model.module.embedding.tkge.tkge_complex import TKGE_ComplEx


class ComplEx_T(TKGE_Model):

    def __init__(self,
                 exp_handler: ExperimentHandler,
                 model_handler: ModelHandler,
                 data_handler: DataHandler,
                 env_handler: EnvironmentHandler,
                 **kwargs
                 ):
        """"""
        super().__init__(exp_handler, model_handler, data_handler, env_handler, **kwargs)

        self.kge = TKGE_ComplEx(model_handler, data_handler, env_handler)

        return

    def init(self):
        """"""
        super().init()
        self.kge.init()
        return

    def forward(self, x):
        """"""
        es_re, es_im, er_re, er_im, et_re, et_im, eo_re, eo_im = self.kge(x)

        m_re = es_re * (er_re * et_re) - es_im * (er_im * et_im)
        m_im = es_re * (er_im * et_im) + es_im * (er_re * et_re)

        m = m_re * eo_re + m_im * eo_im
        m = m.sum(dim=2)
        m = torch.sigmoid(m)

        return m
