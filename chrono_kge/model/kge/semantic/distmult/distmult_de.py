"""
DistMult
"""

import torch

from chrono_kge.model.kge.tkge_model import TKGE_Model
from chrono_kge.main.handler.exp_handler import ExperimentHandler
from chrono_kge.main.handler.model_handler import ModelHandler
from chrono_kge.main.handler.data_handler import DataHandler
from chrono_kge.main.handler.env_handler import EnvironmentHandler
from chrono_kge.model.module.embedding.tkge.tkge_de import TKGE_DE


class DistMult_DE(TKGE_Model):

    def __init__(self,
                 exp_handler: ExperimentHandler,
                 model_handler: ModelHandler,
                 data_handler: DataHandler,
                 env_handler: EnvironmentHandler,
                 **kwargs
                 ):
        """"""
        super().__init__(exp_handler, model_handler, data_handler, env_handler, **kwargs)

        self.kge = TKGE_DE(model_handler, data_handler, env_handler, **kwargs)

        return

    def init(self):
        """"""
        super().init()
        self.kge.init()
        return

    def forward(self, x):
        """"""
        es, er, et, eo = self.kge(x)

        m = - torch.sum(es * eo * er, -1)
        m = torch.sigmoid(m)

        return m
