"""
Base KGE model
"""

import torch.nn as nn

from chrono_kge.main.handler.exp_handler import ExperimentHandler
from chrono_kge.main.handler.model_handler import ModelHandler
from chrono_kge.main.handler.data_handler import DataHandler
from chrono_kge.main.handler.env_handler import EnvironmentHandler


class Base_KGE_Model(nn.Module):

    def __init__(self,
                 exp_handler: ExperimentHandler,
                 model_handler: ModelHandler,
                 data_handler: DataHandler,
                 env_handler: EnvironmentHandler,
                 **kwargs
                 ):
        """"""
        super().__init__()

        self.EXP = exp_handler
        self.MODEL = model_handler
        self.DATA = data_handler
        self.ENV = env_handler

        # kwargs

        self.kwargs = kwargs
        self.kwargs.update({'dh': self.MODEL.dropout_hidden})

        return

    def forward(self, x):
        """"""
        pass
