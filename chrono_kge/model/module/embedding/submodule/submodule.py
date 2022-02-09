"""
Sub Module
"""

from torch import nn

from chrono_kge.main.handler.model_handler import ModelHandler
from chrono_kge.main.handler.data_handler import DataHandler
from chrono_kge.main.handler.env_handler import EnvironmentHandler


class SubModule(nn.Module):

    def __init__(self,
                 model_handler: ModelHandler,
                 data_handler: DataHandler,
                 env_handler: EnvironmentHandler,
                 **kwargs
                 ) -> None:
        """"""
        super().__init__()

        self.MODEL = model_handler
        self.DATA = data_handler
        self.ENV = env_handler

        self.kwargs = kwargs

        return
