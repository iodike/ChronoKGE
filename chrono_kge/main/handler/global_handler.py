"""
Handler
"""

import torch

from torch.optim.lr_scheduler import ExponentialLR

from chrono_kge.main.handler.exp_handler import ExperimentHandler
from chrono_kge.main.handler.model_handler import ModelHandler
from chrono_kge.main.handler.data_handler import DataHandler
from chrono_kge.main.handler.env_handler import EnvironmentHandler

from chrono_kge.model.__model__ import MODEL
from chrono_kge.trainer.__trainer__ import TRAINER


class GlobalHandler:
    """"""

    def __init__(self,
                 exp_handler: ExperimentHandler,
                 model_handler: ModelHandler,
                 data_handler: DataHandler,
                 env_handler: EnvironmentHandler
                 ):
        """"""
        self.EXP = exp_handler
        self.MODEL = model_handler
        self.DATA = data_handler
        self.ENV = env_handler

        self.model = None
        self.trainer = None
        self.optimizer = None
        self.scheduler = None

        self.tuner = None

        return

    def setup(self):
        """"""
        self.ENV.setup()
        self.DATA.setup()
        self.MODEL.setup()
        self.EXP.setup(self.ENV)

        self.load_trainer()
        self.load_model()
        self.init_model()
        self.init_optim()

        return

    def load_model(self):
        """"""
        cx = MODEL.get_model(self.MODEL.uid)
        self.model = cx(
            self.EXP,
            self.MODEL,
            self.DATA,
            self.ENV
        )
        return

    def load_trainer(self):
        """"""
        cx = TRAINER.get_trainer(self.EXP.train_type)
        self.trainer = cx(
            self.EXP,
            self.DATA,
            self.ENV
        )
        return

    def init_model(self) -> None:
        """"""
        if not self.model:
            self.load_model()

        self.model = torch.nn.DataParallel(self.model) if self.ENV.n_gpu > 1 else self.model
        self.model.to(self.ENV.device)
        if hasattr(self.model, 'module'):
            self.model.module.init()
        else:
            self.model.init()
        return

    def init_optim(self) -> None:
        """"""
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.EXP.learning_rate, weight_decay=self.EXP.l2)
        self.scheduler = ExponentialLR(optimizer=self.optimizer, gamma=self.EXP.decay_rate)
        return
