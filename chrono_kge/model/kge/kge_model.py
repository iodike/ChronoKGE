"""
KGE model
"""

import math
import torch.nn as nn

from chrono_kge.model.kge.base_kge_model import Base_KGE_Model
from chrono_kge.utils.vars.modes import REG
from chrono_kge.model.module.embedding.kge.kge import KGE
from chrono_kge.model.module.regularizer.omega import Omega
from chrono_kge.model.module.regularizer.lamda import Lambda
from chrono_kge.utils.vars.defaults import Default
from chrono_kge.utils.logger import logger
from chrono_kge.main.handler.exp_handler import ExperimentHandler
from chrono_kge.main.handler.model_handler import ModelHandler
from chrono_kge.main.handler.data_handler import DataHandler
from chrono_kge.main.handler.env_handler import EnvironmentHandler


class KGE_Model(Base_KGE_Model):

    def __init__(self,
                 exp_handler: ExperimentHandler,
                 model_handler: ModelHandler,
                 data_handler: DataHandler,
                 env_handler: EnvironmentHandler,
                 **kwargs
                 ) -> None:
        """"""
        super().__init__(exp_handler, model_handler, data_handler, env_handler, **kwargs)

        # kge
        self.kge = KGE(self.MODEL, self.DATA, self.ENV)

        self.power = Default.REG_POWER
        self.omega = Omega(weight=self.EXP.reg_emb, power=self.power)
        self.lamda = Lambda(weight=self.EXP.reg_time, power=self.power+1)

        self.reg_loss = 0

        return

    def init(self):
        """"""
        self.kge.init()
        return

    def loss(self, *args):
        """"""
        loss = None

        if self.EXP.train_type == 'ova':
            loss = nn.BCELoss()
            loss = loss(args[0], args[1])
        elif self.EXP.train_type == 'ns':
            loss = nn.CrossEntropyLoss()
            loss = loss(args[0], args[1])
        else:
            logger.error("Unknown training mode: `%s`" % self.train_mode)
            exit(1)

        loss += self.reg_loss

        return loss

    def match(self, m, act=nn.Sigmoid()):
        """"""

        '''cosine similarity (0...1)'''
        s = m @ self.kge.emb_E.weight.t()

        '''non-linear activation'''
        if not self.training or self.EXP.train_type == 'ova':
            s = act(s)

        return s

    def regularize(self, **embeddings):
        """"""
        r = 0

        if self.training:

            '''entity/relation'''
            if REG.isOmega(self.MODEL.reg_mode):
                r += self.omega.forward(factors=[
                    math.pow(2, 1 / self.power) * embeddings['es'],
                    embeddings['er'],
                    math.pow(2, 1 / self.power) * embeddings['eo']
                ])

            '''time'''
            if REG.isLambda(self.MODEL.reg_mode):
                r += self.lamda.forward(factor=self.emb_T.weight)

        return r
