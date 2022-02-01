"""
Temporal KGE model
"""

import math

from chrono_kge.model.kge.kge_model import KGE_Model
from chrono_kge.utils.vars.modes import REG, ENC
from chrono_kge.model.module.embedding.tkge import TKGE
from chrono_kge.model.module.embedding.tkge_comp import TKGE_Components
from chrono_kge.main.handler.exp_handler import ExperimentHandler
from chrono_kge.main.handler.model_handler import ModelHandler
from chrono_kge.main.handler.data_handler import DataHandler
from chrono_kge.main.handler.env_handler import EnvironmentHandler


class TKGE_Model(KGE_Model):

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
        if self.MODEL.enc_mode == ENC.CYCLE:
            self.kge = TKGE_Components(self.MODEL, self.DATA, self.ENV)
        else:
            self.kge = TKGE(self.MODEL, self.DATA, self.ENV)

        return

    def regularize(self, **embeddings):
        """"""
        r = 0

        if self.training:

            '''entity/relation'''
            if REG.isOmega(self.MODEL.reg_mode):
                r += self.omega.forward(factors=[
                    math.pow(2, 1 / self.power) * embeddings['es'],
                    embeddings['er_t'],
                    embeddings['ert'],
                    math.pow(2, 1 / self.power) * embeddings['eo']
                ])

            '''time'''
            if REG.isLambda(self.MODEL.reg_mode):
                r += self.lamda.forward(factor=self.kge.emb_T.weight)

        return r
