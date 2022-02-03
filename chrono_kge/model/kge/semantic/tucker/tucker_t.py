"""
TuckER-T
"""

import torch
from torch import nn

from chrono_kge.model.kge.tkge_model import TKGE_Model
from chrono_kge.utils.vars.defaults import Default
from chrono_kge.model.module.scoring.similarity import CosineSimilarityScorer
from chrono_kge.main.handler.exp_handler import ExperimentHandler
from chrono_kge.main.handler.model_handler import ModelHandler
from chrono_kge.main.handler.data_handler import DataHandler
from chrono_kge.main.handler.env_handler import EnvironmentHandler


class TuckER_T(TKGE_Model):

    def __init__(self,
                 exp_handler: ExperimentHandler,
                 model_handler: ModelHandler,
                 data_handler: DataHandler,
                 env_handler: EnvironmentHandler,
                 **kwargs
                 ) -> None:
        """"""
        super().__init__(exp_handler, model_handler, data_handler, env_handler, **kwargs)

        self.L = nn.Linear(self.MODEL.relation_dim, self.MODEL.entity_dim**2, bias=False, device=self.ENV.device,
                           dtype=Default.DTYPE)

        self.dh1 = nn.Dropout(self.MODEL.dropout_hidden[0])
        self.dh2 = nn.Dropout(self.MODEL.dropout_hidden[1])
        self.bn_out = nn.BatchNorm1d(self.MODEL.entity_dim)

        self.scorer = CosineSimilarityScorer()

        return

    def forward(self, x):
        """"""
        es, er, et, eo = self.kge(x)
        m = es.view(-1, 1, es.size(1))

        # weight relation matrix
        Wr = self.L(er)
        Wr = Wr.view(-1, es.size(1), es.size(1))
        Wr = self.dh1(Wr)

        # out
        m = torch.bmm(m, Wr)
        m = m.view(-1, es.size(1))
        m = self.dh2(self.bn_out(m))

        # score
        s = self.scorer.exec(m, self.kge.emb_E.weight)

        return s
