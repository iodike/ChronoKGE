"""
KG Embedding
"""

import torch

from torch import nn

from chrono_kge.main.handler.model_handler import ModelHandler
from chrono_kge.main.handler.data_handler import DataHandler
from chrono_kge.main.handler.env_handler import EnvironmentHandler


class KGE(nn.Module):

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

        self.emb_E = nn.Embedding(self.DATA.kg.n_entity, self.MODEL.entity_dim,
                                  padding_idx=0, device=self.ENV.device)
        self.emb_R1 = nn.Embedding(self.DATA.kg.n_relation, self.MODEL.relation_dim,
                                   padding_idx=0, device=self.ENV.device)

        # input dropout
        self.drop_in = nn.Dropout(self.MODEL.dropout_input)

        # input normalization
        self.norm_in = nn.BatchNorm1d(self.MODEL.entity_dim)

        return

    def init(self) -> None:
        """Initialise embeddings using normal distribution.
        """
        nn.init.xavier_normal_(self.emb_E.weight.data)
        nn.init.xavier_normal_(self.emb_R1.weight.data)
        return

    def __call__(self, x):
        """"""
        es = self.emb_E(x[:, 0])
        er = self.emb_R1(x[:, 1])
        eo = self.emb_E(x[:, 2])

        '''normalization + dropout'''
        es = self.drop_in(self.norm_in(es))

        return es, er, eo

    @staticmethod
    def sort_by_column(x, c):
        """"""
        xs = x[torch.argsort(x[:, c])]
        return xs
