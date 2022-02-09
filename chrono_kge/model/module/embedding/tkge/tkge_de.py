"""
TKGE - Diachronic Embedding
"""

from torch import nn

from chrono_kge.model.module.embedding.tkge.tkge import TKGE
from chrono_kge.main.handler.model_handler import ModelHandler
from chrono_kge.main.handler.data_handler import DataHandler
from chrono_kge.main.handler.env_handler import EnvironmentHandler
from chrono_kge.model.module.embedding.submodule.diachronic import DIACHRONIC


class TKGE_DE(TKGE):

    def __init__(self,
                 model_handler: ModelHandler,
                 data_handler: DataHandler,
                 env_handler: EnvironmentHandler,
                 **kwargs
                 ) -> None:
        """"""
        super().__init__(model_handler, data_handler, env_handler, **kwargs)

        self.emb_T = nn.Embedding(self.DATA.kg.n_time, self.MODEL.time_dim,
                                  padding_idx=0, device=self.ENV.device)
        self.emb_R2 = nn.Embedding(self.DATA.kg.n_relation, self.MODEL.relation_dim,
                                   padding_idx=0, device=self.ENV.device)

        self.de1 = DIACHRONIC(model_handler, data_handler, env_handler, **kwargs)
        self.de2 = DIACHRONIC(model_handler, data_handler, env_handler, **kwargs)

        return

    def init(self) -> None:
        """Initialise embeddings using normal distribution.
        """
        super().init()
        self.de1.init()
        self.de2.init()
        nn.init.xavier_normal_(self.emb_T.weight.data)
        nn.init.xavier_normal_(self.emb_R2.weight.data)
        return

    def __call__(self, x):
        """"""
        es = self.de1(x[:, 0], x[:, 2])
        er_t = self.emb_R1(x[:, 1])
        er_nt = self.emb_R2(x[:, 1])
        et = self.emb_T(x[:, 2])
        eo = self.de2(x[:, 3], x[:, 2])

        '''normalization + dropout'''
        es = self.drop_in(self.norm_in(es))

        '''modulate'''
        er = self.modulate(er_t, er_nt, et)

        return es, er, et, eo
