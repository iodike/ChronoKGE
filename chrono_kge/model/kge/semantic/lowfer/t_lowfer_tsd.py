"""
T-LowFER - Time Series Decomposition
"""

from chrono_kge.model.kge.semantic.lowfer.t_lowfer import TLowFER
from chrono_kge.model.module.embedding.tkge_series import TKGE_Series
from chrono_kge.main.handler.exp_handler import ExperimentHandler
from chrono_kge.main.handler.model_handler import ModelHandler
from chrono_kge.main.handler.data_handler import DataHandler
from chrono_kge.main.handler.env_handler import EnvironmentHandler


class TLowFER_TSD(TLowFER):

    def __init__(self,
                 exp_handler: ExperimentHandler,
                 model_handler: ModelHandler,
                 data_handler: DataHandler,
                 env_handler: EnvironmentHandler,
                 **kwargs
                 ):
        """"""
        super().__init__(exp_handler, model_handler, data_handler, env_handler, **kwargs)

        # kge
        self.kge = TKGE_Series(model_handler, data_handler, env_handler)

        return

    def forward(self, x):
        """"""

        '''embeddings'''
        es, er, et, eo = self.kge(x)

        '''pooling'''
        m = self.mfbp(es, er)

        '''scoring'''
        s = self.scorer.exec(m, self.kge.emb_E.weight)

        return s
