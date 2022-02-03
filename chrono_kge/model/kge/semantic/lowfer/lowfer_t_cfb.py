"""
LowFER-T - Chained Multi-modal Pooling
"""

from chrono_kge.model.kge.semantic.lowfer.lowfer_t import LowFER_T
from chrono_kge.model.module.pooling.lfcp import FactorizedChainedPooling
from chrono_kge.main.handler.exp_handler import ExperimentHandler
from chrono_kge.main.handler.model_handler import ModelHandler
from chrono_kge.main.handler.data_handler import DataHandler
from chrono_kge.main.handler.env_handler import EnvironmentHandler


class LowFER_T_CFB(LowFER_T):

    def __init__(self,
                 exp_handler: ExperimentHandler,
                 model_handler: ModelHandler,
                 data_handler: DataHandler,
                 env_handler: EnvironmentHandler,
                 **kwargs
                 ):
        """"""
        super().__init__(exp_handler, model_handler, data_handler, env_handler, **kwargs)

        # mfbp
        self.mfcp = FactorizedChainedPooling(param_dim1=self.MODEL.entity_dim, param_dim2=self.MODEL.relation_dim,
                                             param_dim3=self.MODEL.time_dim, out_dim=self.MODEL.entity_dim,
                                             rank=self.MODEL.factor_rank, device=self.ENV.device, **self.kwargs)

        return

    def forward(self, x):
        """"""

        '''embeddings'''
        es, er, et, eo = self.kge(x)

        '''pooling'''
        m = self.mfcp(es, er, **{'emb3': et})

        '''scoring'''
        s = self.scorer.exec(m, self.kge.emb_E.weight)

        return s
