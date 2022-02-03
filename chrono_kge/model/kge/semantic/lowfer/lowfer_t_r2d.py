"""
LowFER-T - Rotation2D (R2D)
"""

from chrono_kge.model.module.calculus.rotation import Rotation2D
from chrono_kge.model.kge.semantic.lowfer.lowfer_t import LowFER_T
from chrono_kge.main.handler.exp_handler import ExperimentHandler
from chrono_kge.main.handler.model_handler import ModelHandler
from chrono_kge.main.handler.data_handler import DataHandler
from chrono_kge.main.handler.env_handler import EnvironmentHandler


class LowFER_T_R2D(LowFER_T):

    def __init__(self,
                 exp_handler: ExperimentHandler,
                 model_handler: ModelHandler,
                 data_handler: DataHandler,
                 env_handler: EnvironmentHandler,
                 **kwargs
                 ):
        """"""
        super().__init__(exp_handler, model_handler, data_handler, env_handler, **kwargs)

        self.r2ds = [Rotation2D(period=2**i) for i in range(0, 2, 2)]
        assert len(self.r2ds) > 0

        return

    def forward(self, x):
        """"""

        '''embeddings'''
        es, er, et, eo = self.kge(x)

        '''rotation'''
        es_tr, er_tr = self.rotate(es, er, et)

        '''pooling'''
        m = self.mfbp(es_tr, er_tr)

        '''scoring'''
        s = self.scorer.exec(m, self.kge.emb_E.weight)

        return s

    def rotate(self, x, y, t):
        """"""
        es_tr = 0
        er_tr = 0

        '''2D rotation'''
        for r2d in self.r2ds:
            s, r = r2d.forward(x, y, t)
            es_tr = es_tr + s
            er_tr = er_tr + r

        return es_tr, er_tr
