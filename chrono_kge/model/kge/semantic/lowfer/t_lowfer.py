"""
Time LowFER
"""

from chrono_kge.model.kge.tkge_model import TKGE_Model
from chrono_kge.model.module.pooling.mfbp import FactorizedBilinearPooling
from chrono_kge.model.module.scoring.similarity import CosineSimilarityScorer
from chrono_kge.main.handler.exp_handler import ExperimentHandler
from chrono_kge.main.handler.model_handler import ModelHandler
from chrono_kge.main.handler.data_handler import DataHandler
from chrono_kge.main.handler.env_handler import EnvironmentHandler


class TLowFER(TKGE_Model):
    """Temporal LowFER class.
    """

    def __init__(self,
                 exp_handler: ExperimentHandler,
                 model_handler: ModelHandler,
                 data_handler: DataHandler,
                 env_handler: EnvironmentHandler,
                 **kwargs
                 ) -> None:
        """Initialise model.
        """
        super().__init__(exp_handler, model_handler, data_handler, env_handler, **kwargs)

        # mfbp
        self.mfbp = FactorizedBilinearPooling(param_dim1=self.MODEL.entity_dim, param_dim2=self.MODEL.relation_dim,
                                              out_dim=self.MODEL.entity_dim, rank=self.MODEL.factor_rank,
                                              device=self.ENV.device, **self.kwargs)

        # scorer
        self.scorer = CosineSimilarityScorer()

        return

    def forward(self, x):
        """Forwards input tensor.
        Output: BS x n_entities
        """

        '''embeddings'''
        es, er, et, eo = self.kge(x)

        '''pooling'''
        m = self.mfbp(es, er)

        '''scoring'''
        s = self.scorer.exec(m, self.kge.emb_E.weight)

        return s
