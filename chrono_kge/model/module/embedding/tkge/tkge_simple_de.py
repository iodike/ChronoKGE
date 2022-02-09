"""
KGE - SimplE-DE
"""

from chrono_kge.main.handler.model_handler import ModelHandler
from chrono_kge.main.handler.data_handler import DataHandler
from chrono_kge.main.handler.env_handler import EnvironmentHandler
from chrono_kge.model.module.embedding.kge.kge_simple import KGE_SimplE
from chrono_kge.model.module.embedding.submodule.diachronic import DIACHRONIC


class TKGE_SimplE_DE(KGE_SimplE):

    def __init__(self,
                 model_handler: ModelHandler,
                 data_handler: DataHandler,
                 env_handler: EnvironmentHandler,
                 **kwargs
                 ) -> None:
        """"""
        super().__init__(model_handler, data_handler, env_handler, **kwargs)

        self.de1 = DIACHRONIC(model_handler, data_handler, env_handler, **kwargs)
        self.de2 = DIACHRONIC(model_handler, data_handler, env_handler, **kwargs)
        self.de3 = DIACHRONIC(model_handler, data_handler, env_handler, **kwargs)
        self.de4 = DIACHRONIC(model_handler, data_handler, env_handler, **kwargs)

        return

    def init(self) -> None:
        """"""
        super().init()
        self.de1.init()
        self.de2.init()
        self.de3.init()
        self.de4.init()
        return

    def __call__(self, x):
        """"""

        '''entities (DE)'''
        e_hh = self.de1(x[:, 0], x[:, 2])
        e_ht = self.de2(x[:, 3], x[:, 2])
        e_th = self.de3(x[:, 0], x[:, 2])
        e_tt = self.de4(x[:, 3], x[:, 2])

        '''relations'''
        e_r = self.rel_embs(x[:, 1])
        e_r_inv = self.rel_inv_embs(x[:, 1])

        '''normalization + dropout'''
        e_hh = self.drop_in(self.norm_in(e_hh))
        e_th = self.drop_in(self.norm_in(e_th))

        return e_hh, e_ht, e_th, e_tt, e_r, e_r_inv
