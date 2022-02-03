"""
KGE - SimplE-DE
"""

from chrono_kge.main.handler.model_handler import ModelHandler
from chrono_kge.main.handler.data_handler import DataHandler
from chrono_kge.main.handler.env_handler import EnvironmentHandler
from chrono_kge.model.module.embedding.kge_simple import KGE_SimplE
from chrono_kge.model.module.embedding.tkge_diachronic import TKGE_DIACHRONIC


class TKGE_SimplE_DE(KGE_SimplE):

    def __init__(self,
                 model_handler: ModelHandler,
                 data_handler: DataHandler,
                 env_handler: EnvironmentHandler,
                 **kwargs
                 ) -> None:
        """"""
        super().__init__(model_handler, data_handler, env_handler, **kwargs)

        self.kge_de = TKGE_DIACHRONIC(model_handler, data_handler, env_handler, **kwargs)

        return

    def init(self) -> None:
        """"""
        super().init()
        self.kge_de.init()
        return

    def __call__(self, x):
        """"""
        e_hh, e_ht, e_th, e_tt, e_r, e_r_inv = super().__call__(x)

        e_hh = self.kge_de.diachronic(e_hh, e_r)
        e_ht = self.kge_de.diachronic(e_ht, e_r)
        e_th = self.kge_de.diachronic(e_th, e_r_inv)
        e_tt = self.kge_de.diachronic(e_tt, e_r_inv)

        return e_hh, e_ht, e_th, e_tt, e_r, e_r_inv
