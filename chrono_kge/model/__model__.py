"""
Models
"""

from chrono_kge.utils.helpers import is_unique
from chrono_kge.utils.logger import logger

# Model
import chrono_kge.model.kge.semantic as semantic


class MODEL(object):
    """
    Models
    """

    REGISTER = {
        'lowfer': semantic.LowFER,
        'lowfer-t': semantic.LowFER_T,
        'lowfer-t-lft': semantic.LowFER_T_LFT,
        'lowfer-t-cfb': semantic.LowFER_T_CFB,
        'lowfer-t-tsd': semantic.LowFER_T_TSD,
        'lowfer-t-r2d': semantic.LowFER_T_R2D,
        'lowfer-t-tcr': semantic.LowFER_T_TCR,
        'complex': semantic.ComplEx,
        'complex-t': semantic.ComplEx_T,
        'distmult': semantic.DistMult,
        'distmult-de': semantic.DistMult_DE,
        'simple': semantic.SimplE,
        'simple-de': semantic.SimplE_DE,
        'tucker': semantic.TuckER,
        'tucker-t': semantic.TuckER_T
    }

    # check uniqueness of keys
    assert(is_unique(list(REGISTER.keys())))

    @staticmethod
    def get_model(key: str):
        """"""
        if key in MODEL.REGISTER.keys():
            return MODEL.REGISTER[key]
        else:
            logger.error("Unknown model `%s`." % key)
            exit(1)
        return
