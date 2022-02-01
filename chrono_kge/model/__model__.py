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
        'tlowfer': semantic.TLowFER,
        'tlowfer-mft': semantic.TLowFER_MFT,
        'tlowfer-cfb': semantic.TLowFER_CFB,
        'tlowfer-tsd': semantic.TLowFER_TSD,
        'tlowfer-r2d': semantic.TLowFER_R2D,
        'tlowfer-tcr': semantic.TLowFER_TCR
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
