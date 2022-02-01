"""

"""

# Tuner
import chrono_kge.tuner as tuner

from chrono_kge.utils.helpers import is_unique
from chrono_kge.utils.logger import logger


class TUNER(object):

    REGISTER = {
        'base': tuner.BaseTuner
    }

    # check uniqueness of keys
    assert(is_unique(list(REGISTER.keys())))

    @staticmethod
    def get_tuner(key: str):
        """"""
        if key in TUNER.REGISTER.keys():
            return TUNER.REGISTER[key]
        else:
            logger.error("Unknown tuner `%s`." % key)
            exit(1)
        return
