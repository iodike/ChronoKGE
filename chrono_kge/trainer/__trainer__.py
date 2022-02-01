"""

"""

import chrono_kge.trainer as trainer

from chrono_kge.utils.helpers import is_unique
from chrono_kge.utils.logger import logger


class TRAINER(object):

    REGISTER = {
        'ova': trainer.OVATrainer
    }

    # check uniqueness of keys
    assert(is_unique(list(REGISTER.keys())))

    @staticmethod
    def get_trainer(key: str):
        """"""
        if key in TRAINER.REGISTER.keys():
            return TRAINER.REGISTER[key]
        else:
            logger.error("Unknown trainer `%s`." % key)
            exit(1)
        return
