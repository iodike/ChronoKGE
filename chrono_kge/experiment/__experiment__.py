"""

"""

import chrono_kge.experiment as exp

from chrono_kge.utils.helpers import is_unique
from chrono_kge.utils.logger import logger


class EXPERIMENT(object):

    REGISTER = {
        'base': (exp.BaseExperiment, exp.TuneExperiment)
    }

    # check uniqueness of keys
    assert(is_unique(list(REGISTER.keys())))

    @staticmethod
    def get_run_experiment(key: str):
        """"""
        if key in EXPERIMENT.REGISTER.keys():
            return EXPERIMENT.REGISTER[key][0]
        else:
            logger.error("Unknown run experiment `%s`." % key)
            exit(1)
        return

    @staticmethod
    def get_tune_experiment(key: str):
        """"""
        if key in EXPERIMENT.REGISTER.keys():
            return EXPERIMENT.REGISTER[key][1]
        else:
            logger.error("Unknown tune experiment `%s`." % key)
            exit(1)
        return
