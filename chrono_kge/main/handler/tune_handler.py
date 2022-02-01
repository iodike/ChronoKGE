"""
Handler
"""

import joblib
import optuna

from optuna.pruners import MedianPruner

from chrono_kge.utils.vars.constants import TUNER, FTYPE
from chrono_kge.utils.vars.defaults import Default
from chrono_kge.utils.vars.params import BENCHMARK_ParameterList
from chrono_kge.experiment.base_experiment import BaseExperiment


class TuneHandler:
    """"""

    def __init__(self,
                 args: dict,
                 exp: BaseExperiment = None
                 ):
        """"""
        self.args = args

        self.uid = 'base'
        self.exp = exp

        self.n_trials = self.args.get('num_trials', 10)
        self.n_hours = self.args.get('timeout', 24)
        self.max_epochs = self.args.get('epochs', 1000)
        self.reload = self.args.get('reload', False)
        self.study_name = ''

        self.name = None
        self.study_path = ""
        self.study = None
        self.param_lists = None

        return

    def setup(self):
        """"""
        if not self.study_name:
            self.name = TUNER.STUDY_PREFIX + self.uid
        else:
            self.name = self.study_name

        self.study_path = self.exp.GLOBAL.ENV.path_save + self.name + FTYPE.PKL

        if self.reload:
            self.study = joblib.load(self.study_name)
        else:
            pruner = MedianPruner(TUNER.START_TRIALS, int(self.max_epochs * TUNER.START_STEPS_RATIO), Default.EVAL_STEP)
            self.study = optuna.create_study(study_name=self.name, direction=TUNER.DIR_MAX, pruner=pruner)
            joblib.dump(self.study, self.study_path)

        self.param_lists = BENCHMARK_ParameterList().get()

        return
