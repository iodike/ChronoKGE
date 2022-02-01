"""
Tune Experiment
"""

import optuna
import torch

from datetime import datetime

from chrono_kge.experiment.base_experiment import BaseExperiment
from chrono_kge.utils.vars.constants import TUNER, METRIC, TIME
from chrono_kge.utils.logger import logger
from chrono_kge.main.handler.global_handler import GlobalHandler


class TuneExperiment(BaseExperiment):

    def __init__(self,
                 global_handler: GlobalHandler
                 ) -> None:
        """"""
        super().__init__(global_handler=global_handler)
        self.cur_step = 0
        self.cur_time = 0
        return

    def run(self) -> dict:
        """"""
        self.before_run()
        results = super().run()
        self.after_run()
        return results

    def before_run(self) -> None:
        """"""
        self.cur_step = 0

        if self.GLOBAL.EXP.trial and self.GLOBAL.EXP.trial.number > 0:
            pass
        else:
            self.save_dict(self.to_dict(), "info")
        return

    def after_run(self) -> None:
        """"""
        if not self.GLOBAL.EXP.trial:
            logger.error("No tuner trial active.")
            exit(1)
        else:
            if self.study_finished():
                best_results = dict((key, self.GLOBAL.EXP.trial.study.best_trial.user_attrs[key])
                                    for key in METRIC.ALL)
                self.save_dict(best_results, "best results")

        return

    def study_finished(self) -> bool:
        """"""
        if self.GLOBAL.EXP.trial.user_attrs[TUNER.ATTR_N_TRIALS]:
            if self.GLOBAL.EXP.trial.number == int(self.GLOBAL.EXP.trial.user_attrs[TUNER.ATTR_N_TRIALS]) - 1:
                return True
        else:
            self.cur_time += (datetime.now() - self.GLOBAL.EXP.trial.datetime_start).total_seconds()
            max_time = int(self.GLOBAL.EXP.trial.user_attrs[TUNER.ATTR_N_HOURS] * TIME.HRS_IN_SEC)
            logger.info("Times c=%d, m=%d" % (self.cur_time, max_time))
            if max_time < self.cur_time:
                return True

        return False

    @torch.no_grad()
    def eval(self, test: bool = False, logging: bool = False) -> dict:
        """"""
        metrics = super().eval(test)

        self.cur_step += self.GLOBAL.EXP.eval_step

        # tuner report
        if self.GLOBAL.EXP.trial:
            self.GLOBAL.EXP.trial.report(metrics['h1'], self.cur_step)

            # tuner pruning
            if self.GLOBAL.EXP.trial.should_prune():
                if self.study_finished():
                    best_results = dict((key, self.GLOBAL.EXP.trial.study.best_trial.user_attrs[key])
                                        for key in METRIC.ALL)
                    self.save_dict(best_results, "best results")
                raise optuna.exceptions.TrialPruned()

        return metrics
