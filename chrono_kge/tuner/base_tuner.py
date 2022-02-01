"""
Base Tuner
"""

import optuna
import optuna.visualization.matplotlib as ovm

from chrono_kge.tuner.abstract_tuner import AbstractTuner
from chrono_kge.utils.vars.constants import FTYPE, TIME, TUNER, METRIC
from chrono_kge.utils.vars.defaults import Default
from chrono_kge.utils.logger import logger
from chrono_kge.main.handler.global_handler import GlobalHandler
from chrono_kge.main.handler.tune_handler import TuneHandler


class BaseTuner(AbstractTuner):

    def __init__(self,
                 global_handler: GlobalHandler,
                 tune_handler: TuneHandler
                 ) -> None:
        """"""
        super().__init__(global_handler, tune_handler)
        return

    def start(self) -> optuna.Study:
        """"""
        self.TUNE.study.optimize(self.objective, n_trials=self.TUNE.n_trials,
                                 timeout=self.TUNE.n_hours * TIME.HRS_IN_SEC)
        return self.TUNE.study

    def tune_params(self, trial: optuna.Trial) -> dict:
        """"""
        return {
            "lr": trial.suggest_categorical("lr", self.TUNE.param_lists['lr']),
            "dr": trial.suggest_categorical("dr", self.TUNE.param_lists['dr']),
            "id": Default.DROPOUT_INPUT,
            "hd": [Default.DROPOUT_HIDDEN]*3,
            "bs": trial.suggest_categorical("bs", self.TUNE.param_lists['bs']),
            "ed": trial.suggest_categorical("ed", self.TUNE.param_lists['ed']),
            "tg": trial.suggest_categorical("tg", self.TUNE.param_lists['tg']),
            "fr": trial.suggest_categorical("fr", self.TUNE.param_lists['fr']),
            "ls": trial.suggest_categorical("ls", self.TUNE.param_lists['ls']),
            "l1": trial.suggest_categorical("l1", self.TUNE.param_lists['l1']),
            "l2": trial.suggest_categorical("l2", self.TUNE.param_lists['l2']),
            "re": trial.suggest_categorical("re", self.TUNE.param_lists['re']),
            "rt": trial.suggest_categorical("rt", self.TUNE.param_lists['rt']),
            "ns": trial.suggest_categorical("ns", self.TUNE.param_lists['ns']),
            "mr": trial.suggest_categorical("mr", self.TUNE.param_lists['mr'])
        }

    def objective(self, trial: optuna.Trial):
        """"""
        trial.set_user_attr(TUNER.ATTR_N_TRIALS, self.TUNE.n_trials)
        trial.set_user_attr(TUNER.ATTR_N_HOURS, self.TUNE.n_hours)
        params = self.tune_params(trial)

        train_args = {'ns': params['ns'], 'mr': params['mr']}

        self.GLOBAL.EXP.batch_size = params['bs']
        self.GLOBAL.EXP.learning_rate = params['lr']
        self.GLOBAL.EXP.decay_rate = params['dr']
        self.GLOBAL.EXP.label_smoothing = params['ls']
        self.GLOBAL.EXP.l1 = params['l1']
        self.GLOBAL.EXP.l2 = params['l2']
        self.GLOBAL.EXP.trial = trial
        self.GLOBAL.EXP.reg_emb = params['re']
        self.GLOBAL.EXP.reg_time = params['rt']
        self.GLOBAL.EXP.train_args = train_args

        self.GLOBAL.MODEL.entity_dim = params['ed']
        self.GLOBAL.MODEL.relation_dim = params['ed']
        self.GLOBAL.MODEL.time_dim = params['ed']
        self.GLOBAL.MODEL.dropout_input = params['id']
        self.GLOBAL.MODEL.dropout_hidden = params['hd']

        self.GLOBAL.DATA.time_gran = params['tg']

        self.GLOBAL.setup()

        metrics: dict = self.TUNE.exp.start()

        # store metrics
        for key, value in metrics.items():
            trial.set_user_attr(key, value)

        return metrics[METRIC.H1]

    def show_stats(self) -> None:
        """"""
        pruned_trials = [t for t in self.TUNE.study.trials if t.state == optuna.structs.TrialState.PRUNED]
        complete_trials = [t for t in self.TUNE.study.trials if t.state == optuna.structs.TrialState.COMPLETE]

        logger.info("Study statistics: ")
        logger.info("Number of finished trials: %d" % len(self.TUNE.study.trials))
        logger.info("Number of pruned trials: %d" % len(pruned_trials))
        logger.info("Number of complete trials: %d" % len(complete_trials))

        if len(self.TUNE.study.trials) > 0:

            trial = self.TUNE.study.best_trial

            logger.info("Best trial: ")
            logger.info("Value: %0.9f" % trial.value)

            logger.info("Params: ")
            for key, value in trial.params.items():
                logger.info("%s: %s" % (key, str(value)))

        return

    def to_csv(self) -> None:
        """"""
        results_path = self.GLOBAL.ENV.path_save + self.TUNE.name + FTYPE.CSV
        trials_df = self.TUNE.study.trials_dataframe()
        trials_df.to_csv(results_path, encoding='utf-8', index=False)
        return

    def create_plots(self) -> None:
        """"""
        if not ovm.is_available():
            logger.warning("Plotly not available.")
            return

        img_path = self.GLOBAL.ENV.path_save

        if len(self.TUNE.study.trials) > 1:
            fig0 = optuna.visualization.plot_param_importances(self.TUNE.study)
            fig0.write_image(img_path + "param_importances" + FTYPE.PNG)

            fig1 = optuna.visualization.plot_intermediate_values(self.TUNE.study)
            fig1.write_image(img_path + "intermediate_values" + FTYPE.PNG)

            fig2 = optuna.visualization.plot_slice(self.TUNE.study)
            fig2.write_image(img_path + "slice" + FTYPE.PNG)

        else:
            logger.info("Too few trials. Number of trials: %d." % len(self.TUNE.study.trials))

        return
