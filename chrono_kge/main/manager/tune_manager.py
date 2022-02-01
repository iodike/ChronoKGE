"""
Tune Manager
"""

from chrono_kge.tuner.__tuner__ import TUNER
from chrono_kge.experiment.__experiment__ import EXPERIMENT
from chrono_kge.main.handler.tune_handler import TuneHandler
from chrono_kge.main.handler.global_handler import GlobalHandler
from chrono_kge.tuner.base_tuner import BaseTuner
from chrono_kge.main.manager.run_manager import RunManager


class TuneManager(RunManager):

    def __init__(self, args: dict) -> None:
        """"""
        super().__init__(args)
        return

    def setup(self) -> None:
        """"""
        exp_handler, model_handler, data_handler, env_handler = super().setup_env()

        self.GLOBAL = GlobalHandler(
            exp_handler=exp_handler,
            model_handler=model_handler,
            data_handler=data_handler,
            env_handler=env_handler
        )

        experiment = self.load_tune_experiment(exp_handler.uid)

        tune_handler = TuneHandler(self.args, exp=experiment)
        tune_handler.setup()

        tuner = self.load_tuner(tune_handler, tune_handler.uid)
        tuner.start()

        tuner.show_stats()
        tuner.to_csv()
        tuner.create_plots()

        return

    def load_tune_experiment(self, uid: str = 'base'):
        """"""
        cx = EXPERIMENT.get_tune_experiment(uid)
        experiment = cx(
            self.GLOBAL
        )
        return experiment

    def load_tuner(self, tune_handler: TuneHandler, uid: str = 'base') -> BaseTuner:
        """"""
        cx = TUNER.get_tuner(uid)
        tuner = cx(
            self.GLOBAL,
            tune_handler
        )
        return tuner
