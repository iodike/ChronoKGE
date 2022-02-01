"""
Run Manager
"""

from chrono_kge.experiment.__experiment__ import EXPERIMENT
from chrono_kge.main.handler.env_handler import EnvironmentHandler
from chrono_kge.main.handler.exp_handler import ExperimentHandler
from chrono_kge.main.handler.data_handler import DataHandler
from chrono_kge.main.handler.model_handler import ModelHandler
from chrono_kge.main.handler.global_handler import GlobalHandler
from chrono_kge.experiment.base_experiment import BaseExperiment


class RunManager:

    def __init__(self, args: dict) -> None:
        """"""
        self.args = args
        self.GLOBAL = None
        return

    def setup(self):
        """"""
        exp_handler, model_handler, data_handler, env_handler = self.setup_env()

        self.GLOBAL = GlobalHandler(
            exp_handler=exp_handler,
            model_handler=model_handler,
            data_handler=data_handler,
            env_handler=env_handler
        )
        self.GLOBAL.setup()

        experiment = self.load_run_experiment(self.GLOBAL.EXP.uid)
        experiment.start()
        return

    def setup_env(self):
        """"""
        env_handler = EnvironmentHandler(self.args)

        data_handler = DataHandler(self.args)

        model_handler = ModelHandler(self.args)

        exp_handler = ExperimentHandler(self.args)

        return exp_handler, model_handler, data_handler, env_handler

    def load_run_experiment(self, uid: str = 'base') -> BaseExperiment:
        """"""
        cx = EXPERIMENT.get_run_experiment(uid)
        experiment = cx(
            self.GLOBAL
        )
        return experiment
