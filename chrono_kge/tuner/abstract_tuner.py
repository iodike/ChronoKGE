"""
Abstract Tuner
"""

from abc import ABC, abstractmethod
from chrono_kge.main.handler.global_handler import GlobalHandler
from chrono_kge.main.handler.tune_handler import TuneHandler


class AbstractTuner(ABC):

    def __init__(self,
                 global_handler: GlobalHandler,
                 tune_handler: TuneHandler
                 ) -> None:
        """"""

        self.GLOBAL = global_handler
        self.TUNE = tune_handler

        return

    @abstractmethod
    def start(self):
        """"""
        pass

    @abstractmethod
    def tune_params(self, *args):
        """"""
        pass

    @abstractmethod
    def objective(self, *args):
        """"""
        pass

    @abstractmethod
    def show_stats(self):
        """"""
        pass

    @abstractmethod
    def to_csv(self):
        """"""
        pass

    @abstractmethod
    def create_plots(self):
        """"""
        pass
