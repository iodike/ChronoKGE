"""
Abstract Trainer
"""

from abc import ABC, abstractmethod


class AbstractTrainer(ABC):

    @abstractmethod
    def rank_targets(self, *args):
        """Ranks targets by score"""
        pass
