"""
Abstract Experiment
"""

from abc import ABC, abstractmethod
import torch


class AbstractExperiment(ABC):

    @abstractmethod
    def start(self) -> dict:
        """"""
        pass

    @abstractmethod
    def run(self) -> dict:
        """Run experiment"""
        pass

    @abstractmethod
    def train(self, *args):
        """Train model"""
        pass

    @abstractmethod
    @torch.no_grad()
    def eval(self, test: bool, logging: bool) -> dict:
        """Evaluate model"""
        pass

    @abstractmethod
    def encode(self, *args):
        """Encoder"""
        pass

    @abstractmethod
    def decode(self, *args):
        """Decoder"""
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        """Returns experiment as dict"""
        pass

    @abstractmethod
    def save_dict(self, ddict: dict, title: str) -> None:
        """Saves a dictionary to file"""
        pass
