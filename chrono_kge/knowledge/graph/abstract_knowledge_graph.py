"""
Abstract knowledge graph
"""

from abc import ABC, abstractmethod


class AbstractKnowledgeGraph(ABC):

    @abstractmethod
    def setup(self) -> None:
        """"""
        pass

    @abstractmethod
    def create_dicts(self) -> None:
        """"""
        pass

    @abstractmethod
    def create_dict(self, filename) -> None:
        """"""
        pass

    @abstractmethod
    def load_dicts(self) -> None:
        """"""
        pass

    @abstractmethod
    def load_dict(self, filename: str, reverse: bool) -> (dict, int):
        """"""
        pass

    @abstractmethod
    def load_triples(self) -> None:
        """"""
        pass

    @abstractmethod
    def process_row(self, *args):
        """"""
        pass

    @abstractmethod
    def create_triples(self, *args):
        """"""
        pass

    @abstractmethod
    def read_file(self, filename: str, file_path: str = ''):
        """"""
        pass

    @abstractmethod
    def check_ids(self, *args) -> bool:
        """"""
        pass
