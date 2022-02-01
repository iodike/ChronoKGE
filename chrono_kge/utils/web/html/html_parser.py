"""
HTML Parser
"""

from bs4 import BeautifulSoup


class HTML_Parser:

    def __init__(self) -> None:
        """"""
        self.soup = None
        return

    def parse_document(self, doc):
        """"""
        raise NotImplementedError

    def parse_doc(self, doc) -> None:
        """"""
        self.soup = BeautifulSoup(doc, 'lxml')
        return
