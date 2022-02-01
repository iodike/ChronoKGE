"""
Scorer
"""

from torch import nn


class Scorer:

    def __init__(self):
        """"""
        return

    def exec(self, a, b):
        """"""
        s = self.score(a, b)
        s = self.activate(s)
        return s

    def score(self, a, b):
        """"""
        pass

    @staticmethod
    def activate(x, act=nn.Sigmoid()):
        """"""
        return act(x)
