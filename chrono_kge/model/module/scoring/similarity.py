"""
Similarity Scorer
"""

from chrono_kge.model.module.scoring.scoring import Scorer


class CosineSimilarityScorer(Scorer):

    def __init__(self):
        """"""
        super().__init__()
        return

    def score(self, a, b):
        """"""
        return a @ b.t()
