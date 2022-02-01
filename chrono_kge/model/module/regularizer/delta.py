"""
Delta regularizer
"""

from chrono_kge.model.module.regularizer.omega import Omega


class Delta(Omega):
    """Delta regularizer
    Natural regularizer for embeddings (baseline), c.f., TNTComplEx page 5, pg. 5.

    Params:
    TNTComplEx: p=2,3,4
    """

    def __init__(self, weight: float, power: int):
        super().__init__(weight, power)
        return

    def __call__(self, factors):
        pass
