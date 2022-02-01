"""
Sigma regularizer
"""

from chrono_kge.model.module.regularizer.regularizer import Regularizer


class Sigma(Regularizer):
    """Sigma regularizer
    """

    def __init__(self, weight: float, power: int):
        super().__init__(weight, power)
        return

    def __call__(self, factors):
        pass
