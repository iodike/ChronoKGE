"""
Params
"""

from chrono_kge.utils.vars.defaults import Default
from chrono_kge.knowledge.knowledge_base import KnowledgeBases


class Parameter:

    def __init__(self, default=None):
        """"""
        self.param = [default]
        return

    def get(self):
        """"""
        return self.param


class ExponentialParameter(Parameter):

    def __init__(self,
                 base: int = 0,
                 exp: tuple = (0, 0),
                 neg: bool = False,
                 rev: int = 0,
                 default=None
                 ) -> None:
        """"""
        super().__init__(default)
        if not default:
            sgn = -1 if neg else 1
            sub = -1 if rev else 1
            self.param = [(rev + sub * (base ** k)) for k in range(sgn * exp[0], sgn * (exp[1] + 1), sgn)]
        return


class LinearParameter(Parameter):

    def __init__(self,
                 alpha: int = 0,
                 rng: tuple = (0, 0),
                 neg: bool = False,
                 rev: int = 0,
                 default=None
                 ) -> None:
        """"""
        super().__init__(default)
        if not default:
            sgn = -1 if neg else 1
            sub = -1 if rev else 1
            self.param = [(rev + sub * (alpha * k)) for k in range(sgn * rng[0], sgn * (rng[1] + 1), sgn)]
        return


class DefaultParameterList:

    def __init__(self):
        """"""
        self.kgs = []
        self.params = {
            'lr': ExponentialParameter(default=Default.LEARNING_RATE).get(),
            'dr': ExponentialParameter(default=Default.DECAY_RATE).get(),
            'bs': ExponentialParameter(default=Default.BATCH_SIZE).get(),
            'ed': LinearParameter(default=Default.EMBEDDING_DIMENSION).get(),
            'tg': ExponentialParameter(default=Default.TIME_GRAN).get(),
            'fr': ExponentialParameter(default=Default.FACTOR_RANK).get(),
            'ls': ExponentialParameter(default=Default.LABEL_SMOOTHING).get(),
            'l1': ExponentialParameter(default=Default.L1).get(),
            'l2': ExponentialParameter(default=Default.L2).get(),
            're': ExponentialParameter(default=Default.REG_EMB).get(),
            'rt': ExponentialParameter(default=Default.REG_TIME).get(),
            'ns': ExponentialParameter(default=Default.NEG_SAMPLES).get(),
            'mr': ExponentialParameter(default=Default.MARGIN).get()
        }

        return

    def get(self):
        """"""
        return self.params


class BENCHMARK_ParameterList(DefaultParameterList):

    def __init__(self):
        """"""
        super().__init__()

        self.kgs = [KnowledgeBases.WIKIDATA12K.name]
        self.params = {
            'lr': ExponentialParameter(base=10,  exp=(1, 4), neg=True, rev=0).get(),
            'dr': ExponentialParameter(base=10, exp=(1, 4), neg=True, rev=1).get(),
            'bs': ExponentialParameter(base=2, exp=(7, 10), neg=False, rev=0).get(),
            'ed': LinearParameter(alpha=100, rng=(3, 3), neg=False, rev=0).get(),
            'tg': ExponentialParameter(base=2, exp=(0, 0), neg=False, rev=0).get(),
            'fr': ExponentialParameter(base=2, exp=(5, 8), neg=False, rev=0).get(),
            'ls': ExponentialParameter(base=10, exp=(1, 3), neg=True, rev=0).get(),
            'l1': ExponentialParameter(base=10, exp=(9, 12), neg=True, rev=0, default=Default.L1).get(),
            'l2': ExponentialParameter(base=10, exp=(9, 12), neg=True, rev=0, default=Default.L2).get(),
            're': ExponentialParameter(base=10, exp=(9, 12), neg=True, rev=0, default=Default.REG_EMB).get(),
            'rt': ExponentialParameter(base=10, exp=(9, 12), neg=True, rev=0, default=Default.REG_TIME).get(),
            'ns': ExponentialParameter(base=2, exp=(1, 4), neg=False, rev=0, default=Default.NEG_SAMPLES).get(),
            'mr': LinearParameter(alpha=10, rng=(1, 12), neg=False, rev=0, default=Default.MARGIN).get()
        }

        return
