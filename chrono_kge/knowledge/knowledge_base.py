"""
Knowledge base
"""

import math
import os

from datetime import date

from chrono_kge.utils.logger import logger
from chrono_kge.utils.vars.constants import DIR, TIME, PATH
from chrono_kge.utils.vars.modes import AUG


class KnowledgeBase:
    """Knowledge base

    :param name: KB name.
    :param start_date: KB start date.
    :param num_days: Time interval in days.
    :param gran: Time granularity in days.
    """

    def __init__(self,
                 name: str,
                 genre: int,
                 start_date: date = None,
                 num_days: int = 0,
                 gran: float = 1.0
                 ) -> None:
        """"""
        self.name: str = name
        self.genre: int = genre
        self.path: str = ''
        self.num_days: int = num_days
        self.gran: float = gran
        self.aug_mode: int = AUG.NONE

        self.start_date: date = start_date
        self.end_date: date = self.get_end_date() if start_date else None
        return

    def setup(self, aug_mode: int):
        """"""
        self.aug_mode = aug_mode
        base_path = PATH.AUGMENT if self.is_augmented() else PATH.DATA
        self.set_data_path(base_path)
        return

    def set_data_path(self, base_path: str) -> None:
        """"""
        if self.genre == KnowledgeBases.STATIC:
            self.path = os.path.join(base_path, DIR.KG_STATIC, self.name)
        elif self.genre == KnowledgeBases.TEMPORAL:
            self.path = os.path.join(base_path, DIR.KG_TEMPORAL, self.name)
        elif self.genre == KnowledgeBases.SYNTHETIC:
            self.path = os.path.join(base_path, DIR.KG_SYNTHETIC, self.name)
        else:
            logger.error("No data path for kb `%s` with genre `%d`." % (self.name, self.genre))
            exit(1)
        return

    def is_static(self) -> bool:
        """"""
        return self.genre == KnowledgeBases.STATIC

    def is_temporal(self) -> bool:
        """"""
        return self.genre == KnowledgeBases.TEMPORAL

    def is_synthetic(self) -> bool:
        """"""
        return self.genre == KnowledgeBases.SYNTHETIC

    def is_augmented(self) -> bool:
        """"""
        return AUG.isAKG(self.aug_mode)

    def is_wikidata(self) -> bool:
        """"""
        return self.name == KnowledgeBases.WIKIDATA12K.name

    def is_icews(self) -> bool:
        """"""
        return self.name in [kb.name for kb in [
            KnowledgeBases.ICEWS14,
            KnowledgeBases.ICEWS05_15,
            KnowledgeBases.ICEWS18
        ]]

    def is_facebook(self) -> bool:
        """"""
        return self.name in [kb.name for kb in [
            KnowledgeBases.FB15K,
            KnowledgeBases.FB15K_237
        ]]

    def is_wordnet(self) -> bool:
        """"""
        return self.name in [kb.name for kb in [
            KnowledgeBases.WN18,
            KnowledgeBases.WN18RR
        ]]

    def is_gdelt(self) -> bool:
        """"""
        return self.name in [kb.name for kb in [
            KnowledgeBases.GDELT5C,
            KnowledgeBases.GDELT8K
        ]]

    def is_yago(self) -> bool:
        """"""
        return self.name in [kb.name for kb in [
            KnowledgeBases.YAGO11K,
            KnowledgeBases.YAGO15K
        ]]

    def get_end_ymd(self, y: int, m: int, d: int) -> (int, int, int):
        """"""
        assert self.start_date is not None

        cdays: int = self.num_days
        cdays += y*TIME.YEAR_IN_DAY
        cdays += m*TIME.MONTH_IN_DAY
        cdays += d

        years = int(math.floor(cdays/TIME.YEAR_IN_DAY))
        cdays -= years * TIME.YEAR_IN_DAY
        months = int(math.floor(cdays/TIME.MONTH_IN_DAY))
        cdays -= months * TIME.MONTH_IN_DAY

        years = 1 if years < 1 else years
        months = 1 if months < 1 else months
        days = 1 if cdays < 1 else cdays

        return years, months, days

    def get_end_date(self) -> date:
        """"""
        assert self.start_date is not None

        ey, em, ed = self.get_end_ymd(self.start_date.year, self.start_date.month, self.start_date.day)
        e_date = date(int(ey), int(em), int(ed))
        return e_date

    def get_n_time(self, time_gran: float = 1.0) -> int:
        """Returns n time units based on KB and sampled time granularity.
        Last time unit included (+1 unit).
        """
        assert self.start_date is not None

        e_date = self.get_end_date()
        d_time = ((e_date - self.start_date).total_seconds() // TIME.DAY_IN_SEC) + 1
        n_time = int(d_time / (self.gran * time_gran))
        return n_time + 1


class KnowledgeBases(object):
    """
    Knowledge bases constant
    """

    '''genre'''

    NONE: int = 0
    STATIC: int = 1
    TEMPORAL: int = 2
    SYNTHETIC: int = 3

    '''static'''

    FB15K: KnowledgeBase = KnowledgeBase(
        name='fb15k',
        genre=STATIC
    )

    FB15K_237: KnowledgeBase = KnowledgeBase(
        name='fb15k-237',
        genre=STATIC
    )

    WN18: KnowledgeBase = KnowledgeBase(
        name='wn18',
        genre=STATIC
    )

    WN18RR: KnowledgeBase = KnowledgeBase(
        name='wn18rr',
        genre=STATIC
    )

    ICEWS14S: KnowledgeBase = KnowledgeBase(
        name='icews14s',
        genre=STATIC
    )

    ICEWS14SF: KnowledgeBase = KnowledgeBase(
        name='icews14sf',
        genre=STATIC
    )

    '''temporal'''

    WIKIDATA12K: KnowledgeBase = KnowledgeBase(
        name='wikidata12k',
        genre=TEMPORAL,
        start_date=date(18, 1, 1),
        num_days=math.floor(2001 * TIME.YEAR_IN_DAY),
        gran=1.0 * TIME.YEAR_IN_DAY
    )

    YAGO11K: KnowledgeBase = KnowledgeBase(
        name='yago11k',
        genre=TEMPORAL,
        start_date=date(1, 1, 1),
        num_days=math.floor(2843 * TIME.YEAR_IN_DAY),
        gran=1.0
    )

    # TODO: Add stats
    YAGO15K: KnowledgeBase = KnowledgeBase(
        name='yago15k',
        genre=TEMPORAL,
        start_date=None,
        num_days=0,
        gran=1.0
    )

    ICEWS14: KnowledgeBase = KnowledgeBase(
        name='icews14',
        genre=TEMPORAL,
        start_date=date(2014, 1, 1),
        num_days=365,
        gran=1.0
    )

    ICEWS05_15: KnowledgeBase = KnowledgeBase(
        name='icews05-15',
        genre=TEMPORAL,
        start_date=date(2005, 1, 1),
        num_days=4017,
        gran=1.0
    )

    ICEWS18: KnowledgeBase = KnowledgeBase(
        name='icews18',
        genre=TEMPORAL,
        start_date=date(2018, 1, 1),
        num_days=303,
        gran=1.0 / TIME.DAY_IN_HRS
    )

    GDELT5C: KnowledgeBase = KnowledgeBase(
        name='gdelt5c',
        genre=TEMPORAL,
        start_date=date(2015, 4, 1),
        num_days=366,
        gran=1.0
    )

    # TODO: verify start date / granularity
    GDELT8K: KnowledgeBase = KnowledgeBase(
        name='gdelt8k',
        genre=TEMPORAL,
        start_date=date(1979, 1, 1),
        num_days=44626 // (TIME.DAY_IN_HRS * 4),
        gran=464 / 44626
    )

    '''synthetic'''

    SMALL_TOWN: KnowledgeBase = KnowledgeBase(
        name='small-town',
        genre=SYNTHETIC,
        start_date=TIME.EPOCH_DATE,
        num_days=math.floor(50*TIME.YEAR_IN_DAY),
        gran=1.0
    )

    DUMMY: KnowledgeBase = KnowledgeBase(
        name='dummy',
        genre=TEMPORAL,
        start_date=TIME.EPOCH_DATE,
        num_days=30,
        gran=1.0
    )

    '''none'''

    NONE: KnowledgeBase = KnowledgeBase(
        name='none',
        genre=NONE,
        start_date=TIME.EPOCH_DATE,
        num_days=0,
        gran=1.0
    )

    '''all'''

    ALL_KB = [
        WIKIDATA12K,
        YAGO11K,
        ICEWS14,
        ICEWS05_15,
        ICEWS18,
        GDELT5C,
        GDELT8K,
        DUMMY,
        SMALL_TOWN,
        ICEWS14S,
        ICEWS14SF,
        FB15K,
        FB15K_237,
        WN18,
        WN18RR,
        NONE
    ]

    @staticmethod
    def has_timestamps(kb: KnowledgeBase) -> bool:
        """
        Whether the kb has timestamps in data sets.
        """
        return kb in [
            KnowledgeBases.ICEWS05_15,
            KnowledgeBases.ICEWS14,
            KnowledgeBases.WIKIDATA12K,
            KnowledgeBases.YAGO11K,
            KnowledgeBases.YAGO15K
        ]

    @staticmethod
    def has_indices(kb: KnowledgeBase) -> bool:
        """
        Whether the kb has entity ids in datasets.
        """
        return kb in [
            KnowledgeBases.ICEWS18,
            KnowledgeBases.GDELT5C,
            KnowledgeBases.GDELT8K,
            KnowledgeBases.DUMMY,
            KnowledgeBases.WN18,
            KnowledgeBases.WN18RR,
            KnowledgeBases.WIKIDATA12K,
            KnowledgeBases.YAGO11K,
        ]

    @staticmethod
    def has_labels(kb: KnowledgeBase) -> bool:
        """
        Whether the kb has entity labels in datasets.
        """
        return kb in [
            KnowledgeBases.ICEWS14,
            KnowledgeBases.ICEWS05_15,
            KnowledgeBases.ICEWS14S,
            KnowledgeBases.ICEWS14SF,
            KnowledgeBases.YAGO15K
        ]

    @staticmethod
    def exists_augmented(kb: KnowledgeBase) -> bool:
        """"""
        return kb.genre in [kb.genre for kb in [
            KnowledgeBases.ICEWS05_15,
            KnowledgeBases.ICEWS14,
            KnowledgeBases.ICEWS18
        ]]

    @staticmethod
    def get_kb_by_name(kb_name: str) -> KnowledgeBase:
        """"""
        for kb in KnowledgeBases.ALL_KB:
            if kb.name == kb_name:
                return kb
        return KnowledgeBases.NONE
