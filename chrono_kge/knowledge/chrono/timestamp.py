"""
Timestamp
"""

import random
import re
import numpy as np
import time

from datetime import date, datetime, timedelta

from chrono_kge.knowledge.knowledge_base import KnowledgeBase
from chrono_kge.knowledge.chrono.time_components import TimeComponents
from chrono_kge.utils.vars.constants import REGEX, TIME


class Timestamp:

    def __init__(self,
                 timestamp: [date, str] = None,
                 kb: KnowledgeBase = None,
                 rand: bool = False,
                 ) -> None:
        """"""
        self.kb = kb
        self.rand = rand
        self.default = self.sample_from_kb()

        '''Init date'''
        if rand or timestamp is None:
            self.dt: date = self.default
        elif isinstance(timestamp, str):
            self.dt: date = self._init_from_str(timestamp)
        elif timestamp:
            self.dt: date = timestamp
        else:
            exit("Timestamp requires atleast 1 valid argument.")

        '''Absolute time since epoch'''
        self.abs_time = self.get_abs_time()

        '''Components'''
        self.tc = TimeComponents(self.dt)

        return

    def _init_from_str(self, timestamp: str):
        """
        :param timestamp: Requires YYYY-mm-dd format
        """
        y, m, d = self.get_year(timestamp), self.get_month(timestamp), self.get_day(timestamp)
        return date(y, m, d)

    def __str__(self):
        """"""
        return self.dt.strftime("%Y-%m-%d")

    def sample_from_kb(self):
        """"""
        return self.random_date(self.kb.start_date, self.kb.end_date) if self.kb else TIME.NONE_DATE

    def get_year(self, timestamp: str) -> int:
        """Timestamp year.
        Get year in YYYY-MM-DD.
        """
        year = self.default.year

        try:
            year = time.strptime(timestamp, '%Y-%m-%d').tm_year
        except ValueError:
            e_match = re.search(REGEX.DATE_YXX, timestamp)
            if e_match:
                year = int(e_match.groups()[0])

        return year

    def get_month(self, timestamp: str) -> int:
        """Timestamp month

        Get month in YYYY-MM-DD.
        """
        month = self.default.month

        try:
            month = time.strptime(timestamp, '%Y-%m-%d').tm_mon
        except ValueError:
            e_match = re.search(REGEX.DATE_XMX, timestamp)
            if e_match:
                month = int(e_match.groups()[0])

        return month

    def get_day(self, timestamp: str) -> int:
        """Timestamp day.

        Get day in YYYY-MM-DD.
        """
        day = self.default.day

        try:
            day = time.strptime(timestamp, '%Y-%m-%d').tm_mday
        except ValueError:
            e_match = re.search(REGEX.DATE_XXD, timestamp)
            if e_match:
                day = int(e_match.groups()[0])

        return day

    def get_abs_time(self, base_date: datetime.date = TIME.EPOCH_DATE) -> int:
        """Absolute time in --days-- since epoch.
        """
        abs_sec = (self.dt - base_date).total_seconds()
        abs_day = int(abs_sec // TIME.DAY_IN_SEC)
        return abs_day

    @staticmethod
    def from_id(tid: int, kb: KnowledgeBase):
        """Create timestamp from KB using time id.
        """
        abs_sec = (kb.start_date - TIME.EPOCH_DATE).total_seconds()
        abs_sec += (tid * kb.gran) * TIME.DAY_IN_SEC
        id_date = date.fromtimestamp(abs_sec)
        return Timestamp(id_date, kb)

    @staticmethod
    def get_diff_days(s_date, e_date) -> int:
        """Compute distance between two timestamps in days.

        TODO: see get_year()

        """
        if isinstance(s_date, str):
            s_time = Timestamp(s_date)
        else:
            s_time = Timestamp(s_date.strftime("%Y-%m-%d"))

        if isinstance(e_date, str):
            e_time = Timestamp(e_date)
        else:
            e_time = Timestamp(e_date.strftime("%Y-%m-%d"))

        diff_days = e_time.get_abs_time(s_time.dt)

        return diff_days

    @staticmethod
    def random_date(start_date: date, end_date: date) -> date:
        """"""
        time_between_dates = end_date - start_date
        days_between_dates = time_between_dates.days
        random_number_of_days = random.randrange(days_between_dates)
        rand_date = start_date + timedelta(days=random_number_of_days)

        if rand_date.month == 2 and rand_date.day > 28:
            return Timestamp.random_date(start_date, end_date)

        return rand_date

    @staticmethod
    def random_days(base_date: date, start_date: date, end_date: date) -> int:
        """"""
        rand_date = Timestamp.random_date(start_date, end_date)
        days = int((time.mktime(rand_date.timetuple()) - time.mktime(base_date.timetuple())) / TIME.DAY_IN_SEC)
        return days

    @staticmethod
    def random():
        """"""
        return Timestamp(None, rand=True)

    @staticmethod
    def years_to_days(years: int) -> int:
        """"""
        days = 0

        for y in range(1, years + 1):
            days += 365

            if y % 400 == 0:
                days += 1
            elif y % 100 == 0:
                continue
            elif y % 4 == 0:
                days += 1
            else:
                continue

        return days

    def get_components_and_bounds(self, components_only=False) -> list:
        """
        components and bounds
        """
        cbs = self.tc.get_components()

        if not components_only:
            return cbs

        return [cbx[0] for cbx in cbs]

    def get_cyclical_components(self):
        """"""
        cyc = []

        for cbs in self.get_components_and_bounds():
            sin, cos = self.get_cycles(cbs[0], cbs[1])
            cyc.append((sin, cos))

        return cyc

    @staticmethod
    def get_cycles(x, p, s=0):
        """"""
        sin = np.sin(2 * np.pi * x / p) + s
        cos = np.cos(2 * np.pi * x / p) + s
        return sin, cos

    @staticmethod
    def get_time_in_sec():
        """"""
        return int(time.time())
