"""
Time Components
"""

import math

from datetime import date, datetime

from chrono_kge.utils.vars.constants import TIME


class TimeComponents:

    def __init__(self, timestamp):
        """"""
        self.gc = GlobalComponents(timestamp)
        self.yc = YearComponents(timestamp, self.gc)
        self.sc = SeasonComponents(timestamp, self.yc)
        self.mc = MonthComponents(timestamp, self.sc)
        self.wc = WeekComponents(timestamp, self.mc)
        self.dc = DayComponents(datetime.combine(timestamp, datetime.min.time()), self.wc)
        return

    def get_components(self):
        """"""
        return self.gc.get_components() + self.yc.get_components() + self.sc.get_components() \
                                        + self.mc.get_components() + self.wc.get_components()


class GlobalComponents:

    def __init__(self, dt: date):
        """"""
        self.dt = dt

        self.millenium = self.get_millenium()
        self.century = self.get_century()
        self.decade = self.get_decade()
        self.unit = self.get_unit()
        return

    def get_components(self) -> list:
        """"""
        return [self.millenium, self.century, self.decade, self.unit]

    def get_millenium(self) -> (int, int):
        """
        0-2 (e.g. 2020-01-01 -> 2)
        """
        return (self.dt.year // 1000) % 10, 10

    def get_century(self) -> (int, int):
        """
        0-29 (e.g. 2020-01-01 -> 0)
        """
        return (self.dt.year // 100) % 10, 10

    def get_decade(self) -> (int, int):
        """
        0-299
        (e.g. 2020-01-01 -> 2)
        """
        return (self.dt.year // 10) % 10, 10

    def get_unit(self) -> (int, int):
        """
        0-2999
        (e.g. 2020-01-01 -> 0)
        """
        return self.dt.year % 10, 10


class YearComponents:

    def __init__(self, dt: date, gc: GlobalComponents):
        """"""
        self.dt = dt
        self.gc = gc

        self.season = self.get_season()
        self.month = self.get_month()
        self.week = self.get_week()
        self.day = self.get_day()
        return

    def get_components(self) -> list:
        """"""
        return [self.season, self.month, self.week, self.day]

    def get_season(self) -> (int, int):
        """
        1-4 (e.g. 2020-01-01 -> 4)
        """
        return int(((self.dt.month - 3) % 12) / 3) + 1, TIME.YEAR_IN_SEASON

    def get_month(self) -> (int, int):
        """"""
        return self.dt.month, TIME.YEAR_IN_MONTH

    def get_week(self) -> (int, int):
        """
        1-52(53) (e.g. 2020-01-01 -> 0)
        """
        return self.dt.isocalendar()[1], math.ceil(TIME.YEAR_IN_WEEK)

    def get_day(self) -> (int, int):
        """
        1-365(366) (e.g. 2020-01-01 -> 1)
        """
        return self.dt.timetuple().tm_yday, math.ceil(TIME.YEAR_IN_DAY)


class SeasonComponents:

    def __init__(self, dt: date, yc: YearComponents):
        """"""
        self.dt = dt
        self.yc = yc

        self.month = self.get_month()
        self.week = self.get_week()
        self.day = self.get_day()
        return

    def get_components(self) -> list:
        """"""
        return [self.month, self.week, self.day]

    def get_month(self) -> (int, int):
        """
        1-3 (e.g. 2020-01-01 -> 2)
        """
        return (self.dt.month % 3) + 1, TIME.SEASON_IN_MONTH

    def get_week(self, p=100) -> (int, int):
        """
        1-13(14) (e.g. 2020-01-01 -> 5)
        """
        return self.get_day(p)[0] // TIME.WEEK_IN_DAY, math.ceil(TIME.SEASON_IN_WEEK)

    def get_day(self, p=100) -> (int, int):
        """
        1-91(92) (e.g. 2020-01-01 -> 32)
        """
        return (((self.yc.get_day()[0] + int(TIME.MONTH_IN_DAY)) * p) % int(TIME.SEASON_IN_DAY * p)) // p, math.ceil(
            TIME.SEASON_IN_DAY)


class MonthComponents:

    def __init__(self, dt: date, sc: SeasonComponents):
        """"""
        self.dt = dt
        self.sc = sc

        self.week = self.get_week()
        self.day = self.get_day()
        return

    def get_components(self) -> list:
        """"""
        return [self.week, self.day]

    def get_week(self, p=100) -> (int, int):
        """
        1-4(5) (e.g. 2020-01-01 -> 0)
        """
        return (((self.sc.yc.get_week()[0] - 1) * p) % int(TIME.MONTH_IN_WEEK * p)) // p, math.ceil(TIME.MONTH_IN_WEEK)

    def get_day(self) -> (int, int):
        """
        1-30(31) (e.g. 2020-01-01 -> 1)
        """
        return self.dt.day, math.ceil(TIME.MONTH_IN_DAY)


class WeekComponents:

    def __init__(self, dt: date, mc: MonthComponents):
        """"""
        self.dt = dt
        self.mc = mc

        self.day = self.get_day()
        return

    def get_components(self) -> list:
        """"""
        return [self.day]

    def get_day(self) -> (int, int):
        """
        1-7 (e.g. 2020-01-01 -> 3)
        """
        return self.dt.isocalendar()[2], TIME.WEEK_IN_DAY


class DayComponents:

    def __init__(self, dt: datetime, wc: WeekComponents):
        """"""
        self.dt = dt
        self.wc = wc
        return

    def get_hour(self):
        """"""
        return self.dt.hour

    def get_min(self):
        """"""
        return self.dt.minute

    def get_sec(self):
        """"""
        return self.dt.second
