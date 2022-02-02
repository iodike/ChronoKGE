"""
Constants
"""

import datetime
import os


class TIME(object):
    """
    Time constants
    """

    '''global'''
    MILLENIUM_IN_YEARS: int = 1000
    CENTURY_IN_YEARS: int = 100
    DECADE_IN_YEARS: int = 10

    '''min'''
    MIN_IN_SEC: int = 60

    '''hour'''
    HRS_IN_MIN: int = 60
    HRS_IN_SEC: int = HRS_IN_MIN * MIN_IN_SEC

    '''day'''
    DAY_IN_HRS: int = 24
    DAY_IN_MIN: int = DAY_IN_HRS * HRS_IN_MIN
    DAY_IN_SEC: int = DAY_IN_HRS * HRS_IN_SEC

    '''week'''
    WEEK_IN_DAY: int = 7
    WEEK_IN_SEC: int = WEEK_IN_DAY * DAY_IN_SEC

    '''year'''
    YEAR_IN_SEASON: int = 4
    YEAR_IN_MONTH: int = 12
    YEAR_IN_DAY: float = 365.25
    YEAR_IN_WEEK: float = YEAR_IN_DAY / WEEK_IN_DAY
    YEAR_IN_SEC: int = YEAR_IN_DAY * DAY_IN_SEC

    '''season'''
    SEASON_IN_MONTH: int = YEAR_IN_MONTH // YEAR_IN_SEASON
    SEASON_IN_WEEK: float = YEAR_IN_WEEK / YEAR_IN_SEASON
    SEASON_IN_DAY: float = YEAR_IN_DAY / YEAR_IN_SEASON

    '''month'''
    MONTH_IN_WEEK: float = YEAR_IN_WEEK / YEAR_IN_MONTH
    MONTH_IN_DAY: float = YEAR_IN_DAY / YEAR_IN_MONTH

    '''max'''
    MIN_DAY, MAX_DAY = 1, 31
    MIN_MONTH, MAX_MONTH = 1, 12
    MIN_YEAR, MAX_YEAR = 1, 2999

    '''defaults'''
    EPOCH_DATE: datetime.date = datetime.date(1970, 1, 1)
    NONE_DATE: datetime.date = EPOCH_DATE


class REGEX(object):
    """
    Regular expressions
    """

    DATE_YXX = "([\\d]{4})-[#|\\d]{2}-[#|\\d]{2}"
    DATE_XMX = "[#|\\d]{4}-([\\d]{2})-[#|\\d]{2}"
    DATE_XXD = "[#|\\d]{4}-[#|\\d]{2}-([\\d]{2})"
    DATE_YMD = "([\\d]{4})-([\\d]{2})-([\\d]{2})"

    DOM_ENT = "([\\w]+)_([\\w&]+)"


class METRIC(object):
    """
    Metrics
    """

    H1 = "h1"
    H3 = "h3"
    H10 = "h10"
    MR = "mr"
    MRR = "mrr"
    MTT = "mtt"

    ALL = [H10, H3, H1, MR, MRR, MTT]


class TUNER(object):
    """
    Tuner constants
    """

    ATTR_N_TRIALS: str = "trials"
    ATTR_N_HOURS: str = 'hours'
    DIR_MAX = 'maximize'
    DIR_MIN = 'minimize'
    TIMEOUT = 1 * TIME.DAY_IN_SEC
    STUDY_PREFIX = "study-"
    START_TRIALS = 10
    START_STEPS_RATIO = 0.1


class TRAINER(object):
    """
    Trainer constants
    """

    NONE = ''
    ONE_VS_ALL = 'ova'
    NEGATIVE_SAMPLING = 'ns'


class FTYPE(object):
    """
    File type constants
    """

    TXT = ".txt"
    PKL = ".pkl"
    CSV = ".csv"
    PNG = ".png"
    JPG = ".jpg"
    PT = ".pt"


class FNAME(object):
    """
    File name constants
    """

    LOCKFILE = "~/.data.lock"


class SETNAME(object):
    """
    Set name constants
    """

    TOTAL = 'total'
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'


class COMMAND(object):
    """
    Execution commands
    """

    RUN: str = 'run'
    TUNE: str = 'tune'
    PLOT: str = 'plot'


class DIR(object):
    """
    Directories
    """

    NONE: str = ''

    SRC: str = 'chrono_kge'

    CONFIG: str = 'config'

    DATA: str = 'data'
    AUGMENT: str = 'augmented'

    OUTPUT: str = 'output'

    UTILS: str = 'utils'
    WEB: str = 'web'
    VARS: str = 'vars'
    DRIVER: str = 'driver'

    KG_STATIC: str = 'static'
    KG_TEMPORAL: str = 'temporal'
    KG_SYNTHETIC: str = 'synthetic'


class PATH(object):
    """
    Paths
    """

    MAIN = os.getcwd()
    PROJECT = os.path.join(MAIN)

    SRC = os.path.join(PROJECT, DIR.SRC)
    CONFIG = os.path.join(PROJECT, DIR.CONFIG)
    DATA = os.path.join(PROJECT, DIR.DATA)
    OUTPUT = os.path.join(PROJECT, DIR.OUTPUT)

    AUGMENT = os.path.join(DATA, DIR.AUGMENT)

    UTILS = os.path.join(SRC, DIR.UTILS)
    WEB = os.path.join(UTILS, DIR.WEB)
    DRIVER = os.path.join(WEB, DIR.DRIVER)


class DRIVER(object):
    """
    Driver
    """

    TIMEOUT = 1
