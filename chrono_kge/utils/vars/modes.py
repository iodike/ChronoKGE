"""
Modes
"""


class MOD(object):
    """
    Modulation modes
    """

    NONE = 0
    T = 1
    TNT = 2


class ENC(object):
    """
    (Time) Encoding modes
    """

    SIMPLE = 0
    CYCLE = 1


class AUG(object):
    """
    Augmentation modes
    """

    NONE: int = 0
    REV: int = 1
    BT: int = 2
    BTC: int = 3
    REV_BT: int = 4

    @staticmethod
    def isNone(mode: int) -> bool:
        return AUG.NONE == mode

    @staticmethod
    def isRev(mode: int) -> bool:
        return AUG.REV == mode or AUG.REV_BT == mode

    @staticmethod
    def isBTC(mode: int) -> bool:
        return AUG.BTC == mode

    @staticmethod
    def isBT(mode: int) -> bool:
        return AUG.isBTC(mode) or AUG.BT == mode or AUG.REV_BT == mode

    @staticmethod
    def isAKG(mode: int) -> bool:
        return AUG.BT == mode or AUG.REV_BT == mode


class REG(object):
    """
    Regularization modes
    """

    NONE: int = 0
    OMEGA: int = 1
    LAMBDA: int = 2
    OMEGA_LAMBDA: int = 3

    @staticmethod
    def isNone(mode: int) -> bool:
        return REG.NONE == mode

    @staticmethod
    def isOmega(mode: int) -> bool:
        return REG.OMEGA == mode or REG.OMEGA_LAMBDA == mode

    @staticmethod
    def isLambda(mode: int) -> bool:
        return REG.LAMBDA == mode or REG.OMEGA_LAMBDA == mode
