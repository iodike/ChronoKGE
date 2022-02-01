"""
Defaults
"""

import torch

from chrono_kge.utils.vars.constants import DIR, PATH, TRAINER
from chrono_kge.utils.vars.modes import MOD, AUG, REG, ENC
from chrono_kge.knowledge.knowledge_base import KnowledgeBases, KnowledgeBase


class Default(object):
    """
    Default arguments.
    """

    '''EXPERIMENT'''

    MODEL: str = 'tlowfer'
    EXPERIMENT: str = 'base'

    KB: KnowledgeBase = KnowledgeBases.ICEWS14
    TRAINER: int = TRAINER.NEGATIVE_SAMPLING

    MOD_MODE: int = MOD.NONE
    AUG_MODE: int = AUG.REV
    REG_MODE: int = REG.NONE
    ENC_MODE: int = ENC.SIMPLE

    EVAL_STEP: int = 10
    TEST_STEP: int = 100

    SAVE: bool = False
    OUTPUT_DIR: str = DIR.OUTPUT

    CUDA = True
    DEVICE: str = "cpu"

    '''HYPER PARAMETERS'''

    MAX_EPOCHS: int = 1000
    BATCH_SIZE: int = 512
    EMBEDDING_DIMENSION: int = 400
    LEARNING_RATE: float = 1e-02
    DECAY_RATE: float = 1.0
    TIME_GRAN: float = 1.0
    FACTOR_RANK: int = 32
    DROPOUT_INPUT: float = 0.2
    DROPOUT_HIDDEN: float = 0.5
    LABEL_SMOOTHING: float = 1e-02
    NEG_SAMPLES: int = 4
    MARGIN: float = 0.1

    '''Normalisation / Regularisation'''

    ALPHA: float = 1e-12
    L1: float = 0.0
    L2: float = 0.0
    REG_EMB: float = 0.0
    REG_TIME: float = 0.0
    REG_POWER: int = 3

    '''KG'''

    DATA_PATH: str = PATH.DATA

    '''TUNE'''

    TUNE_TRIALS: int = None
    TUNE_HOURS: int = 24
    EARLY_STOP: bool = False

    '''PLOT'''

    PLOT_XSCALE: str = "linear"
    PLOT_YSCALE: str = "linear"

    '''TORCH'''

    DTYPE = torch.float
