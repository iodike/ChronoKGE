"""
Metrics
"""

import numpy as np

from chrono_kge.utils.vars.constants import METRIC
from chrono_kge.utils.logger import logger


def mean_rank(rank):
    """Mean rank.
    """

    m_r = 0
    N = len(rank)
    for i in rank:
        m_r = m_r + i / N

    return m_r


def mean_rr(rank):
    """Mean reciprocal rank.
    """

    mrr = 0
    N = len(rank)
    for i in rank:
        mrr = mrr + 1 / i / N

    return mrr


def hits_k(rank, k):
    """Hits@k.
    """

    hit = 0
    for i in rank:
        if i <= k:
            hit = hit + 1

    hit = hit / len(rank)

    return hit


def compute_metrics(hits: list, ranks: list, logging=False) -> dict:
    """Compute metrics.
    """

    metrics = {
        METRIC.H10: np.mean(hits[9]),
        METRIC.H3: np.mean(hits[2]),
        METRIC.H1: np.mean(hits[0]),
        METRIC.MR: np.mean(ranks),
        METRIC.MRR: np.mean(1. / np.array(ranks))
    }

    if logging:
        logger.info('Hits @10: {0}'.format(metrics[METRIC.H10]))
        logger.info('Hits @3: {0}'.format(metrics[METRIC.H3]))
        logger.info('Hits @1: {0}'.format(metrics[METRIC.H1]))
        logger.info('Mean rank: {0}'.format(metrics[METRIC.MR]))
        logger.info('Mean reciprocal rank: {0}'.format(metrics[METRIC.MRR]))

    return metrics


def compute_ranks_hits(pred, real, ranks: list, hits: list):
    """Compute ranks and hits
    """
    for j in range(pred.shape[0]):

        rank = np.where(pred[j] == real[j])[0][0]
        ranks.append(rank + 1)

        for level in range(10):
            if rank <= level:
                hits[level].append(1.0)
            else:
                hits[level].append(0.0)

    return
