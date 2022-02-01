"""
Base Trainer
"""

import torch
import numpy as np

from chrono_kge.trainer.abstract_trainer import AbstractTrainer
from chrono_kge.main.handler.exp_handler import ExperimentHandler
from chrono_kge.main.handler.data_handler import DataHandler
from chrono_kge.main.handler.env_handler import EnvironmentHandler


class BaseTrainer(AbstractTrainer):

    def __init__(self,
                 exp_handler: ExperimentHandler,
                 data_handler: DataHandler,
                 env_handler: EnvironmentHandler
                 ) -> None:
        """"""
        self.EXP = exp_handler
        self.DATA = data_handler
        self.ENV = env_handler
        self.batch_size = self.EXP.batch_size \
            if self.DATA.kg.dataset.TOTAL_SET.n_triples > self.EXP.batch_size \
            else self.DATA.kg.dataset.TOTAL_SET.n_triples
        return

    @staticmethod
    def generate_batches(triples: list, batch_size: int, shuffle: bool = True):
        """Generates mini batches.
        """
        if shuffle:
            triples = np.random.permutation(triples)

        dim0, dim1 = triples.shape
        for i in range(0, dim0, batch_size):
            yield triples[i:i + batch_size]

    def generate_batches_t(self, triples: list, batch_size: int, shuffle: bool = True):
        """"""
        for triples_batch in self.generate_batches(triples, batch_size, shuffle):
            yield torch.LongTensor(triples_batch).to(self.ENV.device)

    def rank_targets(self, triples_batch, triples_batch_scores):
        """"""
        dim0, dim1 = triples_batch.shape

        '''targets'''
        batch_targets = triples_batch[:, -1]

        for j in range(dim0):

            '''get triple'''
            test_subject = tuple(triples_batch[j][:-1])

            '''get objects'''
            test_targets = self.DATA.kg.dataset.TOTAL_SET.target_vocabulary[test_subject]

            '''get real target'''
            test_target = batch_targets[j]

            '''get score'''
            test_target_score = triples_batch_scores[j, test_target].item()

            '''zero all other targets'''
            triples_batch_scores[j, test_targets] = 0.0

            '''set current target's score'''
            triples_batch_scores[j, test_target] = test_target_score

        '''sort scores'''
        score_values, score_ids = torch.sort(triples_batch_scores, dim=1, descending=True)
        score_ids = score_ids.detach().cpu().numpy()

        return score_ids, score_values
