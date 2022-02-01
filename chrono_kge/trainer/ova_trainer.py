"""
One-vs-All Trainer
"""

import torch

from chrono_kge.utils.helpers import lists_to_tuples
from chrono_kge.trainer.base_trainer import BaseTrainer
from chrono_kge.main.handler.exp_handler import ExperimentHandler
from chrono_kge.main.handler.data_handler import DataHandler
from chrono_kge.main.handler.env_handler import EnvironmentHandler


class OVATrainer(BaseTrainer):

    def __init__(self,
                 exp_handler: ExperimentHandler,
                 data_handler: DataHandler,
                 env_handler: EnvironmentHandler
                 ) -> None:
        """"""
        super().__init__(exp_handler, data_handler, env_handler)

        return

    def init_scores(self, triples_batch: list):
        """Generates target scores.
        """
        targets_batch_scores = torch.zeros((len(triples_batch), self.DATA.kg.n_entity))

        for idx, triple in enumerate(triples_batch):
            triple = tuple(triple[:-1])
            targets_batch_scores[idx, self.DATA.kg.dataset.TRAIN_SET.target_vocabulary[triple]] = 1.

        return targets_batch_scores

    def generate_train_batches(self, shuffle: bool = True):
        """"""
        for triples_batch in self.generate_batches(lists_to_tuples(self.DATA.kg.dataset.TRAIN_SET.triples),
                                                   self.batch_size, shuffle):

            targets_batch_scores = self.init_scores(triples_batch)

            if self.EXP.label_smoothing:
                targets_batch_scores = ((1.0 - self.EXP.label_smoothing) * targets_batch_scores) + \
                                      (1.0 / targets_batch_scores.size()[1])

            triples_batch = torch.IntTensor(triples_batch).to(self.ENV.device)
            targets_batch_scores = torch.FloatTensor(targets_batch_scores).to(self.ENV.device)

            yield triples_batch, targets_batch_scores
