"""
Negative Sampling Trainer
"""

import torch
import numpy as np

from collections import defaultdict

from chrono_kge.utils.logger import logger
from chrono_kge.utils.helpers import lists_to_tuples
from chrono_kge.trainer.base_trainer import BaseTrainer
from chrono_kge.main.handler.exp_handler import ExperimentHandler
from chrono_kge.main.handler.data_handler import DataHandler
from chrono_kge.main.handler.env_handler import EnvironmentHandler


class NSTrainer(BaseTrainer):

    def __init__(self,
                 exp_handler: ExperimentHandler,
                 data_handler: DataHandler,
                 env_handler: EnvironmentHandler
                 ) -> None:
        """"""
        super().__init__(exp_handler, data_handler, env_handler)

        '''sample size'''
        logger.info("Negative sampling with rate: %d" % self.EXP.n_samples)

        '''neg'''
        self.neg_train_triples_vocab = defaultdict(list)

        '''sampling'''
        self.sample_negatives()

        return

    def generate_ns_batches(self, triples: list, shuffle: bool = True):
        """"""
        for pos_triple_batch in self.generate_batches(triples, self.batch_size, shuffle):

            '''pos+neg'''
            triple_batch = []

            for idx, triple in enumerate(pos_triple_batch):

                '''pos'''
                triple_batch.append(list(triple))

                '''neg'''
                subject = tuple(triple[:-1])
                for target in self.neg_train_triples_vocab[subject]:
                    triple = list(subject)
                    triple.append(target)
                    triple_batch.append(triple)

            yield triple_batch

    def generate_train_batches(self, shuffle: bool = True):
        """
        TODO: add label smoothing
        """
        for triple_batch in self.generate_ns_batches(lists_to_tuples(self.DATA.kg.dataset.TRAIN_SET.triples), shuffle):

            triple_batch = torch.LongTensor(triple_batch).to(self.ENV.device)
            target_scores = torch.LongTensor(triple_batch[:, -1].detach().cpu().numpy()).to(self.ENV.device)

            yield triple_batch, target_scores

    def sample_negatives(self):
        """"""
        for query, pos_targets in self.DATA.kg.dataset.TRAIN_SET.target_vocabulary.items():
            sample_pool = list(range(self.DATA.kg.n_entity))

            '''remove subject'''
            if list(query)[0] in sample_pool:
                sample_pool.remove(list(query)[0])

            '''remove objects'''
            for target in set(pos_targets):
                if target in sample_pool:
                    sample_pool.remove(target)

            '''sample'''
            neg_targets = self.sample(sample_pool)
            self.neg_train_triples_vocab[query] = neg_targets

        return

    def sample(self, sample_pool: list):
        """"""
        neg_targets = set()
        n_samples = self.EXP.n_samples if len(sample_pool) > self.EXP.n_samples else len(sample_pool)

        while len(neg_targets) < n_samples:
            neg_targets.add(sample_pool[np.random.randint(0, len(sample_pool))])

        return list(neg_targets)
