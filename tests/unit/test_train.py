"""
Test train
"""

import os
import unittest
import numpy as np
import torch

from chrono_kge.knowledge.knowledge_base import KnowledgeBases
from chrono_kge.utils.vars.constants import DIR
from chrono_kge.utils.vars.modes import AUG
from chrono_kge.utils.logger import logger
from chrono_kge.utils.helpers import lists_to_tuples, createUID, makeDirectory
from chrono_kge.utils.metrics import compute_ranks_hits, compute_metrics


class TrainTest(unittest.TestCase):

    def setUp(self):
        """"""

        self.uid = createUID()
        self.kb = KnowledgeBases.get_kb_by_name('dummy')
        self.kb(mode=AUG.REV)
        self.kb.path = os.path.join(os.getcwd(), "../../data", DIR.KG_SYNTHETIC, self.kb.name)

        self.output_dir = "output/%s/v%s/%s/%s/" % (
            'test',
            '3',
            self.kb.name,
            self.uid
        )

        # self.experiment = TLowFER_Experiment(
        #     uid=self.uid,
        #     kb=self.kb,
        #     train_mode=TRAINER.ONE_VS_ALL,
        #     reg_mode=REG.NONE,
        #     eval_step=10,
        #     cuda=False,
        #     save=False,
        #     out_dir=self.output_dir
        # )
        makeDirectory(self.output_dir)

        kwargs = {
            'temporal_dim': 10,
            'time_gran': 1.0
        }

        # self.experiment(
        #     max_epochs=10,
        #     batch_size=6,
        #     learning_rate=0.1,
        #     decay_rate=1.0,
        #     label_smoothing=0.1,
        #     L1=1e-12,
        #     L2=1e-12,
        #     entity_dim=10,
        #     relation_dim=10,
        #     dropout_input=0.2,
        #     dropout_hidden=0.5,
        #     factor_rank=5,
        #     **kwargs
        # )

        return

    def test_eval(self):

        # self.experiment.before_run()

        train_triples = lists_to_tuples(self.experiment.kg.dataset.TRAIN_SET.triples)
        logger.info(train_triples)

        for it in range(1, self.experiment.GLOBAL.EXP.max_epochs + 1):

            losses = []
            logger.info("----Train----")
            for batch_subjects, batch_targets_scores in self.experiment.trainer.generate_batches_and_scores():

                losses.append(self.experiment.train(batch_subjects, batch_targets_scores))

            logger.info("----Mean Loss----")
            logger.info(np.mean(losses))

            self.experiment.scheduler.step()

            '''EVAL START'''
            logger.info("----Validate----")
            _ = self.eval(test=False)
            # exit(0)

            '''EVAL END'''

        logger.info("----Test----")
        self.experiment.results = self.eval(test=True)

        logger.info(self.experiment.results)

        self.experiment.after_run()
        return

    def eval(self, test: bool):
        """"""
        with torch.no_grad():

            self.experiment.model.eval()

            ranks = []
            hits = [[] for _ in range(10)]

            data = self.experiment.kg.dataset.TEST_SET.triples if test else self.experiment.kg.dataset.VALID_SET.triples

            for test_batch in self.experiment.trainer.generate_train_batches(
                    lists_to_tuples(data), self.experiment.batch_size):

                '''The actual triples (s, p, t, o)'''
                logger.info("----Triples----")
                logger.info(test_batch)

                test_batch_scores, _ = self.experiment.encode(test_batch)

                '''The target scores: each object is scored against a given input (s,p,t).
                e.g. for a given input (s,p,t):
                o1: 0.5
                o2: 0.9
                o3: 0.3
                e.t.c. ...
                '''
                logger.info("----Scores----")
                logger.info(test_batch_scores)

                test_batch_targets = test_batch[:, -1]

                '''The targets `o`'''
                logger.info("----Targets----")
                logger.info(test_batch_targets)

                pred_batch_targets, _ = self.experiment.trainer.rank_targets(
                    test_batch.detach().cpu().numpy(),
                    test_batch_scores
                )

                '''The ranked objects:
                Objects are ranked by their score (in descending order)
                '''
                logger.info("----Ranked predictions----")
                logger.info(pred_batch_targets)

                compute_ranks_hits(pred_batch_targets, test_batch_targets.detach().cpu().numpy(), ranks, hits)

            '''The rank of the prediction
            i.e. the position of the predicted object
            '''
            logger.info("----Ranks----")
            logger.info(ranks)

            '''Hits level
            i.e. the number of hits for a given ranking level
            '''
            logger.info("----Hits----")
            logger.info(hits)

        return compute_metrics(hits, ranks, logging=True)


if __name__ == '__main__':
    unittest.main()
