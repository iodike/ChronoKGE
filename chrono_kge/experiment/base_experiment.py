"""
Base Experiment
"""

import os
import time
import numpy as np
import torch

from datetime import datetime
from tqdm import trange

from chrono_kge.experiment.abstract_experiment import AbstractExperiment
from chrono_kge.utils.vars.constants import FTYPE
from chrono_kge.utils.vars.defaults import Default
from chrono_kge.utils.helpers import lists_to_tuples
from chrono_kge.utils.metrics import compute_metrics, compute_ranks_hits
from chrono_kge.main.handler.global_handler import GlobalHandler
from chrono_kge.utils.logger import logger


class BaseExperiment(AbstractExperiment):

    def __init__(self,
                 global_handler: GlobalHandler
                 ) -> None:
        """"""
        super().__init__()

        self.GLOBAL = global_handler
        self.results = None
        self.pbar = None
        return

    def start(self) -> dict:
        """"""
        self.save_dict(self.to_dict(), "info")
        self.results = self.run()
        self.save_dict(self.results, "results")
        return self.results

    def update_pbar(self, epoch, metrics):
        """"""
        self.pbar.set_description(f"Epoch {epoch}")
        self.pbar.set_postfix(metrics)
        return

    def run(self) -> dict:
        """"""
        logger.info("Training started...")
        logger.info("Params: %d", sum(p.numel() for p in self.GLOBAL.model.parameters() if p.requires_grad))

        '''train and evaluate'''
        tts = []
        self.pbar = trange(1, self.GLOBAL.EXP.max_epochs + 1)

        for epoch in self.pbar:
            self.pbar.set_description(f"Epoch {epoch} (Training...)")
            start_train = time.time()

            '''mini-batch training'''
            losses = []
            for batch_triples, target_scores in self.GLOBAL.trainer.generate_train_batches():

                '''train on single batch'''
                losses.append(self.train(batch_triples, target_scores))

            '''update learning rate'''
            self.GLOBAL.scheduler.step()

            tts.append(time.time() - start_train)

            '''validation - interim'''
            if epoch % self.GLOBAL.EXP.eval_step == 0 and epoch != 0:
                metrics = self.eval(test=False)
                metrics['loss'] = float(np.mean(losses))
                self.update_pbar(epoch, metrics)

                '''save'''
                if self.GLOBAL.ENV.save:
                    torch.save(self.GLOBAL.model.state_dict(),
                               os.path.join(self.GLOBAL.ENV.path_save, "%d" + FTYPE.PT % epoch))

            '''test - interim'''
            if epoch % Default.TEST_STEP == 0 and epoch != 0:
                metrics = self.eval(test=True)
                metrics['loss'] = float(np.mean(losses))
                self.update_pbar(epoch, metrics)
                metrics['mtt'] = float(np.mean(tts))
                self.save_dict(metrics, "intermediate test results @ " + str(epoch))

        '''test - final'''
        metrics = self.eval(test=True, logging=True)
        self.update_pbar(self.GLOBAL.EXP.max_epochs, metrics)
        metrics['mtt'] = float(np.mean(tts))
        self.pbar.close()

        '''save'''
        if self.GLOBAL.ENV.save:
            torch.save(self.GLOBAL.model.state_dict(), os.path.join(self.GLOBAL.ENV.path_save, "final" + FTYPE.PT))

        return metrics

    def train(self, batch_triples, target_scores):
        """"""
        self.GLOBAL.model.train()

        '''encoder'''
        subject_scores = self.encode(batch_triples)

        '''decoder'''
        loss = self.decode(subject_scores, target_scores)

        return loss

    @torch.no_grad()
    def eval(self, test: bool = False, logging: bool = False) -> dict:
        """"""
        self.GLOBAL.model.eval()

        ranks = []
        hits = [[] for _ in range(10)]

        data = self.GLOBAL.DATA.kg.dataset.TEST_SET.triples if test else self.GLOBAL.DATA.kg.dataset.VALID_SET.triples

        for test_batch in self.GLOBAL.trainer.generate_batches_t(lists_to_tuples(data), self.GLOBAL.EXP.batch_size):

            test_batch_scores = self.encode(test_batch)

            pred_batch_targets, _ = self.GLOBAL.trainer.rank_targets(
                test_batch.detach().cpu().numpy(),
                test_batch_scores
            )

            test_batch_targets = test_batch[:, -1].detach().cpu().numpy()

            compute_ranks_hits(pred_batch_targets, test_batch_targets, ranks, hits)

        metrics = compute_metrics(hits, ranks, logging=logging)

        return metrics

    def encode(self, triples_batch):
        """"""
        return self.GLOBAL.model.forward(triples_batch)

    def decode(self, *args):
        """"""
        self.GLOBAL.optimizer.zero_grad(set_to_none=True)
        loss = self.GLOBAL.model.loss(*args)
        loss.backward()
        self.GLOBAL.optimizer.step()
        return loss.item()

    def to_dict(self) -> dict:
        """"""
        return dict(
            (key, value if not isinstance(value, dict) else list(value)[:9]) for (key, value) in self.__dict__.items())

    def save_dict(self, ddict: dict, title: str, mode='a') -> None:
        """"""
        with open(self.GLOBAL.ENV.path_save + self.GLOBAL.EXP.uid + FTYPE.TXT, mode) as f:
            f.write("================%s================\n" % title.upper())
            f.write("Datetime: %s\n" % datetime.now())
            for k, v in ddict.items():
                f.write(k + ": " + str(v) + "\n")
            f.close()
        return
