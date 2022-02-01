"""
Textual descriptions
"""

import os

from collections import defaultdict

from chrono_kge.utils.vars.constants import PATH
from chrono_kge.utils.logger import logger


class TextDescriptions:

    def __init__(self, ent_file=None, qid_file=None, txt_file=None):
        """"""
        self.data_dir = 'temporal/icews14'

        if not ent_file:
            self.ent_file = os.path.join(PATH.DATA, self.data_dir, 'entity2id.txt')

        if not qid_file:
            self.qid_file = os.path.join(PATH.AUGMENT, 'wikidata5m', 'wikidata5m_entity.txt')

        if not txt_file:
            self.txt_file = os.path.join(PATH.AUGMENT, 'wikidata5m', 'wikidata5m_text.txt')

        self.i2e = defaultdict(str)
        self.e2q = defaultdict(str)
        self.q2t = defaultdict(str)
        self.i2t = defaultdict(str)
        return

    def __call__(self, *args, **kwargs) -> None:
        """"""

        '''eid -> entity'''
        logger.info("Step 1: Generate I2D...")
        self.ids2ent()

        '''entity -> qid'''
        logger.info("Step 2: Generate E2Q...")
        self.ent2qid()

        '''qid -> text'''
        logger.info("Step 3: Generate Q2T...")
        self.qid2txt()

        '''eid -> text'''
        logger.info("Step 4: Generate E2T...")
        self.ids2txt()

        '''export'''
        logger.info("Step 5: Export...")
        self.export(self.i2t, os.path.join(PATH.AUGMENT, self.data_dir, 'id2txt.txt'))

        return

    def ids2ent(self) -> None:
        """
        e-id -> entity
        """
        dl = self.import_from_file(self.ent_file)

        for key, row in enumerate(dl):
            self.i2e[row[1]] = row[0]

        return

    def ent2qid(self) -> None:
        """
        entity -> qid
        """
        dl = self.import_from_file(self.qid_file)

        for key, row in enumerate(dl):
            qid = str(row[0])
            for e in row[1:]:
                self.e2q[e] = qid
        return

    def qid2txt(self) -> None:
        """
        q-id -> txt
        """
        dl = self.import_from_file(self.txt_file)

        for key, row in enumerate(dl):
            qid = str(row[0])
            for t in row[1:]:
                self.q2t[qid] = t

        return

    def ids2txt(self) -> None:
        """
        e-id -> txt
        """
        s, f = 0, 0
        for i, e in self.i2e.items():
            q = self.e2q[e]
            t = self.q2t[q]

            if t:
                s += 1
                self.i2t[i] = t
            else:
                print(e)
                f += 1

        print("Success: %d / %d" % (s, s+f))
        print("Fail: %d / %d" % (f, s+f))
        return

    @staticmethod
    def export(xd, file_path) -> None:
        """"""
        try:
            with open(file_path, 'a') as f:
                for key, row in xd.items():
                    f.write(str(key) + '\t' + str(row) + '\n')

        except FileNotFoundError:
            logger.info("File not found: `%s`." % file_path)
            exit(1)

        return

    @staticmethod
    def import_from_file(file_path) -> list:
        """"""
        try:
            with open(file_path, 'r') as f:
                dl = [x.strip().split('\t') for x in f]

        except FileNotFoundError:
            logger.info("File not found: `%s`." % file_path)
            exit(1)

        return dl
