"""
Base knowledge graph
"""

import os
import pandas as pd

from collections import defaultdict
from tqdm import tqdm

from chrono_kge.knowledge.graph.abstract_knowledge_graph import AbstractKnowledgeGraph
from chrono_kge.knowledge.augmentation.bt_augmentation import BT_Augmentation
from chrono_kge.knowledge.dataset import Dataset
from chrono_kge.knowledge.knowledge_base import KnowledgeBases, KnowledgeBase
from chrono_kge.utils.vars.constants import FTYPE, PATH
from chrono_kge.utils.vars.modes import AUG
from chrono_kge.utils.helpers import makeDirectory, strip
from chrono_kge.utils.logger import logger


class BaseKnowledgeGraph(AbstractKnowledgeGraph):

    def __init__(self,
                 kb: KnowledgeBase,
                 **kwargs
                 ) -> None:
        """A knowledge graph.
        """

        '''Knowledge base'''
        self.kb = kb

        '''Files'''
        self.entity_file = 'entity2id'
        self.relation_file = 'relation2id'
        self.triple_file = 'triple2id'

        '''Entity sets'''
        self.entity_dict = {}
        self.entity_key_dict = {}

        '''Relation sets'''
        self.relation_dict = {}
        self.relation_key_dict = {}

        '''Numbers'''
        self.n_entity = 0
        self.n_relation = 0

        '''Dataset'''
        self.dataset = None

        return

    def setup(self) -> None:
        """"""

        '''Create data'''
        if AUG.isBTC(self.kb.aug_mode):
            self.create_dicts()
            exit(0)

        '''Load dicts'''
        self.load_dicts()

        '''Process triples'''
        self.load_triples()

        return

    def create_dicts(self) -> None:
        """"""
        if not KnowledgeBases.exists_augmented(self.kb):
            logger.error("KB `%s` not supported for augmentation." % self.kb.name)
            exit(1)

        makeDirectory(self.kb.path)

        self.create_dict(self.entity_file)
        self.create_dict(self.relation_file)
        return

    def create_dict(self, filename: str) -> None:
        """"""
        df = self.read_file(filename, self.kb.path)
        ddict = dict(zip(df[0], df[1]))

        bta = BT_Augmentation()
        _ = bta(ddict, value_first=True, n_values=2, confidence=0,
                filepath=os.path.join(PATH.AUGMENT, self.kb.name, filename))
        return

    def load_dicts(self) -> None:
        """
        Dict: Entity/Relation -> ID
        Key-Dict: Entity (original) -> Entity (translated)
        """

        if KnowledgeBases.has_indices(self.kb) or KnowledgeBases.has_labels(self.kb):
            self.entity_dict, self.entity_key_dict, self.n_entity = self.load_dict(self.entity_file, False)
            self.relation_dict, self.relation_key_dict, self.n_relation = self.load_dict(
                self.relation_file, AUG.isRev(self.kb.aug_mode))
        else:
            logger.info('Building index...')
            self.entity_dict, self.relation_dict = self.create_indices()
            self.n_entity, self.n_relation = len(self.entity_dict), len(self.relation_dict)
            self.n_relation *= 2 if AUG.isRev(self.kb.aug_mode) else 1

        logger.info('#Entities: %d | #Relations: %d' % (self.n_entity, self.n_relation))
        return

    def load_dict(self, filename: str, reverse: bool) -> (dict, dict, int):
        """"""
        dnum = 0
        kdict = defaultdict()

        if AUG.isBT(self.kb.aug_mode):

            akf: pd.DataFrame = self.read_file(filename + '_keys', self.kb.path)
            for key, row in akf.iterrows():
                kdict[row[0]].append(row[1])
                dnum += 1

        df = self.read_file(filename, self.kb.path)
        ddict = {strip(row[0]): row[1] for (i, row) in df.iterrows()}

        dnum += len(ddict)
        dnum *= 2 if reverse else 1

        return ddict, kdict, dnum

    def load_triples(self):
        """Load KG triples.
        """
        self.dataset = Dataset()

        for set_name, subset in self.dataset.ALL_SETS.items():

            df: pd.DataFrame = subset.read(self.kb.path)

            pbar = tqdm(df.iterrows(), total=len(df.index))
            pbar.set_description('Loading %s triples...' % subset.name)

            for key, row in pbar:

                # standard dataset (not augmented)
                for triple in self.process_row(row, mode=AUG.NONE):
                    self.dataset.append(set_name, triple)

                # training set (augmented)
                for triple in self.process_row(row, mode=self.kb.aug_mode):
                    self.dataset.append(set_name, triple)

        return

    def process_row(self, row, mode: int):
        """"""
        if KnowledgeBases.has_indices(self.kb):
            sid, pid, oid = int(row[0]), int(row[1]), int(row[2])
        else:
            sid, pid, oid = self.entity_dict[strip(row[0])], \
                            self.relation_dict[strip(row[1])], \
                            self.entity_dict[strip(row[2])]

        if not self.check_ids(sid, pid, oid):
            exit(1)

        yield from self.create_triples([sid, pid, oid], mode=mode)

    def create_triples(self, ids, mode: int):
        """"""
        # normal
        if AUG.isNone(mode):
            yield [ids[0], ids[1], ids[2]]

        # normal/rev
        if AUG.isRev(mode):
            yield [ids[2], ids[1] + self.n_relation // 2, ids[0]]

        # bt
        if AUG.isBT(mode):
            for asid in self.entity_key_dict[ids[0]]:
                for aoid in self.entity_key_dict[ids[2]]:
                    for apid in self.relation_key_dict[ids[1]]:
                        yield [asid, apid, aoid]

                        # bt/rev
                        if AUG.isRev(mode):
                            yield [aoid, apid + self.n_relation // 2, asid]

    def read_file(self, filename: str, file_path: str = '') -> pd.DataFrame:
        """"""
        table = None

        if not file_path:
            file_path = self.kb.path

        try:
            table = pd.read_table(os.path.join(file_path, filename + FTYPE.TXT), header=None, encoding='utf-8')
        except FileNotFoundError:
            logger.info("The dataset `%s` on path `%s` is missing. Please add or create a new dataset."
                        % (filename, file_path))
            exit(1)

        return table

    def check_ids(self, sid, pid, oid) -> bool:
        """"""
        if sid < 0 or sid >= self.n_entity:
            logger.error("Bad subject index: %d" % sid)
            logger.error("Num. entities: %d" % self.n_entity)
            return False

        if pid < 0 or pid >= self.n_relation:
            logger.error("Bad relation index: %d" % pid)
            logger.error("Num. relations: %d" % self.n_relation)
            return False

        if oid < 0 or oid >= self.n_entity:
            logger.error("Bad object index: %d" % oid)
            logger.error("Num. entities: %d" % self.n_entity)
            return False

        return True

    def create_indices(self) -> (dict, dict):
        """"""
        eset, rset = set(), set()
        for set_name, subset in self.dataset.ALL_SETS.items():
            df: pd.DataFrame = subset.read(self.kb.path)

            for key, row in df.iterrows():
                eset.add(row[0])
                rset.add(row[1])
                eset.add(row[2])

        eids = {e: i for (i, e) in enumerate(sorted(eset))}
        rids = {e: i for (i, e) in enumerate(sorted(rset))}

        return eids, rids
