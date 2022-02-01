"""
Synthetic temporal knowledge graph
"""

import math
import random
import re
import numpy as np
import pandas as pd

from collections import defaultdict
from datetime import date

from chrono_kge.knowledge.graph.temporal_knowledge_graph import TemporalKnowledgeGraph
from chrono_kge.knowledge.knowledge_base import KnowledgeBase
from chrono_kge.knowledge.chrono.timestamp import Timestamp
from chrono_kge.utils.vars.constants import REGEX
from chrono_kge.utils.vars.defaults import Default
from chrono_kge.utils.helpers import split_sets, sort_by_sub
from chrono_kge.utils.logger import logger


class SyntheticKnowledgeGraph(TemporalKnowledgeGraph):

    def __init__(self,
                 kb: KnowledgeBase,
                 generate: bool = True
                 ):
        """"""
        super().__init__(kb)

        self.domain_dict_file = 'domain2id.txt'
        self.ontology_file = 'ontology.txt'
        self.correlation_file = 'correlations.txt'
        self.attribute_dict_file = 'relation2attributes.txt'

        self.generate = generate
        self.attributes = defaultdict(list)
        self.correlations = []
        return

    def __call__(self,
                 t_gran: float = Default.TIME_GRAN,
                 **kwargs
                 ) -> None:
        """"""
        super().__call__(t_gran)

        self.load_dicts()

        # generate dataset
        if self.generate:
            self.generate_data()

        return

    def generate_data(self):
        """Generates datasets.
        Called by init.
        """

        logger.info('-----Generate datasets-----')

        self.load_attributes()

        self.load_correlations()

        triples = self.generate_triples()

        train, valid, test = split_sets(triples)

        self.dataset.TRAIN_SET.triples = train
        self.dataset.VALID_SET.triples = valid
        self.dataset.TEST_SET.triples = test

        self.dataset.write(self.kb.path)

        return

    def load_attributes(self):
        """Loads attributes.

        0: probability exists
        1: probability changes (1-year)
        2: temporal priority (0=highest)
        """

        logger.info('-----Loading attributes-----')

        attributes = defaultdict(list)

        df: pd.DataFrame = self.read_file(self.attribute_dict_file)

        for key, row in df.iterrows():
            for attribute in row[1:]:
                attributes[int(row[0])].append(attribute)

        self.attributes = attributes
        return

    def load_correlations(self):
        """Loads correlations.
        """

        logger.info('-----Loading correlations-----')

        correlations = [[0.0 for _ in range(self.n_relation)] for _ in range(self.n_relation)]

        df: pd.DataFrame = self.read_file(self.correlation_file)

        for key, row in df.iterrows():
            correlations[int(row[0])][int(row[1])] = float(row[2])

        self.correlations = correlations
        return

    def load_ontology(self):
        """Loads ontology.
        Triple format: (entity, relation, entity)

        :return: Ontology sorted by relation priority.
        """

        logger.info('-----Loading ontology-----')

        ontology = []

        df: pd.DataFrame = self.read_file(self.ontology_file)

        for key, triple in df.iterrows():
            rid = int(triple[1])
            prio = self.attributes[rid][2]
            triple.append(prio)
            ontology.append(triple)

        ontology = sort_by_sub(ontology, len(ontology[0])-1)

        for triple in ontology:
            triple.pop()

        return ontology

    def load_domain_dict(self):
        """Loads domains.
        """

        logger.info('-----Loading domains-----')

        df = self.read_file(self.domain_dict_file)
        data_dict = dict(zip(df[0], df[1]))

        return data_dict

    def get_domain2entities(self):
        """Constructs dictionary of domain / entity-list - pairs.
        """

        domain2entities = defaultdict(list)

        domain_dict = self.load_domain_dict()

        for entity_id, entity_name in enumerate(self.entity_dict):

            match = re.match(REGEX.DOM_ENT, entity_name)

            if match:
                domain = str(match.groups()[0])
                domain2entities[domain_dict[domain]].append(self.entity_dict[entity_name])

        return domain2entities

    def generate_triples(self):
        """Generate triples for datasets.
        """

        logger.info('-----Generate triples-----')

        triples = []

        sorted_ontology = self.load_ontology()

        d2e = self.get_domain2entities()

        # get date
        for year in range(self.kb.start_date.timetuple().tm_year,
                          self.kb.start_date.timetuple().tm_year + math.floor(self.kb.num_days/365)):

            last_prio = 0
            latest_dates = [date(year, 1, 1) for _ in range(self.n_entity)]

            # get triple
            for triple in sorted_ontology:

                subject_ids = d2e[triple[0]]
                relation_id = triple[1]
                object_ids = d2e[triple[2]]

                # get stats
                ep, cp, curr_prio = self.attributes[relation_id]

                for sid in subject_ids:

                    for oid in object_ids:

                        # check probability
                        r_count = self.count_triples(triples, [sid, relation_id, oid])
                        if self.is_addable(ep, cp, r_count):

                            # check priority
                            if last_prio < curr_prio:
                                start_date = latest_dates[sid]
                            else:
                                start_date = date(year, 1, 1)

                            end_date = date(year+1, 1, 1)

                            # set time
                            tid = Timestamp.random_days(self.kb.start_date, start_date, end_date)
                            triples.append([sid, relation_id, oid, tid])

                            # set last
                            latest_dates[sid] = max(latest_dates[sid], date.fromtimestamp(tid))
                            last_prio = curr_prio

        np.random.shuffle(triples)

        return triples

    def process_row(self, triple, reverse) -> list:
        """"""
        sid, pid, oid = int(triple[0]), int(triple[1]), int(triple[2])
        tid = int(triple[3] * self.kb.gran / self.t_gran)

        if reverse:
            return [oid, pid + self.n_relation // 2, sid, tid]
        else:
            return [sid, pid, oid, tid]

    @staticmethod
    def count_triples(data: list, triple: list):
        """Count occurrence of given triple in dataset.
        """
        c = 0
        for d in data:
            if triple in d[:-1]:
                c += 1
        return c

    @staticmethod
    def is_addable(ep: float, cp: float, count_metric: int):
        """Checks if triple can be added.
        """
        rand = random.randint(0, 100)

        # exist probability
        if count_metric == 0:
            if rand in (0, int(100 * ep)):
                return True

        # change probability
        else:
            if rand in (0, int(100 * cp)):
                return True
        return False
