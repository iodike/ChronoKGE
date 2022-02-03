"""
Handler
"""

from chrono_kge.knowledge.knowledge_base import KnowledgeBases
from chrono_kge.knowledge.graph.base_knowledge_graph import BaseKnowledgeGraph
from chrono_kge.knowledge.graph.synthetic_knowledge_graph import SyntheticKnowledgeGraph
from chrono_kge.knowledge.graph.temporal_knowledge_graph import TemporalKnowledgeGraph

from chrono_kge.utils.logger import logger


class DataHandler:
    """"""

    def __init__(self, args: dict):
        """"""
        self.args = args

        self.uid = str(self.args.get('dataset')).lower()
        self.aug_mode = self.args.get('aug_mode', 0)

        self.time_gran = self.args.get('time_gran', 1.0)

        self.kb = None
        self.kg = None

        '''Files'''
        self.entity_file = 'entity2id'
        self.relation_file = 'relation2id'
        self.triple_file = 'triple2id'

        return

    def setup(self):
        """"""
        self.load_kb()
        self.load_kg()
        return

    def load_kb(self):
        """"""
        self.kb = KnowledgeBases.get_kb_by_name(self.uid)
        self.kb.setup(self.aug_mode)
        return

    def load_kg(self) -> None:
        """"""
        if not self.kb:
            self.load_kb()

        if self.kb.is_static():
            self.kg = BaseKnowledgeGraph(kb=self.kb)

        elif self.kb.is_temporal():
            self.kg = TemporalKnowledgeGraph(kb=self.kb, t_gran=self.time_gran)

        elif self.kb.is_synthetic():
            self.kg = SyntheticKnowledgeGraph(kb=self.kb)

        else:
            logger.error("KG with name `%s` not known." % self.kb.name)
            exit(1)

        self.kg.setup()

        return
