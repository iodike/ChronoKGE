"""
Temporal knowledge graph
"""

from chrono_kge.knowledge.graph.base_knowledge_graph import BaseKnowledgeGraph
from chrono_kge.knowledge.knowledge_base import KnowledgeBase, KnowledgeBases
from chrono_kge.knowledge.chrono.timestamp import Timestamp
from chrono_kge.utils.vars.defaults import Default
from chrono_kge.utils.logger import logger
from chrono_kge.utils.helpers import strip


class TemporalKnowledgeGraph(BaseKnowledgeGraph):

    def __init__(self,
                 kb: KnowledgeBase,
                 t_gran: float = Default.TIME_GRAN
                 ) -> None:
        """A temporal knowledge graph.
        Triples in (s, p, t, o) - format.
        """
        super().__init__(kb)

        '''Time granularity'''
        self.t_gran = t_gran

        '''Sample time'''
        self.n_time = self.kb.get_n_time(self.t_gran)

        logger.info("Loading time @sampling-rate=%d (total=%d units)." % (self.t_gran, self.n_time - 1))
        return

    def process_row(self, row, mode: int):
        """"""
        if KnowledgeBases.has_indices(self.kb):
            sid, pid, oid = int(row[0]), int(row[1]), int(row[2])
        else:
            sid, pid, oid = self.entity_dict[strip(row[0])], \
                            self.relation_dict[strip(row[1])], \
                            self.entity_dict[strip(row[2])]

        if KnowledgeBases.has_timestamps(self.kb):
            ts = Timestamp(str(row[3]), self.kb)
            tid = int(ts.get_abs_time(self.kb.start_date) / (self.kb.gran * self.t_gran))
            self.dataset.TOTAL_SET.append_ts(tid, ts)

        else:
            tid = int(int(row[3]) / self.t_gran)  # kb granularity already included in time index
            self.dataset.TOTAL_SET.append_ts(tid, Timestamp.from_id(tid, self.kb))

        if not self.check_ids(sid, pid, oid, *[tid]):
            exit(1)

        yield from self.create_triples([sid, pid, oid, tid], mode=mode)

    def create_triples(self, ids, mode: int):
        """"""
        for triple in super().create_triples([ids[0], ids[1], ids[2]], mode=mode):
            triple.insert(2, ids[3])
            yield triple

    def check_ids(self, sid, pid, oid, *args) -> bool:
        """"""
        tid = args[0]

        if not super().check_ids(sid, pid, oid):
            return False

        if tid < 0 or tid >= self.n_time:
            logger.error("Bad time index: %d" % tid)
            ts = self.dataset.TOTAL_SET.timestamp_vocabulary[tid]
            logger.error("Timestamp: %s" % str(ts))
            logger.error("Num. time: %d" % self.n_time)
            return False

        return True
