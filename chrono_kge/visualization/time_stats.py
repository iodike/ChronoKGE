"""
Time Stats
"""

import numpy as np

from chrono_kge.knowledge.graph.temporal_knowledge_graph import TemporalKnowledgeGraph
from chrono_kge.knowledge.chrono.timestamp import Timestamp


class TimeStats:

    def __init__(self, kg: TemporalKnowledgeGraph):
        """"""
        self.kg = kg
        return

    def get_time_stamps(self):
        """"""
        times = []

        for tid in np.asarray(self.kg.dataset.TOTAL_SET.triples)[:, 2]:
            ts: Timestamp = self.kg.dataset.TOTAL_SET.timestamp_vocabulary[tid]
            times.append(ts)
        return times

    def get_time_stats(self):
        """"""
        times = self.get_time_stamps()

        tcaa = []
        for i in range(15):
            tca = []
            for j in range(5000):
                tca.append(0)
            tcaa.append(tca)

        for ts in times:
            tcs = ts.get_components_and_bounds(components_only=True)
            for i, tc in enumerate(tcs):
                tcaa[i][tc] += 1

        for tid in np.asarray(self.kg.dataset.TOTAL_SET.triples)[:, 2]:
            tcaa[len(times[0].get_components_and_bounds(components_only=True))][tid] += 1

        for tca in tcaa:
            print(tca)
            print("="*20)

        return
