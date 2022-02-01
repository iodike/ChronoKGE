"""
Time-component embeddings
"""

import numpy as np
import torch
import torch.nn as nn

from collections import defaultdict

from chrono_kge.knowledge.chrono.timestamp import Timestamp
from chrono_kge.utils.vars.defaults import Default
from chrono_kge.knowledge.graph.temporal_knowledge_graph import TemporalKnowledgeGraph


class TCE(nn.Module):

    def __init__(self,
                 kg: TemporalKnowledgeGraph,
                 c_dim: int,
                 device
                 ):
        """"""
        super().__init__()

        self.device = device

        self.kg = kg

        self.c_dim = c_dim

        self.bounds = [cb[1]+1 for cb in Timestamp.random().get_components_and_bounds()]

        self.time_embeddings = [nn.Embedding(b, self.c_dim, padding_idx=0, device=self.device,
                                dtype=Default.DTYPE) for b in self.bounds]

        return

    def init(self):
        """"""
        [nn.init.xavier_normal_(emb_TC.weight.data) for emb_TC in self.time_embeddings]
        return

    def __call__(self, x):
        """"""

        t_ids = np.asarray(x.detach().cpu())

        tcs = defaultdict(list)
        tce = []

        '''create component dict'''
        for tid in t_ids:
            ts: Timestamp = self.kg.dataset.TOTAL_SET.timestamp_vocabulary[tid]
            for i, t in enumerate(ts.get_components_and_bounds(components_only=True)):
                tcs[i].append(t)

        '''granularity-based gating
        --> gran_bits=00000: G|Y|S|M|W
        '''

        '''create component embeddings'''
        for i, tc in tcs.items():
            tct = torch.IntTensor(np.asarray(tc)).to(self.device)
            tce.append(self.time_embeddings[i](tct))

        return tce
