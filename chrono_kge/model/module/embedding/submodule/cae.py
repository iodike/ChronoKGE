"""
Cycle-aware embeddings
"""

import numpy as np
import torch
import torch.nn as nn

from collections import defaultdict

from chrono_kge.knowledge.chrono.timestamp import Timestamp
from chrono_kge.utils.vars.defaults import Default
from chrono_kge.knowledge.graph.temporal_knowledge_graph import TemporalKnowledgeGraph


class CAE(nn.Module):

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

        '''week, month, season, year'''
        self.cycles = [7, 30, 91, 365]

        self.bounds = [c+1 for c in self.cycles]

        self.time_embeddings = [nn.Embedding(b, self.c_dim, padding_idx=0, device=self.device,
                                dtype=Default.DTYPE) for b in self.bounds+self.bounds]

        return

    def init(self):
        """"""
        [nn.init.xavier_normal_(emb_TC.weight.data) for emb_TC in self.time_embeddings]
        return

    def __call__(self, x):
        """"""

        x_ids = np.asarray(x.detach().cpu())

        tcs = defaultdict(list)
        cae = []

        '''create component dict'''
        for tid in x_ids:
            for i, c in enumerate(self.cycles):
                sin, cos = Timestamp.get_cycles(tid, c, 1)
                tcs[i].append(sin)
                tcs[i + len(self.cycles)].append(cos)

        '''create component embeddings'''
        for i, tc in tcs.items():
            tct = torch.IntTensor(np.asarray(tc)).to(self.device)
            cae.append(self.time_embeddings[i](tct))

        return cae
