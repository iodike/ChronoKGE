"""
TKG Embedding
"""

import numpy as np
import torch

from torch import nn

from chrono_kge.model.module.embedding.tkge import TKGE
from chrono_kge.knowledge.chrono.timeseries import TimeSeries
from chrono_kge.main.handler.model_handler import ModelHandler
from chrono_kge.main.handler.data_handler import DataHandler
from chrono_kge.main.handler.env_handler import EnvironmentHandler


class TKGE_Series(TKGE):

    def __init__(self,
                 model_handler: ModelHandler,
                 data_handler: DataHandler,
                 env_handler: EnvironmentHandler,
                 **kwargs
                 ) -> None:
        """"""
        super().__init__(model_handler, data_handler, env_handler, **kwargs)

        self.f_dim = 100
        self.time_embeddings = [nn.Embedding(self.f_dim, self.MODEL.time_dim, padding_idx=0, device=self.ENV.device)
                                for _ in range(3)]

        self.time_series = TimeSeries.decompose(self.DATA.kg)

        return

    def init(self) -> None:
        """Initialise embeddings using normal distribution.
        """
        super().init()
        [nn.init.xavier_normal_(emb_TC.weight.data) for emb_TC in self.time_embeddings]
        return

    def __call__(self, x):
        """"""
        es = self.emb_E(x[:, 0])
        er_t = self.emb_R1(x[:, 1])
        er_nt = self.emb_R2(x[:, 1])
        et = self.emb_T(x[:, 2])
        eo = self.emb_E(x[:, 3])

        et_s = self.decompose(np.asarray(x[:, 1].detach().cpu()), np.asarray(x[:, 2].detach().cpu()))

        '''normalization + dropout'''
        es = self.drop_in(self.norm_in(es))

        return es, er_t, er_nt, et, et_s, eo

    def decompose(self, r_ids, t_ids):
        """"""

        bs = len(r_ids)
        tcs = np.zeros((3, bs))

        for i in range(bs):

            '''ids'''
            rid = r_ids[i]
            tid = t_ids[i]

            '''date -> (t, s, r)'''
            dvs = self.time_series[rid]

            '''time series dict might be empty -> too few timestamps for decomposition'''
            if not dvs:
                tcs[:, i] = 0.
                continue

            '''date'''
            d = self.DATA.kg.dataset.TOTAL_SET.timestamp_vocabulary[tid].dt

            '''timestamp might not exist for this relation'''
            if d not in dvs.keys():
                tcs[:, i] = 0.
                continue

            '''collect T,S,R components'''
            for j, v in enumerate(list(dvs[d])):

                '''fix NAN values'''
                v = 0. if np.isnan(v) else v
                tcs[j, i] = v

        '''arrays'''
        csa = [np.asarray(tc) for tc in tcs]

        '''fix negative values'''
        csa = [np.interp(cs, (np.min(cs), np.max(cs)), (0, self.f_dim-1)).astype(int) for cs in csa]

        '''tensors'''
        csa = [torch.IntTensor(cs).to(self.ENV.device) for cs in csa]

        '''embeddings'''
        etcs = [self.emb_TC[i](cs) for i, cs in enumerate(csa)]

        '''sum'''
        er_tc = 0
        for etc in etcs:
            er_tc += etc

        return er_tc
