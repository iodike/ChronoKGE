"""
TKG Embedding
"""
import torch
from torch import nn

from chrono_kge.model.module.embedding.tkge import TKGE
from chrono_kge.main.handler.model_handler import ModelHandler
from chrono_kge.main.handler.data_handler import DataHandler
from chrono_kge.main.handler.env_handler import EnvironmentHandler


class TKGE_ComplEx(TKGE):

    def __init__(self,
                 model_handler: ModelHandler,
                 data_handler: DataHandler,
                 env_handler: EnvironmentHandler,
                 **kwargs
                 ) -> None:
        """"""
        super().__init__(model_handler, data_handler, env_handler, **kwargs)

        self.emb_T = nn.Embedding(self.DATA.kg.n_time, self.MODEL.time_dim,
                                  padding_idx=0, device=self.ENV.device)
        self.emb_R2 = nn.Embedding(self.DATA.kg.n_relation, self.MODEL.relation_dim,
                                   padding_idx=0, device=self.ENV.device)

        return

    def __call__(self, x):
        """"""
        es = self.emb_E(x[:, 0])
        er_t = self.emb_R1(x[:, 1])
        er_nt = self.emb_R2(x[:, 1])
        et = self.emb_T(x[:, 2])
        eo = self.emb_E(x[:, 3])

        '''normalization + dropout'''
        es = self.drop_in(self.norm_in(es))

        '''complex'''
        es_re, es_im = torch.chunk(es, 2, dim=2)
        er_t_re, er_t_im = torch.chunk(er_t, 2, dim=2)
        er_nt_re, er_nt_im = torch.chunk(er_nt, 2, dim=2)
        et_re, et_im = torch.chunk(et, 2, dim=2)
        eo_re, eo_im = torch.chunk(eo, 2, dim=2)

        '''modulate'''
        er_re = self.modulate(er_t_re, er_nt_re, et_re)
        er_im = self.modulate(er_t_im, er_nt_im, et_im)

        return es_re, es_im, er_re, er_im, et_re, et_im, eo_re, eo_im
