"""
Handler
"""

import torch

from chrono_kge.utils.helpers import createUID, makeDirectory


class EnvironmentHandler:
    """"""

    def __init__(self, args: dict):
        """"""
        self.args = args

        self.cuda = torch.cuda.is_available() if self.args.get('cuda', False) else False
        self.device = None
        self.n_gpu = 0

        self.seed = self.args.get('seed', 85)
        self.save = self.args.get('save', False)
        self.path_load = self.args.get('path_load', '')
        self.path_save = self.args.get('path_save', '')

        return

    def setup(self):
        """"""
        self.setup_device()
        self.setup_save_path()
        return

    def setup_device(self):
        """"""
        self.device = torch.device("cuda" if self.cuda else "cpu")
        self.n_gpu = torch.cuda.device_count() if self.cuda else self.n_gpu
        return

    def setup_save_path(self):
        """"""

        self.path_save = "output/%s/%s/m%s_a%s_r%s_e%s/%s/%s/" % (
            str(self.args.get('command')),
            str(self.args.get('model')),
            str(self.args.get('mod_mode')),
            str(self.args.get('aug_mode')),
            str(self.args.get('reg_mode')),
            str(self.args.get('enc_mode')),
            str(self.args.get('dataset')).lower(),
            createUID()
        )
        makeDirectory(self.path_save)
        return
