"""
Handler
"""

from chrono_kge.main.handler.env_handler import EnvironmentHandler
from chrono_kge.utils.vars.defaults import Default


class ExperimentHandler:
    """"""

    def __init__(self, args: dict):
        """"""
        self.args = args

        self.uid = Default.EXPERIMENT
        self.eval_step = Default.EVAL_STEP
        self.max_epochs = self.args.get('epochs', 1000)
        self.batch_size = self.args.get('batch_size', 1000)
        self.learning_rate = self.args.get('learning_rate', 1e-2)
        self.decay_rate = self.args.get('decay_rate', 0.99)
        self.label_smoothing = self.args.get('label_smoothing', 1e-2)
        self.l1 = self.args.get('reg_l1', 0.0)
        self.l2 = self.args.get('reg_l1', 0.0)

        self.reg_emb = float(self.args.get('reg_emb', 0.0))
        self.reg_time = float(self.args.get('reg_time', 0.0))

        self.n_samples = 0

        self.train_type = self.args.get('train_type')
        self.train_args = {str(i): v for i, v in enumerate(self.args.get('train_param', []))}

        self.trial = None

        return

    def setup(self, env: EnvironmentHandler):
        """"""
        self.batch_size = self.batch_size * env.n_gpu if env.n_gpu > 1 else self.batch_size
        self.setup_ns()
        return

    def setup_ns(self):
        """"""
        if self.train_type == 'ns':
            self.n_samples = self.train_args.get('0')
        return
