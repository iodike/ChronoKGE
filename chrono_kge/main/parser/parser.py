"""
User Input Parser
"""

from argparse import ArgumentParser

from chrono_kge.utils.vars.defaults import Default
from chrono_kge.knowledge.knowledge_base import KnowledgeBases


class UIParser:
    """"""

    def __init__(self):
        """"""
        super(UIParser, self).__init__()

        self.parser = ArgumentParser()
        self.sub_parser = self.parser.add_subparsers(dest='command')
        self.run_parser = self.sub_parser.add_parser('run')
        self.tune_parser = self.sub_parser.add_parser('tune')
        self.stats_parser = self.sub_parser.add_parser('stats')

        self.add_arguments()
        return

    def add_arguments(self):
        """"""

        '''Main arguments'''

        self.parser.add_argument('-m', '--model', type=str, default=Default.MODEL,
                          help='Learning model.')

        self.parser.add_argument('-d', '--dataset', type=str, default=Default.KB.name,
                          help="Which dataset to use: ["
                          + "|".join(kb.name for kb in KnowledgeBases.ALL_KB)
                          + "].")

        self.parser.add_argument('-e', '--epochs', type=int, default=Default.MAX_EPOCHS,
                          help='Number of total epochs.')

        self.parser.add_argument('-c', '--cuda', action='store_true',
                          help="Whether to use cuda (GPU) or not (CPU).")

        self.parser.add_argument('-tt', '--train_type', type=str, default='ova',
                          help="Training type.")

        self.parser.add_argument('-tp', '--train_param', type=str, nargs='+', default=[],
                          help="Training params.")

        self.parser.add_argument('-am', '--aug_mode', type=int, default=Default.AUG_MODE,
                          help='Augmentation mode.')

        self.parser.add_argument('-rm', '--reg_mode', type=int, default=Default.REG_MODE,
                          help='Regularization mode.')

        self.parser.add_argument('-mm', '--mod_mode', type=int, default=Default.MOD_MODE,
                          help="Modulation mode.")

        self.parser.add_argument('-em', '--enc_mode', type=int, default=Default.ENC_MODE,
                          help="Encoding mode.")

        self.parser.add_argument('-s', '--save', action="store_true",
                          help='Whether or not to save results.')

        '''Sub arguments'''

        self.add_arguments_run()
        self.add_arguments_tune()
        self.add_arguments_stats()

        return

    def add_arguments_run(self):
        """"""
        self.run_parser.add_argument('-bs', '--batch_size', type=int, default=Default.BATCH_SIZE,
                                     help='Batch size.')

        self.run_parser.add_argument('-lr', '--learning_rate', type=float, default=Default.LEARNING_RATE,
                                     help="Learning rate.")

        self.run_parser.add_argument('-dr', '--decay_rate', type=float, default=Default.DECAY_RATE,
                                     help='Decay rate.')

        self.run_parser.add_argument('-ed', '--embedding_dimension', type=int, default=Default.EMBEDDING_DIMENSION,
                                     help='Embedding dimensionality, i.e., number of embedding features.')

        self.run_parser.add_argument('-tg', '--time_gran', type=float, default=Default.TIME_GRAN,
                                     help='Time unit (in days) for temporal datasets.')

        self.run_parser.add_argument('-fr', '--factor_rank', type=int, default=Default.FACTOR_RANK,
                                     help='Latent dimension of MFB.')

        self.run_parser.add_argument('-id', '--input_dropout', type=float, default=Default.DROPOUT_INPUT,
                                     help='Input layer dropout.')

        self.run_parser.add_argument('-hd', '--hidden_dropout', type=list, default=[Default.DROPOUT_HIDDEN]*3,
                                     help='Dropout after the first hidden layer.')

        self.run_parser.add_argument('-ls', '--label_smoothing', type=float, default=Default.LABEL_SMOOTHING,
                                     help='Amount of label smoothing.')

        self.run_parser.add_argument('-l1', '--reg_l1', type=float, default=Default.L1,
                                     help='L1 regularization (Lasso).')

        self.run_parser.add_argument('-l2', '--reg_l2', type=float, default=Default.L2,
                                     help='L2 regularization (Ridge).')

        self.run_parser.add_argument('-re', '--reg_emb', type=float, default=Default.REG_EMB,
                                     help='Embedding regularization (Omega).')

        self.run_parser.add_argument('-rt', '--reg_time', type=float, default=Default.REG_TIME,
                                     help='Time regularization (Lambda).')
        return

    def add_arguments_tune(self):
        """"""

        self.tune_parser.add_argument('-nt', '--num_trials', type=int, default=Default.TUNE_TRIALS,
                                      help='Number of tuning trials.')

        self.tune_parser.add_argument('-to', '--timeout', type=int, default=Default.TUNE_HOURS,
                                      help='Tuner timeout in hrs.')

        self.tune_parser.add_argument('-r', '--reload', action='store_true',
                                      help='Reload previous optimization state (if exists).')
        return

    def add_arguments_stats(self):
        """"""

        self.stats_parser.add_argument('-f', '--file', type=str,
                                       help='Datafile to plot.')

        self.stats_parser.add_argument('-d', '--description', type=str,
                                       help='Datafile to plot.')

        self.stats_parser.add_argument('-xl', '--xlabel', type=str,
                                       help='Label of x-axis.')

        self.stats_parser.add_argument('-yl', '--ylabel', type=str,
                                       help='Label of x-axis.')

        self.stats_parser.add_argument('-xs', '--xscale', type=str, default=Default.PLOT_XSCALE,
                                       help='Scale of x-axis.')

        self.stats_parser.add_argument('-ys', '--yscale', type=str, default=Default.PLOT_YSCALE,
                                       help='Scale of x-axis.')
        return
