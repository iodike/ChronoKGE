"""
Handler
"""


class ModelHandler:
    """"""

    def __init__(self, args: dict):
        """"""
        self.args = args

        self.uid = str(self.args.get('model')).lower()
        self.entity_dim = self.args.get('embedding_dimension', 100)
        self.relation_dim = self.args.get('embedding_dimension', 100)
        self.time_dim = self.args.get('embedding_dimension', 100)
        self.factor_rank = self.args.get('factor_rank', 1)
        self.mod_mode = int(self.args.get('mod_mode', 0))
        self.reg_mode = int(self.args.get('reg_mode', 0))
        self.enc_mode = int(self.args.get('enc_mode', 0))
        self.dropout_input = self.args.get('input_dropout', 0.2)
        self.dropout_hidden = self.args.get('hidden_dropout', [.5, .5, .5])
        self.de_gamma = self.args.get('de_gamma', 0.5)

        return

    def setup(self):
        """"""
        return
