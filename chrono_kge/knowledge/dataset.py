"""
Dataset
"""

import os
import pandas as pd

from collections import defaultdict

from chrono_kge.knowledge.chrono.timestamp import Timestamp
from chrono_kge.utils.helpers import makeDirectory
from chrono_kge.utils.vars.constants import FTYPE, SETNAME


class Subset:
    """
    A subset.
    """

    def __init__(self, name, short) -> None:
        """"""
        self.name = name
        self.short = short
        self.file = short+FTYPE.TXT
        self.triples = []
        self.target_vocabulary = defaultdict(list)
        self.timestamp_vocabulary = defaultdict()
        self.n_triples = 0
        return

    def __len__(self):
        return self.n_triples

    def append(self, triple: list) -> None:
        """"""
        self.triples.append(triple)
        self.target_vocabulary[tuple(triple[:-1])].append(triple[-1])
        self.n_triples += 1
        return

    def append_ts(self, tid: int, ts: Timestamp):
        """"""
        self.timestamp_vocabulary[tid] = ts
        return

    def read(self, data_dir='./') -> pd.DataFrame:
        """"""
        return pd.read_table(os.path.join(data_dir, self.file), header=None, encoding='utf-8')

    def write(self, data_dir='./') -> None:
        """"""
        makeDirectory(data_dir)
        df = pd.DataFrame(self.triples)
        df.to_csv(os.path.join(data_dir, self.file), header=False, index=False, sep='\t', mode='a')
        return


class Dataset:
    """
    Full dataset containing train, validation and test set.
    """

    def __init__(self):
        """"""
        self.TOTAL_SET: Subset = Subset('total', SETNAME.TOTAL)
        self.TRAIN_SET: Subset = Subset('training', SETNAME.TRAIN)
        self.VALID_SET: Subset = Subset('validation', SETNAME.VALID)
        self.TEST_SET: Subset = Subset('testing', SETNAME.TEST)

        self.ALL_SETS: dict = {
            SETNAME.TRAIN: self.TRAIN_SET,
            SETNAME.VALID: self.VALID_SET,
            SETNAME.TEST: self.TEST_SET
        }
        return

    def __len__(self):
        return len(self.TOTAL_SET)

    def append(self, set_name: str, triple: list):
        """"""
        subset: Subset = self.ALL_SETS.get(set_name)
        subset.append(triple)
        self.TOTAL_SET.append(triple)
        return

    def append_ts(self, set_name: str, tid: int, ts: Timestamp):
        """"""
        subset: Subset = self.ALL_SETS.get(set_name)
        subset.append_ts(tid, ts)
        self.TOTAL_SET.append_ts(tid, ts)
        return

    def read(self, data_dir):
        """"""
        for subset in self.ALL_SETS.values():
            yield subset.read(data_dir)

    def write(self, data_dir) -> None:
        """"""
        for subset in self.ALL_SETS.values():
            subset.write(data_dir)
        return
