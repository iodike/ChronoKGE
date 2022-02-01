"""
Time Series
"""

import numpy as np
import pandas as pd

from collections import defaultdict
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


PERIOD = 31


class TimeSeries:

    def __init__(self):
        """"""
        return

    @staticmethod
    def decompose(kg):
        """
        rid -> { date -> (t, s, r) }
        """

        # list: [rid][tid] = v
        r_tv = TimeSeries.get_histogram(kg)

        ts = defaultdict(defaultdict)

        # list:
        for r, tv in r_tv:
            vd = [[v, kg.dataset.TOTAL_SET.timestamp_vocabulary[t].dt] for t, v in tv]

            # dict of 'date -> (t, s, r)'
            ts[r] = TimeSeries.time_series(vd, period=PERIOD, plot=False)

        return ts

    @staticmethod
    def get_histogram(kg):
        """
        list of TID, RID -> (# rids@tid) / (# rids)
        """

        '''triples + sorted over time'''
        triples = np.asarray(kg.dataset.TOTAL_SET.triples)
        triples = triples[triples[:, 2].argsort()]

        rc = defaultdict(int)
        hgd = defaultdict(defaultdict)
        hgl = defaultdict(list)

        '''init'''
        for _, rid, tid, _ in triples:
            rc[rid] = 0
            hgd[rid][tid] = 0

        '''frequency'''
        for _, rid, tid, _ in triples:
            rc[rid] += 1
            hgd[rid][tid] += 1

        '''weight'''
        for rid, td in hgd.items():
            for tid, c in td.items():
                hgd[rid][tid] = c / rc[rid]

        '''to 2D-list'''
        for rid, td in hgd.items():
            hgl[rid] = list(td.items())

        return list(hgl.items())

    @staticmethod
    def time_series(vd_ids, period, plot=False):
        """
        input: list of (value, date)-pairs
        output: dict of 'date -> (t, s, r)'
        """

        df: pd.Dataframe = pd.DataFrame(data=vd_ids, columns=['value', 'date'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        if len(df) < 2 * period:
            return {}

        '''STL-decomposition using Loess'''
        sd = seasonal_decompose(df, model='additive', period=period)

        if plot:
            sd.plot()
            plt.show()

        ds_dict = defaultdict()
        for i, v in enumerate(sd.observed):
            ds_dict[vd_ids[i][1]] = (np.asarray(sd.trend)[i], np.asarray(sd.seasonal)[i], np.asarray(sd.resid)[i])

        return ds_dict
