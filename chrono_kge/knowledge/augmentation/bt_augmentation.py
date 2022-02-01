"""
Back Translation
"""

import threading

from queue import Queue

from chrono_kge.utils.logger import logger
from chrono_kge.utils.vars.constants import FTYPE
from chrono_kge.utils.helpers import isPartOf
from chrono_kge.utils.web.google.google_translate import Translator, BackTranslator


class BT_Augmentation:

    def __init__(self, sl: str = 'en', tls: list = None):
        """"""
        self.translator = Translator(timeout=1)
        self.sl = sl
        self.tls = tls if tls else ['ch-zn', 'es', 'de', 'en']
        return

    def __call__(self,
                 kv_dict: dict,
                 value_first: bool = False,
                 n_values: int = 1,
                 confidence: int = 0,
                 filepath: str = ''
                 ) -> (dict, dict):
        """"""

        '''init'''

        bt_dict: dict = kv_dict.copy()
        bt_size: int = len(kv_dict)

        key_dict: dict = {}
        for i in range(bt_size):
            key_dict[i] = []

        bts = list()
        for i in range(len(self.tls)):
            bt = BackTranslator(sl=self.sl, n_values=n_values, timeout=2)
            bts.append(bt)

        '''translate'''

        i = 0
        for key, value in kv_dict.items():

            '''argument order'''

            v = str(key) if value_first else str(value)
            k = int(value) if value_first else int(key)

            logger.info("Translating: %s." % v)

            '''multi-threading'''

            threads = list()
            que = Queue()

            for t in range(len(self.tls)):
                x = threading.Thread(target=lambda q, a1, a2, a3, a4: q.put(bts[t].translate(a1, a2, a3, a4)),
                                     args=(que, v, self.tls[t], 0, confidence))
                threads.append(x)
                x.start()

            '''wait'''

            for t in threads:
                t.join()

            '''get results'''

            c = 0
            while not que.empty():
                result = que.get()

                if result:
                    for j in range(len(result)):
                        if not isPartOf(result[j], list(bt_dict.keys() if value_first else bt_dict.values())):
                            logger.info("BT from `%s`: %s" % (self.tls[j], result[j]))
                            bt_dict[result[j]] = bt_size + i
                            key_dict[k].append(bt_size + i)

                            i += 1
                            c += 1

                if c >= n_values:
                    break

        logger.info("Found %d new translation(s)." % (len(bt_dict) - len(kv_dict)))

        for bt in bts:
            bt.quit()

        '''save results'''

        if filepath != '':
            with open(filepath + FTYPE.TXT, 'w') as f:
                for key, value in bt_dict.items():
                    f.write('%s\t%d\n' % (key, value))
                f.close()

            with open(filepath + "_keys" + FTYPE.TXT, 'w') as f:
                for key, values in key_dict.items():
                    for value in values:
                        f.write('%s\t%d\n' % (key, value))
                f.close()

        return bt_dict, key_dict
