"""
Google Translate
"""

import random
import string

from chrono_kge.utils.helpers import waitForInternetConnection, dict_to_query
from chrono_kge.utils.logger import logger
from chrono_kge.utils.web.google.google_loader import Google_Loader
from chrono_kge.utils.web.google.google_parser import Google_Parser
from ...vars.constants import DRIVER


G_URL = "https://translate.google.com/"
G_QUERY_DICT = {
    "hl": "de",
    "tab": "TT",
    "sl": "en",
    "tl": "de",
    "op": "translate",
    "text": ""
}


class BackTranslator:

    def __init__(self,
                 sl: str = 'en',
                 n_values: int = 1,
                 timeout: int = DRIVER.TIMEOUT
                 ) -> None:
        """"""
        self.ft = Translator(timeout=timeout)
        self.bt = Translator(timeout=timeout)
        self.sl = sl
        self.n_values = n_values
        return

    def multi_translate(self, word: str,
                        tls: list = None,
                        index: int = 0,
                        confidence: int = 0
                        ) -> list:
        """"""
        ts = []

        if tls:

            for tl in tls:
                t = self.translate(word, tl, index, confidence)
                ts.append(t)

        else:
            logger.warning("List with target languages should not be empty.")

        return ts

    def translate(self,
                  word: str,
                  tl: str = None,
                  index: int = 0,
                  confidence: int = 0
                  ) -> list:
        """"""
        t = word
        ts = []
        all_bts = []

        # forward
        self.ft(word=t, sl=self.sl, tl=tl)
        fts = self.ft.get_translations(index=index, confidence=confidence)

        for ft in fts:

            # backward
            self.bt(word=str(ft), sl=tl, tl=self.sl)
            bts = self.bt.get_translations(index=index, confidence=confidence)

            all_bts.extend(bts)

        # remove duplicates
        all_bts_set = set(all_bts)

        for bt in list(all_bts_set):

            # remove identical
            if not self.checkEquality(bt, t):
                ts.append(bt)

                # max n_values
                if len(ts) >= self.n_values:
                    break

        return ts

    @staticmethod
    def checkEquality(a: str, b: str):
        """"""
        ac = a.translate(str.maketrans('', '', string.punctuation)).lower()
        bc = b.translate(str.maketrans('', '', string.punctuation)).lower()
        return ac == bc

    def quit(self):
        """"""
        self.ft.quit()
        self.bt.quit()
        return


class Translator:

    def __init__(self,
                 timeout: int = DRIVER.TIMEOUT
                 ) -> None:
        """"""
        self.downloader = Google_Loader(timeout=timeout)
        self.parser = Google_Parser()
        self.translations = []
        self.word = None
        return

    def __call__(self,
                 word: str,
                 sl: str = 'en',
                 tl: str = 'de',
                 ):
        """"""
        self.word = word

        '''url'''

        qdict = G_QUERY_DICT
        qdict['text'] = str(self.word)
        qdict['sl'] = str(sl)
        qdict['tl'] = str(tl)

        url = G_URL + dict_to_query(qdict)

        '''loader'''

        doc = ""
        while True:

            try:
                doc = self.downloader.load_document(url)
                break

            except Exception:
                waitForInternetConnection()
                continue

        '''parser'''

        translations = []
        if doc is not None:
            translations = self.parser.parse_document(doc)

        self.translations = translations

        return

    def get_translations(self,
                         index: int = 0,
                         confidence: int = 0,
                         ) -> list:
        """"""
        tls = []

        if self.translations:
            t_len = len(self.translations)

            for i in range(min(index, t_len), min(confidence+1, t_len)):
                tls.append(self.translations[i])

        return tls

    def get_translation(self,
                        index: int = 0,
                        confidence: int = 0,
                        rand: bool = False
                        ) -> (str, bool):
        """"""
        if not self.translations:
            return self.word, True

        max_k = len(self.translations) - 1

        if rand:
            kth = random.randint(min(index, max_k), min(confidence, max_k))
        else:
            kth = index if index <= max_k else 0

        return self.translations[kth], (self.translations[kth] == self.word)

    def quit(self):
        """"""
        self.downloader.quit()
        return
