"""
Wikidata
"""

import os

from collections import defaultdict

from chrono_kge.utils.helpers import makeDirectory
from chrono_kge.utils.vars.constants import PATH, DRIVER, FTYPE
from chrono_kge.utils.logger import logger
from chrono_kge.utils.web.wikidata.wikidata_loader import Wikidata_Loader
from chrono_kge.utils.web.wikidata.wikidata_parser import Wikidata_Parser


WD_URL = "https://www.wikidata.org/"
WD_PATH_WIKI = 'wiki/'
WD_PREFIX_ENTITY = 'Q'
WD_PREFIX_RELATION = 'P'

FILE_TEXT = "wd_text"
FILE_ALIAS = "wd_alias"


class Wikidata:

    def __init__(self,
                 start: int = 1,
                 stop: int = 1000,
                 step: int = 100,
                 timeout: int = DRIVER.TIMEOUT,
                 save_dir: str = "wikidata1k"
                 ) -> None:
        """"""
        self.save_dir = save_dir
        self.range = (start, stop, step)
        self.loader = Wikidata_Loader(timeout)
        self.parser = Wikidata_Parser()
        return

    def get_data(self):
        """"""
        docs = defaultdict(dict)
        base_url = WD_URL + WD_PATH_WIKI

        file_path = os.path.join(PATH.DATA, self.save_dir)
        makeDirectory(file_path)

        c = 0
        for i in range(self.range[0], self.range[1], self.range[2]):

            for j in range(i, i + self.range[2]):

                index = WD_PREFIX_ENTITY + str(j)
                doc = self.loader.load_document(base_url + index)

                if doc is not None:
                    doc_dict = self.parser.parse_document(doc)
                    docs[index] = doc_dict
                    print(doc_dict)

            self.export_text(docs, file_path, FILE_TEXT + "_" + str(c) + FTYPE.TXT)
            self.export_aliases(docs, file_path, FILE_ALIAS + "_" + str(c) + FTYPE.TXT)

            docs.clear()
            c += 1

        return docs

    @staticmethod
    def export_text(xd, file_path, file_name) -> None:
        """"""
        try:
            with open(os.path.join(file_path, file_name), 'a') as f:
                for k, d in xd.items():
                    f.write(str(k) + '\t' + d['title'] + '\t' + d['text'] + '\n')

        except FileNotFoundError:
            logger.info("File not found: `%s`." % file_path)
            exit(1)

        return

    @staticmethod
    def export_aliases(xd, file_path, file_name) -> None:
        """"""
        try:
            with open(os.path.join(file_path, file_name), 'a') as f:
                for k, d in xd.items():
                    f.write(str(k) + '\t' + d['title'] + '\t')
                    for a in d['aliases']:
                        f.write(a + '\t')
                    f.write('\n')

        except FileNotFoundError:
            logger.info("File not found: `%s`." % file_path)
            exit(1)

        return
