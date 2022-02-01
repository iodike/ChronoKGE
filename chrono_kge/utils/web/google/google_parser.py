"""
Google Parser
"""

from chrono_kge.utils.web.html.html_parser import HTML_Parser
from chrono_kge.utils.logger import logger


G_CLASS_ROW_TR = "TKwHGb"
G_CLASS_ROW_SPAN = "kgnlhe"
G_CLASS_RESULT = "VIiyi"


class Google_Parser(HTML_Parser):

    def __init__(self) -> None:
        """"""
        super().__init__()
        return

    def parse_document(self, doc) -> list:
        """"""
        self.parse_doc(doc)

        translations = []

        row_translations = self.get_row_translations()

        if row_translations:
            translations.extend(row_translations)

        else:
            result_translation = self.get_result_translation()

            if result_translation:
                translations.append(result_translation)

        return translations

    def get_result_translation(self) -> str:
        """"""
        translation = ""
        result = self.soup.find('span', {'class': G_CLASS_RESULT})

        if result:
            if result.string:
                translation = str(result.string)
            else:
                chs = result.findChildren('span', recursive=True)
                if chs:
                    if len(chs) > 1:
                        if chs[1]:
                            translation = str(chs[1].string)

        return translation

    def get_row_translations(self) -> list:
        """"""
        translations = []

        rows = self.soup.find_all('tr', {'class': G_CLASS_ROW_TR})

        if rows:
            for row in rows:
                chs = row.find_all('span', {'class': G_CLASS_ROW_SPAN})

                if chs:
                    for ch in chs:
                        if ch.string:
                            translations.append(str(ch.string))
                        else:
                            logger.warning("No substring found.")
                            continue

        else:
            pass

        return translations
