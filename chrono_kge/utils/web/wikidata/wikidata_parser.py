"""
Wikidata Parser
"""

from collections import defaultdict

from chrono_kge.utils.web.html.html_parser import HTML_Parser


WD_TITLE = "wikibase-title-label"
WD_DESCRIPTION = "wikibase-entitytermsview-heading-description"
WD_ALIASES = "wikibase-entitytermsview-aliases"
WD_ALIAS = "wikibase-entitytermsview-aliases-alias"


class Wikidata_Parser(HTML_Parser):

    def __init__(self) -> None:
        """"""
        super().__init__()
        return

    def parse_document(self, doc) -> dict:
        """"""
        doc_dict = defaultdict()

        self.parse_doc(doc)

        doc_dict['title'] = self.get_title()
        doc_dict['text'] = self.get_description()
        doc_dict['aliases'] = self.get_aliases()

        return doc_dict

    def get_title(self) -> str:
        """"""
        title = ""

        result = self.soup.find('span', {'class': WD_TITLE})

        if result:
            title = result.string

        return title

    def get_description(self) -> str:
        """"""
        description = ""

        result = self.soup.find('div', {'class': WD_DESCRIPTION})

        if result:
            description = result.string

        return description

    def get_aliases(self) -> list:
        """"""
        aliases = []

        group = self.soup.find('ul', {'class': WD_ALIASES})

        if group:
            results = group.find_all('li', {'class': WD_ALIAS})

            for result in results:

                if result.string:
                    aliases.append(result.string)

        return aliases
