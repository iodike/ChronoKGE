"""
Wikidata Loader
"""

from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec

from chrono_kge.utils.logger import logger
from chrono_kge.utils.vars.constants import DRIVER
from chrono_kge.utils.web.html.html_loader import HTML_Loader


WD_TITLE = "wikibase-title-label"
WD_DESCRIPTION = "wikibase-entitytermsview-heading-description"


class Wikidata_Loader(HTML_Loader):

    def __init__(self, timeout: int = DRIVER.TIMEOUT) -> None:
        """"""
        super().__init__(timeout)
        return

    def load_document(self, url):
        """"""
        self.load_url(url)

        if not self.exists_content():
            return None

        return self.driver.page_source

    def exists_content(self):
        """"""
        try:
            WebDriverWait(self.driver, self.timeout).until(
                ec.presence_of_element_located((By.CLASS_NAME, WD_TITLE)))

            WebDriverWait(self.driver, self.timeout).until(
                ec.presence_of_element_located((By.CLASS_NAME, WD_DESCRIPTION)))

        except TimeoutException:
            logger.warning("Driver timeout after %d sec." % self.timeout)
            return False

        return True
