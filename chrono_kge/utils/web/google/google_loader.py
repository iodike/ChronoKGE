"""
Google Loader
"""

from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException, ElementNotInteractableException, NoSuchElementException

from chrono_kge.utils.web.html.html_loader import HTML_Loader
from chrono_kge.utils.vars.constants import DRIVER
from chrono_kge.utils.logger import logger


G_TRANS_ADD_ROW = "TKwHGb"
G_TRANS_ADD_RES = "kgnlhe"
G_TRANS_RES = "VIiyi"


class Google_Loader(HTML_Loader):

    def __init__(self, timeout: int = DRIVER.TIMEOUT) -> None:
        """"""
        super().__init__(timeout)
        return

    def load_document(self, url):
        """"""
        self.load_url(url)
        self.handle_cookies()

        if not self.exists_content():
            return None

        return self.driver.page_source

    def handle_cookies(self) -> None:
        """"""
        try:
            elem = self.driver.find_element_by_tag_name("form")
            elem.click()

        except ElementNotInteractableException:
            pass

        except Exception:
            raise NoSuchElementException

        return

    def exists_content(self):
        """"""
        try:
            WebDriverWait(self.driver, self.timeout).until(
                ec.presence_of_element_located((By.CLASS_NAME, G_TRANS_ADD_ROW)))

            WebDriverWait(self.driver, self.timeout).until(
                ec.presence_of_element_located((By.CLASS_NAME, G_TRANS_ADD_RES)))

        except TimeoutException:

            try:
                WebDriverWait(self.driver, self.timeout).until(
                    ec.presence_of_element_located((By.CLASS_NAME, G_TRANS_RES)))

            except TimeoutException:
                logger.warning("Driver timeout after %d sec." % self.timeout)
                return False

        return True
