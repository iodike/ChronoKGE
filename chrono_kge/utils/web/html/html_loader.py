"""
HTML Loader
"""

import os

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from chrono_kge.utils.vars.constants import DRIVER, PATH
from chrono_kge.utils.helpers import getOS, waitForInternetConnection


class HTML_Loader:

    def __init__(self, timeout: int = DRIVER.TIMEOUT) -> None:
        """"""
        self.timeout = timeout

        options = Options()
        options.add_argument('--no-sandbox')
        options.add_argument("--headless")

        bin_path = os.path.join(PATH.DRIVER, getOS(), 'chromedriver')
        self.driver = webdriver.Chrome(bin_path, options=options)
        return

    def load_document(self, url):
        """"""
        raise NotImplementedError

    def load_url(self, url, c=10) -> None:
        """"""
        i = 0
        while i < c:

            try:
                self.driver.get(url)
                break

            except Exception:
                waitForInternetConnection()
                i += 1
                continue

        return

    def quit(self):
        """"""
        if self.driver is not None:
            self.driver.quit()
            self.driver = None
        return
