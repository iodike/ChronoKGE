"""
YAML Parser
"""
import os

import yaml
from chrono_kge.utils.logger import logger
from chrono_kge.utils.vars.constants import PATH


class YML_Parser:
    """"""

    def __init__(self):
        """"""
        super().__init__()
        return

    @staticmethod
    def load_args(file: str) -> dict:
        """"""
        args = {}

        with open(os.path.join(PATH.CONFIG, file), "r") as stream:
            try:
                args = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        logger.info("Loaded config file `%s` successfully!" % file)
        return args
