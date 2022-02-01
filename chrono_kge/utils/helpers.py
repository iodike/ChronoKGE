"""
Helpers
"""

import math
import os
import time
import platform
import requests

from requests import RequestException

from chrono_kge.utils.logger import logger


def lists_to_tuples(data: list) -> list:
    """
    [[a,b],[c,d]] -> [(a,b),(c,d)]

    :param data: List of lists.
    :return: List of tuples.
    """

    tuples = []

    for e in data:
        tuples.append(tuple(e))

    return tuples


def chunks(data: list, size: int):
    """"""

    for i in range(0, len(data), size):
        yield data[i:i + size]


def splits(data: list, ratios=None):
    """"""

    if ratios is None:
        ratios = [0.5, 0.3, 0.2]

    while sum(ratios) > 1.0:
        ratios.pop()

    size = len(data)
    start, end = 0, 0

    for i in range(len(ratios)):
        end = start + math.floor(ratios[i] * size)
        yield data[start:end]
        start = end

    if end != size:
        yield data[start:size]


def split_sets(data: list):
    """"""

    datasets = []

    for dataset in splits(data, [0.5, 0.3, 0.2]):
        datasets.append(dataset)

    return [datasets[i] for i in range(3)]


def sort_by_sub(sub_li: list, pos: int = 1) -> list:
    """

    - reverse = None (Sorts in Ascending order)
    - key is set to sort using second element of
    - sublist lambda has been used

    :param sub_li:
    :param pos:
    :return:
    """

    sub_li.sort(key=lambda x: x[pos])
    return sub_li


def createUID():
    """"""
    return str(hash(time.time()))


def dict_to_query(qdict: dict) -> str:
    """"""
    query = ""

    ct = 0
    for key, value in qdict.items():
        if ct == 0:
            query += "?"
        else:
            query += "&"

        ct += 1
        query += str(key) + "=" + str(value)

    return query


def getOS():
    """"""
    system = platform.system()

    if system == 'Linux':
        dos = 'linux'
    elif system == 'Darwin':
        dos = 'mac'
    elif system == 'Windows':
        dos = 'win'
    else:
        dos = ''

    return dos


def isPartOf(value: str, values: list) -> bool:
    """"""
    for v in values:
        if str(v).lower() == str(value).lower():
            return True
    return False


def waitForInternetConnection(url: str = 'https://8.8.8.8', timeout: int = 1, wait: int = 3):
    """"""
    while True:
        try:
            _ = requests.get(url, timeout=timeout)
            return
        except RequestException:
            logger.info("Waiting for internet connection...")
            time.sleep(wait)
            pass


def makeDirectory(dirname: str, ignore: bool = False):
    """"""
    try:
        os.makedirs(dirname, exist_ok=not ignore)
    except FileExistsError:
        pass
    return


def confirmed():
    """"""
    answer = ""
    while answer not in ["y", "n"]:
        answer = input("Do you want to continue [Y/N]? ").lower()
    return answer == "y"


def strip(s):
    """"""
    if isinstance(s, str):
        s = s.replace(u'\xa0', u'')
    return s


def is_unique(x: list) -> bool:
    """"""
    return len(x) == len(set(x))
