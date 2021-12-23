#! /bin/env python3
from os import makedirs
from os.path import dirname, exists

import consts
import pandas as pd
import numpy as np


def fix_backslash(data):
    """
    This method is used to ignore backslash at the end of the passwords in the eval dataset

    :param data: The evaluation dataset
    :return: a list of password without backslash
    """

    return [pw[:-1] for pw in data]


def extract_data():
    """
    This method opens both train and eval files, extracts data and returns it inside lists.

    :return: lists that contain training passwords and evaluation passwords
    """

    train_set, eval_set = None, None
    with open(consts.train_set) as train_file:
        train_set = train_file.read().splitlines()
    with open(consts.eval_set) as eval_file:
        eval_set = eval_file.read().splitlines()
    return train_set, fix_backslash(eval_set)


def init_dir(path):
    """
    This method creates a directory according to the path parameter if it doesn't exist

    :param path: path to the directory that should be created
    """

    if not exists(dirname(path)):
        makedirs(dirname(path))


def parse_bools(boolean_str):
    """
    This method parse a boolean to a string

    :param boolean_str: the boolean value that must be parsed
    :return: the string version of the boolean
    """

    return consts.bools[str(boolean_str).lower()]
