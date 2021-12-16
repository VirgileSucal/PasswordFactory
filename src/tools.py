#! /bin/env python3
from os import makedirs
from os.path import dirname, exists

import consts
import pandas as pd
import numpy as np


def fix_backslash(data):
    return [pw[:-1] for pw in data]


def extract_data():
    train_set, eval_set = None, None
    with open(consts.train_set) as train_file:
        train_set = train_file.read().splitlines()
    with open(consts.eval_set) as eval_file:
        eval_set = eval_file.read().splitlines()
    return train_set, fix_backslash(eval_set)


def init_dir(path):
    if not exists(dirname(path)):
        makedirs(dirname(path))


def parse_bools(boolean_str):
    return consts.bools[str(boolean_str).lower()]
