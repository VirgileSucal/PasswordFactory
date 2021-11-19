#! /bin/env python3

import consts
import pandas as pd
import numpy as np


def extract_data():
    train_set, eval_set = None, None
    with open(consts.train_set) as train_file:
        train_set = train_file.read().splitlines()
    with open(consts.eval_set) as eval_file:
        eval_set = eval_file.read().splitlines()
    return train_set, eval_set

