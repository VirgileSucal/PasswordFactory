#! /bin/env python3

import consts
import tools
import pandas as pd
import numpy as np
from collections import Counter


def get_pw_sizes(data):
    # return dict(Counter([len(item) for item in data]))
    return [len(item) for item in data]


def violin(data):
    
    pass


if __name__ == '__main__':
    train_set, eval_set = tools.extract_data()
    print(get_pw_sizes(train_set))


