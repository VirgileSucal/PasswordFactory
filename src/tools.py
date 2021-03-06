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


def read_file(path):
    data = None
    with open(path) as file:
        data = file.read().splitlines()
    return data


def write_file(path, lines, mode='w'):
    with open(path, mode) as file:
        file.writelines(lines)


# def extract_data():
#     train_set, pretrain_set, eval_set = None, None, None
#     with open(consts.train_set) as train_file:
#         train_set = train_file.read().splitlines()
#     with open(consts.pretrain_set) as pretrain_file:
#         pretrain_set = pretrain_file.read().split()
#     with open(consts.eval_set) as eval_file:
#         eval_set = eval_file.read().splitlines()
#     return train_set, pretrain_set, fix_backslash(eval_set)


def extract_train_data():
    return read_file(consts.train_set)


def extract_pretrain_data():
    pretrain_set = None
    with open(consts.pretrain_set) as pretrain_file:
        pretrain_set = pretrain_file.read().split()
    return pretrain_set


def extract_selected_train_data():
    return read_file(consts.selected_train_data)


def extract_selected_pretrain_data():
    return read_file(consts.selected_pretrain_data)


def extract_eval_data():
    return fix_backslash(read_file(consts.eval_set))


def extract_data():
    """
    This method opens both train and eval files, extracts data and returns it inside lists.

    :return: lists that contain training passwords and evaluation passwords
    """

    train_set, eval_set = None, None
    train_set = extract_train_data()
    eval_set = extract_eval_data()
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
