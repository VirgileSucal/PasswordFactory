#! /bin/env python3

import consts
import tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def get_pw_sizes(data):
    return [len(item) for item in data]


def make_char_vocab(data):
    char_count = 0
    vocab = []
    all_chars = []
    for password in data:
        all_chars.extend(password)
        char_count += len(password)
        # for char in password:
        #     if char not in vocab:
        #         vocab.append(char)
        #     all_chars.append(char)

    return list(dict.fromkeys(all_chars)), all_chars, len(all_chars)


def get_char_ratio(data):
    return dict(Counter(data))


def violin(data, path):
    tools.init_dir(path)

    fig = plt.figure()

    plt.ylabel("")
    plt.xlabel("password length")

    plt.violinplot(data, showmeans=True, showmedians=True, quantiles=[.25, .75], vert=False)

    plt.savefig(path)
    plt.close()


def get_duplicate(data):
    data_count = {}
    for password in data:
        if password not in data_count:
            data_count[password] = 1
        else:
            data_count[password] += 1
    return data_count, {password: n for password, n in data_count.items() if n > 1}


if __name__ == '__main__':
    train_set, _ = tools.extract_data()
    print(train_set)
    s = get_pw_sizes(train_set)
    s.sort(reverse=True)
    print("Highest password sizes:", s[0:20])
    s.sort(reverse=False)
    print("Lowest password sizes:", s[0:20])
    # print(s)
    # violin(s, consts.fig_path + "test.pdf")
    train_set, eval_set = tools.extract_data()
    print(get_pw_sizes(train_set))

    print(get_duplicate(train_set))

    vocab, chars, length = make_char_vocab(train_set)
    print(vocab)
    print(get_char_ratio(chars))
    print(length)
