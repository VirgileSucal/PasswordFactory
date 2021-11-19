#! /bin/env python3

import consts
import tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, OrderedDict


def get_pw_sizes(data):
    return [len(item) for item in data]


def get_char_ratio(data):
    all_chars = []
    for password in data:
        all_chars.extend(password)

    return all_chars, OrderedDict({char: (n, n / len(all_chars)) for char, n in dict(Counter(all_chars)).items()})


def violin(data, path):
    tools.init_dir(path)

    fig = plt.figure()

    plt.ylabel("")
    plt.xlabel("password length")

    plt.violinplot(data, showmeans=True, showmedians=True, quantiles=[.25, .75], vert=False)

    plt.savefig(path)
    plt.close()


if __name__ == '__main__':
    train_set, _ = tools.extract_data()
    print(train_set)
    sizes = get_pw_sizes(train_set)
    sizes.sort(reverse=True)
    print("Highest password sizes:", sizes[0:20])
    lim = 30
    print("Passwords longer than {}:".format(lim), len([length for length in sizes if length > lim]))
    sizes.sort(reverse=False)
    print("Lowest password sizes:", sizes[0:20])
    # violin(s, consts.fig_path + "test.pdf")

    chars, counter = get_char_ratio(train_set)
    print(list(counter))
    for char, (n, ratio) in counter.items():
        print(char, ":", n, "(", round(ratio * 100, 7), "% )")
    print(len(chars))
