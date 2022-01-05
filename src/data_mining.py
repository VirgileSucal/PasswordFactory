#! /bin/env python3

from os.path import join
import consts
import tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, OrderedDict
from statistics import mean, median

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
    plt.xlabel("Taille des mots de passe")

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


def get_mean_median(password_size):
    return mean(password_size), median(password_size)


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
    violin(sizes, join(consts.fig_path, "sizes.pdf"))
    train_set, eval_set = tools.extract_data()
    print(get_pw_sizes(train_set))

    print(get_duplicate(train_set))

    chars, counter = get_char_ratio(train_set)
    print(list(counter))
    for char, (n, ratio) in counter.items():
        print(char, ":", n, "(", round(ratio * 100, 7), "% )")

    lim_ratio = 0.001
    print("Char that appears less than {}% :".format(lim_ratio))
    for char, (n, ratio) in counter.items():
        if(round(ratio * 100, 7) <= lim_ratio):
            print(char, ":", n, "(", round(ratio * 100, 7), "% )")
    print(len(chars))

    mean_value, median_value = get_mean_median(sizes)
    print("Mean password length: ", mean_value)
    print("Median password length: ", median_value)
