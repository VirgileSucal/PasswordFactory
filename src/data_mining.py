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
    """
    This method returns the size of each password in the dataset

    :param data: the dataset that contains passwords
    :return: a list of int that contains the size of each password
    """

    return [len(item) for item in data]


def get_char_ratio(data):
    """
    This method gets every character inside the passwords and adds it inside a list

    :param data: the dataset that contains passwords
    :return: the list of all characters in the dataset, and a list of all unique character that appears in the dataset
    """

    all_chars = []
    for password in data:
        all_chars.extend(password)

    return all_chars, OrderedDict({char: (n, n / len(all_chars)) for char, n in dict(Counter(all_chars)).items()})


def violin(data, path):
    """
    This method generated a violin plot

    :param data: data for the violin plot
    :param path: path to save the violin plot
    """

    tools.init_dir(path)

    fig = plt.figure()

    plt.ylabel("")
    plt.xlabel("Taille des mots de passe")

    plt.violinplot(data, showmeans=True, showmedians=True, quantiles=[.25, .75], vert=False)

    plt.savefig(path)
    plt.close()


def barplot(data, path):
    """
    This method generated a bar plot

    :param data: data for the bar plot
    :param path: path to save the bar plot
    """

    tools.init_dir(path)

    fig = plt.figure()

    plt.ylabel("")
    plt.xlabel("password length")

    # plt.violinplot(data, showmeans=True, showmedians=True, quantiles=[.25, .75], vert=False)
    plt.bar(data, height = 0.5)

    plt.savefig(path)
    plt.close()


def get_duplicate(data):
    """
    This methods searchs if there are duplicates inside the dataset.
    Each password is added inside a dictionary, and if there is a duplicate,
    this password is also added inside the duplicate dictionary

    :param data: the dataset that contains passwords
    :return: a dictionary that contains each password and the number of occurrences,
    and a dictionary that contains the duplicated passwords
    """

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
    # barplot(sizes, consts.fig_path + "barplot_test.pdf")
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
