#! /bin/env python3

import torch
import string
import consts

from nameGeneration import *
from tools import *
from data_mining import *

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('CUDA AVAILABLE')
else:
    device = torch.device("cpu")
    print('ONLY CPU AVAILABLE')


if __name__ == '__main__':

    train_set, eval_set = extract_data()
    small_train_set = train_set[-1000:]
    # print('train_set: ', len(train_set))
    # print('eval_set: ', len(eval_set))
    # print(small_train_set)
