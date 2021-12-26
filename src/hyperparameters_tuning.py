#! /bin/env python3
import os
import subprocess
from os.path import dirname, abspath, join
import consts

if __name__ == '__main__':
    default_args = {
        "train": consts.default_train_arg,
        "pretrain": False,
        "eval": False,
        "bruteforce": False,
        # "random": True,
        "random": False,
        "debug": False,
        "verbose": consts.default_verbose_arg,
        "nn_class": consts.default_nn_class_arg,
        "hidden_size": consts.default_hidden_size_arg,
        "batch_size": 1,
        "n_layers": consts.default_n_layers_arg,
        "bidirectional": consts.default_bidirectional_arg,
        "dropout_value": consts.default_dropout_value_arg,
        "use_softmax": False,
        # "n_epochs": consts.default_n_epochs_arg,
        "n_pretrain_epochs": consts.default_n_pretrain_epochs_arg,
        "lr": consts.default_lr_arg,
        "epoch_size": 1,
    }

    args_lists = {
        "random": [True, False],
        "nn_class": ["RNN", "LSTM", "GRU"],
        "hidden_size": [64, 256, 512],
        "batch_size": [1, 64, 128],
        "n_layers": [1, 2, 4],
        "bidirectional": [True, False],
        "dropout_value": [0, 0.01, 0.1],
        # "use_softmax": [True, False],
        # "n_epochs": [100_000],
        "n_pretrain_epochs": [1_000],
        # "n_tests": [10_000],
        "lr": [0.005, 0.05, 0.005],
        # "epoch_size": [1],
    }

    def_n_epochs_random = 1_000_000
    def_n_epochs = 20
    n_epochs_random_list = [1_000_000]
    n_epochs_list = [20]

    output = subprocess.run(
        ["sh", join(
            dirname(abspath(__file__)),
            "init.sh"
        )],  # A shell script to initialize workspace.
        capture_output=True
    )
    print(output.args)
    print(output.stdout)
    print(output.stderr)
    print(output.returncode)
    for arg, values in args_lists.items():
        for value in values:
            def_args = {**default_args}
            if arg == "random" and value == True:
                def_args["n_epochs"] = def_n_epochs_random
            else:
                def_args["n_epochs"] = def_n_epochs

            args = "--{} {}".format(arg, str(value)) + " " + " ".join([
                "--{} {}".format(a, str(v)) for a, v in def_args.items() if a != arg
            ])
            args += " -d True "
            print(args)
            # output = subprocess.run(
            #     ["sh", str(join(dirname(abspath(__file__)), "run.sh")) + " " + args],  # A shell script to run the process.
            #     capture_output=True
            # )
            # print(output.stdout)

