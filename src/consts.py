#! /bin/env python3

root_dir = "../"
data_dir = root_dir + "data/"
output_dir = root_dir + "output/"
src_dir = root_dir + "src/"
models_dir = output_dir + "models/"
train_set = data_dir + "TrainEval/train.txt"
pretrain_set = data_dir + "words.txt"
eval_set = data_dir + "TrainEval/eval.txt"
selected_train_data = output_dir + "selected_train_data.txt"
selected_pretrain_data = output_dir + "selected_pretrain_data.txt"
fig_path = output_dir + "fig/"
sdtout_path = output_dir + "sdtout_{}.txt"
default_model_file = "lstm"
default_n_samples = 10_000
bools = {"true": True, "false": False}
vocab_start_idx = 2
