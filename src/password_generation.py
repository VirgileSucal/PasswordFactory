#! /bin/env python3
import math
import string
from os import path
import sys
from os.path import abspath, dirname
from random import randint
from time import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, RandomSampler
from torchtext.legacy.data import BucketIterator
from torch.nn.utils.rnn import pad_sequence
import tools
from data_mining import get_char_ratio
import consts
from argparse import ArgumentParser
from sklearn.model_selection import RandomizedSearchCV
from ray import tune  # https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html , 'pip install ray[tune]'
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('CUDA AVAILABLE')
else:
    device = torch.device("cpu")
    print('ONLY CPU AVAILABLE')

__vocab = None
__vocab_size = None


def time_since(since):
    now = time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def get_vocab(data=None):
    global __vocab
    global __vocab_size
    assert not (__vocab is None and data is None)
    if __vocab is None:
        __vocab = "".join(["\0" for _ in range(consts.vocab_start_idx)]) + "".join(list(get_char_ratio(data)[1].keys()))  # EOS is index 1, because index is used for initializing one-hots.
        __vocab_size = len(__vocab)
    return __vocab


def get_vocab_size(data=None) -> int:
    global __vocab
    global __vocab_size
    assert not (__vocab is None and data is None)
    if __vocab_size is None:
        get_vocab(data)
    return __vocab_size


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size=1, n_layers=1, bidirectional=False, dropout_value=0, use_softmax=False):
        super(LSTM, self).__init__()

        assert input_size > 0
        assert hidden_size > 0
        assert output_size > 0
        assert n_layers > 0
        assert batch_size > 0
        assert 0 <= dropout_value < 1

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.dropout_value = dropout_value
        self.use_softmax = use_softmax
        self.use_softmax = False

        self.encoder = nn.Embedding(self.input_size, self.hidden_size)  # input_size = vocab size, but entry is index, not one_hot
        self.encoder.to(device)
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout_value,
            batch_first=True
        )
        self.lstm.to(device)
        self.decoder = nn.Linear(self.hidden_size, self.output_size)
        self.decoder.to(device)

        if bool(self.dropout_value):
            self.dropout = nn.Dropout(self.dropout_value)
            self.dropout.to(device)
        if self.use_softmax:
            self.softmax = nn.Softmax(dim=1)
            self.softmax.to(device)

    def forward(self, input, hidden, cell):
        input = self.encoder(input)
        # print("f", input)
        # print("f", input.size())
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        output = self.decoder(output)

        if bool(self.dropout_value):
            output = self.dropout(output)
        if self.use_softmax:
            output = self.softmax(output)

        return output, (hidden, cell)

    def init_h_c_with_zeros(self):
        return Variable(torch.zeros(
            (1 + int(self.bidirectional)) * self.n_layers,
            self.batch_size,
            self.hidden_size,
            device=device
        ))

    def init_h_c(self):
        return Variable(torch.rand(
            (1 + int(self.bidirectional)) * self.n_layers,
            self.batch_size,
            self.hidden_size,
            device=device
        ))


def input_tensor(line):
    return torch.LongTensor([get_vocab().find(line[li]) for li in range(0, len(line))])


def target_tensor(line):
    tensor = torch.zeros(len(line), get_vocab_size(), dtype=torch.float64)
    for li in range(1, len(line)):  # Don't encode first one because it doesn't need to be predicted
        letter = line[li]
        tensor[li - 1][get_vocab().find(letter)] = 1.0  # The first letter isn't in target tensor because it isn't predicted.
    tensor[-1][consts.vocab_start_idx] = 1.0  # The last one is \0 (EOS).
    return tensor


def random_epoch_mini_batch(passwords_batches, epoch_size=1):
    assert epoch_size > 0
    index = randint(0, len(passwords_batches) - epoch_size)
    selected_passwords_batches = passwords_batches.dataset[index: index + epoch_size]  # torch.utils.data.RandomSampler

    batches = []
    print(selected_passwords_batches)
    for passwords_batch in selected_passwords_batches:
        input_batch = pad_sequence([input_tensor(password) for password in passwords_batch], batch_first=True).long()
        target_batch = pad_sequence([target_tensor(password) for password in passwords_batch], batch_first=True).long()
        batches.append((input_batch, target_batch))
    return batches


def init_batches(passwords_batches):
    batches = []

    for passwords_batch in passwords_batches:
        input_batch = pad_sequence([input_tensor(password) for password in passwords_batch], batch_first=True).long()
        target_batch = pad_sequence([target_tensor(password) for password in passwords_batch], batch_first=True).long()
        batches.append((input_batch, target_batch))
    return batches


def train_lstm_mini_batch(model, batches, criterion, learning_rate):
    hidden = model.init_h_c()
    cell = model.init_h_c()
    output = None

    model.zero_grad()
    loss = 0

    for input_batch, target_batch in batches:
        # print(input_batch.size(0), hidden.size()[1])
        if input_batch.size(0) != hidden.size()[1]:
            continue
        # print(input_batch)
        # print(input_batch.size())
        output, (hidden, cell) = model(input_batch.to(device), hidden.to(device), cell.to(device))
        # print()
        # print()
        # print(output)
        # print(target_batch)
        # print()
        # print()
        l = criterion(output.to(device), target_batch.type(torch.FloatTensor).to(device))
        loss += l

    loss.backward()

    for p in model.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item() / len(batches)


# def train_lstm(lstm, train_dataloader, n_epochs, criterion, learning_rate, print_every=None, verbose=True):
#
#     assert train_dataloader is not None
#     assert n_epochs > 0
#
#     start = time()
#     all_losses = []
#     total_loss = 0
#     best_loss = (100, 0)
#     n_iters = len(train_dataloader)
#     epoch_size = n_iters // min(n_epochs, n_iters)
#     print("Data size:", n_iters, "; Epoch size:", epoch_size, "; Epochs:", n_epochs, "Batch size:", lstm.batch_size)
#     if print_every is None:
#         print_every = epoch_size
#
#     current_epoch_batches = []
#
#     epoch = 0
#     iter = 0
#     for batch in train_dataloader:
#
#         current_epoch_batches.append(batch)
#         iter += 1
#
#         if iter % epoch_size == 0:  # All batches for this epoch have been loaded.
#             epoch += 1
#
#             batches = init_batches(current_epoch_batches)
#             # if len(batches) == 1 and batches[0].size(0) != lstm.init_h_c().size()[1]:
#             if batches[0][0].size(0) != lstm.init_h_c().size()[1]:
#                 continue
#
#             output, loss = train_lstm_mini_batch(lstm, batches, criterion, learning_rate)
#             total_loss += loss
#             if loss < best_loss[0]:
#                 best_loss = (loss, iter)
#             all_losses.append(loss)
#
#             if verbose and iter % print_every == 0:
#                 print('%s (%d %d%%) %.4f (%.4f)' % (time_since(start), epoch, iter / n_iters * 100, total_loss / iter, loss))
#
#             current_epoch_batches = []
#
#         # if epoch >= 20:  # TODO: Only for tests (remove it)
#         #     break


def train_lstm_epoch(lstm, train_dataloader, criterion, learning_rate, n_mini_batches=None, print_every=None, verbose=True):

    assert train_dataloader is not None

    n_iters = len(train_dataloader)
    if n_mini_batches is None:
        n_mini_batches = n_iters
    assert n_mini_batches > 0

    start = time()
    all_losses = []
    total_loss = 0
    best_loss = (100, 0)
    mini_batches_size = n_iters // min(n_mini_batches, n_iters)
    print("Data size:", n_iters, "; Epoch size:", mini_batches_size, "; Epochs:", n_mini_batches, "Batch size:", lstm.batch_size)
    if print_every is None:
        print_every = mini_batches_size

    current_epoch_batches = []

    mini_batch = 0
    iter = 0
    percent = -1
    for batch in train_dataloader:
        current_epoch_batches.append(batch)
        iter += 1

        if iter % mini_batches_size == 0:  # All batches for this mini_batch have been loaded.
            mini_batch += 1

            batches = init_batches(current_epoch_batches)
            # if len(batches) == 1 and batches[0].size(0) != lstm.init_h_c().size()[1]:
            if batches[0][0].size(0) != lstm.init_h_c().size()[1]:
                continue

            output, loss = train_lstm_mini_batch(lstm, batches, criterion, learning_rate)
            total_loss += loss
            if loss < best_loss[0]:
                best_loss = (loss, iter)
            all_losses.append(loss)

            if verbose and iter % print_every == 0:
                tmp = int(iter / n_iters * 100)
                if tmp > percent:
                    percent = tmp
                    with open("tmp_print.txt", "a") as file:  # TODO: Only for tests, remove it.
                        # file.write(str(time_since(start)) + " ; " + str(percent) + " %\n")
                        file.write('%s (%d %d%%) %.4f (%.4f)' % (time_since(start), mini_batch, iter / n_iters * 100, total_loss / iter, loss) + "\n")
                print('%s (%d %d%%) %.4f (%.4f)' % (time_since(start), mini_batch, iter / n_iters * 100, total_loss / iter, loss))
                # print_progress(total=n_mini_batches, acc=best_loss, start=start, iter=mini_batch, size=len(n_mini_batches))


            current_epoch_batches = []

        # if mini_batch >= 20:  # TODO: Only for tests (remove it)
        #     break


def train_lstm(lstm, train_dataloader, n_epochs, criterion, learning_rate, print_every=None, verbose=True):

    for epoch in range(n_epochs):
        if verbose:
            print("\nEpoch:", epoch + 1, "/", n_epochs)
        train_lstm_epoch(lstm, train_dataloader, criterion, learning_rate, n_mini_batches=None, print_every=print_every, verbose=verbose)


def random_train_lstm(lstm, train_dataloader, n_epochs, criterion, learning_rate, print_every=None, verbose=True):

    assert train_dataloader is not None
    assert n_epochs > 0

    start = time()
    all_losses = []
    total_loss = 0
    best_loss = (100, 0)
    n_iters = len(train_dataloader)
    epoch_size = n_iters // min(n_epochs, n_iters)
    print("Data size:", n_iters, "; Epoch size:", epoch_size, "; Epochs:", n_epochs, "Batch size:", lstm.batch_size)
    if print_every is None:
        print_every = epoch_size

    current_epoch_batches = []

    epoch = 0
    it = 0
    percent = -1
    data_iterator = iter(train_dataloader)
    for batch in data_iterator:

        current_epoch_batches.append(batch)
        it += 1

        if it % epoch_size == 0:  # All batches for this epoch have been loaded.
            epoch += 1

            batches = init_batches(current_epoch_batches)
            # if len(batches) == 1 and batches[0].size(0) != lstm.init_h_c().size()[1]:
            if batches[0][0].size(0) != lstm.init_h_c().size()[1]:
                continue

            output, loss = train_lstm_mini_batch(lstm, batches, criterion, learning_rate)
            total_loss += loss
            if loss < best_loss[0]:
                best_loss = (loss, it)
            all_losses.append(loss)

            if verbose and it % print_every == 0:
                tmp = int(it / n_iters * 100)
                if tmp > percent:
                    percent = tmp
                    with open("tmp_print.txt", "a") as file:  # TODO: Only for tests, remove it.
                        # file.write(str(time_since(start)) + " ; " + str(percent) + " %\n")
                        file.write('%s (%d %d%%) %.4f (%.4f)' % (time_since(start), epoch, it / n_iters * 100, total_loss / it, loss) + "\n")
                print('%s (%d %d%%) %.4f (%.4f)' % (time_since(start), epoch, it / n_iters * 100, total_loss / it, loss))


            current_epoch_batches = []

        # if epoch >= 20:  # TODO: Only for tests (remove it)
        #     break


def pretrain_lstm(lstm, n_epochs, epoch_size, criterion, learning_rate, random=True, print_every=None, verbose=True):
    """
    Pretrain model on a dataset containing obscene words

    Link to dataset: github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/blob/master/en
    """

    pretrain_dataloader = None
    batch_size = lstm.batch_size
    pretrain_set = tools.extract_pretrain_data()
    if random:
        random_sampler = RandomSampler(pretrain_set, num_samples=n_epochs * epoch_size * batch_size, replacement=True)
        pretrain_dataloader = DataLoader(pretrain_set, batch_size=batch_size, shuffle=False, sampler=random_sampler)
        random_train_lstm(lstm, pretrain_dataloader, n_epochs, criterion, learning_rate, print_every=print_every, verbose=verbose)
    else:
        pretrain_dataloader = DataLoader(pretrain_set, batch_size=batch_size, shuffle=True)
        train_lstm(lstm, pretrain_dataloader, n_epochs, criterion, learning_rate, print_every=print_every, verbose=verbose)


def compute_pretrain_set(train_data, verbose=True):
    pretrain_set = tools.extract_pretrain_data()
    # train_data = train_data[:1000]
    # pretrain_set = pretrain_set[:1000]
    selected_train_data, selected_words = [], []
    train_size, words_size = len(train_data), len(pretrain_set)
    train_counter, words_counter, train_percent, words_percent = -1, -1, -1, -1
    print_update = False
    for pw in train_data:
        train_counter += 1
        for word in pretrain_set:
            words_counter += 1
            if pw.lower().find(word.lower()) > -1:
                if pw not in selected_train_data:
                    selected_train_data.append(pw)  # Don't remove it: it must be added several times if it contains several words.
                    tools.write_file(consts.selected_train_data, pw + "\n", 'a')
                if word not in selected_words:
                    selected_words.append(word)
                    tools.write_file(consts.selected_pretrain_data, word + "\n", 'a')
            train_percent_tmp = round((train_counter / train_size) * 100, 2)
            # train_percent_tmp = int((train_counter / train_size) * 100)
            words_percent_tmp = int((words_counter / words_size) * 100)
            if train_percent_tmp > train_percent:
                train_percent = train_percent_tmp
                print_update = True
            # if words_percent_tmp > words_percent:
            #     words_percent = words_percent_tmp
            #     print_update = True
            if print_update:
                print("passwords: {} % ; words: {} %".format(train_percent, words_percent))#, end="\r")
                with open("tmp_print.txt", "a") as file:  # TODO: Only for tests, remove it.
                    file.write("passwords: {} % ; words: {} %\n".format(train_percent, words_percent))
            print_update = False
        words_counter, words_percent = -1, -1
    return selected_train_data, selected_words


def create_pretrain_files(train_data):
    selected_train_data, selected_words = compute_pretrain_set(train_data)
    # tools.write_file(consts.selected_train_data, selected_train_data)
    # tools.write_file(consts.selected_pretrain_data, selected_words)


def argmax(float_list):
    max_val = float_list[0]
    max_idx = 0
    for i in range(len(float_list)):
        if float_list[i] > max_val:
            max_val = float_list[i]
            max_idx = i

    return max_idx


def print_progress(total, acc, start, iter, size):
    global print_progress_current_percent
    print_progress_current_percent = -1
    bar_len = 50
    filled_len = int(round(bar_len * iter / float(total)))
    percents = round(100.0 * iter / float(total), 1)

    current_percent = (100 * acc / size)
    if current_percent <= print_progress_current_percent:
        return
    print_progress_current_percent = current_percent

    if filled_len == 0:
        bar = '>' * filled_len + ' ' * (bar_len - filled_len)
    else:
        bar = '=' * (filled_len - 1) + '>' + ' ' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s => coverage of %.3f %% (%d) on %s \r' % (bar, percents, ' %', current_percent, acc, time_since(start)))
    sys.stdout.flush()


def generate_passwords_batches(decoder, start_letters, max_length: int = 128):
    with torch.no_grad():  # no need to track history in sampling

        hidden = decoder.init_h_c()
        cell = decoder.init_h_c()

        # print(start_letters)

        output_passwords = None
        if type(start_letters) == str:
            output_passwords = [start_letters[0] for _ in range(decoder.batch_size)]
        else:
            output_passwords = [sl for sl in start_letters]
        # # input = input_tensor(start_letters)
        # input = pad_sequence([torch.LongTensor([input_tensor(letter)]) for letter in output_passwords], batch_first=True).long()  # It will generate len(batch) passwords each time.
        input = init_batches([output_passwords])[0][0]
        # # input = init_batches([output_passwords])
        # # print(input)
        # # input = input[0][0]
        # print(input)
        # # print(input.size())
        # # input = torch.unsqueeze(input, dim=-1)
        # # print(input)
        # # print(input.size())

        for i in range(max_length):  # There is a max length to avoid too long computing time.
            output, (hidden, cell) = decoder(input.to(device), hidden.to(device), cell.to(device))
            # print(output)
            # predicted_letters_indices = [argmax(letter_one_hot[0]) for letter_one_hot in output]
            # predicted_letters_indices = [[argmax(letter_one_hot[i]) for letter_one_hot in output] for i in range(len(output_passwords))]
            predicted_letters_indices = [argmax(letter_one_hot[-1]) for letter_one_hot in output]
            # exit(1)
            # print(predicted_letters_indices)
            # if sum(predicted_letters_indices) == 0:  # If letter is EOS
            #     break
            # if predicted_letters_indices[0] < consts.vocab_start_idx:  # If letter is EOS
            #     break
            if sum([pred >= consts.vocab_start_idx for pred in predicted_letters_indices]) == 0:  # To make tests faster.
                # print([pred for pred in predicted_letters_indices])
                break  # TODO: they don't always end on same letter.

            next_letters = [get_vocab()[predicted] for predicted in predicted_letters_indices]
            for i in range(len(output_passwords)):
                output_passwords[i] += next_letters[i]
            # # input = input_tensor(next_letters)
            # # input = pad_sequence([torch.LongTensor([input_tensor(letter)]) for letter in next_letters], batch_first=True).long()  # It will generate len(batch) passwords each time.
            input = init_batches([output_passwords])[0][0]
            # # input = init_batches([output_passwords])
            # # print(input)
            # # input = input[0][0]
            # print(input)

        # return output_passwords
        return [op[:op.find('\0')] for op in output_passwords]
        # return [str(op + '\0')[:op.find('\0')] for op in output_passwords]

        # outputs = []
        # for op in output_passwords:
        #     if '\0' in op:
        #         op = op[:op.find('\0')]
        #     outputs.append(op)
        # return outputs


def get_first_letters(n, first_letters=None):
    assert n > 0
    if n == 1:
        return [letter for letter in get_vocab()]
    if first_letters is None:
        first_letters = []
    new_first_letters = []
    for fl in first_letters:
        for letter in get_vocab():
            new_first_letters.append(fl + letter)
    return get_first_letters(n - 1, first_letters)


def test(model, test_data, n_tests, number_of_first_letters=1, max_length=128, verbose=True):

    start = time()
    accuracy = 0
    nb_samples = len(test_data)
    batch_size = model.batch_size
    print(batch_size)
    current_batch = []
    remainer = int(n_tests)

    print("Eval data size:", nb_samples, "; number of tests:", n_tests)

    # i = 0
    for i in range(1, n_tests + 1):

        starting_letter = ""
        for _ in range(number_of_first_letters):
            random_index = randint(consts.vocab_start_idx, get_vocab_size() - consts.vocab_start_idx)
            starting_letter += get_vocab()[random_index]

        current_batch.append(starting_letter)

        if i % batch_size == 0:  # All batches for this epoch have been loaded.
            remainer -= batch_size

            predicted_passwords = generate_passwords_batches(model, current_batch, max_length=max_length)
            # print("predicted", predicted_passwords)
            # if verbose:
            #     # print("predicted", predicted_passwords[0])
            #     print("predicted", predicted_passwords)

            for predicted in predicted_passwords:
                if predicted in test_data:
                    accuracy = accuracy + 1
            # if predicted_passwords[0] in test_data:
            #     accuracy = accuracy + 1

            current_batch = []

            if verbose:
                print_progress(total=n_tests, acc=accuracy, start=start, iter=i, size=n_tests)

    accuracy = 100 * accuracy / (nb_samples - remainer)
    print('\nCoverage: ', accuracy, '%')


def test_brute_force(test_data, batch_size=1, max_length=128, print_every=None, verbose=True):
    start = time()
    accuracy = 0
    nb_samples = len(test_data)
    size = nb_samples * batch_size
    if print_every is None:
        print_every = epoch_size

    if verbose:
        print("Compute brute force coverage (create {} batches of size {})".format(nb_samples, batch_size))

    # i = 0
    for i in range(1, size + 1):

        predicted = ""
        for _ in range(max_length):
            random_index = randint(consts.vocab_start_idx, get_vocab_size() - consts.vocab_start_idx)
            if random_index < consts.vocab_start_idx:
                break
            predicted += get_vocab()[random_index]
        if predicted in test_data:
            accuracy = accuracy + 1
        if verbose:
            print_progress(total=size, acc=accuracy, start=start, iter=i, size=size)

    accuracy = 100 * accuracy / nb_samples
    print('\nCoverage: ', accuracy, '%')


def get_best_hyper_parameters_sklearn(train_dataset, validation_dataset, model, hyper_parameters, n_iter_search):
    # It looks like this method only works with sklearn classifier (we are working with Pytorch)

    random_search = RandomizedSearchCV(model, param_distributions=hyper_parameters, n_iter=n_iter_search)
    random_search.fit(train_dataset, validation_dataset)
    print(random_search.cv_results_)
    return random_search.cv_results_


def get_best_hyper_parameters_pytorch(train_dataset, validation_dataset, model, hyper_parameters, n_iter_search):
    pass


def save_model(model, model_name):
    model_name += ".pt"
    model_path = path.join(dirname(abspath(__file__)), consts.models_dir, model_name)
    tools.init_dir(model_path)
    # model_path = consts.models_dir + model_name
    torch.save(model, model_path)
    print('Model saved in: ', model_path)


def load_model(model_name):
    model_name += ".pt"
    # model_path = path.join(dirname(abspath(__file__)), model_name)
    # model_path = model_name
    model_path = path.join(dirname(abspath(__file__)), consts.models_dir, model_name)
    if path.exists(model_path):
        model = torch.load(model_path, map_location=device)
        model.eval().to(device)
        print("model "+model_name+" loaded")
        return model
    else:
        print("model \"" + model_name + "\" doesn't exist")


def get_args():
    parser = ArgumentParser()
    parser.add_argument("-r", "--run", default=consts.default_model_file, type=str, help="Name of the model saved file")
    parser.add_argument(
        "-m", "--model", default=str(path.join(consts.models_dir, consts.default_model_file)), type=str,
        help="Path of the model to save for training or to load for evaluating/testing (eval/test) [path/to/the/model]"
    )
    # parser.add_argument(
    #     '-n', '--n', default=consts.default_n_samples, type=int,
    #     help="number of samples to generate [< 1000]."
    # )
    parser.add_argument(
        "-t", "--train", default=True, type=str,
        help="Train the model (if False, load the model) [default True]"
    )
    parser.add_argument(
        "-p", "--pretrain", default=False, type=str,
        help="Pretrain the model (if --train is False, --pretrain is False) [default False]"
    )
    parser.add_argument("-e", "--eval", default=True, type=str, help="Evaluate the model [default True]")
    parser.add_argument(
        "-b", "--bruteforce", default=False, type=str,
        help="Evaluate brute force method coverage [default False]"
    )
    parser.add_argument("-a", "--random", default=False, type=str, help="Use random train [default False]")
    parser.add_argument("-d", "--debug", default=False, type=str, help="Activate debug code [default False]")
    parser.add_argument("-v", "--verbose", default=True, type=str, help="Print log [default True]")
    parser.add_argument("-s", "--test_set", default=None, type=str, help="Path to dataset for tests")
    return parser.parse_args()


if __name__ == '__main__':
    run_id = time()
    # sys.stdout = open(consts.sdtout_path.format(run_id), 'a')

    args = get_args()
    have_to_train = tools.parse_bools(args.train)
    have_to_eval = tools.parse_bools(args.eval)
    test_set = None if args.test_set is None else tools.read_file(args.test_set)
    have_to_test = False if test_set is None else True
    debug = tools.parse_bools(args.debug)
    have_to_pretrain = tools.parse_bools(args.pretrain) if have_to_train else False
    bruteforce = tools.parse_bools(args.bruteforce)
    verbose = tools.parse_bools(args.verbose)
    random_train = args.random
    model_name = args.model
    # tools.init_dir(model_name)
    print(model_name)

    # have_to_train = True
    # have_to_eval = True
    # have_to_train = False
    # have_to_eval = False
    # verbose = False
    # bruteforce = True
    # debug = True

    # train_set, eval_set = extract_data()
    train_set, eval_set = tools.extract_train_data(), tools.extract_eval_data()
    get_vocab(train_set)  # Init vocab
    print(get_vocab(train_set))

    # create_pretrain_files(train_set)
    # selected_train_data, selected_pretrain_data = tools.extract_selected_train_data(), tools.extract_selected_pretrain_data()
    # # print("OK")
    # # exit(0)

    input_size = get_vocab_size()
    hidden_size = 256
    output_size = get_vocab_size()
    batch_size = 1
    # batch_size = 64
    # batch_size = 128
    n_layers = 1
    bidirectional = False
    dropout_value = 0
    use_softmax = False
    # use_softmax = True
    n_epochs = 1
    n_epochs = 1_000
    n_epochs = 10_000
    n_epochs = 1_000_000
    n_pretrain_epochs = 1_000
    n_tests = 10_000
    # n_epochs = 100
    # n_epochs = 20
    # n_epochs = 256
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.005
    # learning_rate = 0.05
    # learning_rate = 0.1
    print_every = 10
    print_every = None
    print_every = 1
    print_every = 100
    print_every = 1000
    number_of_first_letters = 1
    max_length = 128
    epoch_size = 1

    hyper_parameters = {
        "hidden_size": None,
        "batch_size": None,
        "n_layers": None,
        "bidirectional": None,
        "dropout_value": None,
        "use_softmax": None,
        "n_epochs": None,
        "criterion": None,
        "learning_rate": None
    }
    # print(hyper_parameters)

    if debug:
        print("Debug mode !")
        train_set = train_set[-1000:]
        n_epochs = 1_000
        eval_set = eval_set[:100]

    random_sampler = RandomSampler(train_set, num_samples=n_epochs * epoch_size * batch_size, replacement=True)
    # batch_train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    batch_train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=False, sampler=random_sampler)
    # batch_eval_dataloader = DataLoader(eval_set, batch_size=batch_size, shuffle=True)
    batch_eval_dataloader = DataLoader(eval_set, batch_size=1, shuffle=True)

    # print(len(batch_train_dataloader))
    # for batch in batch_train_dataloader:
    #     print(batch)

    lstm1 = LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        batch_size=batch_size,
        n_layers=n_layers,
        bidirectional=bidirectional,
        dropout_value=dropout_value,
        use_softmax=use_softmax
    )

    print("have_to_train:", have_to_train, "({})".format(type(have_to_train)), "have_to_pretrain:", have_to_pretrain, "({})".format(type(have_to_pretrain)), "have_to_eval:", have_to_eval, "({})".format(type(have_to_eval)))

    if have_to_pretrain:
        pretrain_lstm(
            lstm1,
            n_pretrain_epochs,
            epoch_size,
            criterion,
            learning_rate,
            random=random_train,
            print_every=print_every,
            verbose=verbose
        )

    if have_to_train:
        print("\n\nTRAIN\n")
        model_name = "_-_".join([
            "lstm",
            "hidden_size={}".format(hidden_size),
            "n_layers={}".format(n_layers),
            "bidirectional={}".format(bidirectional),
            "dropout_value={}".format(dropout_value),
            "use_softmax={}".format(use_softmax),
            "batch_size={}".format(batch_size),
            "epoch_size={}".format(len(batch_train_dataloader) // min(n_epochs, len(batch_train_dataloader))),
            "random_train={}".format(random_train),
            "train_size={}".format(str(len(batch_train_dataloader)) + "" if not random_train else str(n_epochs * batch_size)),
            "{}".format(run_id)
        ])
        print("Train model \"{}\"".format(model_name))
        if not random_train:
            train_lstm(
                lstm1,
                batch_train_dataloader,
                n_epochs=n_epochs,
                criterion=criterion,
                learning_rate=learning_rate,
                print_every=print_every,
                verbose=verbose
            )
        else:
            random_train_lstm(
                lstm1,
                batch_train_dataloader,
                n_epochs=n_epochs,
                criterion=criterion,
                learning_rate=learning_rate,
                print_every=print_every,
                # epoch_size=epoch_size,
                verbose=verbose
            )
        save_model(lstm1, model_name)
    else:
        print("Load model \"{}\"".format(model_name))
        lstm1 = load_model(model_name)

    if have_to_eval:
        print("\n\nEVAL\n")
        # test(lstm1, batch_eval_dataloader)
        test(lstm1, eval_set, n_tests, number_of_first_letters, max_length, verbose=verbose)
        # if bruteforce:
        #     test_brute_force(eval_set, batch_size=lstm1.batch_size, max_length=max_length, print_every=print_every, verbose=verbose)

    if have_to_test:
        print("\n\nTEST\n")
        # test_dataloader = DataLoader(test_set, batch_size=lstm1.batch_size, shuffle=True)
        test(lstm1, test_set, n_tests, number_of_first_letters, max_length, verbose=verbose)
        eval_set = test_set

    if bruteforce:
        test_brute_force(eval_set, batch_size=lstm1.batch_size, max_length=max_length, print_every=print_every, verbose=verbose)

# sys.stdout.close()

