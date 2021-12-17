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
from torch.utils.data import DataLoader
from torchtext.legacy.data import BucketIterator
from torch.nn.utils.rnn import pad_sequence
import tools
from data_mining import get_char_ratio
from tools import extract_data, parse_bools
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
        __vocab = "\0\0" + "".join(list(get_char_ratio(data)[1].keys()))  # EOS is index 1, because index is used for initializing one-hots.
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
        tensor[li][get_vocab().find(letter)] = 1.0
        # The last one remains a 0 tensor (EOS).
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


def train_lstm_epoch(model, batches, criterion, learning_rate):
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


def train_lstm(lstm, train_dataloader, n_epochs, criterion, learning_rate, print_every=None):

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
    iter = 0
    for batch in train_dataloader:

        current_epoch_batches.append(batch)
        iter += 1

        if iter % epoch_size == 0:  # All batches for this epoch have been loaded.
            epoch += 1

            batches = init_batches(current_epoch_batches)
            # if len(batches) == 1 and batches[0].size(0) != lstm.init_h_c().size()[1]:
            if batches[0][0].size(0) != lstm.init_h_c().size()[1]:
                continue

            output, loss = train_lstm_epoch(lstm, batches, criterion, learning_rate)
            total_loss += loss
            if loss < best_loss[0]:
                best_loss = (loss, iter)
            all_losses.append(loss)

            if iter % print_every == 0:
                print('%s (%d %d%%) %.4f (%.4f)' % (time_since(start), epoch, iter / n_iters * 100, total_loss / iter, loss))

            current_epoch_batches = []

        # if epoch >= 20:  # TODO: Only for tests (remove it)
        #     break


def random_train_lstm(lstm, train_dataloader, n_epochs, criterion, learning_rate, epoch_size):

    assert train_dataloader is not None
    assert n_epochs > 0

    start = time()
    all_losses = []
    total_loss = 0
    best_loss = (100, 0)
    print("Data size:", len(train_dataloader), "; Epoch size:", epoch_size, "; Epochs:", n_epochs, "Batch size:", lstm.batch_size)

    for epoch in range(n_epochs):
        print("epoch:", epoch)
        batches = random_epoch_mini_batch(train_dataloader, epoch_size)
        if batches[0][0].size(0) != lstm.init_h_c().size()[1]:
            print("continue")
            print(batches[0][0].size())
            print(batches[0][0])
            print(lstm.init_h_c().size()[1])
            continue

        output, loss = train_lstm_epoch(lstm, batches, criterion, learning_rate)
        total_loss += loss
        if loss < best_loss[0]:
            best_loss = (loss, iter)
        all_losses.append(loss)

        print('%s (%d %d%%) %.4f (%.4f)' % (time_since(start), epoch, epoch / n_epochs * 100, total_loss / epoch, loss))


def argmax(float_list):
    max_val = float_list[0]
    max_idx = 0
    for i in range(len(float_list)):
        if float_list[i] > max_val:
            max_val = float_list[i]
            max_idx = i

    return max_idx


def progress(total, acc, start, iter, size):
    bar_len = 50
    filled_len = int(round(bar_len * iter / float(total)))
    percents = round(100.0 * iter / float(total), 1)

    if filled_len == 0:
        bar = '>' * filled_len + ' ' * (bar_len - filled_len)
    else:
        bar = '=' * (filled_len - 1) + '>' + ' ' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s => coverage of %.3f %% (%d) on %s \r' % (bar, percents, ' %', (100 * acc / size), acc, time_since(start)))
    sys.stdout.flush()


def generate_passwords(decoder, start_letter: str, max_length: int = 128):
    with torch.no_grad():  # no need to track history in sampling

        hidden = decoder.init_h_c()
        cell = decoder.init_h_c()

        output_passwords = [start_letter[0] for _ in range(decoder.batch_size)]
        # input = input_tensor(start_letters)
        input = pad_sequence([torch.LongTensor([input_tensor(letter)]) for letter in output_passwords], batch_first=True).long()  # It will generate len(batch) passwords each time.
        # print(input)
        # print(input.size())
        # input = torch.unsqueeze(input, dim=-1)
        # print(input)
        # print(input.size())

        for i in range(max_length):
            output, (hidden, cell) = decoder(input.to(device), hidden.to(device), cell.to(device))
            # print(output)
            predicted_letters_indices = [argmax(letter_one_hot[0]) for letter_one_hot in output]
            # print(predicted_letters_indices)
            # if sum(predicted_letters_indices) == 0:  # If letter is EOS
            #     break
            if predicted_letters_indices[0] <= 1:  # If letter is EOS
                break

            next_letters = [get_vocab()[predicted] for predicted in predicted_letters_indices]
            for i in range(len(output_passwords)):
                output_passwords[i] += next_letters[i]
            # input = input_tensor(next_letters)
            input = pad_sequence([torch.LongTensor([input_tensor(letter)]) for letter in next_letters], batch_first=True).long()  # It will generate len(batch) passwords each time.

        return output_passwords


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


def test(model, test_data, number_of_first_letters=1, max_length=128):

    start = time()
    accuracy = 0
    nb_samples = len(test_data)

    print("Eval data size:", nb_samples)

    # i = 0
    for i in range(1, nb_samples + 1):
        starting_letter = ""
        for _ in range(number_of_first_letters):

            random_index = randint(1, get_vocab_size() - 1)
            starting_letter += get_vocab()[random_index]
        i += 1

        predicted_passwords = generate_passwords(model, starting_letter, max_length=max_length)
        # print("predicted", predicted_passwords)
        print("predicted", predicted_passwords[0])

        # for predicted in predicted_passwords:
        #     if predicted in test_data:
        #         accuracy = accuracy + 1
        if predicted_passwords[0] in test_data:
            accuracy = accuracy + 1

        progress(total=nb_samples, acc=accuracy, start=start, iter=i, size=len(test_data))

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
    parser.add_argument("-r", "--run", default=consts.default_model_file, type=str, help="name of the model saved file")
    parser.add_argument("-m", "--model", default=str(path.join(consts.models_dir, consts.default_model_file)), type=str, help="Path of the model to save for training or to load for evaluating/testing (eval/test) [path/to/the/model]")
    # parser.add_argument('-n', '--n', default=consts.default_n_samples, type=int, help="number of samples to generate [< 1000].")
    parser.add_argument("-t", "--train", default=True, type=str, help="Train the model (if False, load the model) [default True]")
    parser.add_argument("-e", "--eval", default=True, type=str, help="Evaluate the model [default True]")
    parser.add_argument("-d", "--debug", default=False, type=str, help="Activate debug code [default False]")
    return parser.parse_args()


if __name__ == '__main__':
    
    args = get_args()
    have_to_train = tools.parse_bools(args.train)
    # have_to_train = args.train
    have_to_eval = tools.parse_bools(args.eval)
    # have_to_eval = args.eval
    model_name = args.model
    debug = tools.parse_bools(args.debug)
    # debug = args.debug
    # tools.init_dir(model_name)
    print(model_name)

    # have_to_train = True
    # have_to_eval = False

    train_set, eval_set = extract_data()
    print(get_vocab(train_set))
    get_vocab(train_set)  # Init vocab

    input_size = get_vocab_size()
    hidden_size = 256
    # hidden_size = input_size
    output_size = get_vocab_size()
    batch_size = 1
    batch_size = 2
    # batch_size = 3
    # batch_size = 10
    # batch_size = 64
    # batch_size = 128
    n_layers = 1
    bidirectional = False
    dropout_value = 0
    use_softmax = False
    # use_softmax = True
    n_epochs = 1
    n_epochs = 1000
    n_epochs = 10000
    n_epochs = 1000000
    # n_epochs = 100
    # n_epochs = 256
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.005
    # learning_rate = 0.05
    # learning_rate = 0.1
    print_every = 10
    print_every = None
    print_every = 1
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

    train_set, eval_set = extract_data()
    get_vocab(train_set)  # Init vocab
    # print(get_vocab(train_set))

    if debug:
        print("Debug mode !")
        # train_set = train_set[-1000:]
        n_epochs = 1000
        # eval_set = eval_set[:100]

    batch_train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # batch_eval_dataloader = DataLoader(eval_set, batch_size=batch_size, shuffle=True)
    batch_eval_dataloader = DataLoader(eval_set, batch_size=1, shuffle=True)

    # print(len(batch_train_dataloader))
    # for batch in batch_train_dataloader:
    #     print(batch)

    lstm1 = None

    print("have_to_train:", have_to_train, "({})".format(type(have_to_train)), "have_to_eval:", have_to_eval, "({})".format(type(have_to_eval)))
    if have_to_train:
        print("\n\nTRAIN\n")
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
        model_name = "_-_".join([
            "hidden_size={}".format(hidden_size),
            "n_layers={}".format(n_layers),
            "bidirectional={}".format(bidirectional),
            "dropout_value={}".format(dropout_value),
            "use_softmax={}".format(use_softmax),
            "batch_size={}".format(batch_size),
            "epoch_size={}".format(len(batch_train_dataloader) // min(n_epochs, len(batch_train_dataloader))),
            "train_size={}".format(len(train_set)),
            "{}".format(time())
        ])
        print("Train model \"{}\"".format(model_name))
        # train_lstm(
        #     lstm1,
        #     batch_train_dataloader,
        #     n_epochs=n_epochs,
        #     criterion=criterion,
        #     learning_rate=learning_rate,
        #     print_every=print_every
        # )
        random_train_lstm(
            lstm1,
            batch_train_dataloader,
            n_epochs=n_epochs,
            criterion=criterion,
            learning_rate=learning_rate,
            epoch_size=epoch_size
        )
        save_model(lstm1, model_name)
    else:
        print("Load model \"{}\"".format(model_name))
        lstm1 = load_model(model_name)

    if have_to_eval:
        print("\n\nEVAL\n")
        # test(lstm1, batch_eval_dataloader)
        test(lstm1, eval_set, number_of_first_letters, max_length)

