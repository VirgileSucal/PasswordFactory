#! /bin/env python3
import string
from os import path
from random import randint
from time import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchtext.legacy.data import BucketIterator
from torch.nn.utils.rnn import pad_sequence
from data_mining import get_char_ratio
from nameGeneration import timeSince, progress, max_length, progressPercent, timeSinceStart, getLines
from tools import extract_data
import consts

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

device = torch.device("cpu")

__vocab = None
__vocab_size = None


def get_vocab(data=None):
    global __vocab
    global __vocab_size
    assert not (__vocab is None and data is None)
    if __vocab is None:
        __vocab = "\0" + "".join(list(get_char_ratio(data)[1].keys()))
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

        self.encoder = nn.Embedding(self.input_size, self.hidden_size)  # input_size = vocab size, but entry is index, not one_hot
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout_value,
            batch_first=True
        )
        self.decoder = nn.Linear(self.hidden_size, self.output_size)

        if bool(self.dropout_value):
            self.dropout = nn.Dropout(self.dropout_value)
        if self.use_softmax:
            self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, cell):
        input = self.encoder(input)
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        output = self.decoder(output)

        if bool(self.dropout_value):
            output = self.dropout(output)
        if self.use_softmax:
            output = self.softmax(output)

        return output, (hidden, cell)

    def init_h_c(self):
        return Variable(torch.zeros(
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


def random_batch(passwords, batch_size=1):
    assert batch_size > 0
    index = randint(0, len(passwords) - batch_size)
    passwords_batch = passwords[index: index + batch_size]
    input_batch = pad_sequence([input_tensor(password) for password in passwords_batch], batch_first=True).long()
    target_batch = pad_sequence([target_tensor(password) for password in passwords_batch], batch_first=True).long()
    return input_batch, target_batch


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
        output, (hidden, cell) = model(input_batch.to(device), hidden.to(device), cell.to(device))
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
    print(n_iters, epoch_size, n_epochs)
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
            output, loss = train_lstm_epoch(lstm, batches, criterion, learning_rate)
            total_loss += loss
            if loss < best_loss[0]:
                best_loss = (loss, iter)
            all_losses.append(loss)

            if iter % print_every == 0:
                print('%s (%d %d%%) %.4f (%.4f)' % (timeSince(start), epoch, epoch / n_epochs * 100, total_loss / iter, loss))

            current_epoch_batches = []


def sample(decoder, start_letters='ABC'):
    with torch.no_grad():  # no need to track history in sampling

        hidden = decoder.init_h_c()
        cell = decoder.init_h_c()

        if len(start_letters) > 1:
            print("1")
            for i in range(len(start_letters)):
                input = input_tensor(start_letters[i])
                # print(start_letters[i], ' ', hidden)
                output, (hidden, cell) = decoder(input.to(device), hidden.to(device), cell.to(device))

            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == get_vocab_size() - 1:
                return start_letters

            letter = get_vocab()[topi]
            input = input_tensor(letter)
        else:
            print("2")
            # input = input_tensor(start_letters)
            input = pad_sequence([torch.LongTensor([input_tensor(start_letters)]) for _ in range(3)], batch_first=True).long()
            print(input)
            print(input.size())

        output_name = start_letters

        for i in range(max_length):
            output, (hidden, cell) = decoder(input.to(device), hidden.to(device), cell.to(device))
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == get_vocab_size() - 1:
                break
            else:
                letter = get_vocab()[topi]
                output_name += letter
            input = input_tensor(letter)

        return output_name


def test(model, nb_samples, test_data, percent):

    start = time()
    accuracy = 0
    predicted = "a"
    predicted_current = []
    nb_samples = len(test_data)

    if nb_samples > 0:

        for i in range(1, nb_samples + 1):
            nc = 1  # randint(1, max_length/2 - 1)

            while predicted in predicted_current:
                starting_letters = ""
                for n in range(nc):
                    rc = randint(1, get_vocab_size())
                    starting_letters = starting_letters + get_vocab()[rc]

                predicted = sample(model, starting_letters).lower()

            predicted_current.append(predicted)

            if predicted in test_data:
                accuracy = accuracy + 1

            progress(total=nb_samples, acc=accuracy, start=start, epoch=i, l=len(test_data))

        accuracy = 100 * accuracy / nb_samples

        print('\nAccuracy: ', accuracy, '%')

    else:
        i = 0
        l = len(test_data)
        p = int(percent / 100 * l)
        while accuracy < p:
            # nc = randint(1, int(max_length / 2 - 1))
            nc = 1

            while predicted in predicted_current:
                starting_letters = ""
                for n in range(nc):
                    rc = randint(0, len(string.ascii_uppercase) - 1)
                    starting_letters = starting_letters + string.ascii_uppercase[rc]

                predicted = sample(model, starting_letters).lower()

            predicted_current.append(predicted)

            if predicted in test_data:
                accuracy = accuracy + 1

            i = i + 1
            progressPercent(totalNames=l, start=start, names=accuracy, p=percent, samplesGenerated=i)

        print(percent + ' % of all names (', len(test_data), ') reached in ', i, 'iterations (', timeSinceStart(start),
              ' s)...')


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
    model_path = consts.models_dir + model_name
    torch.save(model, model_path)
    print('Model saved in: ', model_path)


def load_model(model_name):
    model_name += ".pt"
    model_path = consts.models_dir + model_name
    if path.exists(model_path):
        model = torch.load(model_path)
        model.eval().to(device)
        print("model "+model_name+" loaded")
        return model
    else:
        print("model " + model_name + " doesn't exist")


if __name__ == '__main__':

    train_set, eval_set = extract_data()
    print(get_vocab(train_set))
    get_vocab(train_set)  # Init vocab

    input_size = get_vocab_size()
    hidden_size = 256
    # hidden_size = input_size
    output_size = get_vocab_size()
    batch_size = 1
    batch_size = 3
    n_layers = 1
    bidirectional = False
    dropout_value = 0
    use_softmax = False
    use_softmax = True
    n_epochs = 1
    n_epochs = 10
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.005
    print_every = 10
    print_every = None

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
    } # TODO
    # print(hyper_parameters)

    train_set, eval_set = extract_data()
    get_vocab(train_set)  # Init vocab
    print(get_vocab(train_set))

    train_set = train_set[-1000:]

    batch_train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    batch_eval_dataloader = DataLoader(eval_set, batch_size=batch_size, shuffle=True)

    print(len(batch_train_dataloader))
    for batch in batch_train_dataloader:
        print(batch)

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

    save_model(lstm1, "neptune")
    model_test = load_model("neptune")
    model_test1 = load_model("jupiter")
    # model_test1.eval()

    train_lstm(lstm1, batch_train_dataloader, n_epochs=n_epochs, criterion=criterion, learning_rate=learning_rate, print_every=print_every)

    test(lstm1, len(batch_eval_dataloader), batch_eval_dataloader, 1)

