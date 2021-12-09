#! /bin/env python3

from random import randint
from time import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchtext.legacy.data import BucketIterator
from torch.nn.utils.rnn import pad_sequence
from data_mining import get_char_ratio
from nameGeneration import timeSince
from tools import extract_data

from sklearn.model_selection import RandomizedSearchCV
from ray import tune  # https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
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


def get_vocab(data=None):
    global __vocab
    global __vocab_size
    assert not (__vocab is None and data is None)
    if __vocab is None:
        __vocab = "".join(list(get_char_ratio(data)[1].keys()))
        __vocab_size = len(__vocab) + 1  # Include EOS
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

        self.encoder = nn.Embedding(self.input_size, self.hidden_size)
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
        # print(input)
        print(input.size())
        print(input.view(self.batch_size, 1, -1).size())
        input = self.encoder(input.view(self.batch_size, 1, -1))

        # print(input)
        print(input.size())
        output, (hidden, cell) = self.lstm(input.view(self.batch_size, 1, -1), (hidden, cell))
        output = self.decoder(output.view(self.batch_size, 1, -1))

        if bool(self.dropout_value):
            output = self.dropout(output)
        if self.use_softmax:
            output = self.softmax(output)

        return output, (hidden, cell)

    def init_h_c(self):
        return Variable(torch.zeros(
            self.batch_size,
            (1 + int(self.bidirectional)) * self.n_layers,
            self.hidden_size,
            device=device
        ))


def input_tensor(line):
    tensor = torch.zeros(len(line), 1, get_vocab_size())
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][get_vocab().find(letter)] = 1
    return tensor


def target_tensor(line):
    print(line, line[0])
    letter_indexes = [get_vocab().find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(get_vocab_size() - 1)  # EOS
    return torch.LongTensor(letter_indexes)


def random_batch(passwords, batch_size=1):
    assert batch_size > 0
    print(batch_size)
    index = randint(0, len(passwords) - batch_size)
    print(index, batch_size)
    passwords_batch = passwords[index: index + batch_size]
    print(passwords_batch)
    print([len(p) for p in passwords_batch])
    input_batch = torch.stack([input_tensor(password) for password in passwords_batch]).long()
    # TODO: fix different lengths
    print(input_batch.size())
    target_batch = torch.stack([target_tensor(password) for password in passwords_batch]).long()
    return input_batch, target_batch


def train_lstm_epoch(model, input_batch, target_batch, criterion, learning_rate):
    target_batch.unsqueeze_(-1)
    hidden = model.init_h_c()
    cell = model.init_h_c()
    output = None

    model.zero_grad()

    loss = 0

    for i in range(input_batch.size(0)):
        # print(input_batch)
        output, (hidden, cell) = model(input_batch[i].to(device), hidden, cell)
        l = criterion(output.to(device), target_batch[i].to(device))
        loss += l

    loss.backward()

    for p in model.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item() / input_batch.size(0)


def train_lstm(lstm, train_data, n_epochs, criterion, learning_rate):

    assert train_data is not None
    assert n_epochs > 0

    start = time()
    all_losses = []
    total_loss = 0
    best_loss = (100, 0)
    print_every = n_epochs / 100

    for iter in range(1, n_epochs + 1):
        output, loss = train_lstm_epoch(lstm, *random_batch(train_data, batch_size=batch_size), criterion, learning_rate)
        total_loss += loss
        if loss[0] < best_loss:
            best_loss = (loss, iter)
        all_losses.append(loss)

        if iter % print_every == 0:
            print('%s (%d %d%%) %.4f (%.4f)' % (timeSince(start), iter, iter / n_epochs * 100, total_loss / iter, loss))


def get_best_hyper_parameters_sklearn(train_dataset, validation_dataset, model, hyper_parameters, n_iter_search):
    # It looks like this method only works with sklearn classifier (we are working with Pytorch)

    random_search = RandomizedSearchCV(model, param_distributions=hyper_parameters, n_iter=n_iter_search)
    random_search.fit(train_dataset, validation_dataset)
    print(random_search.cv_results_)
    return random_search.cv_results_


def get_best_hyper_parameters_pytorch():
    pass

if __name__ == '__main__':

    hidden_size = 256
    batch_size = 1
    batch_size = 3
    n_layers = 1
    bidirectional = False
    dropout_value = 0
    use_softmax = False
    n_epochs = 1
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.005

    train_set, eval_set = extract_data()
    get_vocab(train_set)  # Init vocab
    print(get_vocab(train_set))

    train_set = train_set[-1000:]
    # print('train_set: ', len(train_set))
    # print('eval_set: ', len(eval_set))
    # print(small_train_set)

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_set, batch_size=batch_size, shuffle=True)
    train_iterator = BucketIterator(
        train_dataloader,
        batch_size,
    # train_iterator, eval_iterator = BucketIterator.splits(
    #     (train_dataloader, eval_dataloader),
    #     (batch_size, batch_size),
        sort_key=lambda s: len(s),
        shuffle=False,
        sort=False,
        device=device
    )

    print(len(train_iterator))
    train_iterator.create_batches()
    for batch in train_iterator.batches:
        print(batch)

    lstm1 = LSTM(
        input_size=get_vocab_size(),
        hidden_size=hidden_size,
        output_size=get_vocab_size(),
        batch_size=batch_size,
        n_layers=n_layers,
        bidirectional=bidirectional,
        dropout_value=dropout_value,
        use_softmax=use_softmax
    )

    train_lstm(lstm1, train_set, n_epochs=n_epochs, criterion=criterion, learning_rate=learning_rate)
