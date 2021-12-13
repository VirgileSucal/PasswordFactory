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
        __vocab = "\0" + "".join(list(get_char_ratio(data)[1].keys()))
        # __vocab_size = len(__vocab) + 1  # Include EOS
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
        # # print(input)
        # print(input.size())
        # print(input.view(self.batch_size, 1, -1).size())
        # # input = self.encoder(input.view(self.batch_size, 1, -1))  # TODO: Bug
        # # print(self.encoder(input).size())
        # # print(torch.squeeze(self.encoder(input), 2).size())
        # # input = torch.squeeze(self.encoder(input), 2)  # TODO: Bug
        input = self.encoder(input)

        # # print(input)
        # print(input.size())
        # print(input.size(), (hidden.size(), cell.size()))
        # # output, (hidden, cell) = self.lstm(input.view(self.batch_size, 1, -1), (hidden, cell))
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        # print(output.size(), (hidden.size(), cell.size()))
        # print(output.reshape(self.batch_size, 1, -1).size())
        # # output = self.decoder(output.view(self.batch_size, 1, -1))
        # # output = self.decoder(output.reshape(self.batch_size, -1))
        output = self.decoder(output)
        # print(output.size())
        # # print(output)

        if bool(self.dropout_value):
            output = self.dropout(output)
        if self.use_softmax:
            output = self.softmax(output)
        # print(output.size())
        # print(output)

        # print("forward ok")

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
    # print(batch_size)
    index = randint(0, len(passwords) - batch_size)
    # print(index, batch_size)
    passwords_batch = passwords[index: index + batch_size]
    # print(passwords_batch)
    # print([len(p) for p in passwords_batch])
    input_batch = pad_sequence([input_tensor(password) for password in passwords_batch], batch_first=True).long()
    # print("pad input:", input_batch.size())
    target_batch = pad_sequence([target_tensor(password) for password in passwords_batch], batch_first=True).long()
    # print("pad target:", target_batch.size())
    # print(target_batch)
    return input_batch, target_batch


def init_batches(passwords_batches):
    # assert batch_size > 0
    # # print(batch_size)
    # index = randint(0, len(passwords) - batch_size)
    # # print(index, batch_size)
    # passwords_batch = passwords[index: index + batch_size]

    batches = []

    for passwords_batch in passwords_batches:
        # print(passwords_batch)
        # print([len(p) for p in passwords_batch])
        input_batch = pad_sequence([input_tensor(password) for password in passwords_batch], batch_first=True).long()
        # print("pad input:", input_batch.size())
        target_batch = pad_sequence([target_tensor(password) for password in passwords_batch], batch_first=True).long()
        # print("pad target:", target_batch.size())
        # print(target_batch)
        batches.append((input_batch, target_batch))
    # return input_batch, target_batch
    return batches


def train_lstm_epoch(model, batches, criterion, learning_rate):
    hidden = model.init_h_c()
    cell = model.init_h_c()
    output = None

    model.zero_grad()
    loss = 0

    for input_batch, target_batch in batches:
        output, (hidden, cell) = model(input_batch.to(device), hidden, cell)
        # print("output:", output.size())
        # print("target:", target_batch.size())
        # print(target_batch.dtype)
        # # print(target_batch.type(torch.float64).dtype)
        # print(target_batch.type(torch.FloatTensor).dtype)
        # # print(target_batch.type(torch.float64))
        # print(target_batch.type(torch.FloatTensor))
        # # l = criterion(output.to(device), target_batch.to(device))
        l = criterion(output.to(device), target_batch.type(torch.FloatTensor).to(device))
        # print("l:", type(l))
        loss += l
        # print("loss:", type(loss))

    loss.backward()

    for p in model.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item() / len(batches)


# def train_lstm(lstm, train_data, n_epochs, criterion, learning_rate, print_every):
def train_lstm(lstm, train_dataloader, n_epochs, criterion, learning_rate, print_every):

    # assert train_data is not None
    assert train_dataloader is not None
    assert n_epochs > 0

    start = time()
    all_losses = []
    total_loss = 0
    best_loss = (100, 0)
    # print_every = int(n_epochs * print_rate)
    n_iters = len(train_dataloader)
    epoch_size = n_iters // min(n_epochs, n_iters)
    print(n_iters, epoch_size, n_epochs)
    print_every = epoch_size

    current_epoch_batches = []

    # for iter in range(1, n_epochs + 1):
    # for epoch in range(n_epochs):
    epoch = 0
    iter = 0
    for batch in train_dataloader:
        # print(batch)

        current_epoch_batches.append(batch)
        iter += 1

        if iter % epoch_size == 0:  # All batches for this epoch have been loaded.
            epoch += 1

            # output, loss = train_lstm_epoch(lstm, [random_batch(train_dataloader, batch_size=batch_size)], criterion, learning_rate)
            batches = init_batches(current_epoch_batches)
            # print()
            # print(current_epoch_batches, len(current_epoch_batches))
            # bl = [b[0].size() for b in batches]
            # print(bl, len(bl))
            output, loss = train_lstm_epoch(lstm, batches, criterion, learning_rate)
            # print("loss 2:", type(loss))
            total_loss += loss
            if loss < best_loss[0]:
                best_loss = (loss, iter)
            all_losses.append(loss)

            if iter % print_every == 0:
                print('%s (%d %d%%) %.4f (%.4f)' % (timeSince(start), epoch, epoch / n_epochs * 100, total_loss / iter, loss))

            current_epoch_batches = []

        # output, loss = train_lstm_epoch(lstm, [random_batch(train_dataloader, batch_size=batch_size)], criterion, learning_rate)
        # # print("loss 2:", type(loss))
        # total_loss += loss
        # if loss < best_loss[0]:
        #     best_loss = (loss, iter)
        # all_losses.append(loss)
        #
        # if iter % print_every == 0:
        #     print('%s (%d %d%%) %.4f (%.4f)' % (timeSince(start), epoch, epoch / n_epochs * 100, total_loss / iter, loss))


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

    train_set = train_set[-1000:]
    # # print('train_set: ', len(train_set))
    # # print('eval_set: ', len(eval_set))
    # # print(small_train_set)
    #
    # # train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # # eval_dataloader = DataLoader(eval_set, batch_size=batch_size, shuffle=True)
    # # train_iterator = BucketIterator(
    # #     train_dataloader,
    # #     batch_size,
    # batch_train_dataloader, batch_eval_dataloader = BucketIterator.splits(
    #     (train_set, eval_set),
    #     (batch_size, batch_size),
    #     sort_key=lambda s: len(s),
    #     shuffle=True,
    #     sort=False,
    #     device=device
    # )
    #
    # # print(len(batch_train_dataloader))
    # batch_train_dataloader.create_batches()
    # # for batch in batch_train_dataloader.batches:
    # #     print(batch)
    # #
    # print(type(batch_train_dataloader.batches))
    # epoch_train_dataloader = DataLoader(batch_train_dataloader.batches, batch_size=batch_size, shuffle=True)
    # epoch_eval_dataloader = DataLoader(batch_eval_dataloader.batches, batch_size=batch_size, shuffle=True)

    batch_train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    batch_eval_dataloader = DataLoader(eval_set, batch_size=batch_size, shuffle=True)

    print(len(batch_train_dataloader))
    for batch in batch_train_dataloader:
        print(batch)

    # epoch_train_dataloader = DataLoader(batch_train_dataloader, batch_size=batch_size, shuffle=True)
    # epoch_eval_dataloader = DataLoader(batch_eval_dataloader, batch_size=batch_size, shuffle=True)
    # print(len(epoch_train_dataloader))
    # for batch in enumerate(epoch_train_dataloader):
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

    train_lstm(lstm1, batch_train_dataloader, n_epochs=n_epochs, criterion=criterion, learning_rate=learning_rate, print_every=print_every)
