#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import codecs
from torch.autograd import Variable
import time
import data

start_time = time.time()

torch.manual_seed(1)

EMBEDDING_DIM = 512
HIDDEN_DIM = 256
BATCH_SIZE = 256
EPOCH_NUM = 25

class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tag_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, dropout=0)
        self.hidden2tag = nn.Linear(hidden_dim, tag_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, BATCH_SIZE, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, BATCH_SIZE, self.hidden_dim)))

    def forward(self, sentence, lengths):
        embeds = self.word_embeddings(sentence)

        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), BATCH_SIZE, -1), self.hidden)
        out = self.hidden2tag(lstm_out[-1])
        tag_scores = F.log_softmax(out, dim=1)
        return tag_scores


train_file = "./data/train.txt"
test_file = "./data/dev.txt"
training_data = data.load_seqs_data(train_file)
test_data = data.load_seqs_data(test_file)
vocab_dic = data.load_vocab_data()
word_to_ix, tag_to_ix, ix_to_tag = data.load_ix_dics()

model = LSTMClassifier(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)

training_dataloader = data.create_dataset(training_data, word_to_ix, tag_to_ix, BATCH_SIZE)
test_dataloader = data.create_dataset(test_data, word_to_ix, tag_to_ix, BATCH_SIZE)
train_loss_ = []
train_acc_ = []
for epoch in range(EPOCH_NUM):

    total_acc = 0.0
    total_loss = 0.0
    total = 0.0
    for inputs, labels, lens, raw_datas in training_dataloader:
        inputs, labels = Variable(inputs), Variable(labels)

        model.zero_grad()

        model.hidden = model.init_hidden()

        tag_score = model(inputs.t(), lens.numpy())

        loss = loss_function(tag_score, labels)

        loss.backward()
        optimizer.step()

        _, predicted = torch.max(tag_score.data, 1)
        total_acc += (predicted.numpy() == labels.data.numpy()).sum()
        total += len(labels)
        total_loss += loss.data.item()

    print("The total acc {0}, the total {1}".format(total_acc, total))
    train_loss_.append(total_loss / total)
    train_acc_.append(total_acc / total)

    print('[Epoch: %3d/%3d] Training Loss: %.3f, Training Acc: %.3f'
          % (epoch, EPOCH_NUM, train_loss_[epoch], train_acc_[epoch]))

total_acc = 0.0
total = 0.0
with codecs.open("test.result",'w','utf-8') as resultfile:
    for test, labels, lens, raw_datas in test_dataloader:
        test = Variable(test)
        model.hidden = model.init_hidden()
        tag_scores = model(test.t(), lens.numpy())
        for score, label, raw_data in zip(tag_scores, labels, raw_datas):
            resultfile.write('{0} {1} {2}\n'.format(ix_to_tag[torch.max(score, 0)[1].item()], ix_to_tag[label.item()], raw_data))

        _, predicted = torch.max(tag_scores.data, 1)
        total_acc += (predicted.numpy() == labels.numpy()).sum()
        total += len(labels)

print('Testing Acc: %.3f, Testing correct num: %.3f, Testing total num: %.3f' % (total_acc/total, total_acc, total))

print("--- %s seconds ---" % (time.time() - start_time))
