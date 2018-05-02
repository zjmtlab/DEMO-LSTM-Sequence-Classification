import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import codecs
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import time
import data

start_time = time.time()

torch.manual_seed(1)


EMBEDDING_DIM = 256
HIDDEN_DIM = 128
BATCH_SIZE = 10
EPOCH_NUM = 50

class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tag_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, dropout=1)

        self.hidden2tag = nn.Linear(hidden_dim, tag_size)
        #self.liner2 = nn.Linear(64, tag_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, BATCH_SIZE, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, BATCH_SIZE, self.hidden_dim)))

    def forward(self, sentence, lengths):
        embeds = self.word_embeddings(sentence)

        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), BATCH_SIZE, -1), self.hidden)

        #tag_space = self.hidden2tag(lstm_out[-1])
        out = self.hidden2tag(lstm_out[-1])
        #out = F.relu(out)
        #out = self.liner2(out)
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
optimizer = optim.SGD(model.parameters(), lr=0.01)


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
        total_loss += loss.data[0]

    print("The total acc {0}, the total {1}".format(total_acc, total))
    train_loss_.append(total_loss / total)
    train_acc_.append(total_acc / total)

    print('[Epoch: %3d/%3d] Training Loss: %.3f, Training Acc: %.3f'
          % (epoch, EPOCH_NUM, train_loss_[epoch], train_acc_[epoch]))

'''
# See what the scores are after training
total_acc = 0.0
total = 0.0
for seqs, labels, lens, raw_datas in training_dataloader:
    model.hidden = model.init_hidden()
    tag_scores = model(Variable(seqs).t(), lens.numpy())

    _, predicted = torch.max(tag_scores.data, 1)
    total_acc += (predicted.numpy() == labels.numpy()).sum()
    total += len(labels)
print('Training Acc: %.3f, Training correct num: %.3f, Training total num: %.3f' % (total_acc/total, total_acc, total))
'''

resultfile = codecs.open("test.result",'w','utf-8')
    

total_acc = 0.0
total = 0.0
for test, labels, lens, raw_datas in test_dataloader:
    test = Variable(test)
    model.hidden = model.init_hidden()
    tag_scores = model(test.t(), lens.numpy())
    for score, label, raw_data in zip(tag_scores, labels, raw_datas):
        resultfile.write('{0} {1} {2}\n'.format(ix_to_tag[torch.max(score, 0)[1].data.numpy()[0]], ix_to_tag[label], raw_data))

    _, predicted = torch.max(tag_scores.data, 1)
    total_acc += (predicted.numpy() == labels.numpy()).sum()
    total += len(labels)

print('Testing Acc: %.3f, Testing correct num: %.3f, Testing total num: %.3f' % (total_acc/total, total_acc, total))


resultfile.close()

print("--- %s seconds ---" % (time.time() - start_time))
