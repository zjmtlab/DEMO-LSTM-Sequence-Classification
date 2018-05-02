import torch
import codecs
from torch.utils.data import DataLoader, Dataset
import re


def load_seqs_data(filepath):
    file = codecs.open(filepath,'r','utf-8')
    lines = [line.strip() for line in file] 
    file.close()

    trainingSequences = []
    for line in lines:
        
        word = line.split()
        assert len(word) >= 2
        seq = word[1:]
        trainingSequences.append((word[0], seq))

    return sorted(trainingSequences, key=lambda sequence: len(sequence[1]))

def load_vocab_data():
    file = codecs.open('./data/vocab.txt', 'r', 'utf-8')
    lines = [line.strip() for line in file] 
    file.close()

    word_dic = {}
    for line in lines:
        word = line.split()
        assert len(word) == 2
        word_dic[word[0]] = word[1]

    return word_dic

def load_ix_dics():
    vocab_dic = load_vocab_data()
     
    word_to_ix = {}
    word_to_ix["<<pad>>"] = 0
    for word, numbers in vocab_dic.items():
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
    
    tag_to_ix = {}
    ix_to_tag = {}
    tag_dic = load_tag_dic()
    for tag, num in tag_dic.items():
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)
            ix_to_tag[len(tag_to_ix)-1] = tag
    
    return word_to_ix, tag_to_ix, ix_to_tag
 
def load_tag_dic():
    file = codecs.open('./data/id2tag.txt','r','utf-8')
    lines = [line.strip() for line in file] 
    file.close()

    tag_dic = {}
    for line in lines:
        tag = line.split()
        assert len(tag) == 2
        tag_dic[tag[0]] = tag[1]

    return tag_dic


def vectorize_data(data, to_ix):
    return [[to_ix[tok] if tok in to_ix else to_ix['UNK'] for tok in seq] for y, seq in data]

def collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    seqs, label, lens, raw_data = zip(*batch)
    pad_seqs = []
    lens = []
    max_len = len(seqs[0])
    for i in range(len(seqs)):
        temp_seq = [0] * max_len
        temp_seq[:len(seqs[i])] = seqs[i]
        pad_seqs.append(temp_seq)
        lens.append(len(seqs[i]))

    pad_seqs_tensor = torch.LongTensor(pad_seqs)
    label_tensor = torch.LongTensor(label)
    lens_tensor = torch.LongTensor(lens)

    return pad_seqs_tensor, label_tensor, lens_tensor, raw_data


def create_dataset(data, word_to_ix, tag_to_ix, bs=4):
    vectorized_seqs = vectorize_data(data, word_to_ix)
    seq_lengths = torch.LongTensor([len(s) for s in vectorized_seqs])
    target_tensor = torch.LongTensor([tag_to_ix[y] for y, _ in data])
    raw_data = [x for _, x in data]
    return DataLoader(MyDataset(vectorized_seqs, target_tensor, seq_lengths, raw_data),
                      batch_size=bs,
                      shuffle=False,
                      collate_fn=collate_fn,
                      drop_last=True,
                      num_workers=0)


class MyDataset(Dataset):
    def __init__(self, sequences, labels, lens, raw_datas):
        self.seqs = sequences
        self.labels = labels
        self.lens = lens
        self.raw_datas = raw_datas

    def __getitem__(self, index):
        seq, target, len, raw_data = self.seqs[index], self.labels[index], self.lens[index], self.raw_datas[index] 
        return seq, target, len, ''.join(raw_data)

    def __len__(self):
        return len(self.seqs)

