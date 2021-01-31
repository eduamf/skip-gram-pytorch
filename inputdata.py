import collections
import os
import random
import re
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

data_index = 0

# chars and groups
nwl = r'\n'
crr = r'\r'
l2qm = r'00ab'
r2qm = r'00bb'
spc = r'\u0020'
qm2 = r'\u0022'
shyphen = r'\u00ad'
ndash = r'\u2013'
mdash = r'\u2014'
mhyphen = r'\u002d'
spc_hyf_spc = spc + mhyphen + spc
spc_dsh_spc = spc + mdash + spc
valid_chars = "[\wáàãâäéèêëíìïóòôöúùüçñ]"

class Options(object):
    def __init__(self, datafile, vocabulary_size):
        self.vocabulary_size = vocabulary_size
        self.save_path = "data"
        self.vocabulary = self.read_data(datafile)
        data_or, self.count, self.vocab_words = self.build_dataset(self.vocabulary,
                                                                   self.vocabulary_size)
        self.train_data = self.subsampling(data_or)
        # self.train_data = data_or

        self.sample_table = self.init_sample_table()

        self.save_vocab()

    def reg_cleaner(self, text):
      text.replace(shyphen,mdash).replace(ndash,mdash).replace(spc_hyf_spc,spc_dsh_spc)
      text.replace(crr,"").replace(l2qm,"").replace(r2qm,"")
      text.replace(", ",spc).replace(qm2,"").replace(";", "").replace(' " ', spc)
      # mdash will be used to split sentence or reduce window
      return text.lower()

    def read_data(self, filename):
        f_in = open(filename, mode="r", encoding="utf8")
        data = f_in.read()
        data = self.reg_cleaner(data)
        sents = data.split("\n")
        sents = [x for x in sents if len(x) > 4]
        return data

    def build_dataset(self, words, n_words):
        """Process raw inputs into a ."""
        count = [['UNK', -1]]
        count.extend(collections.Counter(words).most_common(n_words - 1))
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        data = list()
        unk_count = 0
        for word in words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count += 1
            data.append(index)
        count[0][1] = unk_count
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return data, count, reversed_dictionary

    def save_vocab(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
        with open(os.path.join(self.save_path, "vocab.txt"), "w") as f:
            for i in xrange(len(self.count)):
                vocab_word = self.vocab_words[i]
                f.write("%s %d\n" % (vocab_word, self.count[i][1]))

    def init_sample_table(self):
        count = [ele[1] for ele in self.count]
        pow_frequency = np.array(count) ** 0.75
        power = sum(pow_frequency)
        ratio = pow_frequency / power
        table_size = 1e8
        count = np.round(ratio * table_size)
        sample_table = []
        for idx, x in enumerate(count):
            sample_table += [idx] * int(x)
        return np.array(sample_table)

    def weight_table(self):
        count = [ele[1] for ele in self.count]
        pow_frequency = np.array(count) ** 0.75
        power = sum(pow_frequency)
        ratio = pow_frequency / power
        return np.array(ratio)

    def subsampling(self, data):
        count = [ele[1] for ele in self.count]
        frequency = np.array(count) / sum(count)
        P = dict()
        for idx, x in enumerate(frequency):
            y = (math.sqrt(x / 0.001) + 1) * 0.001 / x
            P[idx] = y
        subsampled_data = list()
        for word in data:
            if random.random() < P[word]:
                subsampled_data.append(word)
        return subsampled_data

    def generate_batch2(self, skip_window, batch_size):
        global data_index
        data = self.train_data
        batch = np.ndarray(shape=(batch_size), dtype=np.int64)
        labels = np.ndarray(shape=(batch_size, 2 * skip_window), dtype=np.int64)
        span = 2 * skip_window + 1  # [ skip_window target skip_window ]
        buffer = collections.deque(maxlen=span)

        if data_index + span > len(data):
            data_index = 0
        buffer.extend(data[data_index:data_index + span])
        data_index += span
        for i in range(batch_size):
            batch[i] = buffer[skip_window]
            targets = [x for x in range(skip_window)] + [x for x in range(skip_window + 1, span)]
            for idj, j in enumerate(targets):
                labels[i, idj] = buffer[j]
            if data_index == len(data):
                buffer.extend(data[:span])
                data_index = span
                self.process = False
            else:
                buffer.append(data[data_index])
                data_index += 1
        # Backtrack a little bit to avoid skipping words in the end of a batch
        data_index = (data_index + len(data) - span) % len(data)
        return batch, labels

    def generate_batch(self, window_size, batch_size, count):
        data = self.train_data
        global data_index
        span = 2 * window_size + 1
        context = np.ndarray(shape=(batch_size, 2 * window_size), dtype=np.int64)
        labels = np.ndarray(shape=(batch_size), dtype=np.int64)
        pos_pair = []

        if data_index + span > len(data):
            data_index = 0
            self.process = False
        buffer = data[data_index:data_index + span]
        pos_u = []
        pos_v = []

        for i in range(batch_size):
            data_index += 1
            context[i, :] = buffer[:window_size] + buffer[window_size + 1:]
            labels[i] = buffer[window_size]
            if data_index + span > len(data):
                buffer[:] = data[:span]
                data_index = 0
                self.process = False
            else:
                buffer = data[data_index:data_index + span]

            for j in range(span - 1):
                pos_u.append(labels[i])
                pos_v.append(context[i, j])
        neg_v = np.random.choice(self.sample_table, size=(batch_size * 2 * window_size, count))
        return np.array(pos_u), np.array(pos_v), [[int(x) for x in y] for y in neg_v]


import csv
from scipy.stats import spearmanr
import math

def cosine_similarity(v1, v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i];
        y = v2[i]
        sumxx += x * x
        sumyy += y * y
        sumxy += x * y
    return sumxy / math.sqrt(sumxx * sumyy)


def scorefunction(embed):
    f = open('./tmp/vocab.txt')
    line = f.readline()
    vocab = []
    wordindex = dict()
    index = 0
    while line:
        word = line.strip().split()[0]
        wordindex[word] = index
        index = index + 1
        line = f.readline()
    f.close()
    ze = []
    with open('./wordsim353/combined.csv') as csvfile:
        filein = csv.reader(csvfile)
        index = 0
        consim = []
        humansim = []
        for eles in filein:
            if index == 0:
                index = 1
                continue
            if (eles[0] not in wordindex) or (eles[1] not in wordindex):
                continue

            word1 = int(wordindex[eles[0]])
            word2 = int(wordindex[eles[1]])
            humansim.append(float(eles[2]))

            value1 = embed[word1]
            value2 = embed[word2]
            index = index + 1
            score = cosine_similarity(value1, value2)
            consim.append(score)

    cor1, pvalue1 = spearmanr(humansim, consim)

    if 1 == 1:
        lines = open('./rw/rw.txt', 'r').readlines()
        index = 0
        consim = []
        humansim = []
        for line in lines:
            eles = line.strip().split()
            if (eles[0] not in wordindex) or (eles[1] not in wordindex):
                continue
            word1 = int(wordindex[eles[0]])
            word2 = int(wordindex[eles[1]])
            humansim.append(float(eles[2]))

            value1 = embed[word1]
            value2 = embed[word2]
            index = index + 1
            score = cosine_similarity(value1, value2)
            consim.append(score)

    cor2, pvalue2 = spearmanr(humansim, consim)

    return cor1, cor2