from __future__ import division
from codecs import open
from collections import Counter
from numpy import log


def read_documents(doc_file):
    docs = []
    labels = []
    with open(doc_file, encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            docs.append(words[3:])
            labels.append(words[1])
        return docs, labels


def train_nb(documents, labels):
    neg_freq = Counter()
    pos_freq = Counter()
    for i, label in enumerate(labels):
        if label == 'neg':
            neg_freq.update(documents[i])
        elif label == 'pos':
            pos_freq.update(documents[i])

    neg_count = labels.count('neg')
    pos_count = labels.count('pos')

    return neg_freq, neg_count, pos_freq, pos_count


def score_doc_label(document, label, neg_freq, neg_count, pos_freq, pos_count):
    sigma = 0.5
    probability = 0
    if label == 'neg':
        probability = neg_count / (neg_count + pos_count)
        probability = log(probability, 2)

        for word in document:
            print(probability)
            probability = probability + log(neg_freq[word] + sigma, 2)

    elif label == 'pos':
        probability = pos_count / (neg_count + pos_count)
        probability = log(probability, 2)

        for word in document:
            print(probability)
            probability = probability + log(pos_freq[word] + sigma, 2)

    return probability
