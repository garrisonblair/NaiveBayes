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


def score_doc_label(document, label, neg_freq, neg_count, pos_freq, pos_count, neg_denominator, pos_denominator):
    sigma = 0.5
    probability = 0
    if label == 'neg':
        probability = neg_count / (neg_count + pos_count)
        probability = log(probability)

        for word in document:
            word_probability = (neg_freq[word] + sigma) / neg_denominator
            probability += log(word_probability)

    elif label == 'pos':
        probability = pos_count / (neg_count + pos_count)
        probability = log(probability)

        for word in document:
            word_probability = (pos_freq[word] + sigma) / pos_denominator
            probability += log(word_probability)

    return probability


def classify_nb(document, neg_freq, neg_count, pos_freq, pos_count, neg_denominator, pos_denominator):
    neg_score = score_doc_label(document, 'neg', neg_freq, neg_count, pos_freq, pos_count,
                                neg_denominator, pos_denominator)
    pos_score = score_doc_label(document, 'pos', neg_freq, neg_count, pos_freq, pos_count,
                                neg_denominator, pos_denominator)

    if neg_score > pos_score:
        label = 'neg'
        score = neg_score
    else:
        label = 'pos'
        score = pos_score

    return label, score


def classify_documents(documents, neg_freq, neg_count, pos_freq, pos_count):
    vocab_words = len(list(neg_freq))
    neg_denominator = sum(neg_freq.values()) + vocab_words
    pos_denominator = sum(pos_freq.values()) + vocab_words

    labels = []
    scores = []
    for document in documents:
        label, score = classify_nb(document, neg_freq, neg_count, pos_freq, pos_count, neg_denominator, pos_denominator)
        labels.append(label)
        scores.append(scores)

    return labels, scores


def accuracy(true_labels, guessed_labels):
    right_guesses = 0
    wrong_guess_indices = []
    for i, true_label in enumerate(true_labels):
        if guessed_labels[i] == true_label:
            right_guesses += 1
        else:
            wrong_guess_indices.append(i)

    guess_accuracy = right_guesses / len(true_labels)

    return guess_accuracy, wrong_guess_indices
