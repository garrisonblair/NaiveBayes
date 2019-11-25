from file_util import read_documents, train_nb, score_doc_label
from math import exp


def main():
    all_docs, all_labels = read_documents('all_sentiment_shuffled')

    split_point = int(0.80*len(all_docs))
    train_docs = all_docs[:split_point]
    train_labels = all_labels[:split_point]
    eval_docs = all_docs[split_point:]
    eval_labels = all_labels[split_point:]

    probability = score_doc_label(train_docs[0], 'neg', *train_nb(train_docs, train_labels))
    print(probability)
    # print(exp(probability))


if __name__ == "__main__":
    main()
