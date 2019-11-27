from file_util import read_documents, train_nb, classify_documents, accuracy
from pathlib import Path
from math import exp


def main():
    main_folder = Path('venv/naive-bayes/')
    all_docs, all_labels = read_documents(main_folder/'all_sentiment_shuffled')

    split_point = int(0.80*len(all_docs))
    train_docs = all_docs[:split_point]
    train_labels = all_labels[:split_point]
    eval_docs = all_docs[split_point:]
    eval_labels = all_labels[split_point:]

    labels, scores = classify_documents(eval_docs, *train_nb(train_docs, train_labels))

    guess_accuracy, wrong_guess_indices = accuracy(eval_labels, labels)

    total_docs = len(eval_labels)
    wrong_guesses = len(wrong_guess_indices)
    right_guesses = total_docs - wrong_guesses

    print("\nGuessed {} out of {} documents right ({:.0%})".format(right_guesses, total_docs, guess_accuracy))


if __name__ == "__main__":
    main()
