import numpy as np

import dataset
import models


def evaluation(output, target):
    output = np.array(output)
    target = np.array(target)
    true_positive = np.sum(output * target)
    false_positive = np.sum(output * (1 - target))
    false_negative = np.sum((1 - output) * target)
    true_negative = np.sum((1 - output) * (1 - target))
    return true_positive, true_negative, false_positive, false_negative


def main():
    n_folds = 3
    data = dataset.get_dataset("polarity", n_folds=n_folds)
    data.encode()

    for fold in range(n_folds):
        arr = list(range(1, fold + 1)) + list(range(fold + 2, n_folds + 1))
        s = ' fold'.join(['train'] + [str(x) for x in arr])
        print(s + ':')
        train_text, train_label, test_text, test_label = data.get_datasets(
            fold)
        model = models.get_model(
            "naive_bayes", n_corpus=data.n_corpus, n_class=data.n_class)
        model.update(train_text, train_label)

        print("test fold{}:".format(fold))
        output = model.predict(test_text, k=1)

        tp, tn, fp, fn = evaluation(output, test_label)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        print("Precision: {}, Recall: {}, F1: {}, Accuracy: {}.".format(
            precision, recall, f1, accuracy))


if __name__ == "__main__":
    main()
