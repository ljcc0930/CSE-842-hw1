import os
import numpy as np

import arg_parser
import datasets
import models
import train


def train_and_test_fold(model, dataset, fold, args):
    n_folds = args.n_folds
    k = args.k_smooth
    checkpoint = args.checkpoint_path

    train_text, train_label, test_text, test_label = dataset.get_datasets(
        fold)
    kwargs = {}
    if model == "sklearn_naive_bayes":
        kwargs['smooth_k'] = k
    model = models.get_model(model, **kwargs)

    if checkpoint is None or not os.path.exists(checkpoint):
        # training logs
        arr = list(range(1, fold + 1)) + list(range(fold + 2, n_folds + 1))
        s = ' fold'.join(['train'] + [str(x) for x in arr])
        print(s + ':')
        train.train(model, train_text, train_label)
    else:
        model.load(checkpoint)

    # testing logs
    print("test fold{}:".format(fold))
    # testing
    return train.test(model, test_text, test_label)


def main():
    args = arg_parser.parse_args()

    for model in ["sklearn_naive_bayes", "sklearn_svm"]:
        for encode_method in ['bow', 'tfidf']:
            print("----- encode: {} ----- model: {} -----".format(encode_method, model))
            data = datasets.get_dataset(
                "sklearn_polarity", n_folds=args.n_folds, data_dir=args.data_dir)
            data.encode(tfidf=(encode_method == "tfidf"))

            if args.testing_fold is None:
                avg = []
                for fold in range(args.n_folds):
                    avg.append(train_and_test_fold(model, data, fold, args))
                avg = np.mean(avg, axis=0)
                print("{} folds average testing:".format(args.n_folds))
                print(
                    "\tPrecision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, Accuracy: {:.4f}.".format(*avg))
            else:
                train_and_test_fold(data, args.testing_fold, args)


if __name__ == "__main__":
    main()
