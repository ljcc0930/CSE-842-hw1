import os
import numpy as np

import arg_parser
import datasets
import models
import train
import utils


def train_and_test_fold(dataset, fold, args):
    n_folds = args.n_folds
    k = args.k_smooth
    checkpoint = args.checkpoint_path
    save_dir = args.save_dir

    # get data and model
    train_text, train_label, test_text, test_label = dataset.get_datasets(
        fold)
    model = models.get_model(
        "naive_bayes", n_vocabulary=dataset.n_vocabulary, n_class=dataset.n_class, smooth_k=k)

    if checkpoint is None or not os.path.exists(checkpoint):
        # training logs
        arr = list(range(1, fold + 1)) + list(range(fold + 2, n_folds + 1))
        s = ' fold'.join(['train'] + [str(x) for x in arr])
        print(s + ':')
        # training
        train.train(model, train_text, train_label)
        # model.finalize(5, 2)
        # save model
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "liujia45_fold{}.pkl".format(fold))
        model.save(save_path)
    else:
        model.load(checkpoint)

    # testing logs
    print("test fold{}:".format(fold))
    # testing
    return train.test(model, test_text, test_label)


def main():
    """Handcraft dataset / model is not compatible with sklearn dataset / model."""
    args = arg_parser.parse_args()

    utils.set_seed(args.seed)

    data = datasets.get_dataset(
        "sarcasm", n_folds=args.n_folds, data_dir=args.data_dir, n_grams=args.n_grams)
    data.encode()

    print("----- encode: bow ----- model: naive_bayes -----")
    if args.testing_fold is None:
        avg = []
        for fold in range(args.n_folds):
            avg.append(train_and_test_fold(data, fold, args))
        avg = np.mean(avg, axis=0)
        print("{} folds average testing:".format(args.n_folds))
        print(
            "\tPrecision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, Accuracy: {:.4f}.".format(*avg))
    else:
        train_and_test_fold(data, args.testing_fold, args)


if __name__ == "__main__":
    main()
