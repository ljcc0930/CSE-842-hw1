import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='CSE 842 Homework 1')

    # dataset arguments
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Path to save datasets. (default: ./data)')
    parser.add_argument('--dataset', type=str, default='polarity', choices=["polarity"],
                        help='Dataset for training and testing. (default: polarity)')
    parser.add_argument('--n-folds', type=int, default=3,
                        help='Numbers of folds in testing. (default: 3)')
    parser.add_argument('--testing-fold', type=int, default=None,
                        help='The index of which fold should be in the testing set. (default: Enumerate all fold)')

    # model arguments
    parser.add_argument('--k-smooth', type=float, default=1,
                        help='Value of add-k-smoothing in Naive Bayes. (default: 0)')
    parser.add_argument('--save-dir', type=str, default='./models',
                        help='Path to save models. (default: ./models)')

    return parser.parse_args()
