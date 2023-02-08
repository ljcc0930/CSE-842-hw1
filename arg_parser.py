import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='CSE 842 Homework 1')

    parser.add_argument('--seed', default=2, type=int,
                        help='Random seed. (default: 2)')

    # dataset arguments
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Path to save datasets. (default: ./data)')
    # parser.add_argument('--dataset', type=str, default='polarity', choices=["polarity"],
    #                     help='Dataset for training and testing. (default: polarity)')
    parser.add_argument('--n-folds', type=int, default=3,
                        help='Numbers of folds in testing. (default: 3)')
    parser.add_argument('--testing-fold', type=int, default=None,
                        help='The index of which fold should be in the testing set. (default: Enumerate all fold)')

    # model arguments
    parser.add_argument('--k-smooth', type=float, default=1,
                        help='Value of add-k-smoothing in Naive Bayes. (default: 1)')
    parser.add_argument('--save-dir', type=str, default='./model',
                        help='Directory to save models. Only for handcraft NB model. (default: ./model)')
    parser.add_argument('--checkpoint-path', type=str, default=None,
                        help='Path to load pre-trained model. (default: None)')

    return parser.parse_args()
