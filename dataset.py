import os
import itertools

import utils


class Polarity:
    def __init__(self, num_folds, data_dir):
        self.dataset_url = "https://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz"
        self.data_dir = data_dir
        self.num_folds = num_folds

        self.load_data()

    def load_data(self):
        dataset_url = self.dataset_url
        data_dir = self.data_dir
        dataset_compress_name = dataset_url.split('/')[-1]

        utils.ensure_download_data(
            dataset_url, dataset_compress_name, data_dir)

        dataset_dir = os.path.join(data_dir, 'txt_sentoken')

        dirs = [os.path.join(dataset_dir, cls) for cls in ['pos', 'neg']]

        self.folds = [[] for x in range(self.num_folds)], [
            [] for x in range(self.num_folds)]

        for label, class_dir in enumerate(dirs):
            data = []
            for file in os.scandir(class_dir):
                if file.is_file():
                    if file.name.endswith(".txt"):
                        data.append(utils.load_from_txt(file.path))

            cur = 0
            num_data = len(data)
            for idx, (fold0, fold1) in enumerate(zip(*self.folds)):
                nex = cur + (num_data - cur) // (self.num_folds - idx)
                fold0.extend(data[cur: nex])
                fold1.extend([label] * (nex - cur))
                cur = nex

    def get_datasets(self, fold):
        train_text = list(itertools.chain(
            *self.folds[0][:fold], *self.folds[0][fold + 1:]))
        train_label = list(itertools.chain(
            *self.folds[1][:fold], *self.folds[1][fold + 1:]))
        test_text = self.folds[0][fold]
        test_label = self.folds[1][fold]
        return train_text, train_label, test_text, test_label


def get_dataset(dataset, num_folds=3, data_dir='data'):
    if dataset == "polarity":
        return Polarity(num_folds, data_dir)


if __name__ == "__main__":
    n = 3
    ds = get_dataset("polarity", num_folds=n)
    for fold in range(3):
        train_text, train_label, test_text, test_label = ds.get_datasets(fold)
        print(len(train_text))
        print(len(train_label))
        print(len(test_text))
        print(len(test_label))
