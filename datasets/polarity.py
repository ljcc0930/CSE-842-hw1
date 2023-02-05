import os
import numpy as np

import utils


class Polarity:
    def __init__(self, n_folds, data_dir):
        self.dataset_url = "https://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz"
        self.data_dir = data_dir
        self.n_folds = n_folds
        self.n_class = 2

        self.load_data()

    def load_data(self):
        dataset_url = self.dataset_url
        data_dir = self.data_dir
        dataset_compress_name = dataset_url.split('/')[-1]

        utils.ensure_download_data(
            dataset_url, data_dir, dataset_compress_name)

        dataset_dir = os.path.join(data_dir, 'txt_sentoken')

        dirs = [os.path.join(dataset_dir, cls)
                for cls in ['neg', 'pos']]  # 0 = neg, 1 = pos

        self.corpus = []

        self.folds = [[] for x in range(self.n_folds)], [
            [] for x in range(self.n_folds)]

        for label, class_dir in enumerate(dirs):
            data = []
            for file in os.scandir(class_dir):
                if file.is_file():
                    if file.name.endswith(".txt"):
                        doc = utils.load_from_txt(file.path)
                        data.append(doc)
                        self.corpus += utils.concat_lists(*doc)

            cur = 0
            n_data = len(data)
            for idx, (fold0, fold1) in enumerate(zip(*self.folds)):
                nex = cur + (n_data - cur) // (self.n_folds - idx)
                fold0.extend(data[cur: nex])
                fold1.extend([label] * (nex - cur))
                cur = nex

        self.corpus = list(np.unique(self.corpus))
        self.n_corpus = len(self.corpus)
        self.encode_mapping = {s: i for i, s in enumerate(self.corpus)}
        self._is_encoded = False

    def get_datasets(self, fold):
        train_text = utils.concat_lists(
            *self.folds[0][:fold], *self.folds[0][fold + 1:])
        train_label = utils.concat_lists(
            *self.folds[1][:fold], *self.folds[1][fold + 1:])
        test_text = self.folds[0][fold]
        test_label = self.folds[1][fold]
        return train_text, train_label, test_text, test_label

    def encode(self):
        if self._is_encoded:
            return

        for fold in self.folds[0]:
            for doc in fold:
                for line in doc:
                    for idx, word in enumerate(line):
                        line[idx] = self.encode_mapping[word]

        self._is_encoded = True

    def decode(self):
        if not self._is_encoded:
            return

        for fold in self.folds[0]:
            for doc in fold:
                for line in doc:
                    for idx, item in enumerate(line):
                        line[idx] = self.corpus[item]

        self._is_encoded = False
