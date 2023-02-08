import os
import re
import numpy as np
import nltk
import random

import utils


class Dataset:
    def __init__(self, n_folds, data_dir):
        self.data_dir = data_dir
        self.n_folds = n_folds
        self.n_class = 2

        self._is_encoded = False

        self.prepair_data()

    def prepair_data(self):
        self.download_data()
        self.load_data()
        self.prepair_folds()
        self.prepair_vocabulary()

    def download_data(self):
        dataset_url = self.dataset_url
        data_dir = self.data_dir
        dataset_compress_name = dataset_url.split('/')[-1]

        utils.ensure_download_data(
            dataset_url, data_dir, dataset_compress_name)

    def load_data(self):
        raise NotImplementedError(
            "{} load data is not implemented!".format(self.__class__))

    def prepair_folds(self):
        self.folds = [[] for x in range(self.n_folds)], [
            [] for x in range(self.n_folds)]
        for label, cls_data in enumerate(self.data):
            random.shuffle(cls_data)
            cur = 0
            n_data = len(cls_data)
            for idx, (fold0, fold1) in enumerate(zip(*self.folds)):
                nex = cur + (n_data - cur) // (self.n_folds - idx)
                fold0.extend(cls_data[cur: nex])
                fold1.extend([label] * (nex - cur))
                cur = nex

    def prepair_vocabulary(self):
        vocabulary = []
        for cls_data in self.data:
            for doc in cls_data:
                vocabulary += doc
        self.vocabulary = list(np.unique(vocabulary))
        self.n_vocabulary = len(self.vocabulary)
        self.encode_mapping = {s: i for i, s in enumerate(self.vocabulary)}
        self._is_encoded = False

    def get_datasets(self, fold):
        assert self._is_encoded

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
                for idx, word in enumerate(doc):
                    doc[idx] = self.encode_mapping[word]

        self._is_encoded = True
