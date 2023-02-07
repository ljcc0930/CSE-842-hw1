import os

import sklearn.datasets
import sklearn.model_selection
import sklearn.feature_extraction

import nltk

from .polarity import Polarity
import utils


class SklearnPolarity(Polarity):
    def __init__(self, n_folds, data_dir):
        super().__init__(n_folds, data_dir)

    def load_data(self):
        dataset_url = self.dataset_url
        data_dir = self.data_dir
        dataset_compress_name = dataset_url.split('/')[-1]

        utils.ensure_download_data(
            dataset_url, data_dir, dataset_compress_name)

        dataset_dir = os.path.join(data_dir, 'txt_sentoken')
        self.dataset = sklearn.datasets.load_files(dataset_dir, shuffle=True)

    def get_datasets(self, fold):
        assert self._is_encoded

        train_idx, test_idx = self.folds[fold]

        train_text = self.encoded[train_idx]
        test_text = self.encoded[test_idx]
        train_label = self.dataset.target[train_idx]
        test_label = self.dataset.target[test_idx]

        return train_text, train_label, test_text, test_label

    def encode(self, tfidf=False):
        if self._is_encoded:
            return
        transformer = sklearn.feature_extraction.text.CountVectorizer(
            min_df=3, tokenizer=nltk.word_tokenize)
        self.encoded = transformer.fit_transform(self.dataset.data)

        if tfidf:
            transformer = sklearn.feature_extraction.text.TfidfTransformer()
            self.encoded = transformer.fit_transform(self.encoded)

        kf = sklearn.model_selection.KFold(n_splits=self.n_folds)
        self.folds = list(kf.split(self.encoded, self.dataset.target))

        self._is_encoded = True
